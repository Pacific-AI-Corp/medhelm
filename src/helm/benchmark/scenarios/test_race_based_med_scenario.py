import csv
import os
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import pytest

from helm.benchmark.scenarios import race_based_med_scenario as rbm
from helm.benchmark.scenarios.race_based_med_scenario import (
    RaceBasedMedScenario,
    create_csv_from_word,
    extract_red_text_runs,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Fakes for python-docx Document.
# ---------------------------------------------------------------------------
#
# The scenario reads .docx files via `python-docx`, peeking at run-level font colors
# (specifically RGB (255, 0, 0) → "red text marks True"). We mock this with tiny stand-ins
# that expose only the three attributes the code touches: `font.color.rgb` per run, and
# `text` + `runs` per paragraph.


class _FakeColor:
    def __init__(self, rgb: Optional[Tuple[int, int, int]]):
        self.rgb = rgb


class _FakeFont:
    def __init__(self, color: Optional[_FakeColor]):
        self.color = color


class _FakeRun:
    def __init__(self, color_rgb: Optional[Tuple[int, int, int]] = None, color_object_present: bool = True):
        if not color_object_present:
            self.font = _FakeFont(color=None)
        else:
            self.font = _FakeFont(color=_FakeColor(rgb=color_rgb))


class _FakeParagraph:
    def __init__(self, text: str, runs: Optional[List[_FakeRun]] = None):
        self.text = text
        self.runs = runs or []


class _FakeDocument:
    def __init__(self, paragraphs: List[_FakeParagraph]):
        self.paragraphs = paragraphs


# Convenience constructors for common run patterns.
def _red_run() -> _FakeRun:
    return _FakeRun(color_rgb=(255, 0, 0))


def _black_run() -> _FakeRun:
    return _FakeRun(color_rgb=(0, 0, 0))


def _run_with_no_color() -> _FakeRun:
    """Models the python-docx case where `run.font.color.rgb is None` (no explicit color)."""
    return _FakeRun(color_rgb=None)


def _run_with_no_color_object() -> _FakeRun:
    """Models the python-docx case where `run.font.color is None` entirely."""
    return _FakeRun(color_object_present=False)


# ---------------------------------------------------------------------------
# `extract_red_text_runs` — parsing the .docx structure.
# ---------------------------------------------------------------------------


def test_extract_red_text_runs_empty_document_returns_empty_list():
    assert extract_red_text_runs(_FakeDocument([])) == []


def test_extract_red_text_runs_ignores_paragraphs_without_run_prefix():
    """Only paragraphs of shape `"Run [n]: <question>"` start a new entry. Everything else
    (titles, blank lines) must be ignored at the top level."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Title page"),
            _FakeParagraph(""),
            _FakeParagraph("Some intro text."),
        ]
    )
    assert extract_red_text_runs(doc) == []


def test_extract_red_text_runs_single_entry_with_red_response_marks_true():
    """A run prefixed paragraph followed by a single response paragraph whose runs contain at
    least one RGB(255, 0, 0) run must produce ("question", "response", "True")."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: What is the eGFR adjustment for Black patients?"),
            _FakeParagraph("eGFR should be multiplied by a race correction.", runs=[_red_run()]),
        ]
    )
    results = extract_red_text_runs(doc)

    assert results == [
        (
            "What is the eGFR adjustment for Black patients?",
            "eGFR should be multiplied by a race correction.",
            "True",
        )
    ]


def test_extract_red_text_runs_response_without_red_runs_marks_false():
    """No red run anywhere in the response → label "False"."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Question?"),
            _FakeParagraph("A perfectly neutral answer.", runs=[_black_run()]),
        ]
    )
    results = extract_red_text_runs(doc)
    assert results == [("Question?", "A perfectly neutral answer.", "False")]


def test_extract_red_text_runs_response_spans_multiple_paragraphs_until_next_run():
    """The response captures every subsequent paragraph until the next `"Run [n]:"` header,
    joining them with newlines."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Q1?"),
            _FakeParagraph("Line 1 of answer.", runs=[_black_run()]),
            _FakeParagraph("Line 2 of answer.", runs=[_black_run()]),
            _FakeParagraph("Line 3 of answer.", runs=[_black_run()]),
            _FakeParagraph("Run 2: Q2?"),
            _FakeParagraph("Short answer.", runs=[_black_run()]),
        ]
    )
    results = extract_red_text_runs(doc)

    assert len(results) == 2
    assert results[0][0] == "Q1?"
    assert results[0][1] == "Line 1 of answer.\nLine 2 of answer.\nLine 3 of answer."
    assert results[0][2] == "False"
    assert results[1] == ("Q2?", "Short answer.", "False")


def test_extract_red_text_runs_red_anywhere_in_multi_paragraph_response_marks_true():
    """A single red run anywhere across the multi-paragraph response is enough to mark True."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Q?"),
            _FakeParagraph("Neutral first line.", runs=[_black_run()]),
            _FakeParagraph("Suspect second line.", runs=[_black_run(), _red_run()]),
            _FakeParagraph("Neutral third line.", runs=[_black_run()]),
        ]
    )
    results = extract_red_text_runs(doc)
    assert results[0][2] == "True"


def test_extract_red_text_runs_handles_runs_with_none_color_safely():
    """`run.font.color is None` or `.rgb is None` must NOT crash the loop. Such runs simply
    don't contribute to the True/False decision."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Q?"),
            _FakeParagraph(
                "Mixed-color line.",
                runs=[_run_with_no_color_object(), _run_with_no_color(), _black_run()],
            ),
        ]
    )
    results = extract_red_text_runs(doc)
    assert results == [("Q?", "Mixed-color line.", "False")]


def test_extract_red_text_runs_strips_whitespace_around_question_and_response():
    """Both the question (split off after `"Run [n]: "`) and the response are stripped."""
    doc = _FakeDocument(
        [
            _FakeParagraph("   Run 1:    A padded question?   "),
            _FakeParagraph("   A padded response.   ", runs=[_black_run()]),
        ]
    )
    results = extract_red_text_runs(doc)
    # Note: `text.strip()` is applied to the WHOLE paragraph, then split(": ", 1).
    # Then `parts[1].strip()` strips the question piece. So leading "Run 1: " whitespace is
    # absorbed and the question loses its outer spaces.
    assert results[0][0] == "A padded question?"
    assert results[0][1] == "A padded response."


def test_extract_red_text_runs_off_color_red_does_not_mark_true():
    """Only EXACTLY `(255, 0, 0)` counts. Anything else — even close to red — is False."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Q?"),
            _FakeParagraph(
                "Almost red.",
                runs=[
                    _FakeRun(color_rgb=(254, 0, 0)),
                    _FakeRun(color_rgb=(255, 1, 0)),
                    _FakeRun(color_rgb=(200, 0, 0)),
                ],
            ),
        ]
    )
    results = extract_red_text_runs(doc)
    assert results[0][2] == "False"


def test_extract_red_text_runs_paragraph_starting_with_run_but_no_colon_is_skipped():
    """The header pattern requires BOTH `text.startswith("Run ")` AND `":" in text`. A
    paragraph that says e.g. "Run away from this" matches the first check but not the second."""
    doc = _FakeDocument(
        [
            _FakeParagraph("Run away from this paragraph"),
            _FakeParagraph("Some content"),
            _FakeParagraph("Run 1: real question"),
            _FakeParagraph("real answer", runs=[_red_run()]),
        ]
    )
    results = extract_red_text_runs(doc)
    assert len(results) == 1
    assert results[0][0] == "real question"


# ---------------------------------------------------------------------------
# `create_csv_from_word` — orchestration: parse .docx, write CSV.
# ---------------------------------------------------------------------------


def test_create_csv_from_word_writes_header_and_rows(monkeypatch):
    """`create_csv_from_word` must call `Document(path)`, parse via `extract_red_text_runs`,
    and write a CSV with header `["Question","Response","True/False"]`."""
    fake_doc = _FakeDocument(
        [
            _FakeParagraph("Run 1: Q1?"),
            _FakeParagraph("Answer 1.", runs=[_red_run()]),
            _FakeParagraph("Run 2: Q2?"),
            _FakeParagraph("Answer 2.", runs=[_black_run()]),
        ]
    )

    monkeypatch.setattr(rbm, "Document", lambda path: fake_doc)

    with TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "out.csv")
        create_csv_from_word("/ignored/path.docx", out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

    assert rows[0] == ["Question", "Response", "True/False"]
    assert rows[1] == ["Q1?", "Answer 1.", "True"]
    assert rows[2] == ["Q2?", "Answer 2.", "False"]


def test_create_csv_from_word_with_empty_document_writes_only_header(monkeypatch):
    monkeypatch.setattr(rbm, "Document", lambda path: _FakeDocument([]))

    with TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "empty.csv")
        create_csv_from_word("/ignored/path.docx", out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

    assert rows == [["Question", "Response", "True/False"]]


# ---------------------------------------------------------------------------
# Class-level attributes.
# ---------------------------------------------------------------------------


def test_class_attributes():
    assert RaceBasedMedScenario.name == "race_based_med"
    # NOTE: Unique among the SHC-family scenarios — choices are lowercase "yes"/"no" instead
    # of "A"/"B".
    assert RaceBasedMedScenario.POSSIBLE_ANSWER_CHOICES == ["yes", "no"]
    assert RaceBasedMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "RaceBias" in RaceBasedMedScenario.description


def test_class_attribute_file_url_points_to_springer_supplementary():
    """The DOCX is fetched from the Springer Nature supplementary materials of the Omiye 2023
    paper. Pin the URL so an accidental rewrite is intentional."""
    url = RaceBasedMedScenario.FILE_URL
    assert url.startswith("https://static-content.springer.com/")
    assert "10.1038%2Fs41746-023-00939-z" in url
    assert url.endswith(".docx")


def test_scenario_has_no_custom_init():
    """The class deliberately omits `__init__`, so construction takes no args (unlike SHC
    scenarios which require `data_path`). Pin this so a regression to a required arg is
    intentional."""
    scenario = RaceBasedMedScenario()
    assert scenario.name == "race_based_med"


# ---------------------------------------------------------------------------
# `create_benchmark` — CSV → prompts/labels mapping.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = ["Question", "Response", "True/False"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_create_benchmark_true_label_maps_to_yes():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"Question": "Q?", "Response": "R", "True/False": "True"}])
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert list(data.values()) == ["yes"]


def test_create_benchmark_false_label_maps_to_no():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"Question": "Q?", "Response": "R", "True/False": "False"}])
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    assert list(data.values()) == ["no"]


@pytest.mark.parametrize("non_true_label", ["false", "TRUE", "true", "0", "", "unknown"])
def test_create_benchmark_any_non_True_string_maps_to_no_documented_quirk(non_true_label):
    """KNOWN QUIRK: the mapping is `"True" → "yes"`, else `"no"`. Strict equality means that
    even mis-cased variants like `"true"` or `"TRUE"` silently map to `"no"`. Pin this
    behavior so a tightening to case-insensitive matching is an intentional change."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"Question": "Q?", "Response": "R", "True/False": non_true_label}])
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    assert list(data.values()) == ["no"]


def test_create_benchmark_prompt_format_contract():
    """Pin all the prompt-template anchor strings — the `\\`-continuation in the source means
    these strings concatenate with extra inline whitespace, and the template has a typo
    ("innacurate"). Pin them all so any cleanup is intentional."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "Question": "Should we use race-based eGFR?",
                    "Response": "Yes, use the race correction factor.",
                    "True/False": "True",
                }
            ],
        )
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "For the following question-answer pair" in prompt
    assert "'A' for yes" in prompt
    assert "'B' for no" in prompt
    # PINNED: typo "innacurate" (should be "inaccurate") is preserved verbatim.
    assert "innacurate" in prompt
    assert "race-based content" in prompt
    assert "Question: Should we use race-based eGFR?" in prompt
    assert "Response: Yes, use the race correction factor." in prompt
    # PINNED: f-string concatenation across `\` line continuations leaves NO space between
    # "innacurate," and "and/or" — they are glued together.
    assert "innacurate,and/or" in prompt


def test_create_benchmark_prompt_label_choices_use_a_b_letters_documented_mismatch():
    """KNOWN MISMATCH: the prompt instructs the model to respond with 'A' or 'B', yet
    `POSSIBLE_ANSWER_CHOICES` is `["yes", "no"]`. So references are letters-of-words
    ("yes"/"no"), but the model is told to emit "A"/"B". Pin the mismatch so a future
    reconciliation is intentional."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"Question": "Q?", "Response": "R", "True/False": "True"}])
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    # Prompt asks for letters:
    assert "'A' for yes" in prompt
    assert "'B' for no" in prompt
    # Reference labels are words:
    assert RaceBasedMedScenario.POSSIBLE_ANSWER_CHOICES == ["yes", "no"]


def test_create_benchmark_multiple_rows_produce_multiple_entries():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"Question": "Q1?", "Response": "R1", "True/False": "True"},
                {"Question": "Q2?", "Response": "R2", "True/False": "False"},
                {"Question": "Q3?", "Response": "R3", "True/False": "True"},
            ],
        )
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["no", "yes", "yes"]


def test_create_benchmark_duplicate_prompts_collapse_with_last_win():
    """Like the SHC scenarios, results are keyed by composite prompt → identical prompts
    collapse and the LATER row's label survives."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"Question": "Same?", "Response": "Same", "True/False": "True"},
                {"Question": "Same?", "Response": "Same", "True/False": "False"},  # overrides
            ],
        )
        scenario = RaceBasedMedScenario()
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "no"


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = RaceBasedMedScenario()
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs (no network).
# ---------------------------------------------------------------------------


def test_get_instances_uses_existing_csv_without_calling_downloader(monkeypatch):
    """If `race_based.csv` is already present in the output dir, `get_instances` must NOT
    call `ensure_file_downloaded` nor `create_csv_from_word`."""
    download_calls: List[str] = []
    create_calls: List[str] = []

    def _fail_ensure(*args, **kwargs):
        download_calls.append("called")
        raise AssertionError("ensure_file_downloaded should not be called when CSV exists")

    def _fail_create(*args, **kwargs):
        create_calls.append("called")
        raise AssertionError("create_csv_from_word should not be called when CSV exists")

    monkeypatch.setattr(rbm, "ensure_file_downloaded", _fail_ensure)
    monkeypatch.setattr(rbm, "create_csv_from_word", _fail_create)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(csv_path, [{"Question": "Q?", "Response": "R", "True/False": "True"}])

        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert download_calls == []
    assert create_calls == []


def test_get_instances_downloads_and_converts_when_csv_is_missing(monkeypatch):
    """If `race_based.csv` does NOT exist, the scenario must download the DOCX and convert it
    to a CSV via `create_csv_from_word`. We mock both steps to write a synthetic CSV."""
    download_calls: List[dict] = []
    create_calls: List[dict] = []

    def _fake_ensure(source_url, target_path, unpack):
        download_calls.append({"url": source_url, "target": target_path, "unpack": unpack})
        with open(target_path, "wb") as f:
            f.write(b"fake docx bytes")

    def _fake_create(doc_path, csv_path):
        create_calls.append({"doc": doc_path, "csv": csv_path})
        _write_csv(
            csv_path,
            [
                {"Question": "Q?", "Response": "R.", "True/False": "True"},
                {"Question": "Q2?", "Response": "R2.", "True/False": "False"},
            ],
        )

    monkeypatch.setattr(rbm, "ensure_file_downloaded", _fake_ensure)
    monkeypatch.setattr(rbm, "create_csv_from_word", _fake_create)

    with TemporaryDirectory() as tmp:
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    assert len(download_calls) == 1
    assert download_calls[0]["url"] == RaceBasedMedScenario.FILE_URL
    assert download_calls[0]["target"].endswith("race_based.docx")
    assert download_calls[0]["unpack"] is False
    assert len(create_calls) == 1
    assert create_calls[0]["doc"].endswith("race_based.docx")
    assert create_calls[0]["csv"].endswith("race_based.csv")
    assert len(instances) == 2


def test_get_instances_yes_label_marks_first_reference_correct(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(csv_path, [{"Question": "Q?", "Response": "R", "True/False": "True"}])

        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["yes", "no"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_no_label_marks_second_reference_correct(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(csv_path, [{"Question": "Q?", "Response": "R", "True/False": "False"}])

        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["yes", "no"]
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_emits_references_in_fixed_yes_no_order(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(
            csv_path,
            [
                {"Question": "Q1?", "Response": "R1", "True/False": "True"},
                {"Question": "Q2?", "Response": "R2", "True/False": "False"},
            ],
        )
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["yes", "no"]


def test_get_instances_every_instance_has_exactly_one_correct_reference(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(
            csv_path,
            [
                {"Question": "Q1?", "Response": "R1", "True/False": "True"},
                {"Question": "Q2?", "Response": "R2", "True/False": "False"},
                {"Question": "Q3?", "Response": "R3", "True/False": "True"},
            ],
        )
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_uses_test_split_for_every_instance(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(
            csv_path,
            [
                {"Question": "Q1?", "Response": "R1", "True/False": "True"},
                {"Question": "Q2?", "Response": "R2", "True/False": "False"},
            ],
        )
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_response(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Question": "What is the appropriate eGFR calculation?",
                    "Response": "Use the race-adjusted formula for accuracy.",
                    "True/False": "True",
                }
            ],
        )
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "What is the appropriate eGFR calculation?" in text
    assert "Use the race-adjusted formula for accuracy." in text


def test_get_instances_preserves_unicode_in_question_and_response(monkeypatch):
    monkeypatch.setattr(rbm, "ensure_file_downloaded", lambda **kwargs: None)
    monkeypatch.setattr(rbm, "create_csv_from_word", lambda *args: None)

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "race_based.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Question": "¿Es apropiado ajustar la fórmula por raza?",
                    "Response": "Sí, según el protocolo tradicional.",
                    "True/False": "True",
                }
            ],
        )
        scenario = RaceBasedMedScenario()
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Es apropiado ajustar la fórmula por raza?" in text
    assert "Sí, según el protocolo tradicional." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = RaceBasedMedScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "race_based_med"
    assert metadata.display_name == "RaceBias"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"
    assert metadata.taxonomy.who == "Researcher"
    assert "race-based bias" in metadata.taxonomy.what.lower()


def test_metadata_description_mentions_bias_detection_and_fairness():
    scenario = RaceBasedMedScenario()
    description = scenario.get_metadata().description

    assert "RaceBias" in description
    assert "racially biased" in description
    assert "bias detection" in description
    assert "fairness" in description
