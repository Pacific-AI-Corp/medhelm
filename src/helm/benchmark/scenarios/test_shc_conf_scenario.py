import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_conf_scenario import SHCCONFMedScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = ["prompt", "context", "label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Class-level / constructor tests.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = SHCCONFMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCCONFMedScenario.name == "shc_conf_med"
    assert SHCCONFMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCCONFMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "MedConfInfo" in SHCCONFMedScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — CSV parsing logic.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_produces_single_entry():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Does the note contain confidential PHI?",
                    "context": "Note: substance use history.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Does the note contain confidential PHI?" in prompt
    assert "Note: substance use history." in prompt


def test_create_benchmark_prompt_uses_generic_phrasing_and_omits_adhd_framing():
    """SHC-CONF is the *generic* sibling of SHC-PTBM: its prompt must NOT mention ADHD
    (which is specific to the PTBM scenario). Pin this so the templates don't drift."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCCONFMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "Provide an answer to the following question" in prompt
    assert "'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional details" in prompt
    # The ADHD-specific framing must NOT appear (would mean cross-contamination with PTBM).
    assert "attention deficit hyperactivity disorder" not in prompt.lower()
    assert "ADHD" not in prompt


def test_create_benchmark_multiple_rows_produce_multiple_entries():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "A"},
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """`create_benchmark` stores results in a dict keyed by the composite prompt. Two rows that
    produce the same prompt collapse into one entry, and the LATER label wins."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overwrites
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` is set at module load to allow huge clinical notes
    in the `context` column. Verify a multi-megabyte field round-trips correctly."""
    huge_context = "x" * (2 * 1024 * 1024)  # 2 MiB
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCCONFMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCCONFMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_references_always_have_two_choices_in_fixed_order():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["A", "B"]


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "A"},
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_uses_test_split_for_every_instance():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Is this PHI?",
                    "context": "Patient confides about substance use.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Is this PHI?" in text
    assert "Patient confides about substance use." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """The assertion `answer in POSSIBLE_ANSWER_CHOICES` enforces an exact `"A"` / `"B"` value.
    Anything else, including lowercase or longer strings, must crash."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCCONFMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCCONFMedScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    """Because `create_benchmark` uses a dict keyed by prompt, two rows that produce identical
    composite prompts collapse into a single instance and only the LATEST label is retained."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same?", "context": "Same", "label": "A"},
                {"prompt": "Same?", "context": "Same", "label": "B"},  # overrides
                {"prompt": "Different?", "context": "Same", "label": "A"},
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    # Only 2 distinct prompts survive; the duplicate keeps the LAST label.
    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Adolescent notes may contain non-ASCII content (names, transliterated terms)."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Información confidencial?",
                    "context": "Nota: el paciente menciona drogadicción.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCONFMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Información confidencial?" in text
    assert "Nota: el paciente menciona drogadicción." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCCONFMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_conf_med"
    assert metadata.display_name == "MedConfInfo"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.who == "Clinician"
    assert metadata.taxonomy.when == "Any"


def test_metadata_description_cites_rabbani_paper():
    """Pin the inline citation link to the JAMA Pediatrics article."""
    scenario = SHCCONFMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Rabbani" in description
    assert "jamanetwork.com" in description
    assert "2814109" in description
