import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_ptbm_scenario import SHCPTBMMedScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict]) -> None:
    """Write rows into a CSV using the exact columns the scenario expects."""
    fieldnames = ["prompt", "context", "label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_csv_in_tmpdir(rows: List[dict]) -> str:
    """Convenience: create a temp dir, write a CSV inside, and return the path. Caller is
    responsible for keeping the TemporaryDirectory alive — preferred pattern is to use the
    `with TemporaryDirectory()` form directly in each test."""
    raise NotImplementedError  # not used directly; here for documentation only


# ---------------------------------------------------------------------------
# Class-level / constructor tests.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = SHCPTBMMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCPTBMMedScenario.name == "shc_ptbm_med"
    assert SHCPTBMMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCPTBMMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "ADHD-Behavior" in SHCPTBMMedScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — pure-ish CSV parsing logic.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_produces_single_entry():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Does the note recommend PTBM?",
                    "context": "Mother reports difficulty.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPTBMMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    # The composite prompt embeds both the question and the context verbatim.
    assert "Does the note recommend PTBM?" in prompt
    assert "Mother reports difficulty." in prompt


def test_create_benchmark_prompt_contains_expected_static_phrases():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCPTBMMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    # Domain framing.
    assert "attention deficit hyperactivity disorder (ADHD)" in prompt
    # Output format instructions.
    assert "'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional details or response" in prompt


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
        scenario = SHCPTBMMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_into_one_entry():
    """`create_benchmark` stores results in a dict keyed by the composite prompt string. If two
    input rows produce the same prompt (same question + context), the *later* row's label wins.

    This is a meaningful behavior: it dedupes accidental duplicates in the source CSV, but it
    also silently overrides labels. Pin this so future de-dedup logic is intentional."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overwrites
            ],
        )
        scenario = SHCPTBMMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """The module sets `csv.field_size_limit(sys.maxsize)` to allow huge clinical notes. Verify
    that a multi-megabyte context column is read without error."""
    huge_context = "x" * (2 * 1024 * 1024)  # 2 MiB
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCPTBMMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt = next(iter(data.keys()))
    assert huge_context in prompt


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCPTBMMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCPTBMMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCPTBMMedScenario(data_path=path)
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
        scenario = SHCPTBMMedScenario(data_path=path)
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
        scenario = SHCPTBMMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_uses_test_split_for_every_instance():
    """The dataset is held out for evaluation only; everything ships as `TEST_SPLIT`."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCPTBMMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Is PTBM recommended?",
                    "context": "Patient John Doe shows hyperactivity.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPTBMMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Is PTBM recommended?" in text
    assert "Patient John Doe shows hyperactivity." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """`get_instances` asserts every label is one of {"A", "B"}. A label outside that set must
    halt processing rather than silently skipping the row."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCPTBMMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    """`check_file_exists` should raise before any CSV parsing is attempted, surfacing the
    misconfiguration to the user."""
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCPTBMMedScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    """Because `create_benchmark` uses a dict keyed by prompt, two CSV rows with the same
    prompt + context collapse into a single instance and only the *latest* label is retained.
    This test pins the current behavior."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same?", "context": "Same", "label": "A"},
                {"prompt": "Same?", "context": "Same", "label": "B"},
                {"prompt": "Different?", "context": "Same", "label": "A"},
            ],
        )
        scenario = SHCPTBMMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    # Only 2 distinct prompts survive; the duplicate "Same?" keeps the LAST label ("B").
    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCPTBMMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_ptbm_med"
    assert metadata.display_name == "ADHD-Behavior"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "During Treatment"


def test_metadata_description_cites_pillai_paper():
    """Pin the inline citation so it doesn't get accidentally stripped on a documentation
    refactor."""
    scenario = SHCPTBMMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Pillai" in description
    assert "doi.org/10.1093/jamia/ocae001" in description
