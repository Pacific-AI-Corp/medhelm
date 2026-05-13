import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_sequoia_scenario import SHCSequoiaMedScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = ["question", "context", "label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Class-level / constructor tests.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = SHCSequoiaMedScenario(data_path="/tmp/referrals.csv")
    assert scenario.data_path == "/tmp/referrals.csv"


def test_class_attributes():
    assert SHCSequoiaMedScenario.name == "shc_sequoia_med"
    assert SHCSequoiaMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCSequoiaMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "Sequoia Clinic" in SHCSequoiaMedScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — CSV parsing and numbered prompts.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_numbered_prompt_and_answer():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "question": "Is referral appropriate?",
                    "context": "Patient notes from palliative care.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert prompt.startswith(" 1 Provide an answer to the following question:")
    assert "Is referral appropriate?" in prompt
    assert "Patient notes from palliative care." in prompt
    assert "'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional details" in prompt


def test_create_benchmark_prompt_template_matches_sequoia_scenario():
    """Pin the composite prompt so CSV column wiring (question vs context) cannot drift."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"question": "Q?", "context": "Ctx", "label": "B"}])
        scenario = SHCSequoiaMedScenario(data_path=path)
        prompt = next(iter(scenario.create_benchmark(path).keys()))

    assert "Provide an answer to the following question: Q? with the following context:" in prompt
    assert " Ctx , Answer the question with a 'A' for yes or 'B' for no." in prompt


def test_create_benchmark_counter_increments_per_row():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"question": "Q1?", "context": "C1", "label": "A"},
                {"question": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompts = list(data.keys())
    assert prompts[0].startswith(" 1 ")
    assert prompts[1].startswith(" 2 ")
    assert sorted(data.values()) == ["A", "B"]


def test_create_benchmark_identical_question_and_context_still_distinct_prompts():
    """Unlike dict-keyed prompts without a row index, Sequoia prefixes each row with a counter so
    duplicate rows remain separate instances."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"question": "Same?", "context": "Same ctx", "label": "A"},
                {"question": "Same?", "context": "Same ctx", "label": "B"},
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 2


def test_create_benchmark_handles_csv_with_very_large_field():
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"question": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCSequoiaMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCSequoiaMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"question": "Q?", "context": "C", "label": "A"}])
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"question": "Q?", "context": "C", "label": "B"}])
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_empty_benchmark_returns_no_instances():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"question": "Q1?", "context": "C1", "label": "A"},
                {"question": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_uses_test_split_for_every_instance():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"question": "Q1?", "context": "C1", "label": "A"},
                {"question": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_contains_question_context_and_numbering():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [{"question": "Refer patient?", "context": "Stable on opioids.", "label": "A"}],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert " 1 " in text
    assert "Refer patient?" in text
    assert "Stable on opioids." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"question": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCSequoiaMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "missing.csv")
        scenario = SHCSequoiaMedScenario(data_path=missing)
        with pytest.raises(FileNotFoundError, match="SHCSequoiaMedScenario"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_preserves_unicode_in_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "question": "¿Derivación adecuada?",
                    "context": "Notas de cuidados paliativos.",
                    "label": "B",
                }
            ],
        )
        scenario = SHCSequoiaMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Derivación adecuada?" in text
    assert "Notas de cuidados paliativos." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCSequoiaMedScenario(data_path="/tmp/x.csv")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_sequoia_med"
    assert metadata.display_name == "ClinicReferral"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.what == "Provide answers on clinic referrals"
    assert metadata.taxonomy.when == "Pre-referral"
    assert metadata.taxonomy.who == "Hospital Admistrator"


def test_metadata_description_mentions_sequoia_clinic():
    scenario = SHCSequoiaMedScenario(data_path="/tmp/x.csv")
    description = scenario.get_metadata().description
    assert "Sequoia Clinic" in description
    assert "ClinicReferral" in description
