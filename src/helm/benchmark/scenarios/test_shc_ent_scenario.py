import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_ent_scenario import SHCENTMedScenario
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
    scenario = SHCENTMedScenario(data_path="/tmp/ent.csv")
    assert scenario.data_path == "/tmp/ent.csv"


def test_class_attributes():
    assert SHCENTMedScenario.name == "shc_ent_med"
    assert SHCENTMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B", "C"]
    assert SHCENTMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "ENT" in SHCENTMedScenario.description or "Ear, Nose" in SHCENTMedScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — numbering, ENT wording, empty-label skips.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_numbered_prompt_and_three_way_answer():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Does this patient warrant ENT referral?",
                    "context": "Epistaxis documented twice this admission.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert prompt.startswith("1 Provide an answer to the following question:")
    assert "Does this patient warrant ENT referral?" in prompt
    assert "Epistaxis documented twice this admission." in prompt
    assert "'C' for no mention" in prompt
    assert "either 'A' for yes, 'B' for no, or 'C' for no mention" in prompt
    assert "A, B, or C response" in prompt


def test_create_benchmark_prompt_template_pins_ent_instruction():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "Ctx", "label": "B"}])
        scenario = SHCENTMedScenario(data_path=path)
        prompt = next(iter(scenario.create_benchmark(path).keys()))

    expected_segment = (
        "Provide an answer to the following question: Q? with the following context: Ctx , "
        "Answer the question with either 'A' for yes, 'B' for no, or 'C' for no mention. "
        "Do not provide any additional details or response, just a simple A, B, or C response."
    )
    assert expected_segment in prompt


def test_create_benchmark_skips_rows_with_empty_label_and_keeps_counter_linear():
    """Rows with `label == \"\"` are dropped (encoding/fixtures note in scenario); counter advances only for kept rows."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Bad row?", "context": "X", "label": ""},
                {"prompt": "Good?", "context": "Y", "label": "A"},
                {"prompt": "Also?", "context": "Z", "label": "C"},
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 2
    prompts = list(data.keys())
    assert prompts[0].startswith("1 Provide")
    assert prompts[1].startswith("2 Provide")


def test_create_benchmark_only_empty_labels_returns_empty_dict():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": ""},
                {"prompt": "Q2?", "context": "C2", "label": ""},
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


def test_create_benchmark_two_rows_same_question_context_remain_distinct_prompts():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same?", "context": "Same ctx", "label": "A"},
                {"prompt": "Same?", "context": "Same ctx", "label": "B"},
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 2
    assert list(data.keys())[0].startswith("1 Provide")
    assert list(data.keys())[1].startswith("2 Provide")


def test_create_benchmark_handles_csv_with_very_large_field():
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "B"}])
        scenario = SHCENTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCENTMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected_correct_idx",
    [("A", 0), ("B", 1), ("C", 2)],
)
def test_get_instances_labels_mark_correct_reference(label, expected_correct_idx):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": label}])
        scenario = SHCENTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B", "C"]
    for i, ref in enumerate(refs):
        assert ref.is_correct == (i == expected_correct_idx)


def test_get_instances_empty_benchmark_returns_no_instances():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCENTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_only_skipped_rows_returns_no_instances():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": ""}])
        scenario = SHCENTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
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
        scenario = SHCENTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


@pytest.mark.parametrize("bad_label", ["yes", "no", "D", "1", "AB", "a", "b", "c"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCENTMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "missing.csv")
        scenario = SHCENTMedScenario(data_path=missing)
        with pytest.raises(FileNotFoundError, match="SHCENTMedScenario"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_preserves_unicode_in_prompt_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Hay indicación de derivación ORL?",
                    "context": "Otoscopia: perforación timpánica.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCENTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Hay indicación de derivación ORL?" in text
    assert "Otoscopia: perforación timpánica." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCENTMedScenario(data_path="/tmp/x.csv")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_ent_med"
    assert metadata.display_name == "ENT-Referral"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.what == "Identify referrals for ENT specialists"
    assert metadata.taxonomy.when == "Any"
    assert metadata.taxonomy.who == "Hospital Admistrator"


def test_metadata_description_mentions_ent():
    scenario = SHCENTMedScenario(data_path="/tmp/x.csv")
    description = scenario.get_metadata().description
    assert "ENT-Referral" in description
    assert "Ear, Nose, and Throat" in description or "ENT" in description
