import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_cdi_scenario import SHCCDIMedScenario
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
    scenario = SHCCDIMedScenario(data_path="/tmp/cdi.csv")
    assert scenario.data_path == "/tmp/cdi.csv"


def test_class_attributes():
    assert SHCCDIMedScenario.name == "shc_cdi_med"
    assert SHCCDIMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCCDIMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "Clinical Documentation Integrity" in SHCCDIMedScenario.description
    assert "CDI-QA" in SHCCDIMedScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — CSV parsing and CDI-specific prompt wording.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_maps_prompt_context_label():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Was acute renal failure documented per payer guidelines?",
                    "context": "Attestation mentions AKI stage 3 in discharge summary.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCDIMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Was acute renal failure documented per payer guidelines?" in prompt
    assert "Attestation mentions AKI stage 3 in discharge summary." in prompt


def test_create_benchmark_prompt_template_uses_cdi_wording():
    """CDI uses \"either 'A'\" and `context , Answer`; pins divergence from SHC-CONF wording."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "Ctx", "label": "B"}])
        scenario = SHCCDIMedScenario(data_path=path)
        prompt = next(iter(scenario.create_benchmark(path).keys()))

    assert (
        "Provide an answer to the following question: Q? with the following context: Ctx , "
        "Answer the question with either 'A' for yes or 'B' for no." in prompt
    )
    assert "either 'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional details" in prompt
    # SHC-CONF says \"with a 'A'\" without \"either\"; keep regression guard narrow.
    assert "Answer the question with a '" not in prompt


def test_create_benchmark_multiple_rows():
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
        scenario = SHCCDIMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """Keyed by composite prompt; identical rows collapse and the later label wins."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same?", "context": "Same C", "label": "A"},
                {"prompt": "Same?", "context": "Same C", "label": "B"},
            ],
        )
        scenario = SHCCDIMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCCDIMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCCDIMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_empty_benchmark_returns_no_instances():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_references_always_two_choices_in_order():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SHCCDIMedScenario(data_path=path)
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
        scenario = SHCCDIMedScenario(data_path=path)
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
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_contains_prompt_and_context_columns():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Does documentation support MCC capture?",
                    "context": "Progress note lists CHF exacerbation.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Does documentation support MCC capture?" in text
    assert "Progress note lists CHF exacerbation." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCCDIMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "missing.csv")
        scenario = SHCCDIMedScenario(data_path=missing)
        with pytest.raises(FileNotFoundError, match="SHCCDIMedScenario"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts():
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
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_prompt_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿La documentación respalda el diagnóstico?",
                    "context": "Nota: paciente con insuficiencia cardíaca.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCCDIMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿La documentación respalda el diagnóstico?" in text
    assert "Nota: paciente con insuficiencia cardíaca." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCCDIMedScenario(data_path="/tmp/x.csv")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_cdi_med"
    assert metadata.display_name == "CDI-QA"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.what == "Answer verification questions from CDI notes"
    assert metadata.taxonomy.when == "Any"
    assert metadata.taxonomy.who == "Hospital Admistrator"


def test_metadata_description_mentions_cdi():
    scenario = SHCCDIMedScenario(data_path="/tmp/x.csv")
    description = scenario.get_metadata().description
    assert "CDI-QA" in description
    assert "Clinical Documentation Integrity" in description
