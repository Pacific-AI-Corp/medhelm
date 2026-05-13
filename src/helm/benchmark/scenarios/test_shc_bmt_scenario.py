import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_bmt_scenario import SHCBMTMedScenario
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
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = SHCBMTMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCBMTMedScenario.name == "shc_bmt_med"
    assert SHCBMTMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCBMTMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    # Class-level description mentions the BMT-Status benchmark and the transplant aliases.
    assert "BMT-Status" in SHCBMTMedScenario.description
    assert "bone marrow transplant" in SHCBMTMedScenario.description


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
                    "prompt": "Did the patient receive a subsequent BMT?",
                    "context": "Patient had HSCT in 2019, then HCT in 2021.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCBMTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Did the patient receive a subsequent BMT?" in prompt
    assert "Patient had HSCT in 2019, then HCT in 2021." in prompt


def test_create_benchmark_prompt_uses_generic_framing_with_no_sibling_keywords():
    """SHC-BMT uses the *generic* binary-question prompt template — identical in shape to
    SHC-CONF. The framings of the three other SHC siblings (ADHD/PTBM, proxy user,
    confidential information) must NOT leak in, and the BMT-specific vocabulary lives in
    `prompt`/`context` (user-supplied), not in the template wrapper."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCBMTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    # Generic-template anchors:
    assert "Provide an answer to the following question" in prompt
    assert "'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional" in prompt
    # No sibling-template framings:
    assert "attention deficit hyperactivity disorder" not in prompt.lower()
    assert "ADHD" not in prompt
    assert "proxy user" not in prompt
    assert "confidential information" not in prompt
    # The wrapper itself should NOT hard-code BMT vocabulary: those words come only from the
    # user-supplied `prompt`/`context` columns. With placeholder content, they must be absent.
    assert "bone marrow" not in prompt.lower()
    assert "HSCT" not in prompt
    assert "transplant" not in prompt.lower()


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
        scenario = SHCBMTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """Dict-keyed storage: identical composite prompts collapse, LATER label wins."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overrides
            ],
        )
        scenario = SHCBMTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` at module load allows multi-MB clinical notes."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCBMTMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCBMTMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCBMTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCBMTMedScenario(data_path=path)
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
        scenario = SHCBMTMedScenario(data_path=path)
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
        scenario = SHCBMTMedScenario(data_path=path)
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
        scenario = SHCBMTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Did the patient receive a subsequent HCT?",
                    "context": "Patient underwent an allogeneic HSCT in March 2023.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCBMTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Did the patient receive a subsequent HCT?" in text
    assert "Patient underwent an allogeneic HSCT in March 2023." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """Strict equality vs `{"A", "B"}`: lowercase, empty, or extra characters must fail."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCBMTMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCBMTMedScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overrides
                {"prompt": "Different?", "context": "Same", "label": "A"},
            ],
        )
        scenario = SHCBMTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same Q?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Recibió el paciente un trasplante posterior?",
                    "context": "Trasplante alogénico realizado en 2022.",
                    "label": "B",
                }
            ],
        )
        scenario = SHCBMTMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Recibió el paciente un trasplante posterior?" in text
    assert "Trasplante alogénico realizado en 2022." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCBMTMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_bmt_med"
    assert metadata.display_name == "BMT-Status"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    # NOTE: This scenario uses lowercase "question answering" — distinct from other scenarios
    # that use "Question answering" (e.g. HeadQA) or "Classification" (e.g. SHC-CONF).
    # Pinned to surface any accidental capitalization shift.
    assert metadata.taxonomy.task == "question answering"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"
    # Distinct `who` value: this scenario targets *Researchers* (no Clinician/Caregiver).
    assert metadata.taxonomy.who == "Researcher"
    assert "bone marrow transplant" in metadata.taxonomy.what.lower()


def test_metadata_description_lists_three_transplant_aliases():
    """The description mentions BMT and its two clinical aliases HSCT and HCT."""
    scenario = SHCBMTMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description

    assert "BMT-Status" in description
    assert "bone marrow transplant" in description
    assert "BMT" in description
    assert "hematopoietic stem cell transplant" in description
    assert "HSCT" in description
    assert "hematopoietic cell transplant" in description
    assert "HCT" in description


def test_metadata_description_has_no_citation_link():
    """Unlike its SHC-PTBM / SHC-CONF / SHC-PRIVACY / SHC-PROXY siblings, BMT-Status has NO
    inline citation link in the description. Pin this so that adding a citation later is an
    intentional change."""
    scenario = SHCBMTMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description

    assert "doi.org" not in description
    assert "jamanetwork.com" not in description
    assert "[(" not in description  # markdown citation pattern
    assert ")](" not in description  # markdown link pattern
