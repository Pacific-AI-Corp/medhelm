import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_gip_scenario import SHCGIPMedScenario
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
    scenario = SHCGIPMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCGIPMedScenario.name == "shc_gip_med"
    assert SHCGIPMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCGIPMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "HospiceReferral" in SHCGIPMedScenario.description
    assert "hospice" in SHCGIPMedScenario.description


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
                    "prompt": "Is the patient eligible for hospice referral?",
                    "context": "End-stage CHF with poor prognosis.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCGIPMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Is the patient eligible for hospice referral?" in prompt
    assert "End-stage CHF with poor prognosis." in prompt


def test_create_benchmark_prompt_uses_generic_template_with_no_sibling_keywords():
    """SHC-GIP reuses the *generic* binary-question template (same shape as SHC-CONF and
    SHC-BMT). The hospice-specific vocabulary (`hospice`, `palliative`) must come from the
    user-supplied `prompt`/`context` columns — NOT from the template wrapper itself."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCGIPMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    # Generic-template anchors:
    assert "Provide an answer to the following question" in prompt
    assert "'A' for yes or 'B' for no" in prompt
    assert "Do not provide any additional" in prompt
    # No sibling-template framings (canary against cross-contamination):
    assert "attention deficit hyperactivity disorder" not in prompt.lower()
    assert "ADHD" not in prompt
    assert "proxy user" not in prompt
    assert "confidential information" not in prompt
    # And the wrapper does NOT hard-code hospice vocabulary either.
    assert "hospice" not in prompt.lower()
    assert "palliative" not in prompt.lower()
    assert "end-of-life" not in prompt.lower()


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
        scenario = SHCGIPMedScenario(data_path=path)
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
        scenario = SHCGIPMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` at module load allows multi-MB clinical notes."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCGIPMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCGIPMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCGIPMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCGIPMedScenario(data_path=path)
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
        scenario = SHCGIPMedScenario(data_path=path)
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
        scenario = SHCGIPMedScenario(data_path=path)
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
        scenario = SHCGIPMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Should this patient be referred to hospice?",
                    "context": "Stage IV cancer, declining function, comfort-focused goals.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCGIPMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Should this patient be referred to hospice?" in text
    assert "Stage IV cancer, declining function, comfort-focused goals." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """Strict equality vs `{"A", "B"}`: lowercase, empty, or extra characters must fail."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCGIPMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCGIPMedScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    """Two rows with the same composite prompt collapse into a single instance, and the LATEST
    label is retained (dict assignment in `create_benchmark`)."""
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
        scenario = SHCGIPMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same Q?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Palliative-care notes may contain non-ASCII names and accented words."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Debe ser derivado a hospicio?",
                    "context": "Enfermedad terminal, mal pronóstico.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCGIPMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Debe ser derivado a hospicio?" in text
    assert "Enfermedad terminal, mal pronóstico." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCGIPMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_gip_med"
    assert metadata.display_name == "HospiceReferral"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.what == "Assess hospice referral appropriateness"


def test_metadata_when_field_uses_unique_end_of_care_value():
    """UNIQUE: this scenario sets `when="End-of-care"` — distinct from the typical "Any" used
    by the other SHC scenarios and from "Pre-Trial" used by N2C2-CT. Pin this so a normalization
    pass (e.g. to "End-of-life") is intentional."""
    scenario = SHCGIPMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()
    assert metadata.taxonomy.when == "End-of-care"


def test_metadata_who_field_has_typo_documented_bug():
    """KNOWN TYPO pinned for visibility: `who="Hospital Admistrator"` — missing the 'n' (should
    be "Administrator"). Pin so any cleanup is intentional and surfaces as a test failure that
    needs an updated expectation."""
    scenario = SHCGIPMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()
    # Pinned exact value (typo "Admistrator"):
    assert metadata.taxonomy.who == "Hospital Admistrator"
    # And as a guard: the correctly-spelled word is NOT present.
    assert "Administrator" not in metadata.taxonomy.who


def test_metadata_description_mentions_hospice_and_palliative_care():
    scenario = SHCGIPMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description

    assert "HospiceReferral" in description
    assert "hospice care" in description
    assert "palliative care" in description
    assert "end-of-life" in description


def test_metadata_description_has_no_citation_link():
    """Like SHC-BMT (and unlike SHC-PTBM/CONF/PRIVACY/PROXY which cite the Rabbani or Tse
    papers), SHC-GIP has NO inline citation link in the description. Pin so that adding one
    later is intentional."""
    scenario = SHCGIPMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description

    assert "doi.org" not in description
    assert "jamanetwork.com" not in description
    assert "[(" not in description
    assert ")](" not in description
