import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_privacy_scenario import SHCPRIVACYMedScenario
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
    scenario = SHCPRIVACYMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCPRIVACYMedScenario.name == "shc_privacy_med"
    # 2 choices, matching SHC-PTBM / SHC-CONF; distinct from SHC-PROXY (3 choices).
    assert SHCPRIVACYMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SHCPRIVACYMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "PrivacyDetection" in SHCPRIVACYMedScenario.description


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
                    "prompt": "Does the note leak private information?",
                    "context": "Note: patient's full SSN is 123-45-6789.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPRIVACYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Does the note leak private information?" in prompt
    assert "Note: patient's full SSN is 123-45-6789." in prompt


def test_create_benchmark_prompt_uses_privacy_framing_and_no_sibling_framings():
    """SHC-PRIVACY's prompt must be about *confidential information* detection in clinical
    messages — it should NOT contain ADHD framing (SHC-PTBM) nor proxy-user framing
    (SHC-PROXY). This is a canary against cross-contamination from sibling templates."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCPRIVACYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "confidential information was included" in prompt
    assert "clinical messages" in prompt
    assert "'A' for yes or 'B' for no" in prompt
    # ADHD framing (PTBM) must NOT appear.
    assert "attention deficit hyperactivity disorder" not in prompt.lower()
    assert "ADHD" not in prompt
    # Proxy-user framing must NOT appear.
    assert "proxy user" not in prompt


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
        scenario = SHCPRIVACYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """`create_benchmark` keys results by composite prompt; identical prompts collapse to one
    entry with the LATER label winning."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overrides
            ],
        )
        scenario = SHCPRIVACYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` is set at module load to allow large clinical
    messages."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCPRIVACYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCPRIVACYMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCPRIVACYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SHCPRIVACYMedScenario(data_path=path)
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
        scenario = SHCPRIVACYMedScenario(data_path=path)
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
        scenario = SHCPRIVACYMedScenario(data_path=path)
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
        scenario = SHCPRIVACYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Is private info exposed?",
                    "context": "Note: patient DOB and address are mentioned.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPRIVACYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Is private info exposed?" in text
    assert "Note: patient DOB and address are mentioned." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """The assertion `answer in POSSIBLE_ANSWER_CHOICES` enforces an exact `"A"` / `"B"` value.
    Anything else — including lowercase or longer strings — must crash."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCPRIVACYMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCPRIVACYMedScenario(data_path=non_existent)
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
        scenario = SHCPRIVACYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same Q?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Patient-portal messages may contain non-ASCII names and accented words."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Hay información privada?",
                    "context": "Nota: la paciente menciona dirección y teléfono.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPRIVACYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Hay información privada?" in text
    assert "Nota: la paciente menciona dirección y teléfono." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCPRIVACYMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_privacy_med"
    assert metadata.display_name == "PrivacyDetection"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"
    assert "Clinician" in metadata.taxonomy.who
    assert "Caregiver" in metadata.taxonomy.who
    assert "leaks private information" in metadata.taxonomy.what


def test_metadata_description_cites_tse_paper():
    scenario = SHCPRIVACYMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Tse" in description
    assert "doi.org/10.1001/jamapediatrics.2024.4438" in description
