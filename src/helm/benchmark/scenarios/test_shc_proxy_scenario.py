import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.shc_proxy_scenario import SHCPROXYMedScenario
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
    scenario = SHCPROXYMedScenario(data_path="/tmp/note.csv")
    assert scenario.data_path == "/tmp/note.csv"


def test_class_attributes():
    assert SHCPROXYMedScenario.name == "shc_proxy_med"
    # NOTE: This scenario uniquely declares 3 answer choices (A/B/C), unlike its SHC-PTBM and
    # SHC-CONF siblings which only have 2 (A/B). Pin this so a future reduction is intentional.
    assert SHCPROXYMedScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B", "C"]
    assert SHCPROXYMedScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "ProxySender" in SHCPROXYMedScenario.description


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
                    "prompt": "Was this message sent by a proxy?",
                    "context": "Parent writes about child's symptoms.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Was this message sent by a proxy?" in prompt
    assert "Parent writes about child's symptoms." in prompt


def test_create_benchmark_prompt_uses_proxy_framing_and_no_adhd_framing():
    """SHC-PROXY's prompt must be about proxy detection — it should mention the proxy/clinical
    message context and NOT include the ADHD framing from SHC-PTBM."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "proxy user" in prompt
    assert "clinical messages" in prompt
    # The ADHD framing must NOT appear (would mean cross-contamination with PTBM).
    assert "attention deficit hyperactivity disorder" not in prompt.lower()
    assert "ADHD" not in prompt


def test_create_benchmark_prompt_instructs_a_or_b_response_documented_discrepancy():
    """KNOWN DISCREPANCY pinned for regression / intentional fix:

    The composite prompt tells the model to "Answer the question with a 'A' for yes or 'B' for
    no", suggesting only two valid outputs. Yet `POSSIBLE_ANSWER_CHOICES` defines THREE valid
    labels (A/B/C). Either the prompt undersells the option space (C is allowed) or `C` is a
    leftover from an earlier 3-class version. Pin the current state so any reconciliation is
    intentional."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "'A' for yes or 'B' for no" in prompt
    # No mention of a third option C in the *prompt itself*, even though POSSIBLE_ANSWER_CHOICES
    # contains C.
    assert "'C'" not in prompt


def test_create_benchmark_multiple_rows_produce_multiple_entries():
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
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "B", "C"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """`create_benchmark` keys results by composite prompt; identical prompts collapse to one
    entry with the LATER label winning."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "C"},  # overrides
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "C"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` is set at module load to allow large clinical
    messages."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "A"}])
        scenario = SHCPROXYMedScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SHCPROXYMedScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,correct_idx",
    [
        ("A", 0),
        ("B", 1),
        ("C", 2),
    ],
)
def test_get_instances_marks_correct_reference_per_label(label, correct_idx):
    """Every label in `POSSIBLE_ANSWER_CHOICES` must mark the corresponding reference correct,
    and ONLY that one."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "Ctx", "label": label}])
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert len(refs) == 3
    for i, ref in enumerate(refs):
        if i == correct_idx:
            assert ref.is_correct, f"reference {i} should be correct for label {label!r}"
        else:
            assert not ref.is_correct, f"reference {i} should NOT be correct for label {label!r}"


def test_get_instances_references_always_have_three_choices_in_fixed_order():
    """Every instance must emit references in the exact order ['A', 'B', 'C']. This is what
    distinguishes SHC-PROXY from its 2-choice SHC-PTBM/SHC-CONF siblings."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "C"},
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["A", "B", "C"]


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
        scenario = SHCPROXYMedScenario(data_path=path)
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
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Is this a proxy message?",
                    "context": "Father writes: my son has a fever.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Is this a proxy message?" in text
    assert "Father writes: my son has a fever." in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "D", "1", "AB", "a", "b", "c"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """Comparison is strict equality against `{"A", "B", "C"}`. Anything outside that set —
    including lowercase forms of the valid letters — must crash."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SHCPROXYMedScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SHCPROXYMedScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    """Two rows with the same composite prompt collapse into a single instance, and the LATEST
    label is retained (because `create_benchmark` uses dict assignment)."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same?", "context": "Same", "label": "A"},
                {"prompt": "Same?", "context": "Same", "label": "C"},  # overrides
                {"prompt": "Different?", "context": "Same", "label": "B"},
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "C"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Patient-portal messages may contain non-ASCII names and accented words."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Es un mensaje de un proxy?",
                    "context": "Padre escribe: mi hijo tiene fiebre.",
                    "label": "A",
                }
            ],
        )
        scenario = SHCPROXYMedScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Es un mensaje de un proxy?" in text
    assert "Padre escribe: mi hijo tiene fiebre." in text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SHCPROXYMedScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "shc_proxy_med"
    assert metadata.display_name == "ProxySender"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"
    # `who` mentions both Clinician and Caregiver.
    assert "Clinician" in metadata.taxonomy.who
    assert "Caregiver" in metadata.taxonomy.who


def test_metadata_description_cites_tse_paper():
    scenario = SHCPROXYMedScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Tse" in description
    assert "doi.org/10.1001/jamapediatrics.2024.4438" in description
