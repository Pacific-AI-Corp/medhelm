import json
import os
import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.med_dialog_scenario import MedDialogScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


VALID_SUBSETS = ["healthcaremagic", "icliniq"]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _patch_with_entries(monkeypatch, entries: list, recorded_calls=None):
    """Mock `ensure_file_downloaded` so it writes `{"data": entries}` to the requested path."""

    def _fake(source_url, target_path, **kwargs):
        if recorded_calls is not None:
            recorded_calls.append({"source_url": source_url, "target_path": target_path})
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump({"data": entries}, f)

    monkeypatch.setattr(
        "helm.benchmark.scenarios.med_dialog_scenario.ensure_file_downloaded", _fake
    )


def _entry(src: str = "Patient: ...\nDoctor: ...", tgt: str = "Summary."):
    return {"src": src, "tgt": tgt}


# ---------------------------------------------------------------------------
# Tests for `__init__` validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subset", VALID_SUBSETS)
def test_init_accepts_valid_subsets(subset):
    scenario = MedDialogScenario(subset=subset)

    assert scenario.subset == subset
    assert scenario.name == "med_dialog"


def test_init_rejects_invalid_subset():
    with pytest.raises(AssertionError, match="Invalid subset"):
        MedDialogScenario(subset="unknown-source")


def test_init_rejects_empty_subset():
    with pytest.raises(AssertionError):
        MedDialogScenario(subset="")


# ---------------------------------------------------------------------------
# Mocked end-to-end tests for `get_instances`.
# ---------------------------------------------------------------------------


def test_get_instances_basic_processing(monkeypatch):
    _patch_with_entries(
        monkeypatch,
        [
            _entry(src="Patient: I have a cough.", tgt="Cough complaint."),
            _entry(src="Patient: My knee hurts.", tgt="Knee pain."),
        ],
    )

    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert instances[0].input.text == "Patient: I have a cough."
    assert instances[0].references == [
        Reference(output=Output(text="Cough complaint."), tags=[CORRECT_TAG]),
    ]
    assert instances[1].input.text == "Patient: My knee hurts."


@pytest.mark.parametrize("subset", VALID_SUBSETS)
def test_get_instances_builds_subset_specific_url(monkeypatch, subset):
    """The download URL must include the chosen subset, otherwise the scenario would always
    fetch the same data regardless of the subset argument."""
    recorded: list = []
    _patch_with_entries(monkeypatch, entries=[_entry()], recorded_calls=recorded)

    scenario = MedDialogScenario(subset=subset)
    with TemporaryDirectory() as tmpdir:
        scenario.get_instances(tmpdir)

    assert len(recorded) == 1
    assert f"/blob/{subset}/test.json" in recorded[0]["source_url"]
    assert recorded[0]["target_path"].endswith(os.path.join(subset, "test.json"))


def test_get_instances_returns_empty_when_data_is_empty(monkeypatch):
    _patch_with_entries(monkeypatch, [])

    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


def test_get_instances_preserves_entry_order(monkeypatch):
    _patch_with_entries(
        monkeypatch,
        [_entry(src=f"src-{i}", tgt=f"tgt-{i}") for i in range(5)],
    )

    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert [i.input.text for i in instances] == [f"src-{i}" for i in range(5)]
    assert [i.references[0].output.text for i in instances] == [f"tgt-{i}" for i in range(5)]


def test_get_instances_only_downloads_test_split(monkeypatch):
    """The scenario iterates over ALL_SPLITS but the `if split == "test"` branch guards every
    download. Mock should therefore receive exactly one HTTP request."""
    recorded: list = []
    _patch_with_entries(monkeypatch, [_entry()], recorded_calls=recorded)

    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        scenario.get_instances(tmpdir)

    assert len(recorded) == 1
    assert recorded[0]["target_path"].endswith("test.json")


def test_get_instances_handles_dialogue_special_characters(monkeypatch):
    """Real conversations contain unicode, newlines and quotes; the JSON round-trip must preserve them."""
    dialogue = "Patient: ¿Cómo está? \"I have pain.\"\nDoctor: Take this med ± once.\n• Note 1\n• Note 2"
    summary = "Spanish-English mixed query about pain."
    _patch_with_entries(monkeypatch, [_entry(src=dialogue, tgt=summary)])

    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances[0].input.text == dialogue
    assert instances[0].references[0].output.text == summary


def test_get_instances_writes_subset_specific_subdirectory(monkeypatch):
    """The cache path should be scoped per-subset so two scenarios can co-exist on disk."""
    recorded: list = []
    _patch_with_entries(monkeypatch, [_entry()], recorded_calls=recorded)

    scenario = MedDialogScenario(subset="healthcaremagic")
    with TemporaryDirectory() as tmpdir:
        scenario.get_instances(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "healthcaremagic"))


# ---------------------------------------------------------------------------
# Integration test against the real Codalab bundle.
#
# Only iCliniq is exercised because HealthCareMagic's test split is ~7x larger and the Codalab
# endpoint is slower than GitHub raw URLs.
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_med_dialog_scenario_icliniq_integration():
    scenario = MedDialogScenario(subset="icliniq")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 3108
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(instance.input.text for instance in instances)
    assert all(instance.references[0].output.text for instance in instances)


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MedDialogScenario(subset="icliniq")
    metadata = scenario.get_metadata()

    assert metadata.name == "med_dialog"
    assert metadata.display_name == "MedDialog"
    assert metadata.short_display_name == "MedDialog"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "med_dialog_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MedDialogScenario(subset="healthcaremagic")

    assert scenario.name == "med_dialog"
    assert scenario.subset == "healthcaremagic"
    assert "dialogue" in scenario.tags
    assert "biomedical" in scenario.tags
