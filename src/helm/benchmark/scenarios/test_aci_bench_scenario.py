import json
import os
import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.aci_bench_scenario import ACIBenchScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helper: write a synthetic JSON file in the format expected by `process_json`.
# ---------------------------------------------------------------------------


def _write_json(path: str, entries: list[dict]) -> None:
    """The scenario expects `{"data": [{"src": ..., "tgt": ...}, ...]}`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"data": entries}, f)


# ---------------------------------------------------------------------------
# Integration tests against the real dataset (slow, opt-in).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_aci_bench_scenario_get_instances():
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 187
    assert instances[0].split == TRAIN_SPLIT
    assert instances[0].input.text.startswith("Doctor-patient dialogue:\n\n[doctor] hi , martha . how are you ?")
    assert instances[0].references[0].output.text.startswith("CHIEF COMPLAINT\n\nAnnual exam.")
    assert instances[0].references[0].tags == [CORRECT_TAG]


@pytest.mark.scenarios
def test_aci_bench_scenario_split_distribution():
    """ACI-Bench is one of the few scenarios with both TRAIN and TEST splits; the test set itself
    aggregates three separate source files."""
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    train = [i for i in instances if i.split == TRAIN_SPLIT]
    test = [i for i in instances if i.split == TEST_SPLIT]

    assert len(train) == 67
    assert len(test) == 120
    assert len(train) + len(test) == len(instances)


@pytest.mark.scenarios
def test_aci_bench_scenario_instance_structure():
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert all(instance.split in {TRAIN_SPLIT, TEST_SPLIT} for instance in instances)
    assert all(instance.input.text.startswith("Doctor-patient dialogue:") for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(instance.references[0].tags == [CORRECT_TAG] for instance in instances)
    assert all(instance.references[0].output.text for instance in instances)


# ---------------------------------------------------------------------------
# Unit tests for `process_json` with synthetic data (fast, no network).
# ---------------------------------------------------------------------------


def test_process_json_single_entry():
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tiny.json")
        _write_json(
            path,
            [
                {"src": "[doctor] hello\n[patient] hi", "tgt": "CHIEF COMPLAINT\n\nGreeting."},
            ],
        )
        instances = scenario.process_json(path, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].input.text == "Doctor-patient dialogue:\n\n[doctor] hello\n[patient] hi"
    assert instances[0].references == [
        Reference(output=Output(text="CHIEF COMPLAINT\n\nGreeting."), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == TEST_SPLIT


def test_process_json_multiple_entries():
    scenario = ACIBenchScenario()
    entries = [{"src": f"dialogue {i}", "tgt": f"note {i}"} for i in range(5)]
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "many.json")
        _write_json(path, entries)
        instances = scenario.process_json(path, TRAIN_SPLIT)

    assert len(instances) == 5
    assert all(instance.split == TRAIN_SPLIT for instance in instances)
    for idx, instance in enumerate(instances):
        assert instance.input.text == f"Doctor-patient dialogue:\n\ndialogue {idx}"
        assert instance.references[0].output.text == f"note {idx}"


def test_process_json_empty_data_list():
    """An empty `data` array must return zero instances without crashing."""
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.json")
        _write_json(path, [])
        instances = scenario.process_json(path, TEST_SPLIT)

    assert instances == []


def test_process_json_preserves_special_characters():
    """Multi-line dialogues, unicode and quotes must round-trip through the JSON parser."""
    scenario = ACIBenchScenario()
    dialogue = '[doctor] ¿cómo estás?\n[patient] "fine, thanks"\n[doctor] line 3'
    note = "Assessment: paciente está bien.\n• bullet 1\n• bullet 2"
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "unicode.json")
        _write_json(path, [{"src": dialogue, "tgt": note}])
        instances = scenario.process_json(path, TEST_SPLIT)

    assert instances[0].input.text == f"Doctor-patient dialogue:\n\n{dialogue}"
    assert instances[0].references[0].output.text == note


@pytest.mark.parametrize("split", [TRAIN_SPLIT, TEST_SPLIT])
def test_process_json_propagates_split(split):
    scenario = ACIBenchScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "split.json")
        _write_json(path, [{"src": "d", "tgt": "n"}])
        instances = scenario.process_json(path, split)

    assert instances[0].split == split


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = ACIBenchScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "aci_bench"
    assert metadata.display_name == "ACI-Bench"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "aci_bench_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = ACIBenchScenario()

    assert scenario.name == "aci_bench"
    assert scenario.tags == ["summarization", "medicine"]
    assert ACIBenchScenario.TRAIN_URL.endswith("train_full.json")
    assert len(ACIBenchScenario.TEST_URLS) == 3
    assert all(url.endswith(".json") for url in ACIBenchScenario.TEST_URLS)
    # The git commit hash pins the dataset version; surface it in the URLs.
    assert "e75b383172195414a7a68843ec4876e83e5409f7" in ACIBenchScenario.PREFIX
    assert all(ACIBenchScenario.PREFIX in url for url in ACIBenchScenario.TEST_URLS)
