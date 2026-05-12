import json
import os
from tempfile import TemporaryDirectory
from typing import Any, Optional

import pytest

from helm.benchmark.scenarios.med_mcqa_scenario import MedMCQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TRAIN_SPLIT, VALID_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _entry(question="Q?", opa="A", opb="B", opc="C", opd="D", cop=1) -> dict:
    """Build a JSONL row matching the MedMCQA schema."""
    return {
        "question": question,
        "opa": opa,
        "opb": opb,
        "opc": opc,
        "opd": opd,
        "cop": cop,
    }


def _write_jsonl(path: str, rows: list) -> None:
    """The scenario advertises `.json` but the files are actually JSONL (one JSON per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _patch_with_jsonl(monkeypatch, train_rows: Optional[list[Any]] = None, dev_rows: Optional[list[Any]] = None):
    """Mock `ensure_file_downloaded` so it materialises `data/train.json` and `data/dev.json`."""
    train_rows = train_rows or []
    dev_rows = dev_rows or []

    def _fake(source_url, target_path, **kwargs):
        # The scenario passes the *directory* as `target_path` and asks for unzip; emulate the
        # post-unpack on-disk layout instead.
        os.makedirs(target_path, exist_ok=True)
        _write_jsonl(os.path.join(target_path, "train.json"), train_rows)
        _write_jsonl(os.path.join(target_path, "dev.json"), dev_rows)

    monkeypatch.setattr("helm.benchmark.scenarios.med_mcqa_scenario.ensure_file_downloaded", _fake)


# ---------------------------------------------------------------------------
# Mocked end-to-end tests for `get_instances`.
# ---------------------------------------------------------------------------


def test_get_instances_basic_processing(monkeypatch):
    _patch_with_jsonl(
        monkeypatch,
        train_rows=[
            _entry(question="Train Q1", opa="A1", opb="B1", opc="C1", opd="D1", cop=2),
        ],
        dev_rows=[
            _entry(question="Dev Q1", opa="A1", opb="B1", opc="C1", opd="D1", cop=4),
        ],
    )

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    train = [i for i in instances if i.split == TRAIN_SPLIT]
    valid = [i for i in instances if i.split == VALID_SPLIT]
    assert len(train) == 1
    assert len(valid) == 1

    # Train instance: cop=2 → opb is correct.
    assert train[0].input.text == "Train Q1"
    assert train[0].references == [
        Reference(output=Output(text="A1"), tags=[]),
        Reference(output=Output(text="B1"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="C1"), tags=[]),
        Reference(output=Output(text="D1"), tags=[]),
    ]

    # Valid instance: cop=4 → opd is correct.
    assert valid[0].references[3].is_correct
    assert valid[0].references[3].output.text == "D1"


@pytest.mark.parametrize(
    "cop,correct_option_text",
    [
        (1, "opa-value"),
        (2, "opb-value"),
        (3, "opc-value"),
        (4, "opd-value"),
    ],
)
def test_get_instances_marks_correct_option_per_cop(monkeypatch, cop, correct_option_text):
    """The `cop` field is 1-indexed and must select the matching option text from {opa..opd}."""
    _patch_with_jsonl(
        monkeypatch,
        train_rows=[
            _entry(
                opa="opa-value",
                opb="opb-value",
                opc="opc-value",
                opd="opd-value",
                cop=cop,
            )
        ],
    )

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    train = [i for i in instances if i.split == TRAIN_SPLIT]
    correct = [ref for ref in train[0].references if CORRECT_TAG in ref.tags]
    assert len(correct) == 1
    assert correct[0].output.text == correct_option_text


def test_get_instances_preserves_option_order(monkeypatch):
    """References are emitted in opa, opb, opc, opd order regardless of which is correct."""
    _patch_with_jsonl(
        monkeypatch,
        train_rows=[_entry(opa="α", opb="β", opc="γ", opd="δ", cop=3)],
    )

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    texts = [ref.output.text for ref in instances[0].references]
    assert texts == ["α", "β", "γ", "δ"]


def test_get_instances_handles_multiple_jsonl_rows(monkeypatch):
    """Each line of the file is its own row and becomes its own Instance."""
    _patch_with_jsonl(
        monkeypatch,
        train_rows=[_entry(question=f"T{i}", cop=(i % 4) + 1) for i in range(5)],
        dev_rows=[_entry(question=f"V{i}", cop=(i % 4) + 1) for i in range(3)],
    )

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 8
    train_questions = [i.input.text for i in instances if i.split == TRAIN_SPLIT]
    valid_questions = [i.input.text for i in instances if i.split == VALID_SPLIT]
    assert train_questions == ["T0", "T1", "T2", "T3", "T4"]
    assert valid_questions == ["V0", "V1", "V2"]


def test_get_instances_with_empty_splits_returns_no_instances(monkeypatch):
    _patch_with_jsonl(monkeypatch, train_rows=[], dev_rows=[])

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


def test_get_instances_requires_unzip_unpack(monkeypatch):
    """The dataset is shipped as a ZIP archive; the scenario must request `unpack=True` /
    `unpack_type="unzip"` so the post-download tree contains `train.json` and `dev.json`."""
    recorded: list = []

    def _fake(source_url, target_path, **kwargs):
        recorded.append({"source_url": source_url, "target_path": target_path, "kwargs": kwargs})
        os.makedirs(target_path, exist_ok=True)
        _write_jsonl(os.path.join(target_path, "train.json"), [])
        _write_jsonl(os.path.join(target_path, "dev.json"), [])

    monkeypatch.setattr("helm.benchmark.scenarios.med_mcqa_scenario.ensure_file_downloaded", _fake)

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        scenario.get_instances(tmpdir)

    assert len(recorded) == 1
    assert recorded[0]["source_url"] == MedMCQAScenario.DATASET_DOWNLOAD_URL
    assert recorded[0]["kwargs"]["unpack"] is True
    assert recorded[0]["kwargs"]["unpack_type"] == "unzip"


def test_get_instances_reads_dev_json_for_valid_split(monkeypatch):
    """The HuggingFace-style split name is `valid`, but the on-disk filename is `dev.json`.
    A regression that uses `valid.json` would fail to find the file."""
    file_writes: list = []

    def _fake(source_url, target_path, **kwargs):
        os.makedirs(target_path, exist_ok=True)
        # Only write `dev.json` — if the scenario expected `valid.json`, it would crash with
        # FileNotFoundError.
        _write_jsonl(os.path.join(target_path, "train.json"), [])
        _write_jsonl(os.path.join(target_path, "dev.json"), [_entry(question="from-dev-file")])
        file_writes.append(os.listdir(target_path))

    monkeypatch.setattr("helm.benchmark.scenarios.med_mcqa_scenario.ensure_file_downloaded", _fake)

    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert any(i.split == VALID_SPLIT and i.input.text == "from-dev-file" for i in instances)
    assert "dev.json" in file_writes[0]


# ---------------------------------------------------------------------------
# Real-network integration test.
# Slow: the dataset is a 55MB ZIP on Google Drive and contains 187k+ rows.
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_med_mcqa_scenario_integration():
    scenario = MedMCQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    train = [i for i in instances if i.split == TRAIN_SPLIT]
    valid = [i for i in instances if i.split == VALID_SPLIT]

    assert len(train) == 182822
    assert len(valid) == 4183
    assert len(instances) == 187005
    # Each instance has exactly 4 references and exactly one is correct.
    assert all(len(instance.references) == 4 for instance in instances)
    assert all(sum(1 for ref in instance.references if CORRECT_TAG in ref.tags) == 1 for instance in instances)


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MedMCQAScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "med_mcqa"
    assert metadata.display_name == "MedMCQA"
    # MedMCQA is one of the few scenarios whose main split is `valid`, not `test`, because the
    # test ground truth is not released to the public.
    assert metadata.main_split == "valid"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.language == "English"


def test_answer_option_to_index_mapping():
    """The mapping pins both order and 1-indexing of options. Both matter for `cop` lookups."""
    assert MedMCQAScenario.ANSWER_OPTION_TO_INDEX == {"opa": 1, "opb": 2, "opc": 3, "opd": 4}
    # Keys must keep insertion order (relied upon by the reference-building loop).
    assert list(MedMCQAScenario.ANSWER_OPTION_TO_INDEX.keys()) == ["opa", "opb", "opc", "opd"]
    # Values are 1-indexed.
    assert min(MedMCQAScenario.ANSWER_OPTION_TO_INDEX.values()) == 1
    assert max(MedMCQAScenario.ANSWER_OPTION_TO_INDEX.values()) == 4


def test_basic_attributes():
    scenario = MedMCQAScenario()

    assert scenario.name == "med_mcqa"
    assert "biomedical" in scenario.tags
    assert "question_answering" in scenario.tags
    assert MedMCQAScenario.DATASET_DOWNLOAD_URL.startswith("https://drive.google.com")
    assert "id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky" in MedMCQAScenario.DATASET_DOWNLOAD_URL
