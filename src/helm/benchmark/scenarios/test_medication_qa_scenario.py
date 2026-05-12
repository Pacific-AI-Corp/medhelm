import os
import pytest
from tempfile import TemporaryDirectory

import pandas as pd

from helm.benchmark.scenarios.medication_qa_scenario import MedicationQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _patch_with_dataframe(monkeypatch, df: pd.DataFrame) -> None:
    """Mock `ensure_file_downloaded` so it writes the supplied DataFrame as the expected xlsx file."""

    def _fake(source_url, target_path, **kwargs):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        df.to_excel(target_path, index=False, engine="openpyxl")

    monkeypatch.setattr(
        "helm.benchmark.scenarios.medication_qa_scenario.ensure_file_downloaded", _fake
    )


# ---------------------------------------------------------------------------
# Integration test against the real MedInfo 2019 spreadsheet (slow, opt-in).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_medication_qa_scenario_get_instances():
    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 689
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(instance.references[0].output.text for instance in instances)


# ---------------------------------------------------------------------------
# Mocked tests for `get_instances` (driven by synthetic Excel files).
# ---------------------------------------------------------------------------


def test_get_instances_basic_dataframe(monkeypatch):
    df = pd.DataFrame(
        {
            "Question": ["What is aspirin used for?", "How to store insulin?"],
            "Answer": ["For pain relief.", "Refrigerate it."],
        }
    )
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert instances[0].input.text == "What is aspirin used for?"
    assert instances[0].references == [
        Reference(output=Output(text="For pain relief."), tags=[CORRECT_TAG]),
    ]
    assert instances[1].input.text == "How to store insulin?"
    assert instances[1].references[0].output.text == "Refrigerate it."


def test_get_instances_filters_rows_with_missing_answers(monkeypatch):
    """Rows whose Answer column is NaN must be dropped before instances are built."""
    df = pd.DataFrame(
        {
            "Question": ["Q1 valid", "Q2 missing answer", "Q3 valid"],
            "Answer": ["A1.", None, "A3."],
        }
    )
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert [i.input.text for i in instances] == ["Q1 valid", "Q3 valid"]
    assert [i.references[0].output.text for i in instances] == ["A1.", "A3."]


def test_get_instances_returns_empty_when_all_answers_missing(monkeypatch):
    df = pd.DataFrame({"Question": ["Q1", "Q2"], "Answer": [None, None]})
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


def test_get_instances_returns_empty_for_empty_spreadsheet(monkeypatch):
    df = pd.DataFrame({"Question": [], "Answer": []})
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


def test_get_instances_preserves_row_order(monkeypatch):
    """Pandas iteration order should match Excel row order, even after NaN filtering."""
    df = pd.DataFrame(
        {
            "Question": [f"Q{i}" for i in range(10)],
            "Answer": [f"A{i}" if i % 2 == 0 else None for i in range(10)],
        }
    )
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert [i.input.text for i in instances] == ["Q0", "Q2", "Q4", "Q6", "Q8"]


def test_get_instances_handles_long_freeform_answers(monkeypatch):
    """Consumer health answers are often paragraph-length and may contain newlines and unicode."""
    long_answer = (
        "Take this medication exactly as prescribed.\n"
        "Common side effects include nausea — please contact your doctor if they persist.\n"
        "Do not exceed the recommended dose."
    )
    df = pd.DataFrame({"Question": ["How should I take X?"], "Answer": [long_answer]})
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances[0].references[0].output.text == long_answer


def test_get_instances_only_questions_with_answers_become_instances(monkeypatch):
    """Even if other columns (Focus, Type) are present in the source, only Question/Answer are used."""
    df = pd.DataFrame(
        {
            "Question": ["Q1"],
            "Answer": ["A1."],
            "Focus": ["aspirin"],  # extra column present in the real dataset
            "Type": ["Indication"],
        }
    )
    _patch_with_dataframe(monkeypatch, df)

    scenario = MedicationQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1
    assert instances[0].input.text == "Q1"
    assert instances[0].references[0].output.text == "A1."


# ---------------------------------------------------------------------------
# Mocked test for `download_medication_qa` (verifies URL construction).
# ---------------------------------------------------------------------------


def test_download_medication_qa_builds_correct_source_and_target(monkeypatch):
    scenario = MedicationQAScenario()
    calls: list = []

    def _fake(source_url, target_path, **kwargs):
        calls.append({"source_url": source_url, "target_path": target_path, "kwargs": kwargs})

    monkeypatch.setattr(
        "helm.benchmark.scenarios.medication_qa_scenario.ensure_file_downloaded", _fake
    )

    with TemporaryDirectory() as tmpdir:
        scenario.download_medication_qa(tmpdir)

    assert len(calls) == 1
    assert calls[0]["source_url"] == os.path.join(
        MedicationQAScenario.SOURCE_REPO_URL, MedicationQAScenario.FILENAME
    )
    assert calls[0]["target_path"] == os.path.join(tmpdir, MedicationQAScenario.FILENAME)
    assert calls[0]["kwargs"]["unpack"] is False


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MedicationQAScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medication_qa"
    assert metadata.display_name == "MedicationQA"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "medication_qa_accuracy"
    assert metadata.taxonomy.task == "Question answering"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MedicationQAScenario()

    assert scenario.name == "medication_qa"
    assert "biomedical" in scenario.tags
    assert "question_answering" in scenario.tags
    assert "generation" in scenario.tags
    assert MedicationQAScenario.FILENAME == "MedInfo2019-QA-Medications.xlsx"
    assert MedicationQAScenario.SOURCE_REPO_URL.startswith("https://github.com/abachaa/Medication_QA_MedInfo2019")
