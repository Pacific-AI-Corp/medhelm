import csv
import os
from tempfile import TemporaryDirectory
from typing import List, Optional

import pytest

from helm.benchmark.scenarios.starr_patient_instructions_scenario import (
    StarrPatientInstructionsScenario,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


REQUIRED_FIELDS = [
    "Diagnosis",
    "ActualProcedure",
    "HistoryPhysicalNoteText",
    "OperativeNoteText",
    "DischargeInstructionNoteText",
]

ALL_COLUMNS = ["QC"] + REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _valid_row(**overrides) -> dict:
    """Return a fully valid row that satisfies every filter. Override individual fields by
    passing them as kwargs."""
    row = {
        "QC": "TRUE",
        "Diagnosis": "Acute appendicitis",
        "ActualProcedure": "Laparoscopic appendectomy",
        "HistoryPhysicalNoteText": "H&P: 22yo M with RLQ pain.",
        "OperativeNoteText": "Op: uncomplicated appendectomy.",
        "DischargeInstructionNoteText": "Rest at home; pain meds as prescribed.",
    }
    row.update(overrides)
    return row


def _write_csv(path: str, rows: List[dict], columns: Optional[List[str]] = None) -> None:
    cols = columns if columns is not None else ALL_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_scenario(rows: List[dict], dir_path: str) -> StarrPatientInstructionsScenario:
    csv_path = os.path.join(dir_path, "starr.csv")
    _write_csv(csv_path, rows)
    return StarrPatientInstructionsScenario(data_path=csv_path)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = StarrPatientInstructionsScenario(data_path="/tmp/x.csv")
    assert scenario.data_path == "/tmp/x.csv"


def test_class_attributes():
    assert StarrPatientInstructionsScenario.name == "starr_patient_instructions"
    # Pin all four tags to detect accidental drops in a refactor.
    assert StarrPatientInstructionsScenario.tags == [
        "patient_communication",
        "healthcare",
        "instruction_generation",
        "surgery",
    ]
    assert "PatientInstruct" in StarrPatientInstructionsScenario.description


# ---------------------------------------------------------------------------
# `get_instances` — happy paths.
# ---------------------------------------------------------------------------


def test_get_instances_valid_row_produces_single_instance():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([_valid_row()], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.split == TEST_SPLIT
    assert len(instance.references) == 1
    assert CORRECT_TAG in instance.references[0].tags


def test_get_instances_input_text_uses_expected_template():
    """The prompt concatenates the four context fields in a fixed order with labeled sections.
    The discharge instruction text MUST NOT appear in the input."""
    row = _valid_row(
        Diagnosis="D",
        ActualProcedure="P",
        HistoryPhysicalNoteText="HP",
        OperativeNoteText="OP",
        DischargeInstructionNoteText="DISCHARGE_GOLD",
    )
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert text == "Diagnosis: D\nProcedure: P\nHistory & Physical: HP\nOperative Report: OP\n\n"
    # Critical: the gold reference must NOT leak into the input.
    assert "DISCHARGE_GOLD" not in text


def test_get_instances_reference_is_discharge_instruction():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([_valid_row(DischargeInstructionNoteText="Take meds and rest.")], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].references[0].output.text == "Take meds and rest."


def test_get_instances_emits_one_instance_per_valid_row():
    rows = [_valid_row(Diagnosis=f"D{i}", DischargeInstructionNoteText=f"Inst{i}") for i in range(4)]
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario(rows, tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 4
    assert [i.references[0].output.text for i in instances] == [f"Inst{i}" for i in range(4)]


def test_get_instances_all_instances_share_test_split():
    rows = [_valid_row(), _valid_row()]
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario(rows, tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


# ---------------------------------------------------------------------------
# QC filter — must equal "TRUE" (case-insensitive after strip).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qc_value", ["TRUE", "true", "True", "TrUe", "  TRUE  ", "true\t"])
def test_get_instances_accepts_truthy_qc_case_insensitive_and_with_padding(qc_value):
    """The scenario applies `.strip().upper()` then compares to `"TRUE"`. Any casing or
    whitespace padding must round-trip to the accepted form."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([_valid_row(QC=qc_value)], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1


@pytest.mark.parametrize("qc_value", ["FALSE", "false", "", " ", "no", "0", "yes", "1"])
def test_get_instances_skips_rows_when_qc_is_not_true(qc_value):
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([_valid_row(QC=qc_value)], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_handles_missing_qc_column_as_skip():
    """If the CSV has no `QC` column at all, every row defaults to "" and is skipped (no
    rows survive the filter)."""
    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "starr.csv")
        # Build a CSV without the QC column. Use only required fields.
        cols = REQUIRED_FIELDS
        row = {k: v for k, v in _valid_row().items() if k != "QC"}
        _write_csv(csv_path, [row], columns=cols)
        scenario = StarrPatientInstructionsScenario(data_path=csv_path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


# ---------------------------------------------------------------------------
# Required-field filter — every required field must be non-empty after strip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_get_instances_skips_rows_where_a_required_field_is_empty(missing_field):
    """A blank value in any of the 5 required fields drops the row entirely."""
    row = _valid_row(**{missing_field: ""})
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_get_instances_skips_rows_where_a_required_field_is_whitespace(missing_field):
    """`.strip()` reduces whitespace-only values to "", which then fails the truthiness check."""
    row = _valid_row(**{missing_field: "   \t   "})
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_strips_required_fields_before_emitting():
    """Whitespace at the ends of the input fields must be trimmed in the composed prompt."""
    row = _valid_row(
        Diagnosis="   DX   ",
        ActualProcedure="\tPROC\t",
        HistoryPhysicalNoteText="\nHP\n",
        OperativeNoteText="  OP  ",
        DischargeInstructionNoteText="  DISCH  ",
    )
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Diagnosis: DX\n" in text
    assert "Procedure: PROC\n" in text
    assert "History & Physical: HP\n" in text
    assert "Operative Report: OP\n" in text
    assert instances[0].references[0].output.text == "DISCH"


# ---------------------------------------------------------------------------
# Mixed-row filtering.
# ---------------------------------------------------------------------------


def test_get_instances_only_returns_rows_that_pass_every_filter():
    rows = [
        _valid_row(Diagnosis="D1"),  # valid
        _valid_row(QC="FALSE", Diagnosis="D2"),  # QC filter
        _valid_row(Diagnosis="", DischargeInstructionNoteText="D3"),  # missing diagnosis
        _valid_row(Diagnosis="D4"),  # valid
    ]
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario(rows, tmp)
        instances = scenario.get_instances(output_path=tmp)

    surviving_diagnoses = [line.split("Diagnosis: ")[1].split("\n")[0] for line in (i.input.text for i in instances)]
    assert surviving_diagnoses == ["D1", "D4"]


# ---------------------------------------------------------------------------
# Empty / missing file / unicode.
# ---------------------------------------------------------------------------


def test_get_instances_returns_empty_list_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        scenario = StarrPatientInstructionsScenario(data_path=os.path.join(tmp, "missing.csv"))
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_preserves_unicode_in_fields():
    row = _valid_row(
        Diagnosis="Migraña sévère",
        ActualProcedure="Procédure d'urgence",
        DischargeInstructionNoteText="Repos à la maison; éviter l'effort.",
    )
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert "Migraña sévère" in instances[0].input.text
    assert "Procédure d'urgence" in instances[0].input.text
    assert instances[0].references[0].output.text == "Repos à la maison; éviter l'effort."


def test_get_instances_supports_very_long_clinical_notes():
    """The default csv module would normally cap field size; large clinical notes must still
    round-trip in this scenario."""
    long_hp = "H&P: " + ("x" * 50_000)
    row = _valid_row(HistoryPhysicalNoteText=long_hp)
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario([row], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert long_hp in instances[0].input.text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = StarrPatientInstructionsScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "starr_patient_instructions"
    assert metadata.display_name == "PatientInstruct"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "starr_patient_instructions_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.when == "Post-procedure"
    assert metadata.taxonomy.who == "Clinician"
    assert metadata.taxonomy.language == "English"
