import os
from tempfile import TemporaryDirectory
from typing import Tuple

import pandas as pd
import pytest

from helm.benchmark.scenarios.dischargeme_scenario import (
    DischargeMeScenario,
    create_prompt,
    file_preprocessing,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers — build the on-disk dataset layout the pipeline expects.
# ---------------------------------------------------------------------------


PIPELINE_RELATIVE_DIR = ("files", "discharge-me", "1.3", "test_phase_1")
ALL_REQUIRED_FILES = [
    "diagnosis.csv.gz",
    "discharge.csv.gz",
    "discharge_target.csv.gz",
    "radiology.csv.gz",
    "edstays.csv.gz",
    "triage.csv.gz",
]


def _phase_dir(data_path: str) -> str:
    """Return the absolute path to the phase-1 directory under `data_path`."""
    return os.path.join(data_path, *PIPELINE_RELATIVE_DIR)


def _write_gz_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, compression="gzip", index=False)


def _build_minimal_dataset(data_path: str) -> Tuple[str, dict]:
    """Create a minimal but coherent dataset under `data_path` that the pipeline can consume.

    Returns the phase-1 directory and a dict of source DataFrames for assertions."""
    phase_dir = _phase_dir(data_path)
    os.makedirs(phase_dir, exist_ok=True)

    # Two distinct admissions for two distinct subjects.
    df_diagnosis = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 101, "icd_code": "I10"},
            {"subject_id": 2, "hadm_id": 202, "icd_code": "E78"},
        ]
    )
    df_triage = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 101, "temperature": 98.6},
            {"subject_id": 2, "hadm_id": 202, "temperature": 99.1},
        ]
    )
    df_discharge = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 101,
                # The text deliberately contains both task-objective answers so we can verify
                # `remove_substring` for each.
                "text": "Note for patient 1. BHC=stable course. DI=rest at home.",
            },
            {
                "subject_id": 2,
                "hadm_id": 202,
                "text": "Note for patient 2. BHC=hypertension monitored. DI=take meds daily.",
            },
        ]
    )
    df_radiology = pd.DataFrame(
        [
            {"hadm_id": 101, "text": "Radiology report 1: clean chest."},
            {"hadm_id": 202, "text": "Radiology report 2: mild infiltrate."},
        ]
    )
    df_ed = pd.DataFrame(
        [
            {"hadm_id": 101, "stay_id": 1},
            {"hadm_id": 202, "stay_id": 2},
        ]
    )
    df_target = pd.DataFrame(
        [
            {
                "hadm_id": 101,
                "brief_hospital_course": "stable course",
                "discharge_instructions": "rest at home",
            },
            {
                "hadm_id": 202,
                "brief_hospital_course": "hypertension monitored",
                "discharge_instructions": "take meds daily",
            },
        ]
    )

    _write_gz_csv(os.path.join(phase_dir, "diagnosis.csv.gz"), df_diagnosis)
    _write_gz_csv(os.path.join(phase_dir, "triage.csv.gz"), df_triage)
    _write_gz_csv(os.path.join(phase_dir, "discharge.csv.gz"), df_discharge)
    _write_gz_csv(os.path.join(phase_dir, "radiology.csv.gz"), df_radiology)
    _write_gz_csv(os.path.join(phase_dir, "edstays.csv.gz"), df_ed)
    _write_gz_csv(os.path.join(phase_dir, "discharge_target.csv.gz"), df_target)

    return phase_dir, {
        "diagnosis": df_diagnosis,
        "triage": df_triage,
        "discharge": df_discharge,
        "radiology": df_radiology,
        "ed": df_ed,
        "target": df_target,
    }


# ---------------------------------------------------------------------------
# `create_prompt` — pure function, easy to exercise.
# ---------------------------------------------------------------------------


def test_create_prompt_contains_discharge_and_radiology_sections():
    prompt = create_prompt(
        text="DISCHARGE_TEXT",
        text_df_radiology="RADIOLOGY_TEXT",
        task_objective="Brief Hospital Course",
    )

    assert "Discharge Text:\nDISCHARGE_TEXT" in prompt
    assert "Radiology Report:\nRADIOLOGY_TEXT" in prompt


def test_create_prompt_substitutes_task_objective_in_intro_and_trailer():
    """The task objective appears in both the intro sentence (the verb 'Generate the ...') and
    again as a trailer just before the model's expected output."""
    prompt = create_prompt(text="X", text_df_radiology="Y", task_objective="Discharge Instructions")

    assert prompt.startswith("Generate the Discharge Instructions from the following")
    assert prompt.rstrip().endswith("Discharge Instructions:")


def test_create_prompt_preserves_multiline_input_text_verbatim():
    multi = "Line 1.\nLine 2.\n- bullet"
    prompt = create_prompt(text=multi, text_df_radiology="rad", task_objective="X")
    assert multi in prompt


def test_create_prompt_handles_empty_strings():
    prompt = create_prompt(text="", text_df_radiology="", task_objective="BHC")
    assert "Discharge Text:\n\n\n" in prompt
    assert "Radiology Report:\n\n\n" in prompt
    assert prompt.rstrip().endswith("BHC:")


# ---------------------------------------------------------------------------
# `file_preprocessing` — exercises the full 6-way join + substring removal.
# ---------------------------------------------------------------------------


def test_file_preprocessing_brief_hospital_course_removes_objective_substring_from_text():
    """After preprocessing, `final_df['text']` must have the `brief_hospital_course` value
    stripped out (since the task is to generate it, leaking it in the input is forbidden)."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)

        result = file_preprocessing(tmp, task_objective="brief_hospital_course")

    row_101 = result[result["hadm_id"] == 101].iloc[0]
    assert "stable course" not in row_101["text"]
    # The rest of the discharge note must still be present.
    assert "Note for patient 1" in row_101["text"]
    assert "DI=rest at home" in row_101["text"]


def test_file_preprocessing_discharge_instructions_removes_correct_substring():
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)

        result = file_preprocessing(tmp, task_objective="discharge_instructions")

    row_101 = result[result["hadm_id"] == 101].iloc[0]
    assert "rest at home" not in row_101["text"]
    # The other answer must be left in place (only the active task's answer is stripped).
    assert "BHC=stable course" in row_101["text"]


def test_file_preprocessing_keeps_radiology_text_in_separate_column():
    """The radiology report must end up under `text_df_radiology` (note the suffix added by the
    merge on `hadm_id`), not silently overwriting `text`."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)

        result = file_preprocessing(tmp, task_objective="brief_hospital_course")

    assert "text_df_radiology" in result.columns
    radiology_for_101 = result[result["hadm_id"] == 101].iloc[0]["text_df_radiology"]
    assert radiology_for_101 == "Radiology report 1: clean chest."


def test_file_preprocessing_drops_duplicate_hadm_ids():
    """If the source diagnosis CSV has multiple rows per `hadm_id`, the pipeline must collapse
    them down so each admission produces a single instance downstream."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)
        # Append a second diagnosis row for hadm_id=101 (still subject_id=1).
        phase_dir = _phase_dir(tmp)
        df_diagnosis_2 = pd.DataFrame(
            [
                {"subject_id": 1, "hadm_id": 101, "icd_code": "I10"},
                {"subject_id": 1, "hadm_id": 101, "icd_code": "E78"},  # duplicate hadm_id
                {"subject_id": 2, "hadm_id": 202, "icd_code": "E78"},
            ]
        )
        _write_gz_csv(os.path.join(phase_dir, "diagnosis.csv.gz"), df_diagnosis_2)

        result = file_preprocessing(tmp, task_objective="brief_hospital_course")

    assert (result["hadm_id"].value_counts() == 1).all()


def test_file_preprocessing_inner_join_drops_admissions_missing_from_target():
    """`final_df = pd.merge(df_input, df_target, on='hadm_id', how='inner')` — admissions
    without a matching target row must be excluded."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)
        # Remove hadm_id=202 from the target file.
        phase_dir = _phase_dir(tmp)
        partial_target = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "brief_hospital_course": "stable course",
                    "discharge_instructions": "rest at home",
                }
            ]
        )
        _write_gz_csv(os.path.join(phase_dir, "discharge_target.csv.gz"), partial_target)

        result = file_preprocessing(tmp, task_objective="brief_hospital_course")

    assert result["hadm_id"].tolist() == [101]


def test_file_preprocessing_crashes_on_empty_join_documented_bug():
    """KNOWN BUG pinned for regression / intentional fix detection:

    When every inner join produces zero rows (e.g. when `hadm_id`s in radiology do not match
    any in discharge), the empty `final_df.apply(..., axis=1)` returns an empty DataFrame
    rather than an empty Series. The subsequent assignment

        final_df["text"] = final_df.apply(...)

    raises `ValueError: Columns must be same length as key`. A correct implementation would
    either guard against the empty case or use a Series-returning fallback such as
    `final_df["text"].mask(...)`. If/when that fix lands, update this test."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)
        phase_dir = _phase_dir(tmp)
        # Make radiology reference a non-existent hadm_id so every join collapses to zero rows.
        df_radiology = pd.DataFrame([{"hadm_id": 999, "text": "lonely report"}])
        _write_gz_csv(os.path.join(phase_dir, "radiology.csv.gz"), df_radiology)

        with pytest.raises(ValueError, match="Columns must be same length as key"):
            file_preprocessing(tmp, task_objective="brief_hospital_course")


@pytest.mark.parametrize("missing_file", ALL_REQUIRED_FILES)
def test_file_preprocessing_raises_when_required_csv_is_missing(missing_file):
    """Every one of the six gzip CSVs is checked via `check_file_exists`; deleting any one of
    them must abort preprocessing with a clear error."""
    with TemporaryDirectory() as tmp:
        _build_minimal_dataset(tmp)
        os.remove(os.path.join(_phase_dir(tmp), missing_file))

        with pytest.raises(Exception):
            file_preprocessing(tmp, task_objective="brief_hospital_course")


# ---------------------------------------------------------------------------
# `DischargeMeScenario` — constructor + simple methods.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = DischargeMeScenario(data_path="/tmp/physionet.org")
    assert scenario.data_path == "/tmp/physionet.org"


def test_class_attributes():
    assert DischargeMeScenario.name == "dischargeme"
    assert DischargeMeScenario.tags == ["biomedical"]
    assert "MIMIC-IV" in DischargeMeScenario.description or "discharge" in DischargeMeScenario.description


def test_read_file_strips_and_returns_lines():
    """`read_file` reads a text file and returns the list of stripped lines."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("  line one  \nline two\n   \nlast line   ")

        scenario = DischargeMeScenario(data_path="/tmp/x")
        lines = scenario.read_file(path)

    assert lines == ["line one", "line two", "", "last line"]


# ---------------------------------------------------------------------------
# `DischargeMeScenario.get_instances` — end-to-end via the synthetic dataset.
# Each input row produces TWO instances (BHC + DI), so 2 admissions → 4 instances.
# ---------------------------------------------------------------------------


def test_get_instances_emits_two_instances_per_admission():
    with TemporaryDirectory() as tmp, TemporaryDirectory() as output_dir:
        _build_minimal_dataset(tmp)
        scenario = DischargeMeScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=output_dir)

    assert len(instances) == 4  # 2 admissions × 2 tasks


def test_get_instances_alternates_brief_hospital_course_and_discharge_instructions():
    """The loop interleaves: index 0 = BHC for admission 0, index 1 = DI for admission 0, etc."""
    with TemporaryDirectory() as tmp, TemporaryDirectory() as output_dir:
        _build_minimal_dataset(tmp)
        scenario = DischargeMeScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=output_dir)

    # Pairwise: even-indexed instances are BHC, odd-indexed are DI.
    assert instances[0].input.text.startswith("Generate the Brief Hospital Course from the following")
    assert instances[1].input.text.startswith("Generate the Discharge Instructions from the following")
    assert instances[2].input.text.startswith("Generate the Brief Hospital Course from the following")
    assert instances[3].input.text.startswith("Generate the Discharge Instructions from the following")


def test_get_instances_references_carry_the_gold_target_answer():
    with TemporaryDirectory() as tmp, TemporaryDirectory() as output_dir:
        _build_minimal_dataset(tmp)
        scenario = DischargeMeScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=output_dir)

    references_seen = {inst.references[0].output.text for inst in instances}
    assert {
        "stable course",
        "rest at home",
        "hypertension monitored",
        "take meds daily",
    } <= references_seen


def test_get_instances_every_instance_has_one_correct_reference_and_test_split():
    with TemporaryDirectory() as tmp, TemporaryDirectory() as output_dir:
        _build_minimal_dataset(tmp)
        scenario = DischargeMeScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=output_dir)

    for instance in instances:
        assert instance.split == TEST_SPLIT
        assert len(instance.references) == 1
        assert CORRECT_TAG in instance.references[0].tags


def test_get_instances_prompt_excludes_task_objective_answer_from_input_text():
    """Critical: the gold answer that the model must produce MUST NOT appear in the discharge
    text that the model is allowed to read. Otherwise the task is trivial."""
    with TemporaryDirectory() as tmp, TemporaryDirectory() as output_dir:
        _build_minimal_dataset(tmp)
        scenario = DischargeMeScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=output_dir)

    # `instances[0]` is the BHC prompt for the first admission; its `text` should NOT contain
    # the BHC answer for that admission.
    bhc_instance = instances[0]
    answer_bhc = bhc_instance.references[0].output.text
    # The answer must appear at the *end* (in the reference) but NOT inside the Discharge Text
    # section of the prompt. We slice the prompt up to the trailer "Brief Hospital Course:" so
    # we only inspect the input part.
    input_part = bhc_instance.input.text.rsplit("Brief Hospital Course:", 1)[0]
    assert answer_bhc not in input_part


def test_get_instances_calls_file_preprocessing_once_per_task(monkeypatch):
    """The scenario must invoke `file_preprocessing` exactly twice — once for each task
    objective — so that the discharge text is stripped of the relevant answer for each prompt."""
    calls = []

    real_module = "helm.benchmark.scenarios.dischargeme_scenario"

    def _fake(data_path, task_objective):
        calls.append(task_objective)
        # Return a tiny synthetic DataFrame with the columns the loop needs.
        return pd.DataFrame(
            [
                {
                    "text": f"discharge_for_{task_objective}",
                    "text_df_radiology": f"radiology_for_{task_objective}",
                    "hadm_id": 1,
                    "brief_hospital_course": "BHC_GOLD",
                    "discharge_instructions": "DI_GOLD",
                }
            ]
        )

    monkeypatch.setattr(f"{real_module}.file_preprocessing", _fake)

    with TemporaryDirectory() as output_dir:
        scenario = DischargeMeScenario(data_path="/unused/because/mocked")
        instances = scenario.get_instances(output_path=output_dir)

    assert sorted(calls) == ["brief_hospital_course", "discharge_instructions"]
    assert len(instances) == 2  # one admission × two tasks
    # The BHC instance carries the BHC gold answer; the DI instance carries the DI gold answer.
    assert instances[0].references[0].output.text == "BHC_GOLD"
    assert instances[1].references[0].output.text == "DI_GOLD"


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = DischargeMeScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "dischargeme"
    assert metadata.display_name == "DischargeMe"
    assert metadata.short_display_name == "DischargeMe"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "dischargeme_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Upon hospital discharge"


def test_get_metadata_description_cites_physionet_link():
    scenario = DischargeMeScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "physionet.org" in description
    assert "discharge-me" in description
