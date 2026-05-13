import os
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest

from helm.benchmark.scenarios.chw_care_plan_scenario import (
    CHWCarePlanScenario,
    create_prompt_text,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict], columns=None) -> None:
    """Write `rows` as a CSV at `path`. If `columns` is None it defaults to ['MO Note']."""
    cols = columns if columns is not None else ["MO Note"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


def _make_scenario_with_csv(rows: List[dict], dir_path: str) -> CHWCarePlanScenario:
    csv_path = os.path.join(dir_path, "chw.csv")
    _write_csv(csv_path, rows)
    return CHWCarePlanScenario(data_path=csv_path)


# ---------------------------------------------------------------------------
# `create_prompt_text` — pure function, easy to test.
# ---------------------------------------------------------------------------


def test_create_prompt_text_contains_clinical_note_verbatim():
    note = "Patient presents with chest pain."
    prompt = create_prompt_text(note)
    assert "Clinical Note:" in prompt
    assert note in prompt


def test_create_prompt_text_includes_all_hpi_fields():
    """The Response Format must spell out all seven OPQRST-style HPI sections."""
    prompt = create_prompt_text("any note")

    for section in [
        "Chief Complaint",
        "History of Present Illness",
        "Onset:",
        "Provoking/Palliating Factors:",
        "Quality:",
        "Region/Radiation:",
        "Severity:",
        "Timing:",
        "Related Symptoms:",
    ]:
        assert section in prompt, f"Missing section: {section}"


def test_create_prompt_text_warns_against_hallucination():
    """Pin the anti-hallucination guardrail language — a key benchmark goal."""
    prompt = create_prompt_text("any note")
    assert "Do not hallucinate" in prompt
    assert "Not mentioned" in prompt
    assert "Do not introduce external knowledge" in prompt


def test_create_prompt_text_preserves_multiline_input_verbatim():
    note = "Line 1.\nLine 2.\n- bullet"
    prompt = create_prompt_text(note)
    assert note in prompt


def test_create_prompt_text_handles_empty_note():
    """Empty input must not crash; the template is still emitted with an empty Clinical Note
    section."""
    prompt = create_prompt_text("")
    assert prompt.rstrip().endswith("Clinical Note:")


def test_create_prompt_text_preserves_unicode():
    note = "Migrăña sévère, le matin"
    prompt = create_prompt_text(note)
    assert note in prompt


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = CHWCarePlanScenario(data_path="/tmp/chw.csv")
    assert scenario.data_path == "/tmp/chw.csv"


def test_class_attributes():
    assert CHWCarePlanScenario.name == "chw_care_plan"
    assert CHWCarePlanScenario.tags == ["question_answering", "biomedical"]
    assert "NoteExtract" in CHWCarePlanScenario.description


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSV files.
# ---------------------------------------------------------------------------


def test_get_instances_single_row_produces_single_instance():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": "Patient reports headache for 3 days."}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.split == TEST_SPLIT
    assert len(instance.references) == 1
    assert CORRECT_TAG in instance.references[0].tags


def test_get_instances_input_is_full_prompt_and_reference_is_raw_note():
    """Critical contract: the model sees the full templated prompt (with instructions), but the
    gold reference is the ORIGINAL clinical note — used by entailment-style metrics."""
    note = "Patient reports headache for 3 days."
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": note}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    # Input contains the prompt template + the note.
    assert "Clinical Note:" in instances[0].input.text
    assert note in instances[0].input.text
    assert "Do not hallucinate" in instances[0].input.text
    # Reference is the raw note only.
    assert instances[0].references[0].output.text == note


def test_get_instances_emits_one_instance_per_row():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": f"Note {i}"} for i in range(5)], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 5
    for i, instance in enumerate(instances):
        assert instance.references[0].output.text == f"Note {i}"


def test_get_instances_assigns_test_split_to_every_instance():
    """All data is held out for evaluation; everything must land in `TEST_SPLIT`."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": "N1"}, {"MO Note": "N2"}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_skips_rows_with_nan_in_mo_note(capsys):
    """Rows with NaN in `MO Note` are skipped, and a notice is printed for visibility."""
    with TemporaryDirectory() as tmp:
        # Use `None` so pandas writes/reads it back as NaN.
        scenario = _make_scenario_with_csv(
            [
                {"MO Note": "First valid note."},
                {"MO Note": None},
                {"MO Note": "Second valid note."},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert [i.references[0].output.text for i in instances] == [
        "First valid note.",
        "Second valid note.",
    ]
    # The scenario explicitly prints "Skipping row X due to NaN..." — check we saw it.
    captured = capsys.readouterr()
    assert "Skipping row" in captured.out
    assert "NaN" in captured.out


@pytest.mark.parametrize("blank_value", ["", "   ", "\t"])
def test_get_instances_drops_blank_notes_via_pandas_default_nan_inference(blank_value):
    """KNOWN behaviour pinned: when pandas' `read_csv` parses an empty or whitespace-only cell
    with default settings, it returns NaN. The scenario's `pd.isna(note_text)` check then skips
    those rows. If a future change reads with `keep_default_na=False`, those rows will start
    surviving and this test will alert the maintainer to confirm the new behaviour."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": blank_value}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_does_not_skip_truthy_short_notes():
    """A non-blank but very short note is kept — only NaN values are dropped, not 'too short'
    ones. Pin this so a future quality filter would be intentional."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": "ok"}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].references[0].output.text == "ok"


def test_get_instances_returns_empty_list_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_raises_when_data_file_is_missing():
    """`check_file_exists` should raise before any pandas parsing happens."""
    with TemporaryDirectory() as tmp:
        scenario = CHWCarePlanScenario(data_path=os.path.join(tmp, "missing.csv"))
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_mo_note_column_is_missing():
    """If the CSV lacks the `MO Note` column entirely, `row["MO Note"]` raises `KeyError`."""
    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "chw.csv")
        _write_csv(csv_path, [{"other_col": "x"}], columns=["other_col"])
        scenario = CHWCarePlanScenario(data_path=csv_path)

        with pytest.raises(KeyError, match="MO Note"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_preserves_unicode_in_notes():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": "Patient signale céphalée sévère"}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].references[0].output.text == "Patient signale céphalée sévère"
    assert "Patient signale céphalée sévère" in instances[0].input.text


def test_get_instances_handles_multiline_notes():
    """Notes may contain newlines (carried over from real clinical workflows); they must be
    embedded into the prompt and surface verbatim in the reference."""
    multiline_note = "Line 1: chest pain.\nLine 2: dyspnea.\nLine 3: diaphoresis."
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_csv([{"MO Note": multiline_note}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].references[0].output.text == multiline_note
    assert multiline_note in instances[0].input.text


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = CHWCarePlanScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "chw_care_plan"
    assert metadata.display_name == "NoteExtract"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "chw_care_plan_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"


def test_get_metadata_description_warns_against_hallucination():
    """Pin the central project promise (no hallucinations) in the public description."""
    scenario = CHWCarePlanScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "hallucination" in description.lower()
