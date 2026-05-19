import json
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.mimic_bhc_scenario import MIMICBHCScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_jsonl(path: str, rows: List[dict]) -> None:
    """Write each dict as a single JSON line."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_scenario_with_jsonl(rows: List[dict], dir_path: str) -> MIMICBHCScenario:
    """Materialise the JSONL file under `dir_path` and return a scenario wired to it."""
    data_path = os.path.join(dir_path, "mimic_bhc.jsonl")
    _write_jsonl(data_path, rows)
    return MIMICBHCScenario(data_path=data_path)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = MIMICBHCScenario(data_path="/tmp/mimic_bhc.jsonl")
    assert scenario.data_path == "/tmp/mimic_bhc.jsonl"


def test_class_attributes():
    assert MIMICBHCScenario.name == "mimic_bhc"
    assert MIMICBHCScenario.tags == ["summarization", "biomedical"]
    assert "Brief" in MIMICBHCScenario.description or "BHC" in MIMICBHCScenario.description


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic JSONL files.
# ---------------------------------------------------------------------------


def test_get_instances_pairs_input_with_target_as_correct_reference():
    """A single JSONL row produces one Instance: `input` becomes the prompt, `target` becomes
    the correct reference, marked with `CORRECT_TAG`."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {
                    "input": "Patient admitted with chest pain.",
                    "target": "Mr. X underwent uneventful workup.",
                }
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].input.text == "Patient admitted with chest pain."
    assert instances[0].references[0].output.text == "Mr. X underwent uneventful workup."
    assert CORRECT_TAG in instances[0].references[0].tags


def test_get_instances_emits_one_instance_per_jsonl_row_in_order():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [{"input": f"Note {i}", "target": f"Summary {i}"} for i in range(5)],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 5
    for i, instance in enumerate(instances):
        assert instance.input.text == f"Note {i}"
        assert instance.references[0].output.text == f"Summary {i}"


def test_get_instances_assigns_test_split_to_every_instance():
    """The scenario is zero-shot — only the `test` entry of the `splits` dict is uncommented,
    so every Instance must land in `TEST_SPLIT`."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "N1", "target": "S1"},
                {"input": "N2", "target": "S2"},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "N1", "target": "S1"},
                {"input": "N2", "target": "S2"},
                {"input": "N3", "target": "S3"},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert len(instance.references) == 1
        assert CORRECT_TAG in instance.references[0].tags


def test_get_instances_skips_rows_with_empty_input():
    """`if not clinical_note or not bhc_summary: continue`. An empty `input` falls into the
    falsy branch and the row is dropped."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "Has note.", "target": "Has summary."},
                {"input": "", "target": "Stray summary."},  # skipped
                {"input": "Has note 2.", "target": "Has summary 2."},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert [i.input.text for i in instances] == ["Has note.", "Has note 2."]


def test_get_instances_skips_rows_with_empty_target():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "N1", "target": "S1"},
                {"input": "N2", "target": ""},  # skipped
                {"input": "N3", "target": "S3"},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert [i.references[0].output.text for i in instances] == ["S1", "S3"]


def test_get_instances_does_not_strip_whitespace_only_inputs():
    """KNOWN behaviour (different from MIMIC-RRS): this scenario does NOT strip the strings
    before the falsy check, so an input that contains only whitespace is *truthy* and the row
    is kept. Pin this so a future strip() addition is intentional."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "   ", "target": "Summary."},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].input.text == "   "


def test_get_instances_returns_empty_list_for_empty_jsonl_file():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl([], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_returns_empty_list_when_every_row_is_filtered():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "", "target": ""},
                {"input": "Has note.", "target": ""},
                {"input": "", "target": "Has summary."},
            ],
            tmp,
        )
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_raises_when_data_file_is_missing():
    """`check_file_exists` raises (typically `FileNotFoundError`) before any JSON is parsed."""
    with TemporaryDirectory() as tmp:
        scenario = MIMICBHCScenario(data_path=os.path.join(tmp, "does_not_exist.jsonl"))
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_row_is_missing_input_field():
    """If a JSONL row lacks the `input` key, the list comprehension raises `KeyError` before
    any Instance is built."""
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [
                {"input": "N1", "target": "S1"},
                {"target": "Only summary present."},  # missing `input`
            ],
            tmp,
        )
        with pytest.raises(KeyError, match="input"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_row_is_missing_target_field():
    with TemporaryDirectory() as tmp:
        scenario = _make_scenario_with_jsonl(
            [{"input": "Note without target."}],
            tmp,
        )
        with pytest.raises(KeyError, match="target"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_on_malformed_jsonl_line():
    """A line that is not valid JSON must surface `json.JSONDecodeError`."""
    with TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "broken.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write('{"input": "N1", "target": "S1"}\n')
            f.write("this is not json at all\n")
        scenario = MIMICBHCScenario(data_path=data_path)

        with pytest.raises(json.JSONDecodeError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_preserves_unicode_and_multiline_content():
    """MIMIC notes contain anonymisation tokens like `___` and multi-line content; nothing in
    the pipeline should rewrite them."""
    with TemporaryDirectory() as tmp:
        note = "<SEX> M\nMr. ___ was admitted.\nAllergies: aucune."
        summary = "Patient stable.\n- discharged home"
        scenario = _make_scenario_with_jsonl([{"input": note, "target": summary}], tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].input.text == note
    assert instances[0].references[0].output.text == summary


def test_get_instances_skips_blank_lines_in_jsonl():
    """A blank line in the middle of a JSONL file would normally cause `json.loads("")` to
    fail. Pin the current behaviour (the scenario does NOT tolerate blank lines), so any future
    skip-on-blank fix is intentional."""
    with TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "with_blanks.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write('{"input": "N1", "target": "S1"}\n')
            f.write("\n")  # blank line
            f.write('{"input": "N2", "target": "S2"}\n')
        scenario = MIMICBHCScenario(data_path=data_path)

        with pytest.raises(json.JSONDecodeError):
            scenario.get_instances(output_path=tmp)


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = MIMICBHCScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "mimic_bhc"
    assert metadata.display_name == "MIMIC-IV-BHC"
    assert metadata.short_display_name == "MIMIC-BHC"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "mimic_bhc_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.who == "Clinician"
    assert metadata.taxonomy.when == "Upon hospital discharge"
    assert metadata.taxonomy.language == "English"


def test_get_metadata_description_cites_aali_paper():
    scenario = MIMICBHCScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Aali" in description
    assert "doi.org/10.1093/jamia/ocae312" in description
