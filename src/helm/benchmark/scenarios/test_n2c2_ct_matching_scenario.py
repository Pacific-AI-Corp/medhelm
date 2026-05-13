import os
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import pytest

from helm.benchmark.scenarios.n2c2_ct_matching_scenario import (
    LONG_DEFINITIONS,
    ORIGINAL_DEFINITIONS,
    N2C2CTMatchingScenario,
    XMLDataLoader,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers for synthesizing the n2c2 XML files on disk.
# ---------------------------------------------------------------------------


def _write_xml(path: str, text: str, tags: Dict[str, str]) -> None:
    """Write a synthetic n2c2-style XML file: <root> with <TEXT> + <TAGS> children.

    Each tag in `tags` becomes a child of <TAGS> with `met="met"` or `met="not met"`.
    """
    root = ET.Element("PatientMatching")
    text_elem = ET.SubElement(root, "TEXT")
    text_elem.text = text
    tags_elem = ET.SubElement(root, "TAGS")
    for tag_name, met in tags.items():
        ET.SubElement(tags_elem, tag_name, met=met)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _all_met_tags(value: str = "met") -> Dict[str, str]:
    """Return a dict that fills every criterion in LONG_DEFINITIONS with the same `met` value."""
    return {key: value for key in LONG_DEFINITIONS.keys()}


# ---------------------------------------------------------------------------
# Static data: ORIGINAL_DEFINITIONS and LONG_DEFINITIONS.
# ---------------------------------------------------------------------------


def test_original_and_long_definitions_have_identical_keys():
    """Both dicts must cover the same 13 inclusion criteria. A mismatch would mean the prompt
    builder (`LONG_DEFINITIONS[self.subject]`) breaks for criteria only declared in the other
    dict."""
    assert set(ORIGINAL_DEFINITIONS.keys()) == set(LONG_DEFINITIONS.keys())


def test_definitions_contain_expected_n2c2_criteria():
    """Pin the exact 13 inclusion criteria from the n2c2 2018 shared task."""
    expected = {
        "ABDOMINAL",
        "ADVANCED-CAD",
        "ALCOHOL-ABUSE",
        "ASP-FOR-MI",
        "CREATININE",
        "DIETSUPP-2MOS",
        "DRUG-ABUSE",
        "ENGLISH",
        "HBA1C",
        "KETO-1YR",
        "MAJOR-DIABETES",
        "MAKES-DECISIONS",
        "MI-6MOS",
    }
    assert set(LONG_DEFINITIONS.keys()) == expected
    assert len(LONG_DEFINITIONS) == 13


def test_definitions_are_all_non_empty_strings():
    for key, value in ORIGINAL_DEFINITIONS.items():
        assert isinstance(value, str) and value.strip(), f"{key}: empty/non-string original def"
    for key, value in LONG_DEFINITIONS.items():
        assert isinstance(value, str) and value.strip(), f"{key}: empty/non-string long def"


def test_long_definitions_are_at_least_as_long_for_a_sample_of_keys():
    """Spot check: LONG_DEFINITIONS is meant to be richer than ORIGINAL_DEFINITIONS for a few
    targeted criteria. Pin this for ABDOMINAL / DIETSUPP-2MOS / ENGLISH / MAKES-DECISIONS
    which all received added language."""
    for key in ["ABDOMINAL", "DIETSUPP-2MOS", "ENGLISH", "MAKES-DECISIONS"]:
        assert len(LONG_DEFINITIONS[key]) > len(
            ORIGINAL_DEFINITIONS[key]
        ), f"LONG_DEFINITIONS[{key!r}] expected to be longer than the original"


# ---------------------------------------------------------------------------
# `XMLDataLoader.__init__`.
# ---------------------------------------------------------------------------


def test_xml_data_loader_init_defaults_true():
    loader = XMLDataLoader(path_to_folder="/tmp/x")
    assert loader.path_to_folder == "/tmp/x"
    assert loader.is_convert_to_numbers is True
    assert loader.is_split_text is True
    assert loader.is_remove_excessive_new_lines is True


def test_xml_data_loader_init_accepts_overrides():
    loader = XMLDataLoader(
        path_to_folder="/p",
        is_convert_to_numbers=False,
        is_split_text=False,
        is_remove_excessive_new_lines=False,
    )
    assert loader.is_convert_to_numbers is False
    assert loader.is_split_text is False
    assert loader.is_remove_excessive_new_lines is False


# ---------------------------------------------------------------------------
# `XMLDataLoader.split_text`.
# ---------------------------------------------------------------------------


SPLIT_CHAR = "*" * 100  # what the scenario uses as a section delimiter


def test_split_text_with_no_delimiter_returns_single_part():
    loader = XMLDataLoader(path_to_folder="/p")
    parts = loader.split_text("just one note")
    assert parts == ["just one note"]


def test_split_text_splits_on_exactly_100_asterisks_and_strips_parts():
    loader = XMLDataLoader(path_to_folder="/p")
    text = f"   note 1   {SPLIT_CHAR}   note 2  {SPLIT_CHAR}note 3"
    parts = loader.split_text(text)
    assert parts == ["note 1", "note 2", "note 3"]


def test_split_text_drops_empty_parts():
    """Adjacent delimiters create empty parts; these are filtered out by the `if x.strip() !=
    ""` predicate."""
    loader = XMLDataLoader(path_to_folder="/p")
    text = f"{SPLIT_CHAR}{SPLIT_CHAR}note{SPLIT_CHAR}"
    parts = loader.split_text(text)
    assert parts == ["note"]


def test_split_text_does_not_split_on_fewer_than_100_asterisks():
    """99 asterisks are NOT a delimiter — the split character is strictly 100 stars."""
    loader = XMLDataLoader(path_to_folder="/p")
    text = "left" + ("*" * 99) + "right"
    assert loader.split_text(text) == [text]


# ---------------------------------------------------------------------------
# `XMLDataLoader.remove_excessive_newlines`.
# ---------------------------------------------------------------------------


def test_remove_excessive_newlines_collapses_triple_newlines_to_single():
    loader = XMLDataLoader(path_to_folder="/p")
    assert loader.remove_excessive_newlines("a\n\n\nb") == "a\nb"


def test_remove_excessive_newlines_preserves_double_newlines():
    """Only the literal sequence `\\n\\n\\n` is replaced. `\\n\\n` (paragraph breaks) is kept."""
    loader = XMLDataLoader(path_to_folder="/p")
    assert loader.remove_excessive_newlines("a\n\nb") == "a\n\nb"


def test_remove_excessive_newlines_handles_multiple_runs():
    """Pin behavior: only a single `.replace()` call → `\\n\\n\\n\\n` becomes `\\n\\n`
    (the FIRST three newlines collapse, the last newline survives)."""
    loader = XMLDataLoader(path_to_folder="/p")
    assert loader.remove_excessive_newlines("a\n\n\n\nb") == "a\n\nb"


# ---------------------------------------------------------------------------
# `XMLDataLoader.read_tags`.
# ---------------------------------------------------------------------------


def test_read_tags_converts_met_to_one_and_not_met_to_zero_by_default():
    loader = XMLDataLoader(path_to_folder="/p")
    root = ET.fromstring('<root><TAGS><ABDOMINAL met="met"/><ENGLISH met="not met"/></TAGS></root>')
    tags = loader.read_tags(root)
    assert tags == {"ABDOMINAL": 1, "ENGLISH": 0}


def test_read_tags_keeps_string_values_when_convert_disabled():
    """When `is_convert_to_numbers=False`, the loader passes through the raw `met` attribute
    string."""
    loader = XMLDataLoader(path_to_folder="/p", is_convert_to_numbers=False)
    root = ET.fromstring('<root><TAGS><ABDOMINAL met="met"/><ENGLISH met="not met"/></TAGS></root>')
    tags = loader.read_tags(root)
    assert tags == {"ABDOMINAL": "met", "ENGLISH": "not met"}


def test_read_tags_treats_any_non_met_string_as_zero():
    """The conversion is strict equality `met_value == "met"`. Anything else (typos, empty
    strings, capitalized variants) maps to 0."""
    loader = XMLDataLoader(path_to_folder="/p")
    root = ET.fromstring('<root><TAGS><A met=""/><B met="Met"/><C met="MET"/><D met="not met"/></TAGS></root>')
    tags = loader.read_tags(root)
    assert tags == {"A": 0, "B": 0, "C": 0, "D": 0}


# ---------------------------------------------------------------------------
# `XMLDataLoader.parse_xml`.
# ---------------------------------------------------------------------------


def test_parse_xml_returns_text_split_and_tags():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p1.xml")
        text = f"Record date: 2020-01-01\nNote A{SPLIT_CHAR}Record date: 2020-02-02\nNote B"
        _write_xml(path, text, {"ABDOMINAL": "met", "ENGLISH": "not met"})

        loader = XMLDataLoader(path_to_folder=tmp)
        result_text, tags = loader.parse_xml(path)

    assert len(result_text) == 2
    assert "Record date: 2020-01-01" in result_text[0]
    assert "Record date: 2020-02-02" in result_text[1]
    assert tags == {"ABDOMINAL": 1, "ENGLISH": 0}


def test_parse_xml_returns_single_chunk_when_split_disabled():
    """`is_split_text=False` keeps the whole TEXT body as a single string in a 1-element list."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p1.xml")
        text = f"Note A{SPLIT_CHAR}Note B"
        _write_xml(path, text, {"ABDOMINAL": "met"})

        loader = XMLDataLoader(path_to_folder=tmp, is_split_text=False)
        result_text, tags = loader.parse_xml(path)

    assert len(result_text) == 1
    assert "Note A" in result_text[0]
    assert "Note B" in result_text[0]


def test_parse_xml_handles_empty_text_element():
    """Empty <TEXT/> must produce an empty notes list (or list with empty string when split is
    off) rather than crashing."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p1.xml")
        # Use minimal XML where <TEXT> is empty.
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                '<?xml version="1.0"?><PatientMatching><TEXT></TEXT>'
                '<TAGS><ABDOMINAL met="met"/></TAGS></PatientMatching>'
            )

        loader = XMLDataLoader(path_to_folder=tmp)
        result_text, tags = loader.parse_xml(path)

    assert result_text == []  # split_text on "" filters out the only empty chunk
    assert tags == {"ABDOMINAL": 1}


# ---------------------------------------------------------------------------
# `XMLDataLoader.load_data`.
# ---------------------------------------------------------------------------


def test_load_data_returns_one_record_per_xml_file_sorted_alphabetically():
    """`load_data` filters to `.xml` files only and sorts them alphabetically so the iteration
    order is deterministic."""
    with TemporaryDirectory() as tmp:
        _write_xml(os.path.join(tmp, "patient_b.xml"), "B body", {"ABDOMINAL": "met"})
        _write_xml(os.path.join(tmp, "patient_a.xml"), "A body", {"ABDOMINAL": "not met"})
        _write_xml(os.path.join(tmp, "patient_c.xml"), "C body", {"ABDOMINAL": "met"})

        loader = XMLDataLoader(path_to_folder=tmp)
        data = loader.load_data()

    assert [d["patient_id"] for d in data] == ["patient_a", "patient_b", "patient_c"]
    assert data[0]["ehr"] == ["A body"]
    assert data[0]["labels"] == {"ABDOMINAL": 0}
    assert data[1]["labels"] == {"ABDOMINAL": 1}


def test_load_data_skips_non_xml_files():
    """The directory may contain README / hidden files / other artifacts. Only `.xml` files
    are loaded."""
    with TemporaryDirectory() as tmp:
        _write_xml(os.path.join(tmp, "p1.xml"), "body", {"ABDOMINAL": "met"})
        with open(os.path.join(tmp, "README.txt"), "w") as f:
            f.write("ignore me")
        with open(os.path.join(tmp, "notes.json"), "w") as f:
            f.write("{}")

        loader = XMLDataLoader(path_to_folder=tmp)
        data = loader.load_data()

    assert len(data) == 1
    assert data[0]["patient_id"] == "p1"


def test_load_data_returns_empty_list_for_empty_directory():
    with TemporaryDirectory() as tmp:
        loader = XMLDataLoader(path_to_folder=tmp)
        assert loader.load_data() == []


def test_load_data_strips_xml_extension_from_patient_id():
    """The `patient_id` is derived from the filename by chopping the `.xml` suffix."""
    with TemporaryDirectory() as tmp:
        _write_xml(os.path.join(tmp, "201.xml"), "body", {"ABDOMINAL": "met"})
        loader = XMLDataLoader(path_to_folder=tmp)
        data = loader.load_data()
    assert data[0]["patient_id"] == "201"


# ---------------------------------------------------------------------------
# `XMLDataLoader.get_date_of_note`.
# ---------------------------------------------------------------------------


def test_get_date_of_note_extracts_iso_date_via_regex():
    patient = {"patient_id": "p1", "ehr": ["Record date: 2023-04-15\nbody"]}
    assert XMLDataLoader.get_date_of_note(patient, 0) == "2023-04-15"


def test_get_date_of_note_returns_none_when_no_date_present(capsys):
    """When the regex `Record date: \\d{4}-\\d{2}-\\d{2}` doesn't match, the function returns
    None and prints an error to stdout."""
    patient = {"patient_id": "p1", "ehr": ["no date here"]}
    assert XMLDataLoader.get_date_of_note(patient, 0) is None
    captured = capsys.readouterr()
    assert "Could not find the date" in captured.out


def test_get_date_of_note_assert_allows_index_equal_to_length_documented_quirk():
    """KNOWN QUIRK: the assertion is `note_idx <= len(patient["ehr"])` (LE, not LT). So
    `note_idx == len(ehr)` passes the assert but then crashes with IndexError when accessing
    `patient["ehr"][note_idx]`. Pin so a tightening to `<` is intentional."""
    patient = {"patient_id": "p1", "ehr": ["only one note"]}
    with pytest.raises(IndexError):
        XMLDataLoader.get_date_of_note(patient, 1)  # passes assert, crashes the access


def test_get_date_of_note_assert_fires_for_index_well_past_length():
    """Out-of-bounds beyond `len(ehr)` triggers the assert immediately (never reaches the
    indexing line)."""
    patient = {"patient_id": "p1", "ehr": ["only one note"]}
    with pytest.raises(AssertionError):
        XMLDataLoader.get_date_of_note(patient, 5)


# ---------------------------------------------------------------------------
# `XMLDataLoader.get_current_date_for_patient`.
# ---------------------------------------------------------------------------


def test_get_current_date_for_patient_returns_last_matched_date():
    """`get_current_date_for_patient` iterates ALL notes and keeps the LAST match. The patient's
    notes are expected to be in chronological order, so the last note's date is the most
    recent."""
    patient = {
        "patient_id": "p1",
        "ehr": [
            "Record date: 2020-01-01\nFirst note",
            "Record date: 2020-06-15\nSecond note",
            "Record date: 2021-12-31\nThird note",
        ],
    }
    assert XMLDataLoader.get_current_date_for_patient(patient) == "2021-12-31"


def test_get_current_date_for_patient_skips_notes_without_a_date():
    """Notes without a Record date don't override the running value."""
    patient = {
        "patient_id": "p1",
        "ehr": [
            "Record date: 2020-01-01\nA",
            "(undated continuation)",
            "Record date: 2020-06-15\nB",
            "(undated tail)",
        ],
    }
    assert XMLDataLoader.get_current_date_for_patient(patient) == "2020-06-15"


def test_get_current_date_for_patient_returns_none_when_no_note_has_date(capsys):
    patient: dict[str, Any] = {"patient_id": "p1", "ehr": ["no date", "still no date"]}
    assert XMLDataLoader.get_current_date_for_patient(patient) is None
    captured = capsys.readouterr()
    assert "Could not find the date" in captured.out


def test_get_current_date_for_patient_returns_none_for_empty_ehr():
    patient: dict[str, Any] = {"patient_id": "p1", "ehr": []}
    # Empty EHR means no iteration → most_recent_date stays None.
    # We just verify it doesn't crash and returns None.
    result: Optional[str] = XMLDataLoader.get_current_date_for_patient(patient)
    assert result is None


# ---------------------------------------------------------------------------
# `N2C2CTMatchingScenario.__init__`.
# ---------------------------------------------------------------------------


def test_scenario_init_stores_data_path_and_subject():
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    assert scenario.data_path == "/tmp/d"
    assert scenario.subject == "ABDOMINAL"


def test_scenario_init_constructs_train_and_test_dirs_with_trailing_slashes():
    """The `__init__` joins the base path with `train/` and `test/` (with trailing slashes)."""
    scenario = N2C2CTMatchingScenario(data_path="/data/n2c2", subject="ABDOMINAL")
    assert scenario.path_to_train_dir == os.path.join("/data/n2c2", "train/")
    assert scenario.path_to_test_dir == os.path.join("/data/n2c2", "test/")


def test_scenario_class_attributes():
    assert N2C2CTMatchingScenario.name == "n2c2_ct_matching"
    # PINNED: lowercase, same shape as race_based_med (unlike SHC-x which uses A/B).
    assert N2C2CTMatchingScenario.POSSIBLE_ANSWER_CHOICES == ["yes", "no"]
    # KNOWN: tags is an empty list (marked TODO in source). Pin so a future tag list is an
    # intentional change.
    assert N2C2CTMatchingScenario.tags == []
    assert "N2C2-CT" in N2C2CTMatchingScenario.description


# ---------------------------------------------------------------------------
# `N2C2CTMatchingScenario.create_prompt`.
# ---------------------------------------------------------------------------


def _patient(notes: List[str], patient_id: str = "p1") -> Dict[str, object]:
    return {"patient_id": patient_id, "ehr": notes, "labels": {}}


def test_create_prompt_includes_all_required_sections():
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    patient = _patient(["Record date: 2020-05-01\nNote body."])

    prompt = scenario.create_prompt(patient)

    assert "# Task" in prompt
    assert "# Inclusion Criterion" in prompt
    assert "# Patient Clinical Notes" in prompt
    assert "# Current Date" in prompt
    assert "# Question" in prompt
    assert "Your job is to decide" in prompt


def test_create_prompt_inlines_subject_and_long_definition():
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    patient = _patient(["Record date: 2020-05-01\nNote body."])

    prompt = scenario.create_prompt(patient)

    # Subject inlined twice (in the Criterion block AND in the Question block).
    assert prompt.count('"ABDOMINAL"') >= 2
    # The (long) definition is inlined verbatim.
    assert LONG_DEFINITIONS["ABDOMINAL"] in prompt


def test_create_prompt_inlines_note_count_and_dates():
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    patient = _patient(
        [
            "Record date: 2020-01-01\nFirst note",
            "Record date: 2020-06-15\nSecond note",
        ]
    )

    prompt = scenario.create_prompt(patient)

    assert "Below is a set of 2 clinical notes" in prompt
    assert "## Note #1" in prompt
    assert "## Note #2" in prompt
    assert "Date: 2020-01-01" in prompt
    assert "Date: 2020-06-15" in prompt
    # Current date is the most recent.
    assert "Assume that the current date is: 2020-06-15" in prompt


def test_create_prompt_falls_back_to_empty_string_when_note_date_missing():
    """When a note has no Record date, `get_date_of_note` returns None — the template falls
    back to an empty string via `or ''`."""
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    patient = _patient(["Note without a date header.", "Record date: 2021-01-01\nDated note"])

    prompt = scenario.create_prompt(patient)

    assert "Date: \n" in prompt  # the undated note got an empty date
    assert "Date: 2021-01-01" in prompt
    assert "Assume that the current date is: 2021-01-01" in prompt


def test_create_prompt_separates_notes_with_long_asterisk_lines():
    """Notes are joined by a newline + 50 asterisks + blank line."""
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="ABDOMINAL")
    patient = _patient(
        [
            "Record date: 2020-01-01\nNote 1",
            "Record date: 2020-02-01\nNote 2",
        ]
    )

    prompt = scenario.create_prompt(patient)
    assert "*" * 50 in prompt
    # And the 100-asterisk lines that bracket the notes block.
    assert "-" * 100 in prompt


def test_create_prompt_raises_key_error_for_unknown_subject():
    """`LONG_DEFINITIONS[self.subject]` raises KeyError if `subject` isn't a valid criterion."""
    scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject="NOT-A-CRITERION")
    with pytest.raises(KeyError):
        scenario.create_prompt(_patient(["Record date: 2020-01-01\nfoo"]))


def test_create_prompt_supports_every_subject_in_long_definitions():
    """Every defined subject must produce a valid prompt — no KeyError, all sections present.

    This is a self-healing test: if a new criterion is added to LONG_DEFINITIONS, it's tested
    automatically without changing the test file."""
    patient = _patient(["Record date: 2020-01-01\nbody"])
    for subject in LONG_DEFINITIONS.keys():
        scenario = N2C2CTMatchingScenario(data_path="/tmp/d", subject=subject)
        prompt = scenario.create_prompt(patient)
        assert subject in prompt
        assert LONG_DEFINITIONS[subject] in prompt


# ---------------------------------------------------------------------------
# `N2C2CTMatchingScenario.get_instances`.
# ---------------------------------------------------------------------------


def _build_split_dirs(tmp: str) -> str:
    """Create the expected `train/` and `test/` subdirs under `tmp`, return base path."""
    base = os.path.join(tmp, "n2c2")
    os.makedirs(os.path.join(base, "train"), exist_ok=True)
    os.makedirs(os.path.join(base, "test"), exist_ok=True)
    return base


def test_get_instances_processes_only_test_split_documented_quirk():
    """KNOWN QUIRK: the source iterates `for split in ["train", "test"]` but immediately gates
    with `if split == "test"`. So the train directory is read NEVER — even if it contains
    data, it produces zero instances. Pin this so a future enabling of train is intentional.

    Strategy: put one patient in TRAIN with all criteria "met", and a different patient in
    TEST with all "not met". If train was processed, we'd see a "yes" instance; if only test,
    we see a single "no" instance."""
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)

        _write_xml(
            os.path.join(base, "train", "p_train.xml"),
            "Record date: 2020-01-01\nTrain body",
            _all_met_tags("met"),
        )
        _write_xml(
            os.path.join(base, "test", "p_test.xml"),
            "Record date: 2020-01-01\nTest body",
            _all_met_tags("not met"),
        )

        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].split == TEST_SPLIT
    correct = next(ref for ref in instances[0].references if CORRECT_TAG in ref.tags)
    assert correct.output.text == "no"  # because the test patient was "not met"
    # And the train patient's text is NOT in any prompt.
    assert "Train body" not in instances[0].input.text


def test_get_instances_emits_one_instance_per_test_patient():
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        for i in range(3):
            _write_xml(
                os.path.join(base, "test", f"p_{i}.xml"),
                f"Record date: 2020-0{i + 1}-01\nNote {i}",
                {**_all_met_tags("not met"), "ABDOMINAL": "met"},
            )
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        assert instance.split == TEST_SPLIT
        # All 3 patients have ABDOMINAL=met, so "yes" must be the correct answer.
        correct = next(ref for ref in instance.references if CORRECT_TAG in ref.tags)
        assert correct.output.text == "yes"


def test_get_instances_yes_label_maps_to_first_reference_correct():
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        _write_xml(
            os.path.join(base, "test", "p1.xml"),
            "Record date: 2020-01-01\nNote",
            {**_all_met_tags("not met"), "ABDOMINAL": "met"},
        )
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["yes", "no"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_no_label_maps_to_second_reference_correct():
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        _write_xml(
            os.path.join(base, "test", "p1.xml"),
            "Record date: 2020-01-01\nNote",
            _all_met_tags("not met"),
        )
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_references_always_in_yes_no_order_with_exactly_one_correct():
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        # Mix: one "yes" patient and one "no" patient.
        _write_xml(
            os.path.join(base, "test", "p_yes.xml"),
            "Record date: 2020-01-01\nA",
            {**_all_met_tags("not met"), "ABDOMINAL": "met"},
        )
        _write_xml(
            os.path.join(base, "test", "p_no.xml"),
            "Record date: 2020-01-01\nB",
            _all_met_tags("not met"),
        )
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["yes", "no"]
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_input_text_contains_per_patient_note_body():
    """Each instance's `input.text` must include that patient's specific note body."""
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        _write_xml(
            os.path.join(base, "test", "p_a.xml"),
            "Record date: 2020-01-01\nUNIQUE_A",
            {**_all_met_tags("not met"), "ABDOMINAL": "met"},
        )
        _write_xml(
            os.path.join(base, "test", "p_b.xml"),
            "Record date: 2020-02-01\nUNIQUE_B",
            {**_all_met_tags("not met"), "ABDOMINAL": "met"},
        )
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)

    texts = [inst.input.text for inst in instances]
    assert any("UNIQUE_A" in t for t in texts)
    assert any("UNIQUE_B" in t for t in texts)
    # And they cross-validate: each note appears in exactly one prompt.
    assert sum(1 for t in texts if "UNIQUE_A" in t) == 1
    assert sum(1 for t in texts if "UNIQUE_B" in t) == 1


def test_get_instances_returns_empty_list_when_test_directory_is_empty():
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        instances = scenario.get_instances(output_path=tmp)
    assert instances == []


def test_get_instances_creates_test_dir_if_missing_via_ensure_directory_exists():
    """`ensure_directory_exists` is called on the test dir, so a missing folder is auto-created.
    Without this auto-creation, `os.listdir` inside `XMLDataLoader.load_data` would raise
    `FileNotFoundError`."""
    with TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "n2c2")
        os.makedirs(base, exist_ok=True)
        # NB: deliberately NOT creating base/test/
        scenario = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        assert not os.path.exists(scenario.path_to_test_dir)
        # Must not raise even though `base/test/` doesn't yet exist.
        instances = scenario.get_instances(output_path=tmp)
        assert instances == []
        # After the call, the test directory must exist (assert before the tempdir is cleaned).
        assert os.path.isdir(scenario.path_to_test_dir)


def test_get_instances_supports_multiple_subjects_with_same_test_corpus():
    """Pin that the `subject` arg selects the correct row of the labels dict — different
    subjects can yield opposite labels for the same patient."""
    with TemporaryDirectory() as tmp:
        base = _build_split_dirs(tmp)
        _write_xml(
            os.path.join(base, "test", "p1.xml"),
            "Record date: 2020-01-01\nNote",
            {**_all_met_tags("not met"), "ABDOMINAL": "met", "ENGLISH": "not met"},
        )

        scenario_abd = N2C2CTMatchingScenario(data_path=base, subject="ABDOMINAL")
        scenario_eng = N2C2CTMatchingScenario(data_path=base, subject="ENGLISH")
        inst_abd = scenario_abd.get_instances(output_path=tmp)[0]
        inst_eng = scenario_eng.get_instances(output_path=tmp)[0]

    correct_abd = next(ref for ref in inst_abd.references if CORRECT_TAG in ref.tags)
    correct_eng = next(ref for ref in inst_eng.references if CORRECT_TAG in ref.tags)
    assert correct_abd.output.text == "yes"
    assert correct_eng.output.text == "no"


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = N2C2CTMatchingScenario(data_path="/tmp", subject="ABDOMINAL")
    metadata = scenario.get_metadata()

    assert metadata.name == "n2c2_ct_matching"
    assert metadata.display_name == "N2C2-CT Matching"
    # UNIQUE: this scenario also sets `short_display_name`, distinct from the long form.
    assert metadata.short_display_name == "N2C2-CT"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    # UNIQUE: this scenario uses `"Pre-Trial"` for `when` (distinct from the typical "Any").
    assert metadata.taxonomy.when == "Pre-Trial"
    assert metadata.taxonomy.who == "Researcher"
    assert "clinical trial" in metadata.taxonomy.what.lower()


def test_metadata_description_mentions_candidate_classification():
    scenario = N2C2CTMatchingScenario(data_path="/tmp", subject="ABDOMINAL")
    description = scenario.get_metadata().description
    assert "clinical notes" in description
    assert "valid candidate" in description
    assert "clinical trial" in description
