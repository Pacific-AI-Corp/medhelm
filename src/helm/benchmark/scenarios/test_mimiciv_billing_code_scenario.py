import os
from tempfile import TemporaryDirectory
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from helm.benchmark.scenarios.mimiciv_billing_code_scenario import MIMICIVBillingCodeScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_feather(path: str, texts: List[str], targets: List[Any]) -> None:
    """Write a synthetic MIMIC-IV-style feather file with `text` and `target` columns.

    `targets` can mix lists, numpy arrays, strings, or scalars to exercise the type-coercion
    branches in `get_instances`. The column is stored as `object` so it can hold heterogeneous
    types per row.
    """
    df = pd.DataFrame(
        {
            "text": texts,
            "target": pd.Series(targets, dtype=object),
        }
    )
    df.to_feather(path)


def _make_scenario_with_feather(rows: List[dict]) -> tuple:
    """Convenience: spin up a temp dir, write a feather with `text`/`target` columns from
    `rows`, return (scenario, output_path)."""
    tmp = TemporaryDirectory()
    path = os.path.join(tmp.name, "data.feather")
    _write_feather(path, [r["text"] for r in rows], [r["target"] for r in rows])
    return MIMICIVBillingCodeScenario(data_path=path), tmp


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = MIMICIVBillingCodeScenario(data_path="/tmp/mimic.feather")
    assert scenario.data_path == "/tmp/mimic.feather"


def test_class_attributes():
    assert MIMICIVBillingCodeScenario.name == "mimiciv_billing_code"
    # UNIQUE: tags use the underscore `"question_answering"` style, distinct from the SHC
    # family which uses `["knowledge", "reasoning", "biomedical"]`. Pin the wording.
    assert MIMICIVBillingCodeScenario.tags == ["question_answering", "biomedical"]
    assert "MIMIC-IV" in MIMICIVBillingCodeScenario.description
    assert "ICD-10" in MIMICIVBillingCodeScenario.description


# ---------------------------------------------------------------------------
# `get_instances` — happy paths.
# ---------------------------------------------------------------------------


def test_get_instances_emits_one_instance_per_row():
    rows = [
        {"text": "Note A", "target": ["I10", "E11"]},
        {"text": "Note B", "target": ["J18"]},
        {"text": "Note C", "target": ["N18.6", "I12"]},
    ]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert len(instances) == 3
        assert [inst.input.text for inst in instances] == ["Note A", "Note B", "Note C"]
    finally:
        tmp.cleanup()


def test_get_instances_joins_icd10_codes_with_comma():
    """The reference text is the comma-joined ICD-10 codes (no space after the comma)."""
    rows = [{"text": "Note", "target": ["I10", "E11", "J18"]}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "I10,E11,J18"
    finally:
        tmp.cleanup()


def test_get_instances_handles_single_element_list():
    rows = [{"text": "Note", "target": ["I10"]}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "I10"
    finally:
        tmp.cleanup()


def test_get_instances_emits_exactly_one_correct_reference_per_instance():
    rows = [
        {"text": "Note A", "target": ["I10"]},
        {"text": "Note B", "target": ["J18"]},
    ]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        for instance in instances:
            assert len(instance.references) == 1
            assert CORRECT_TAG in instance.references[0].tags
    finally:
        tmp.cleanup()


def test_get_instances_uses_test_split_for_every_instance():
    """The entire dataset is treated as a single TEST_SPLIT — no train/val carve-out."""
    rows = [
        {"text": "Note A", "target": ["I10"]},
        {"text": "Note B", "target": ["J18"]},
    ]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert all(inst.split == TEST_SPLIT for inst in instances)
    finally:
        tmp.cleanup()


def test_get_instances_input_text_preserves_full_note():
    """The `text` column is used verbatim as input — no truncation, stripping, or escaping."""
    long_note = ("Admission summary: " + "x " * 5000).strip()
    rows = [{"text": long_note, "target": ["Z00"]}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].input.text == long_note
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# `get_instances` — `target` column type-coercion branches.
# ---------------------------------------------------------------------------


def test_get_instances_converts_numpy_array_target_to_list():
    """The first branch: `isinstance(icd10_codes, np.ndarray)` → `.tolist()` and join."""
    rows = [{"text": "Note", "target": np.array(["I10", "E11"])}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "I10,E11"
    finally:
        tmp.cleanup()


def test_get_instances_wraps_single_string_target_in_list():
    """The fallback branch: a non-list, non-ndarray `target` is wrapped as `[str(...)]`. For a
    single-string target, this yields a one-element reference whose text is the string itself."""
    rows = [{"text": "Note", "target": "I10"}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "I10"
    finally:
        tmp.cleanup()


def test_get_instances_wraps_integer_target_via_str_coercion():
    """For a numeric target, `str(int)` is applied. Pin this so a future refusal to coerce is
    intentional."""
    rows = [{"text": "Note", "target": 42}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "42"
    finally:
        tmp.cleanup()


def test_get_instances_mixed_target_types_across_rows(monkeypatch):
    """Same DataFrame, three rows, three distinct `target` types — all should produce sensible
    references.

    NB: pyarrow's feather format requires uniform list-element types per column, so we can't
    write a true heterogeneous-target DataFrame to disk. Instead, we monkeypatch `read_feather`
    to return a hand-built DataFrame and create an empty stand-in file to satisfy
    `check_file_exists`."""
    import helm.benchmark.scenarios.mimiciv_billing_code_scenario as mod

    fake_df = pd.DataFrame(
        {
            "text": ["Row 1 list", "Row 2 ndarray", "Row 3 scalar"],
            "target": pd.Series([["I10", "E11"], np.array(["J18"]), "Z00"], dtype=object),
        }
    )
    monkeypatch.setattr(mod.pd, "read_feather", lambda path: fake_df)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.feather")
        open(path, "w").close()
        scenario = MIMICIVBillingCodeScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert [inst.references[0].output.text for inst in instances] == [
        "I10,E11",
        "J18",
        "Z00",
    ]


def test_get_instances_handles_codes_with_decimal_points():
    """ICD-10 codes commonly contain a period (e.g. "I10.0", "N18.6"). Pin that the join
    preserves them verbatim and uses comma (NOT period) as the delimiter."""
    rows = [{"text": "Note", "target": ["I10.0", "N18.6", "E11.9"]}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == "I10.0,N18.6,E11.9"
    finally:
        tmp.cleanup()


def test_get_instances_coerces_non_string_list_elements_to_str(monkeypatch):
    """The join uses `str(code) for code in icd10_codes`, so int/float elements inside a
    list-target are stringified.

    NB: feather can't store a mixed-type list directly, so we monkeypatch `read_feather`."""
    import helm.benchmark.scenarios.mimiciv_billing_code_scenario as mod

    fake_df = pd.DataFrame(
        {
            "text": ["Note"],
            "target": pd.Series([["I10", 42, 3.14]], dtype=object),
        }
    )
    monkeypatch.setattr(mod.pd, "read_feather", lambda path: fake_df)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.feather")
        open(path, "w").close()
        scenario = MIMICIVBillingCodeScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].references[0].output.text == "I10,42,3.14"


def test_get_instances_empty_list_target_yields_empty_reference():
    """An empty `target` list joins to "". This pins what happens to admissions without
    coding."""
    rows = [{"text": "Note", "target": []}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].references[0].output.text == ""
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# `get_instances` — edge cases and error handling.
# ---------------------------------------------------------------------------


def test_get_instances_returns_empty_list_for_empty_dataframe():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.feather")
        df = pd.DataFrame({"text": pd.Series([], dtype=str), "target": pd.Series([], dtype=object)})
        df.to_feather(path)

        scenario = MIMICIVBillingCodeScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_raises_when_data_file_is_missing():
    with TemporaryDirectory() as tmp:
        scenario = MIMICIVBillingCodeScenario(data_path=os.path.join(tmp, "missing.feather"))
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_skips_rows_that_raise_during_processing(capsys, monkeypatch):
    """The per-row body is wrapped in `try/except`: a failing row is printed and skipped, the
    rest of the rows still produce instances.

    Strategy: insert a code object whose `__str__` raises. The scenario calls
    `str(code) for code in icd10_codes` inside the join, which triggers the exception. We use
    `monkeypatch` on `read_feather` because we can't roundtrip arbitrary Python objects
    through feather."""

    class _ExplodingStr:
        def __str__(self):
            raise RuntimeError("boom on str()")

    import helm.benchmark.scenarios.mimiciv_billing_code_scenario as mod

    fake_df = pd.DataFrame(
        {
            "text": ["Note OK 1", "Note WILL EXPLODE", "Note OK 2"],
            "target": pd.Series([["I10"], [_ExplodingStr()], ["J18"]], dtype=object),
        }
    )
    monkeypatch.setattr(mod.pd, "read_feather", lambda path: fake_df)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.feather")
        open(path, "w").close()
        scenario = MIMICIVBillingCodeScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    # The exploding row must NOT crash the whole scenario; only the good rows survive.
    assert len(instances) == 2
    assert [inst.input.text for inst in instances] == ["Note OK 1", "Note OK 2"]
    # The print statement from the except branch must surface "Error processing row".
    captured = capsys.readouterr()
    assert "Error processing row 1" in captured.out


def test_get_instances_preserves_unicode_in_note_text():
    """Discharge notes may contain non-ASCII characters (accented names, foreign drug
    names)."""
    rows = [{"text": "Patient: María García. Diagnóstico: hipertensión.", "target": ["I10"]}]
    scenario, tmp = _make_scenario_with_feather(rows)
    try:
        instances = scenario.get_instances(output_path=tmp.name)
        assert instances[0].input.text == "Patient: María García. Diagnóstico: hipertensión."
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MIMICIVBillingCodeScenario(data_path="/tmp/x.feather")
    metadata = scenario.get_metadata()

    assert metadata.name == "mimiciv_billing_code"
    assert metadata.display_name == "MIMIC-IV Billing Code"
    assert metadata.main_split == "test"
    # UNIQUE: this scenario uses a custom F1 metric, not "exact_match" like the SHC family.
    assert metadata.main_metric == "mimiciv_billing_code_f1"
    # The task is marked Classification even though it's effectively code extraction.
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert "ICD-10" in metadata.taxonomy.what


def test_metadata_when_field_uses_unique_discharge_phrasing():
    """UNIQUE: this scenario sets `when="During or after patient discharge"` — a descriptive
    phrase, distinct from the more terse values used elsewhere (`"Any"`, `"Pre-Trial"`,
    `"End-of-care"`). Pin this so a normalization is intentional."""
    scenario = MIMICIVBillingCodeScenario(data_path="/tmp/x.feather")
    assert scenario.get_metadata().taxonomy.when == "During or after patient discharge"


def test_metadata_who_field_has_typo_documented_bug():
    """KNOWN TYPO pinned for visibility: `who="Hospital Admistrator"` — missing the 'n' (same
    typo that appears in `shc_gip_scenario.py`). Pin so any cleanup is intentional and surfaces
    as a test failure that needs an updated expectation."""
    scenario = MIMICIVBillingCodeScenario(data_path="/tmp/x.feather")
    metadata = scenario.get_metadata()
    assert metadata.taxonomy.who == "Hospital Admistrator"
    # Guard: the correctly-spelled word must NOT be present (regression in case it's fixed
    # without updating the test).
    assert "Administrator" not in metadata.taxonomy.who


def test_metadata_description_mentions_billing_and_reimbursement():
    scenario = MIMICIVBillingCodeScenario(data_path="/tmp/x.feather")
    description = scenario.get_metadata().description

    assert "MIMIC-IV" in description
    assert "ICD-10" in description
    assert "discharge" in description
    assert "billing codes" in description
    assert "reimbursement" in description


# ---------------------------------------------------------------------------
# Documented typos / quirks in the source code itself.
# ---------------------------------------------------------------------------


def test_error_message_class_name_has_typo_documented_bug():
    """KNOWN TYPO in the error message string: `[MIMICIVBilligCodeScenario]` (single 'n')
    vs the actual class name `MIMICIVBillingCodeScenario` (double 'n'). Surface the typo via
    a deliberate mismatch test."""
    with TemporaryDirectory() as tmp:
        scenario = MIMICIVBillingCodeScenario(data_path=os.path.join(tmp, "nope.feather"))
        try:
            scenario.get_instances(output_path=tmp)
        except FileNotFoundError as exc:
            message = str(exc)
            # PINNED: the typo is present in the error message.
            assert "MIMICIVBilligCodeScenario" in message  # WRONG spelling (single 'n')
            # Guard: the corrected spelling is NOT in the message yet.
            assert "MIMICIVBillingCodeScenario" not in message
        else:
            pytest.fail("Expected FileNotFoundError for missing data file")
