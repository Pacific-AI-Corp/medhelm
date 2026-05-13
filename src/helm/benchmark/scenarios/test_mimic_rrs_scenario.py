import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.mimic_rrs_scenario import MIMICRRSScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_lines(path: str, lines: List[str]) -> None:
    """Write each list element as a separate line in the file at `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_split(data_dir: str, split_name: str, findings: List[str], impressions: List[str]) -> None:
    """Write the two `.tok` files MIMIC-RRS expects for a given split."""
    _write_lines(os.path.join(data_dir, f"{split_name}.findings.tok"), findings)
    _write_lines(os.path.join(data_dir, f"{split_name}.impression.tok"), impressions)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = MIMICRRSScenario(data_path="/tmp/mimic_rrs")
    assert scenario.data_path == "/tmp/mimic_rrs"


def test_class_attributes():
    assert MIMICRRSScenario.name == "mimic_rrs"
    assert MIMICRRSScenario.tags == ["question_answering", "biomedical"]
    assert "MIMIC-III" in MIMICRRSScenario.description


# ---------------------------------------------------------------------------
# `read_file` — simple text reader that strips each line.
# ---------------------------------------------------------------------------


def test_read_file_strips_each_line():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.txt")
        _write_lines(path, ["  hello world  ", "second  ", "   ", "fourth"])

        lines = MIMICRRSScenario(data_path="/tmp/x").read_file(path)

    assert lines == ["hello world", "second", "", "fourth"]


def test_read_file_returns_empty_list_for_empty_file():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.txt")
        open(path, "w", encoding="utf-8").close()

        assert MIMICRRSScenario(data_path="/tmp/x").read_file(path) == []


def test_read_file_preserves_unicode():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unicode.txt")
        _write_lines(path, ["Migrăña", "Lésion détectée"])

        lines = MIMICRRSScenario(data_path="/tmp/x").read_file(path)

    assert lines == ["Migrăña", "Lésion détectée"]


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic .tok files.
# ---------------------------------------------------------------------------


def test_get_instances_pairs_findings_with_impressions():
    """One input row + one impression row produces one Instance with the impression as the
    correct reference."""
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["The heart is normal. Lungs clear."],
            impressions=["No acute findings."],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].input.text == "The heart is normal. Lungs clear."
    assert instances[0].references[0].output.text == "No acute findings."
    assert CORRECT_TAG in instances[0].references[0].tags


def test_get_instances_emits_one_instance_per_row():
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=[f"Finding {i}" for i in range(5)],
            impressions=[f"Impression {i}" for i in range(5)],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 5
    for i, instance in enumerate(instances):
        assert instance.input.text == f"Finding {i}"
        assert instance.references[0].output.text == f"Impression {i}"


def test_get_instances_assigns_test_split_to_every_instance():
    """The scenario is zero-shot: only `test` is loaded, and every row maps to `TEST_SPLIT`."""
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["F1", "F2"],
            impressions=["I1", "I2"],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    with TemporaryDirectory() as tmp:
        _build_split(tmp, "test", findings=["F1", "F2"], impressions=["I1", "I2"])
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert len(instance.references) == 1
        assert CORRECT_TAG in instance.references[0].tags


def test_get_instances_skips_rows_where_finding_is_empty_after_strip():
    """`get_instances` performs `if not finding or not impression: continue`. Pure-whitespace
    lines collapse to "" after `read_file`'s strip and must be skipped."""
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["F1", "   ", "F3"],
            impressions=["I1", "I2", "I3"],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert [i.input.text for i in instances] == ["F1", "F3"]
    assert [i.references[0].output.text for i in instances] == ["I1", "I3"]


def test_get_instances_skips_rows_where_impression_is_empty_after_strip():
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["F1", "F2", "F3"],
            impressions=["I1", "", "I3"],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert [i.input.text for i in instances] == ["F1", "F3"]


def test_get_instances_raises_assertion_when_lengths_differ():
    """The two files must have the same number of lines; otherwise the `assert` blows up to
    prevent silently producing mis-paired data."""
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["F1", "F2"],
            impressions=["I1"],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        with pytest.raises(AssertionError, match="same length"):
            scenario.get_instances(output_path=tmp)


def test_get_instances_only_processes_test_split_even_if_train_files_exist():
    """The `splits` dict has `train`/`validate` commented out. Even if those files happen to
    exist alongside `test`, the scenario must NOT read them."""
    with TemporaryDirectory() as tmp:
        _build_split(tmp, "test", findings=["TEST_F"], impressions=["TEST_I"])
        # Add train and validate files that the scenario must NOT touch.
        _build_split(tmp, "train", findings=["TRAIN_F"], impressions=["TRAIN_I"])
        _build_split(tmp, "validate", findings=["VAL_F"], impressions=["VAL_I"])

        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert instances[0].input.text == "TEST_F"
    # If `train` or `validate` were being read, we'd see extra instances here.


def test_get_instances_raises_when_findings_file_is_missing():
    with TemporaryDirectory() as tmp:
        # Only create the impression file.
        _write_lines(os.path.join(tmp, "test.impression.tok"), ["I1"])

        scenario = MIMICRRSScenario(data_path=tmp)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_impression_file_is_missing():
    with TemporaryDirectory() as tmp:
        # Only create the findings file.
        _write_lines(os.path.join(tmp, "test.findings.tok"), ["F1"])

        scenario = MIMICRRSScenario(data_path=tmp)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_returns_empty_list_for_files_with_only_blanks():
    """Edge case: when every line in both files strips down to "", every row hits the
    `continue` branch and zero instances survive."""
    with TemporaryDirectory() as tmp:
        # Use the same number of "blank-after-strip" lines on each side so the length-equality
        # assert is satisfied; both files end up as three pure-whitespace rows.
        _build_split(
            tmp,
            "test",
            findings=["   ", "\t", "   "],
            impressions=["   ", "\t", "   "],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances == []


def test_get_instances_preserves_unicode_in_inputs_and_references():
    with TemporaryDirectory() as tmp:
        _build_split(
            tmp,
            "test",
            findings=["Légère atélectasie bilatérale."],
            impressions=["Sans anomalie aiguë."],
        )
        scenario = MIMICRRSScenario(data_path=tmp)
        instances = scenario.get_instances(output_path=tmp)

    assert instances[0].input.text == "Légère atélectasie bilatérale."
    assert instances[0].references[0].output.text == "Sans anomalie aiguë."


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = MIMICRRSScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "mimic_rrs"
    assert metadata.display_name == "MIMIC-RRS"
    assert metadata.short_display_name == "MIMIC-RRS"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "mimic_rrs_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.who == "Radiologist"
    assert metadata.taxonomy.when == "Post-imaging"
    assert metadata.taxonomy.language == "English"


def test_get_metadata_description_cites_chen_paper():
    scenario = MIMICRRSScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "Chen" in description
    assert "arxiv.org/abs/2211.08584" in description
