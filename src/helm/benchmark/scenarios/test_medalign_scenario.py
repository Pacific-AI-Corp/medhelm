import pytest
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd

from helm.benchmark.scenarios.medalign_scenario import MedalignScenario
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    PassageQuestionInput,
    TEST_SPLIT,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_scenario(max_length: int = 4096, data_path: str = "/tmp/medalign") -> MedalignScenario:
    return MedalignScenario(max_length=max_length, data_path=data_path)


def _make_df(rows: List[dict]) -> pd.DataFrame:
    """Build the minimal DataFrame shape that `process_tsv` consumes."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Class-level / constructor tests.
# ---------------------------------------------------------------------------


def test_init_stores_max_length_and_data_path():
    scenario = MedalignScenario(max_length=8192, data_path="/tmp/medalign")
    assert scenario.max_length == 8192
    assert scenario.data_path == "/tmp/medalign"


def test_class_attributes():
    assert MedalignScenario.name == "medalign"
    assert "MedAlign" in MedalignScenario.description
    assert MedalignScenario.tags == ["knowledge", "reasoning", "biomedical"]


# ---------------------------------------------------------------------------
# `process_tsv` — pure logic, easy to exercise with synthetic DataFrames.
# ---------------------------------------------------------------------------


def test_process_tsv_single_row_creates_single_instance():
    scenario = _make_scenario()
    df = _make_df(
        [
            {
                "prompt": "Summarize the patient note.",
                "clinician_response": "Patient stable; discharge approved.",
            }
        ]
    )

    instances = scenario.process_tsv(df)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.split == TEST_SPLIT
    assert len(instance.references) == 1
    assert instance.references[0].output.text == "Patient stable; discharge approved."
    assert CORRECT_TAG in instance.references[0].tags
    assert instance.references[0].is_correct


def test_process_tsv_returns_passage_question_input_with_empty_passage():
    """The scenario builds `PassageQuestionInput(passage="", question=...)`. Because the parent
    class concatenates `passage_prefix + passage + separator + question_prefix + question`, with
    `passage=""` and the default `separator="\\n"`, the prompt text always starts with a leading
    newline followed by `Question: `."""
    scenario = _make_scenario()
    df = _make_df([{"prompt": "What is the diagnosis?", "clinician_response": "Hypertension."}])

    instances = scenario.process_tsv(df)

    assert isinstance(instances[0].input, PassageQuestionInput)
    assert instances[0].input.text == "\nQuestion: What is the diagnosis?"


def test_process_tsv_preserves_row_order():
    scenario = _make_scenario()
    df = _make_df(
        [
            {"prompt": "Q1", "clinician_response": "A1"},
            {"prompt": "Q2", "clinician_response": "A2"},
            {"prompt": "Q3", "clinician_response": "A3"},
        ]
    )

    instances = scenario.process_tsv(df)

    assert [i.references[0].output.text for i in instances] == ["A1", "A2", "A3"]
    assert [i.input.text for i in instances] == [
        "\nQuestion: Q1",
        "\nQuestion: Q2",
        "\nQuestion: Q3",
    ]


def test_process_tsv_returns_empty_list_for_empty_dataframe():
    scenario = _make_scenario()
    df = _make_df([])
    # Ensure the empty DataFrame still has the expected columns so `iterrows` is safe.
    df = pd.DataFrame(columns=["prompt", "clinician_response"])
    assert scenario.process_tsv(df) == []


def test_process_tsv_marks_every_instance_as_test_split():
    """MedAlign is a zero-shot evaluation dataset; every instance must land in `TEST_SPLIT`."""
    scenario = _make_scenario()
    df = _make_df([{"prompt": f"Q{i}", "clinician_response": f"A{i}"} for i in range(5)])

    instances = scenario.process_tsv(df)

    assert len(instances) == 5
    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_process_tsv_every_instance_has_exactly_one_correct_reference():
    scenario = _make_scenario()
    df = _make_df([{"prompt": f"Q{i}", "clinician_response": f"A{i}"} for i in range(3)])

    instances = scenario.process_tsv(df)

    for instance in instances:
        assert len(instance.references) == 1
        assert CORRECT_TAG in instance.references[0].tags


def test_process_tsv_preserves_multiline_responses_verbatim():
    """The clinician_response may contain newlines, lists, etc.; nothing in `process_tsv` should
    rewrite or strip them."""
    scenario = _make_scenario()
    multi = "Step 1.\nStep 2.\n- bullet"
    df = _make_df([{"prompt": "Plan?", "clinician_response": multi}])

    instances = scenario.process_tsv(df)
    assert instances[0].references[0].output.text == multi


def test_process_tsv_preserves_unicode_in_prompt_and_response():
    scenario = _make_scenario()
    df = _make_df([{"prompt": "¿Cuál es el diagnóstico?", "clinician_response": "Migrăña — sí."}])

    instances = scenario.process_tsv(df)

    assert instances[0].input.text == "\nQuestion: ¿Cuál es el diagnóstico?"
    assert instances[0].references[0].output.text == "Migrăña — sí."


# ---------------------------------------------------------------------------
# `get_instances` — mock the heavy helper so we never touch the real dataset.
# ---------------------------------------------------------------------------


def test_get_instances_delegates_to_return_dataset_dataframe(monkeypatch):
    """`get_instances` must call `return_dataset_dataframe(max_length, data_path)` exactly once
    and then feed the resulting DataFrame through `process_tsv` to produce instances."""
    recorded = {}

    def _fake(max_length, data_path):
        recorded["max_length"] = max_length
        recorded["data_path"] = data_path
        return _make_df(
            [
                {"prompt": "Why is potassium high?", "clinician_response": "Renal failure."},
                {"prompt": "Best next step?", "clinician_response": "IV insulin."},
            ]
        )

    monkeypatch.setattr("helm.benchmark.scenarios.medalign_scenario.return_dataset_dataframe", _fake)
    scenario = MedalignScenario(max_length=2048, data_path="/some/path")

    with TemporaryDirectory() as tmp:
        instances = scenario.get_instances(output_path=tmp)

    assert recorded == {"max_length": 2048, "data_path": "/some/path"}
    assert len(instances) == 2
    assert instances[0].references[0].output.text == "Renal failure."
    assert instances[1].references[0].output.text == "IV insulin."


def test_get_instances_returns_empty_list_for_empty_helper_result(monkeypatch):
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medalign_scenario.return_dataset_dataframe",
        lambda max_length, data_path: pd.DataFrame(columns=["prompt", "clinician_response"]),
    )
    scenario = _make_scenario()
    with TemporaryDirectory() as tmp:
        assert scenario.get_instances(output_path=tmp) == []


def test_get_instances_propagates_helper_errors(monkeypatch):
    """If the helper raises (e.g. missing TSV file), `get_instances` should surface the error
    rather than silently returning [], so the user sees the misconfiguration."""

    def _broken_helper(max_length, data_path):
        raise FileNotFoundError("instructions tsv missing")

    monkeypatch.setattr(
        "helm.benchmark.scenarios.medalign_scenario.return_dataset_dataframe",
        _broken_helper,
    )
    scenario = _make_scenario()
    with TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="instructions tsv missing"):
            scenario.get_instances(output_path=tmp)


# ---------------------------------------------------------------------------
# `get_metadata`.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = _make_scenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medalign"
    assert metadata.display_name == "MedAlign"
    assert metadata.short_display_name == "MedAlign"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "medalign_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"
    assert "Clinician" in metadata.taxonomy.who


def test_get_metadata_description_cites_fleming_paper():
    """The dashboard renders a markdown citation link to the arXiv paper; pin it here so a
    bad-merge that removes the link gets caught."""
    scenario = _make_scenario()
    description = scenario.get_metadata().description
    assert "Fleming" in description
    assert "arxiv.org/abs/2308.14089" in description
