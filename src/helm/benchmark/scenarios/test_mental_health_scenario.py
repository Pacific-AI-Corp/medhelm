import os
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest

from helm.benchmark.scenarios.mental_health_scenario import MentalHealthScenario
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    PassageQuestionInput,
    TEST_SPLIT,
)


REQUIRED_COLUMNS = ["topic", "dialogue_type", "context", "gold_counselor_response"]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _row(
    topic: str = "Anxiety",
    dialogue_type: str | int = "type_1",
    context: str = "counselor: Hi.\nclient: I feel stressed.",
    gold_counselor_response: str = "I hear you; let's talk about it.",
) -> dict:
    return {
        "topic": topic,
        "dialogue_type": dialogue_type,
        "context": context,
        "gold_counselor_response": gold_counselor_response,
    }


def _make_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _write_csv(path: str, rows: List[dict]) -> None:
    df = _make_df(rows)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = MentalHealthScenario(data_path="/tmp/mh.csv")
    assert scenario.data_path == "/tmp/mh.csv"


def test_class_attributes():
    assert MentalHealthScenario.name == "mental_health"
    # Pin all 5 tags to detect drift.
    assert MentalHealthScenario.tags == [
        "dialogue",
        "counseling",
        "mental_health",
        "empathy",
        "healthcare",
    ]
    assert "MentalHealth" in MentalHealthScenario.description


# ---------------------------------------------------------------------------
# `process_dialogue_data` — pure logic, easy to exercise.
# ---------------------------------------------------------------------------


def test_process_dialogue_data_single_row_produces_single_instance():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row()])

    instances = scenario.process_dialogue_data(df)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.split == TEST_SPLIT
    assert len(instance.references) == 1
    assert CORRECT_TAG in instance.references[0].tags


def test_process_dialogue_data_reference_text_is_gold_response():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row(gold_counselor_response="That sounds really difficult.")])

    instances = scenario.process_dialogue_data(df)

    assert instances[0].references[0].output.text == "That sounds really difficult."


def test_process_dialogue_data_input_wraps_in_passage_question_input():
    """The scenario uses `PassageQuestionInput(passage="", question=...)`. With
    `passage=""`, the resulting `.text` is prefixed with `"\\nQuestion: "` (the default
    `question_prefix` of `PassageQuestionInput`)."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row()])

    instances = scenario.process_dialogue_data(df)

    assert isinstance(instances[0].input, PassageQuestionInput)
    assert instances[0].input.text.startswith("\nQuestion: ")


def test_process_dialogue_data_input_text_contains_all_four_sections():
    """The composed input must include: `Topic: ...`, `Type: ...`, the conversation history,
    and the final instruction to generate a counselor response."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df(
        [
            _row(
                topic="Workplace",
                dialogue_type="conversational",
                context="counselor: Welcome.\nclient: I am stuck.",
            )
        ]
    )

    instances = scenario.process_dialogue_data(df)
    text = instances[0].input.text

    assert "Topic: Workplace" in text
    assert "Type: conversational" in text
    assert "Previous conversation:\ncounselor: Welcome.\nclient: I am stuck." in text
    assert "Generate an empathetic and appropriate counselor response:" in text


def test_process_dialogue_data_input_section_ordering():
    """Sections must appear in the documented order: Topic → Type → Previous conversation →
    Generate instruction (otherwise the model receives a malformed prompt)."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row(topic="Anxiety", dialogue_type="type_1")])

    text = scenario.process_dialogue_data(df)[0].input.text
    idx_topic = text.index("Topic:")
    idx_type = text.index("Type:")
    idx_conv = text.index("Previous conversation:")
    idx_gen = text.index("Generate an empathetic")

    assert idx_topic < idx_type < idx_conv < idx_gen


def test_process_dialogue_data_preserves_multiline_context_verbatim():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    multi_turn_context = (
        "counselor: How are you?\n" "client: Tired.\n" "counselor: Tell me more.\n" "client: I haven't slept in days."
    )
    df = _make_df([_row(context=multi_turn_context)])

    text = scenario.process_dialogue_data(df)[0].input.text
    assert multi_turn_context in text


def test_process_dialogue_data_emits_one_instance_per_row_preserving_order():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    rows = [_row(topic=f"Topic{i}", gold_counselor_response=f"Response {i}") for i in range(4)]
    df = _make_df(rows)

    instances = scenario.process_dialogue_data(df)

    assert len(instances) == 4
    assert [i.references[0].output.text for i in instances] == [f"Response {i}" for i in range(4)]


def test_process_dialogue_data_empty_dataframe_returns_no_instances():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    empty_df = _make_df([])
    assert scenario.process_dialogue_data(empty_df) == []


def test_process_dialogue_data_assigns_test_split_to_every_instance():
    """All examples are held out for zero-shot evaluation."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row(), _row()])

    instances = scenario.process_dialogue_data(df)
    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_process_dialogue_data_every_instance_has_exactly_one_correct_reference():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row(), _row(), _row()])

    instances = scenario.process_dialogue_data(df)
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_process_dialogue_data_preserves_unicode_in_fields():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df(
        [
            _row(
                topic="Ansiedad",
                context="cliente: Estoy abrumado.\nconsejero: Te escucho.",
                gold_counselor_response="Lamento que estés pasando por esto.",
            )
        ]
    )

    text = scenario.process_dialogue_data(df)[0].input.text
    assert "Topic: Ansiedad" in text
    assert "Estoy abrumado." in text
    assert scenario.process_dialogue_data(df)[0].references[0].output.text == "Lamento que estés pasando por esto."


def test_process_dialogue_data_accepts_numeric_dialogue_type():
    """`dialogue_type` is documented as a numerical identifier; the f-string formatter must
    render it as a stringified number rather than crashing."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    df = _make_df([_row(dialogue_type=42)])

    text = scenario.process_dialogue_data(df)[0].input.text
    assert "Type: 42" in text


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against a synthetic CSV.
# ---------------------------------------------------------------------------


def test_get_instances_end_to_end_with_synthetic_csv():
    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "mh.csv")
        _write_csv(
            csv_path,
            [
                _row(topic="Anxiety", gold_counselor_response="R1"),
                _row(topic="Sleep", gold_counselor_response="R2"),
            ],
        )
        scenario = MentalHealthScenario(data_path=csv_path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    assert "Topic: Anxiety" in instances[0].input.text
    assert "Topic: Sleep" in instances[1].input.text
    assert [i.references[0].output.text for i in instances] == ["R1", "R2"]


def test_get_instances_returns_empty_list_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "mh.csv")
        _write_csv(csv_path, [])
        scenario = MentalHealthScenario(data_path=csv_path)
        assert scenario.get_instances(output_path=tmp) == []


def test_get_instances_raises_when_data_file_is_missing():
    """`check_file_exists` raises before any pandas parsing happens."""
    with TemporaryDirectory() as tmp:
        scenario = MentalHealthScenario(data_path=os.path.join(tmp, "missing.csv"))
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


@pytest.mark.parametrize("missing_column", REQUIRED_COLUMNS)
def test_get_instances_raises_when_a_required_column_is_missing(missing_column):
    """If any of the four required columns is missing, `row[col]` inside `process_dialogue_data`
    raises `KeyError`."""
    cols = [c for c in REQUIRED_COLUMNS if c != missing_column]
    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "mh.csv")
        # Build a CSV with only the surviving columns.
        df = pd.DataFrame([{c: "x" for c in cols}], columns=cols)
        df.to_csv(csv_path, index=False)
        scenario = MentalHealthScenario(data_path=csv_path)

        with pytest.raises(KeyError, match=missing_column):
            scenario.get_instances(output_path=tmp)


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_get_metadata_returns_expected_taxonomy():
    scenario = MentalHealthScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "mental_health"
    assert metadata.display_name == "MentalHealth"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "mental_health_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"
    # `who` lists both counselors and patients.
    assert "Counselors" in metadata.taxonomy.who
    assert "Patients" in metadata.taxonomy.who


def test_get_metadata_description_mentions_empathetic_communication():
    """A canary on the public-facing copy: the description must mention empathy/counseling so
    the dashboard reads correctly."""
    scenario = MentalHealthScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "empathetic" in description.lower()
    assert "counseling" in description.lower()
