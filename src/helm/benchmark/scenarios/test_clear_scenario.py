import os
import pytest
from tempfile import TemporaryDirectory

import pandas as pd

from helm.benchmark.scenarios.clear_scenario import CLEARScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_scenario(condition: str, data_path: str) -> CLEARScenario:
    return CLEARScenario(condition=condition, data_path=data_path)


def _write_excel(data_path: str, condition: str, rows: list) -> str:
    """Write a synthetic xlsx file named `{condition}.xlsx` in `data_path`. Returns the path."""
    df = pd.DataFrame(rows)
    excel_path = os.path.join(data_path, f"{condition}.xlsx")
    df.to_excel(excel_path, index=False, engine="openpyxl")
    return excel_path


# ---------------------------------------------------------------------------
# Tests for `__init__` validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", CLEARScenario.CONDITIONS)
def test_init_accepts_every_supported_condition(condition):
    scenario = _make_scenario(condition=condition, data_path="/tmp/dummy")

    assert scenario.condition == condition
    assert scenario.data_path == "/tmp/dummy"


def test_init_rejects_unknown_condition():
    with pytest.raises(ValueError, match="not supported"):
        _make_scenario(condition="not_a_condition", data_path="/tmp/x")


def test_init_rejects_empty_condition():
    with pytest.raises(ValueError):
        _make_scenario(condition="", data_path="/tmp/x")


# ---------------------------------------------------------------------------
# Dynamic attributes set in `__init__`.
# ---------------------------------------------------------------------------


def test_init_sets_per_condition_scenario_name():
    """The scenario name encodes the condition so each one can be tracked separately."""
    scenario = _make_scenario(condition="major_depression", data_path="/tmp/x")
    assert scenario.name == "clear_major_depression"


def test_init_sets_tags_with_hyphenated_condition():
    """The third tag must turn snake_case into kebab-case (e.g. for slug-style consumption)."""
    scenario = _make_scenario(condition="post_traumatic_stress_disorder", data_path="/tmp/x")

    assert scenario.tags == ["classification", "biomedical", "post-traumatic-stress-disorder"]


def test_init_sets_description_with_clear_keyword():
    scenario = _make_scenario(condition="alcohol_dependence", data_path="/tmp/x")
    assert "CLEAR" in scenario.description


# ---------------------------------------------------------------------------
# `CONDITIONS` and `CONDITION_PROMPTS` consistency.
# ---------------------------------------------------------------------------


def test_conditions_list_has_thirteen_entries():
    assert len(CLEARScenario.CONDITIONS) == 13


def test_condition_prompts_covers_every_condition():
    """Each condition must have a human-readable prompt; missing entries would crash
    `get_answer_choices` and `get_instances` for that condition."""
    assert set(CLEARScenario.CONDITION_PROMPTS.keys()) == set(CLEARScenario.CONDITIONS)


def test_condition_prompts_have_no_empty_values():
    for condition, prompt in CLEARScenario.CONDITION_PROMPTS.items():
        assert prompt, f"Empty human-readable prompt for {condition!r}"


# ---------------------------------------------------------------------------
# `get_answer_choices` (pure logic, no I/O).
# ---------------------------------------------------------------------------


def test_get_answer_choices_returns_three_options():
    scenario = _make_scenario(condition="alcohol_dependence", data_path="/tmp/x")
    choices = scenario.get_answer_choices()

    assert len(choices) == 3
    assert choices[0] == "Has a history of alcohol dependence"
    assert choices[1] == "Does not have a history of alcohol dependence"
    assert choices[2] == "Uncertain"


def test_get_answer_choices_uses_human_readable_condition_text():
    """The PTSD condition has a parenthesised abbreviation; the choices must include it verbatim."""
    scenario = _make_scenario(condition="post_traumatic_stress_disorder", data_path="/tmp/x")
    choices = scenario.get_answer_choices()

    assert choices[0] == "Has a history of post-traumatic stress disorder (PTSD)"
    assert choices[1] == "Does not have a history of post-traumatic stress disorder (PTSD)"
    assert choices[2] == "Uncertain"


@pytest.mark.parametrize("condition", CLEARScenario.CONDITIONS)
def test_get_answer_choices_always_returns_three(condition):
    scenario = _make_scenario(condition=condition, data_path="/tmp/x")
    assert len(scenario.get_answer_choices()) == 3


# ---------------------------------------------------------------------------
# `get_instances` with synthetic xlsx files.
# ---------------------------------------------------------------------------


def test_get_instances_label_one_marks_choice_a():
    """Label "1" must mark "Has a history of <condition>" as the correct reference."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "alcohol_dependence",
            [{"text": "Patient drinks daily.", "result_human": 1}],
        )
        scenario = _make_scenario(condition="alcohol_dependence", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert len(instances) == 1
    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert len(correct) == 1
    assert correct[0].output.text == "Has a history of alcohol dependence"


def test_get_instances_label_zero_marks_choice_b():
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "alcohol_dependence",
            [{"text": "Patient denies use.", "result_human": 0}],
        )
        scenario = _make_scenario(condition="alcohol_dependence", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert correct[0].output.text == "Does not have a history of alcohol dependence"


def test_get_instances_label_two_marks_choice_c_uncertain():
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "alcohol_dependence",
            [{"text": "Note is ambiguous.", "result_human": 2}],
        )
        scenario = _make_scenario(condition="alcohol_dependence", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert correct[0].output.text == "Uncertain"


def test_get_instances_references_always_have_three_options_in_fixed_order():
    """Even when only one is correct, all 3 references are emitted in A/B/C order."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "bipolar_disorder",
            [{"text": "Some note.", "result_human": 0}],
        )
        scenario = _make_scenario(condition="bipolar_disorder", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    texts = [ref.output.text for ref in instances[0].references]
    assert texts == [
        "Has a history of bipolar disorder",
        "Does not have a history of bipolar disorder",
        "Uncertain",
    ]


def test_get_instances_unknown_label_falls_back_to_literal():
    """If `result_human` is not in {0, 1, 2}, the literal value is used as `mapped_label`. None of
    the canonical choices match it, so NO reference is marked correct."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "homelessness",
            [{"text": "Note.", "result_human": 9}],
        )
        scenario = _make_scenario(condition="homelessness", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert len(instances) == 1
    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert correct == []


def test_get_instances_skips_rows_with_whitespace_only_text():
    """`str(text).strip()` collapses pure whitespace to "" → row is skipped by `if not text`."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "chronic_pain",
            [
                {"text": "Has note.", "result_human": 1},
                {"text": "   \t  ", "result_human": 0},  # whitespace only -> skipped
                {"text": "Has note 2.", "result_human": 2},
            ],
        )
        scenario = _make_scenario(condition="chronic_pain", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert len(instances) == 2
    assert instances[0].input.text.count("Has note.") == 1
    assert instances[1].input.text.count("Has note 2.") == 1


def test_get_instances_does_not_skip_rows_with_nan_values_documented_limitation():
    """KNOWN LIMITATION (pinned to detect regressions / intentional fixes):

    pandas turns blank/`None` xlsx cells into NaN. The scenario calls `str(NaN).strip()` which
    yields the literal string "nan" — a truthy non-empty value — so rows with NaN labels are
    NOT skipped by `if not text or not label`. Worse, the presence of a NaN in the column
    promotes the *entire* `result_human` column to `float`, so every integer label is also read
    back as e.g. `1.0`, then stringified as "1.0", which is NOT in `{"0", "1", "2"}`. The net
    effect: even valid rows fail to receive a correct reference whenever any row has a NaN.

    If a future change makes the scenario coerce labels to int or skip NaN rows, this test
    should be updated intentionally."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "tobacco_dependence",
            [
                {"text": "Note A.", "result_human": 1},
                {"text": "Note B.", "result_human": None},
            ],
        )
        scenario = _make_scenario(condition="tobacco_dependence", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    # Both rows survive — the NaN row is NOT skipped.
    assert len(instances) == 2
    # Neither row is properly classified, because the promotion to float collapses "1" to "1.0"
    # and the NaN row stringifies to "nan", and neither matches the {"0", "1", "2"} keys.
    for instance in instances:
        assert [ref for ref in instance.references if ref.is_correct] == []


def test_get_instances_empty_dataframe_returns_no_instances():
    with TemporaryDirectory() as data_dir:
        _write_excel(data_dir, "liver_disease", [])
        scenario = _make_scenario(condition="liver_disease", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert instances == []


def test_get_instances_propagates_test_split():
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "suicidal_behavior",
            [{"text": "Note.", "result_human": 1}],
        )
        scenario = _make_scenario(condition="suicidal_behavior", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert instances[0].split == TEST_SPLIT


def test_get_instances_input_text_contains_condition_choices_and_answer_prompt():
    """The composite prompt should include the condition descriptor, all three A/B/C lines, and
    the final 'Answer:' marker so the model knows where to write."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "personality_disorder",
            [{"text": "Patient note.", "result_human": 1}],
        )
        scenario = _make_scenario(condition="personality_disorder", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    text = instances[0].input.text
    assert "Determine whether the patient has a history of personality disorder." in text
    assert "Original Clinical Note:\nPatient note." in text
    assert "A. Has a history of personality disorder" in text
    assert "B. Does not have a history of personality disorder" in text
    assert "C. Uncertain" in text
    assert text.endswith("Answer:")


def test_get_instances_strips_whitespace_from_text_and_label():
    """`str(...).strip()` is applied to both columns; pad both ends with whitespace and verify."""
    with TemporaryDirectory() as data_dir:
        _write_excel(
            data_dir,
            "unemployment",
            [{"text": "  Patient is jobless.  ", "result_human": "  1  "}],
        )
        scenario = _make_scenario(condition="unemployment", data_path=data_dir)
        instances = scenario.get_instances(output_path=data_dir)

    assert len(instances) == 1
    assert "Patient is jobless." in instances[0].input.text
    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert correct[0].output.text == "Has a history of unemployment"


# ---------------------------------------------------------------------------
# Missing-file behavior (the scenario *does not* download data on its own).
# ---------------------------------------------------------------------------


def test_get_instances_raises_when_required_xlsx_is_missing():
    """`check_file_exists` raises when the per-condition file is not present in `data_path`."""
    with TemporaryDirectory() as empty_dir:
        scenario = _make_scenario(condition="bipolar_disorder", data_path=empty_dir)

        # We do not pin the exception class because `check_file_exists` may evolve; we just want
        # to confirm that *some* error surfaces instead of returning silently with no instances.
        with pytest.raises(Exception):
            scenario.get_instances(output_path=empty_dir)


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    """Note: `get_metadata` returns the scenario family name `"clear"`, NOT the per-condition
    `self.name`. That asymmetry is intentional — the dashboard groups all conditions together."""
    scenario = _make_scenario(condition="alcohol_dependence", data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "clear"
    assert metadata.display_name == "CLEAR"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
