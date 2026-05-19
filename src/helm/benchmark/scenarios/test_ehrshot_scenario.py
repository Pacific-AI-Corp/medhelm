import datetime
import os
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest

from helm.benchmark.scenarios.ehrshot_scenario import (
    ACTION_COT_TMPL,
    ACTION_TMPL,
    CLINICAL_DEFS,
    CLINICAL_SHORT_DEFS,
    CODE_DEFS,
    CONFIG,
    EHRSHOTScenario,
    PERSONAS,
    TASK_DEFS,
    TASK_FULL_NAMES,
    TASK_QUESTIONS,
    _process_prior_events_chunk,
    base_prompt,
    codes_and_timestamps,
    codes_only,
    count_tokens,
    get_code_def,
    get_prior_events,
    get_task_config,
    lumia_prompt,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Static-data consistency. These are quick canary tests that catch typos and
# accidental partial edits to the per-task constants.
# ---------------------------------------------------------------------------


NEW_TASKS = {
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
}
GUO_TASKS = {"guo_los", "guo_readmission", "guo_icu"}
LAB_TASKS = {
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
}
ALL_TASKS = NEW_TASKS | GUO_TASKS | LAB_TASKS


def test_task_full_names_covers_every_task():
    assert set(TASK_FULL_NAMES.keys()) == ALL_TASKS


def test_task_questions_covers_every_task():
    assert set(TASK_QUESTIONS.keys()) == ALL_TASKS


def test_personas_covers_every_task_and_each_has_at_least_one_entry():
    assert set(PERSONAS.keys()) == ALL_TASKS
    for task, roles in PERSONAS.items():
        assert isinstance(roles, list)
        assert len(roles) >= 1, f"{task} has no personas"


def test_task_defs_covers_only_guo_and_lab_tasks():
    """`TASK_DEFS` is only used to populate clinical context for guo/lab tasks (the new_* tasks
    use the wikipedia `CLINICAL_DEFS` instead)."""
    assert set(TASK_DEFS.keys()) == GUO_TASKS | LAB_TASKS


def test_clinical_defs_and_short_defs_cover_only_new_tasks():
    assert set(CLINICAL_DEFS.keys()) == NEW_TASKS
    assert set(CLINICAL_SHORT_DEFS.keys()) == NEW_TASKS


def test_code_defs_covers_only_new_tasks():
    assert set(CODE_DEFS.keys()) == NEW_TASKS


def test_code_defs_have_non_empty_descendant_sets():
    """Each new_* task must have at least one parent → descendants entry; an empty descendants set
    would silently produce an empty Medical Code Definition section."""
    for task, parents in CODE_DEFS.items():
        assert parents, f"{task} has no parent codes"
        for parent, payload in parents.items():
            assert "descendants" in payload
            assert len(payload["descendants"]) >= 1


# ---------------------------------------------------------------------------
# `get_task_config` — routes to the correct config bucket based on prefix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", sorted(GUO_TASKS))
def test_get_task_config_returns_guo_config_for_guo_tasks(task):
    assert get_task_config(task) is CONFIG["guo"]


@pytest.mark.parametrize("task", sorted(LAB_TASKS))
def test_get_task_config_returns_lab_config_for_lab_tasks(task):
    assert get_task_config(task) is CONFIG["lab"]


@pytest.mark.parametrize("task", sorted(NEW_TASKS))
def test_get_task_config_returns_new_config_for_new_tasks(task):
    assert get_task_config(task) is CONFIG["new"]


def test_get_task_config_falls_back_to_new_for_unknown_prefix():
    """Anything not starting with 'guo' or 'lab' falls into the `new` bucket — including unknown
    names. Pin the behaviour so a future change here is intentional."""
    assert get_task_config("totally_unknown_task") is CONFIG["new"]


# ---------------------------------------------------------------------------
# `get_code_def` — flattens parent→descendants and sorts deterministically.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", sorted(NEW_TASKS))
def test_get_code_def_returns_sorted_unique_string(task):
    code_def = get_code_def(task)

    codes = code_def.split(", ")
    assert codes == sorted(codes), "codes are not lexicographically sorted"
    assert len(codes) == len(set(codes)), "codes contain duplicates"


def test_get_code_def_contains_the_parent_code_for_acutemi():
    """The parent SNOMED concept must appear in its own descendants set (sanity-check)."""
    assert "SNOMED/57054005" in get_code_def("new_acutemi")


def test_get_code_def_includes_every_descendant_for_celiac():
    """Celiac has the smallest descendants set; verify the whole set ends up in the output."""
    descendants = CODE_DEFS["new_celiac"]["SNOMED/396331005"]["descendants"]
    code_def = get_code_def("new_celiac")
    for code in descendants:
        assert code in code_def


# ---------------------------------------------------------------------------
# EHR converters: `codes_only` and `codes_and_timestamps`.
# ---------------------------------------------------------------------------


def test_codes_only_handles_dict_events():
    events = [{"code": "ICD10CM/C25.9"}, {"code": "ICD10CM/C25.0"}]
    assert codes_only(events) == "- ICD10CM/C25.9\n- ICD10CM/C25.0"


def test_codes_only_handles_plain_string_events():
    events = ["ICD10CM/C25.9", "SNOMED/1268532006"]
    assert codes_only(events) == "- ICD10CM/C25.9\n- SNOMED/1268532006"


def test_codes_only_supports_mixed_event_types():
    """The docstring promises both shapes can coexist in the same list."""
    events = ["ICD10CM/A.1", {"code": "ICD10CM/B.2"}]
    assert codes_only(events) == "- ICD10CM/A.1\n- ICD10CM/B.2"


def test_codes_only_returns_empty_string_for_empty_list():
    assert codes_only([]) == ""


def test_codes_and_timestamps_formats_iso_date_per_event():
    events = [
        {"time": datetime.datetime(2024, 1, 1), "code": "ICD10CM/C25.9"},
        {"time": datetime.datetime(2024, 12, 31), "code": "ICD10CM/C25.0"},
    ]
    assert codes_and_timestamps(events) == "- 2024-01-01 ICD10CM/C25.9\n- 2024-12-31 ICD10CM/C25.0"


def test_codes_and_timestamps_returns_empty_string_for_empty_list():
    assert codes_and_timestamps([]) == ""


# ---------------------------------------------------------------------------
# `base_prompt` — covers every combination of flags.
# ---------------------------------------------------------------------------


def test_base_prompt_returns_expected_dict_keys():
    out = base_prompt("new_hypertension")
    assert set(out.keys()) == {"instruction", "example", "delimiter"}
    assert out["example"] == "Patient EHR:\n{ehr}\n\n"
    assert out["delimiter"] == "\n##\n"


def test_base_prompt_persona_uses_first_role_lowercased():
    """The first persona for hypertension is 'Primary Care Physician', and the prompt must
    lowercase it when embedding into the persona sentence."""
    instruction = base_prompt("new_hypertension")["instruction"]
    assert "expert primary care physician at Stanford Healthcare" in instruction


def test_base_prompt_uses_task_full_name_lowercased_in_persona():
    instruction = base_prompt("new_hypertension")["instruction"]
    assert "You specialize in predicting hypertension." in instruction


def test_base_prompt_omits_persona_when_disabled():
    instruction = base_prompt("new_hypertension", is_include_persona=False)["instruction"]
    assert "Stanford Healthcare" not in instruction


def test_base_prompt_short_clinical_def_used_when_flag_set():
    instruction = base_prompt(
        "new_hypertension",
        is_include_clinical_def=True,
        is_use_short_clinical_def=True,
    )["instruction"]

    assert "Clinical Definition: " in instruction
    assert CLINICAL_SHORT_DEFS["new_hypertension"] in instruction
    # The long version must NOT also be appended.
    assert CLINICAL_DEFS["new_hypertension"] not in instruction


def test_base_prompt_full_clinical_def_used_when_short_flag_unset():
    instruction = base_prompt(
        "new_hypertension",
        is_include_clinical_def=True,
        is_use_short_clinical_def=False,
    )["instruction"]

    assert CLINICAL_DEFS["new_hypertension"] in instruction


def test_base_prompt_omits_clinical_def_when_disabled():
    instruction = base_prompt("new_hypertension", is_include_clinical_def=False)["instruction"]
    assert "Clinical Definition: " not in instruction


def test_base_prompt_code_def_included_when_flag_set():
    instruction = base_prompt(
        "new_hypertension",
        is_include_code_def=True,
    )["instruction"]
    assert "Medical Code Definition:" in instruction
    # Sample of a code we know is in the descendants set.
    assert "ICD10CM/I10" in instruction


def test_base_prompt_code_def_omitted_when_disabled():
    instruction = base_prompt("new_hypertension", is_include_code_def=False)["instruction"]
    assert "Medical Code Definition:" not in instruction


def test_base_prompt_uses_action_template_by_default():
    instruction = base_prompt("new_hypertension", is_include_cot=False)["instruction"]
    expected_question = TASK_QUESTIONS["new_hypertension"]
    assert ACTION_TMPL.format(question=expected_question) in instruction
    # The CoT-specific phrasing must NOT appear.
    assert "numbered list of the steps" not in instruction


def test_base_prompt_uses_cot_template_when_flag_set():
    instruction = base_prompt("new_hypertension", is_include_cot=True)["instruction"]
    expected_question = TASK_QUESTIONS["new_hypertension"]
    assert ACTION_COT_TMPL.format(question=expected_question) in instruction
    assert "numbered list of the steps" in instruction


def test_base_prompt_with_guo_task_only_includes_question_section():
    """guo_* tasks have neither a clinical def nor a code def in `CLINICAL_DEFS`/`CODE_DEFS`, so
    we must call `base_prompt` with those flags off — which `get_task_config("guo*")` does."""
    task_config = get_task_config("guo_icu")
    instruction = base_prompt("guo_icu", **task_config)["instruction"]
    assert "Clinical Definition:" not in instruction
    assert "Medical Code Definition:" not in instruction
    assert "Instruction: " in instruction
    assert "expert intensivist" in instruction  # first persona


# ---------------------------------------------------------------------------
# `lumia_prompt` — full prompt orchestration.
# ---------------------------------------------------------------------------


def test_lumia_prompt_uses_codes_only_converter_by_default():
    config = {"ehr_converter": "codes_only"}
    timeline = {"ehr": [{"code": "ICD10CM/C25.9"}, {"code": "ICD10CM/A.1"}]}

    prompt = lumia_prompt("new_acutemi", config, [], timeline)
    assert "- ICD10CM/C25.9" in prompt
    assert "- ICD10CM/A.1" in prompt
    # No timestamps.
    assert "2024" not in prompt


def test_lumia_prompt_uses_codes_and_timestamps_converter_when_configured():
    config = {"ehr_converter": "codes_and_timestamps"}
    timeline = {
        "ehr": [
            {"time": datetime.datetime(2024, 1, 1), "code": "ICD10CM/C25.9"},
        ],
    }

    prompt = lumia_prompt("new_acutemi", config, [], timeline)
    assert "2024-01-01 ICD10CM/C25.9" in prompt


def test_lumia_prompt_raises_on_unknown_converter_strategy():
    with pytest.raises(ValueError, match="Invalid `ehr_converter`"):
        lumia_prompt("new_acutemi", {"ehr_converter": "bogus"}, [], {"ehr": []})


def test_lumia_prompt_yes_no_replacement_is_a_noop_documented_quirk():
    """KNOWN QUIRK pinned to detect regressions:

    `lumia_prompt` calls
        tmpl["instruction"].replace(
            'Then respond with "yes" or "no" as your final output',
            'Then respond with "A" for yes or "B" for no as your final output',
        )
    But neither `ACTION_TMPL` nor `ACTION_COT_TMPL` actually contain that sentence — the action
    templates only say "answer the question: {question}". So the `replace` is a no-op, and the
    final prompt contains NEITHER the original phrase NOR the rewritten "A for yes / B for no"
    string. If a future change introduces a "yes/no" sentence to the templates, this test should
    be updated intentionally."""
    prompt = lumia_prompt(
        "new_acutemi",
        {"ehr_converter": "codes_only"},
        [],
        {"ehr": []},
    )
    assert 'Then respond with "yes" or "no"' not in prompt
    assert '"A" for yes or "B" for no' not in prompt


def test_lumia_prompt_has_sections_in_expected_order():
    """The final prompt structure is `# Instructions\\n...\\n# Your Task\\nPatient EHR:\\n...`."""
    prompt = lumia_prompt(
        "new_acutemi",
        {"ehr_converter": "codes_only"},
        [],
        {"ehr": [{"code": "X"}]},
    )
    instructions_idx = prompt.index("# Instructions")
    your_task_idx = prompt.index("# Your Task")
    ehr_idx = prompt.index("Patient EHR:")
    assert instructions_idx < your_task_idx < ehr_idx


# ---------------------------------------------------------------------------
# `count_tokens` — thin wrapper around tiktoken.
# ---------------------------------------------------------------------------


def test_count_tokens_returns_zero_for_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_returns_positive_for_non_empty_string():
    assert count_tokens("hello world") > 0


def test_count_tokens_scales_with_input_length():
    short_count = count_tokens("hello")
    long_count = count_tokens("hello " * 50)
    assert long_count > short_count


# ---------------------------------------------------------------------------
# `_process_prior_events_chunk` and `get_prior_events`.
# ---------------------------------------------------------------------------


def _make_data_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": 1, "time": pd.Timestamp("2024-01-01"), "code": "A"},
            {"subject_id": 1, "time": pd.Timestamp("2024-01-02"), "code": "B"},
            {"subject_id": 1, "time": pd.Timestamp("2024-01-10"), "code": "C"},
            {"subject_id": 2, "time": pd.Timestamp("2024-01-01"), "code": "X"},
        ]
    )


def test_process_prior_events_chunk_returns_codes_before_or_equal_to_prediction_time():
    df_data = _make_data_df()
    df_labels = pd.DataFrame(
        [
            {"subject_id": 1, "prediction_time": pd.Timestamp("2024-01-02")},
        ]
    )
    grouped = df_data.sort_values(["subject_id", "time"]).groupby("subject_id")

    result = _process_prior_events_chunk(df_labels, grouped)

    # Both 2024-01-01 (A) and 2024-01-02 (B) come at-or-before the prediction time; C does not.
    assert result == [["A", "B"]]


def test_process_prior_events_chunk_returns_empty_list_for_unknown_subject():
    df_data = _make_data_df()
    df_labels = pd.DataFrame([{"subject_id": 999, "prediction_time": pd.Timestamp("2024-01-02")}])
    grouped = df_data.sort_values(["subject_id", "time"]).groupby("subject_id")
    assert _process_prior_events_chunk(df_labels, grouped) == [[]]


def test_get_prior_events_serial_path_preserves_label_order():
    df_data = _make_data_df()
    df_labels = pd.DataFrame(
        [
            {"subject_id": 1, "prediction_time": pd.Timestamp("2024-01-02")},
            {"subject_id": 2, "prediction_time": pd.Timestamp("2024-01-01")},
            {"subject_id": 1, "prediction_time": pd.Timestamp("2024-01-15")},
        ]
    )

    # `get_prior_events` sorts labels by (subject_id, prediction_time) internally. We use
    # `n_procs=1` for deterministic, single-process behaviour.
    result = get_prior_events(df_data, df_labels, n_procs=1)

    # Convert results to a set-by-content lookup since order follows the sorted label table.
    counts = sorted(len(codes) for codes in result)
    assert counts == [
        1,
        2,
        3,
    ]  # subject 2 (1 event), subject 1 @ 2024-01-02 (2 events), subject 1 @ 2024-01-15 (3 events)


def test_get_prior_events_accepts_dataframe_with_no_matching_subjects():
    df_data = _make_data_df()
    df_labels = pd.DataFrame([{"subject_id": 12345, "prediction_time": pd.Timestamp("2024-01-02")}])
    assert get_prior_events(df_data, df_labels, n_procs=1) == [[]]


# ---------------------------------------------------------------------------
# `EHRSHOTScenario` class — construction + metadata.
# ---------------------------------------------------------------------------


def test_scenario_init_stores_subject_data_path_and_max_length():
    scenario = EHRSHOTScenario(subject="new_hypertension", data_path="/tmp/x", max_length=4096)
    assert scenario.subject == "new_hypertension"
    assert scenario.data_path == "/tmp/x"
    assert scenario.max_length == 4096


def test_scenario_init_max_length_defaults_to_none():
    scenario = EHRSHOTScenario(subject="new_hypertension", data_path="/tmp/x")
    assert scenario.max_length is None


def test_scenario_class_attributes():
    assert EHRSHOTScenario.name == "ehrshot"
    assert EHRSHOTScenario.POSSIBLE_ANSWER_CHOICES == ["yes", "no"]
    assert "EHRSHOT" in EHRSHOTScenario.description


def test_scenario_metadata():
    scenario = EHRSHOTScenario(subject="new_hypertension", data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "ehrshot"
    assert metadata.display_name == "EHRSHOT"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"


# ---------------------------------------------------------------------------
# `EHRSHOTScenario.get_instances` — exercised against a tiny synthetic parquet
# dataset, so no network / multiprocessing pool is required.
# ---------------------------------------------------------------------------


def _build_minimal_ehrshot_layout(data_path: str, task: str) -> None:
    """Create the directory layout and the parquet files that `create_benchmark` reads:

    data_path/
        data/data.parquet
        metadata/subject_splits.parquet
        labels/<task>/labels.parquet
    """
    os.makedirs(os.path.join(data_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "labels", task), exist_ok=True)

    df_data = pd.DataFrame(
        [
            {"subject_id": 1, "time": pd.Timestamp("2024-01-01"), "code": "ICD10CM/I10"},
            {"subject_id": 1, "time": pd.Timestamp("2024-02-01"), "code": "SNOMED/59621000"},
            {"subject_id": 2, "time": pd.Timestamp("2024-01-15"), "code": "ICD10CM/I10"},
        ]
    )
    df_data.to_parquet(os.path.join(data_path, "data", "data.parquet"))

    df_splits = pd.DataFrame(
        [
            {"subject_id": 1, "split": "held_out"},
            {"subject_id": 2, "split": "train"},
        ]
    ).set_index("subject_id")
    df_splits.to_parquet(os.path.join(data_path, "metadata", "subject_splits.parquet"))

    df_labels = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "prediction_time": pd.Timestamp("2024-03-01"),
                "boolean_value": True,
            },
            {
                "subject_id": 2,
                "prediction_time": pd.Timestamp("2024-02-01"),
                "boolean_value": False,
            },
        ]
    )
    df_labels.to_parquet(os.path.join(data_path, "labels", task, "labels.parquet"))


def test_get_instances_end_to_end_with_synthetic_parquets():
    """End-to-end smoke test: build a tiny dataset on disk, then run `get_instances`. This
    exercises `create_benchmark`, `get_prior_events` (with n_procs=1 via multiprocessing pool
    of 4 — pandas tolerates this for tiny inputs), label/split joining, prompt generation, and
    reference tagging."""
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        _build_minimal_ehrshot_layout(data_dir, task="new_hypertension")

        scenario = EHRSHOTScenario(subject="new_hypertension", data_path=data_dir)
        instances = scenario.get_instances(output_path=output_dir)

    assert len(instances) == 2
    # Every instance is placed in TEST_SPLIT regardless of source split.
    assert all(instance.split == TEST_SPLIT for instance in instances)
    # Each instance carries two references (yes/no) with exactly one tagged correct.
    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["yes", "no"]
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1
    # The first label is True ("yes"); the second is False ("no").
    assert instances[0].references[0].is_correct  # "yes" is correct
    assert instances[1].references[1].is_correct  # "no" is correct


def test_get_instances_creates_cached_parquet_on_first_run():
    """The first run materialises a `medhelm_prompts.parquet` under `output_path/<subject>/`.
    On a second run, `get_instances` should *not* rebuild it from scratch."""
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        _build_minimal_ehrshot_layout(data_dir, task="new_hypertension")
        scenario = EHRSHOTScenario(subject="new_hypertension", data_path=data_dir)

        scenario.get_instances(output_path=output_dir)
        cached_path = os.path.join(output_dir, "new_hypertension", "medhelm_prompts.parquet")
        assert os.path.exists(cached_path), "expected cached prompts parquet"

        # A second run reads the cache and produces the same number of instances.
        cached_mtime_before = os.path.getmtime(cached_path)
        instances_again = scenario.get_instances(output_path=output_dir)
        assert len(instances_again) == 2
        assert os.path.getmtime(cached_path) == cached_mtime_before, "cache should not be rewritten"


def test_get_instances_skips_prompts_exceeding_max_length():
    """`max_length` is enforced via `count_tokens` after the prompt is built. We set it to 1 so
    every prompt — which contains a long instruction block — is over budget and skipped."""
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        _build_minimal_ehrshot_layout(data_dir, task="new_hypertension")
        scenario = EHRSHOTScenario(subject="new_hypertension", data_path=data_dir, max_length=1)
        instances = scenario.get_instances(output_path=output_dir)

    assert instances == []


def test_get_instances_prompt_contains_task_specific_section():
    """The generated prompt must include both the persona for the task and at least one of the
    task's medical codes (since the new_* config enables `is_include_code_def`)."""
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        _build_minimal_ehrshot_layout(data_dir, task="new_hypertension")
        scenario = EHRSHOTScenario(subject="new_hypertension", data_path=data_dir)
        instances = scenario.get_instances(output_path=output_dir)

    prompt_text = instances[0].input.text
    assert "expert primary care physician at Stanford Healthcare" in prompt_text
    assert "Medical Code Definition" in prompt_text
    assert "ICD10CM/I10" in prompt_text
    assert "Patient EHR:" in prompt_text


def test_get_instances_raises_when_data_parquet_is_missing():
    with TemporaryDirectory() as data_dir, TemporaryDirectory() as output_dir:
        # Intentionally do NOT populate the layout.
        scenario = EHRSHOTScenario(subject="new_hypertension", data_path=data_dir)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=output_dir)


# ---------------------------------------------------------------------------
# Constants self-consistency: every persona is non-empty.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", sorted(ALL_TASKS))
def test_personas_first_entry_is_non_empty_for_every_task(task):
    """`base_prompt` blindly indexes `PERSONAS[task][0]` and would crash on an empty value."""
    role: List[str] = PERSONAS[task]
    assert role
    assert role[0] and role[0].strip()
