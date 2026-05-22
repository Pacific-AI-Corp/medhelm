"""Run spec functions for medical scenarios outside the MedHELM leaderboard schema."""

from typing import Any, Dict

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_basic_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_generative_harms_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.lex_glue_scenario import TaskType
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("custom_mcqa")
def get_custom_mcqa_spec(
    path: str,
    num_train_instances: int = 0,
    method: str = ADAPT_MULTIPLE_CHOICE_JOINT,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.custom_mcqa_scenario.CustomMCQAScenario",
        args={
            "path": path,
            "num_train_instances": num_train_instances,
        },
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers).",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=num_train_instances,
    )

    return RunSpec(
        name=f"custom_mcqa,path={path},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["custom"],
    )


@run_spec_function("covid_dialog")
def get_covid_dialog_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.covid_dialog_scenario.COVIDDialogScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Generate a response given a patient's questions and concerns.",
        input_noun="Patient",
        output_noun="Doctor",
        max_tokens=128,
    )

    return RunSpec(
        name="covid_dialog",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["COVIDDialog"],
    )


@run_spec_function("me_q_sum")
def get_me_q_sum_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.me_q_sum_scenario.MeQSumScenario", args={})

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=128,
        temperature=0.3,
    )

    return RunSpec(
        name="me_q_sum",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MeQSum"],
    )


@run_spec_function("med_mcqa")
def get_med_mcqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.med_mcqa_scenario.MedMCQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Give a letter answer among A, B, C or D.",
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="med_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["med_mcqa"],
    )


@run_spec_function("med_paragraph_simplification")
def get_med_paragraph_simplification_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_paragraph_simplification_scenario.MedParagraphSimplificationScenario",
        args={},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=10,
        max_tokens=512,
        temperature=0.3,
    )

    return RunSpec(
        name="med_paragraph_simplification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["MedParagraphSimplification"],
    )


@run_spec_function("live_qa")
def get_live_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.live_qa_scenario.LiveQAScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="Please answer the following consumer health question.",
        input_noun="Question",
        output_noun="Answer",
        max_train_instances=0,
        max_tokens=512,
    )
    annotator_specs = [AnnotatorSpec(class_name="helm.benchmark.annotation.live_qa_annotator.LiveQAAnnotator")]
    metric_specs = get_open_ended_generation_metric_specs() + [
        MetricSpec(class_name="helm.benchmark.metrics.live_qa_metrics.LiveQAScoreMetric")
    ]

    return RunSpec(
        name="live_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["live_qa"],
    )


@run_spec_function("lex_glue")
def get_lex_glue_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.lex_glue_scenario import (
        get_lex_glue_instructions,
        get_lex_glue_max_tokens,
        get_lex_glue_max_train_instances,
        get_lex_glue_task_type,
    )

    task_type = get_lex_glue_task_type(subset)

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lex_glue_scenario.LexGLUEScenario",
        args={"subset": subset},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_lex_glue_instructions(subset),
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=get_lex_glue_max_tokens(subset),
        max_train_instances=get_lex_glue_max_train_instances(subset),
        multi_label=(task_type == TaskType.MLTC),
    )

    metric_specs = get_basic_generation_metric_specs([]) + get_generic_metric_specs()
    if task_type == TaskType.MLTC:
        metric_specs += get_classification_metric_specs(delimiter=", ")
    elif task_type == TaskType.SLTC:
        metric_specs += get_classification_metric_specs()

    return RunSpec(
        name=f"lex_glue:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lex_glue"],
    )


@run_spec_function("code")
def get_code_spec(dataset: str, timeout=3) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.code_scenario.CodeScenario", args={"dataset": dataset}
    )

    if dataset == "humaneval":
        adapter_spec = get_completion_adapter_spec(
            temperature=0.2,
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint"],
            max_tokens=600,
        )
    else:
        adapter_spec = get_completion_adapter_spec(
            max_train_instances=2,
            temperature=0.2,
            stop_sequences=["'''", "---", '"""', "\n\n\n"],
            max_tokens=600,
        )

    if dataset == "humaneval":
        code_metric_specs = get_basic_metric_specs(["code_eval_acc", "pass"])
    else:
        args: Dict[str, Any] = {"names": ["test_avg", "strict_acc"], "timeout": timeout}
        code_metric_specs = [MetricSpec(class_name="helm.benchmark.metrics.code_metrics.APPSMetric", args=args)]

    return RunSpec(
        name=f"code:dataset={dataset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=code_metric_specs + get_generative_harms_metric_specs(),
        groups=[f"code_{dataset}"],
    )
