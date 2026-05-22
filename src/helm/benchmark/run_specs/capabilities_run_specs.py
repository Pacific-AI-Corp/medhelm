"""Run spec functions for MMLU-Pro."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT, ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _bool_to_str(value: bool):
    return str(value).lower()


@run_spec_function("mmlu_pro")
def get_mmlu_pro_spec(subject: str = "all", use_chain_of_thought: str = "true", use_few_shot: str = "false") -> RunSpec:
    use_chain_of_thought_bool: bool = use_chain_of_thought.lower() == "true"
    use_few_shot_bool: bool = use_few_shot.lower() == "true"

    run_spec_name = (
        f"mmlu_pro:subset={subject},use_chain_of_thought={_bool_to_str(use_chain_of_thought_bool)},"
        f"use_few_shot={_bool_to_str(use_few_shot_bool)}"
    )
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_pro_scenario.MMLUProScenario", args={"subject": subject}
    )
    max_train_instance_num = 5 if use_few_shot_bool else 0

    if use_chain_of_thought_bool:
        adapter_spec = AdapterSpec(
            method=ADAPT_MULTIPLE_CHOICE_JOINT_CHAIN_OF_THOUGHT,
            max_train_instances=max_train_instance_num,
            max_tokens=4096,
            input_prefix="What is the correct answer to this question: ",
            input_suffix="\nChoices:\n",
            output_prefix="",
            global_suffix=(
                "Let’s think step by step. Based on your reasoning, what is the single, "
                "most likely answer choice? Format your response as follows: "
                '"The correct answer is (insert answer here)".'
            ),
        )
        return RunSpec(
            name=run_spec_name,
            scenario_spec=scenario_spec,
            adapter_spec=adapter_spec,
            metric_specs=get_basic_metric_specs([])
            + [
                MetricSpec(
                    class_name="helm.benchmark.metrics.gpqa_chain_of_thought_metric.GPQAChainOfThoughtMetric", args={}
                ),
            ],
            groups=["mmlu_pro"],
        )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        max_train_instances=max_train_instance_num,
        max_tokens=4096,
        input_prefix="What is the correct answer to this question: ",
        input_suffix="\nChoices:\n",
        output_prefix="",
        global_suffix=('Format your response as follows: "The correct answer is (insert answer here)".'),
    )
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu_pro"],
    )
