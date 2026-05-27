"""Run spec functions for medical radiology and speech multimodal benchmarks."""

from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.image_generation_parameters import ImageGenerationParameters


def _get_image_generation_adapter_spec(num_outputs: int = 1) -> AdapterSpec:
    image_generation_parameters = ImageGenerationParameters()
    return AdapterSpec(
        method=ADAPT_GENERATION,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=num_outputs,
        max_tokens=0,
        image_generation_parameters=image_generation_parameters,
    )


def _get_core_heim_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.aesthetics_metrics.AestheticsMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.efficiency_metrics.EfficiencyMetric", args={}),
        MetricSpec(
            class_name="helm.benchmark.metrics.image_generation.fractal_dimension_metric.FractalDimensionMetric",
            args={},
        ),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.nudity_metrics.NudityMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.watermark_metrics.WatermarkMetric", args={}),
    ] + get_basic_metric_specs(names=[])


def _get_heim_bias_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.gender_metrics.GenderMetric", args={}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.skin_tone_metrics.SkinToneMetric", args={}),
    ]


def _get_generation_adapter_spec(
    instructions: str = "",
    max_tokens: int = 100,
    stop_sequences: Optional[List[str]] = None,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        global_prefix="",
        instructions=instructions,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences if stop_sequences is not None else [],
        temperature=0.0,
        random=None,
    )


def _get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        [
            "exact_match",
            "quasi_exact_match",
            "quasi_leave_articles_exact_match",
            "f1_score",
            "rouge_l",
            "bleu_1",
            "bleu_4",
            "cider",
        ]
    )


@run_spec_function("vqa_rad")
def get_vqa_rad_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.vqa_rad_scenario.VQARadScenario",
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Answer the question using a single word or sentence. Just give a short answer "
        "without answering in a complete sentence.",
        max_tokens=20,
    )
    return RunSpec(
        name="vqa_rad",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=_get_open_ended_generation_metric_specs(),
        groups=["vqa_rad"],
    )


@run_spec_function("radiology")
def get_radiology_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.radiology_scenario.RadiologyScenario", args={}
    )
    adapter_spec = _get_image_generation_adapter_spec(num_outputs=4)
    return RunSpec(
        name="radiology",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=_get_core_heim_metric_specs(),
        groups=["radiology"],
    )


@run_spec_function("mental_disorders")
def get_mental_disorders_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.image_generation.mental_disorders_scenario.MentalDisordersScenario",
        args={},
    )
    adapter_spec = _get_image_generation_adapter_spec(num_outputs=8)
    return RunSpec(
        name="mental_disorders",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=_get_heim_bias_metric_specs() + _get_core_heim_metric_specs(),
        groups=["mental_disorders"],
    )
