import json
import os
from typing import List

import requests

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


class HealthBenchScenario(Scenario):
    name = "health_bench"
    description = "HealthBench-style rubric evaluation (LLM-as-judge)"
    tags = ["health", "rubric", "llm-judge"]

    DATA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

    def download_data(self, cache_path: str) -> str:
        file_path = os.path.join(cache_path, "healthbench.jsonl")

        if os.path.exists(file_path):
            return file_path

        print(f"Downloading dataset to {file_path}...")
        response = requests.get(self.DATA_URL, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return file_path

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        file_path = self.download_data(cache_dir)

        instances: List[Instance] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                messages = row["prompt"]

                instances.append(
                    Instance(
                        input=Input(messages=messages),
                        references=[],
                        split=TEST_SPLIT,
                        id=str(idx),
                        extra_data={
                            "rubrics": row.get("rubrics", []),
                            "example_tags": row.get("example_tags", []),
                            "prompt_id": row.get("prompt_id"),
                        },
                    )
                )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="health_bench",
            display_name="HealthBenchScenario",
            description="HealthBench: a new benchmark designed to better measure the capabilities of AI systems for health. Built in partnership with 262 physicians who have practiced in 60 countries, HealthBench includes 5,000 realistic health conversations, each with a custom physician-created rubric to grade model responses.",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Verify whether answers to questions from LLMs are correct according to a rubric",
                when="Any",
                who="Researcher",
                language="Any",
            ),
            main_metric="medhelm_health_score",
            main_split="test",
        )
