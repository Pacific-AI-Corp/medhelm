# The following code includes templates and evaluation logic reproduced with minor modifications from:
# https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
#
# MIT License
#
# Copyright (c) 2024 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import json
import os
from typing import List
import zipfile
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

from helm.common.general import ensure_file_downloaded


class HealthBenchScenario(Scenario):
    name = "health_bench"
    description = "HealthBench-style rubric evaluation (LLM-as-judge)"
    tags = ["health", "rubric", "llm-judge"]

    DATASET_DOWNLOAD_URL = (
        "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
    )
    FILENAME = "healthbench.jsonl"

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
        data_path = os.path.join(output_path, self.FILENAME)
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        instances: List[Instance] = []

        with open(data_path, "r", encoding="utf-8") as f:
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


class HealthBenchProfessionalScenario(Scenario):
    name = "health_bench_professional"
    description = "HealthBench Professional rubric evaluation (LLM-as-judge)"
    tags = ["health", "rubric", "llm-judge"]

    DATASET_DOWNLOAD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench_professional/assets.zip"
    FILENAME = "assets.zip"
    DATA_FILE = "healthbench_professional_eval.jsonl"

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, self.FILENAME)
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=True,
        )

        instances: List[Instance] = []
        ensure_directory_exists(data_path)
        data_file_path = os.path.join(data_path, self.DATA_FILE)
        with open(data_file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                messages = row["conversation"].get("messages", [])

                instances.append(
                    Instance(
                        input=Input(messages=messages),
                        references=[],
                        split=TEST_SPLIT,
                        id=str(idx),
                        extra_data={
                            "rubrics": row.get("rubric_items", []),
                            "prompt_id": row.get("id"),
                            "physician_response": row.get("physician_response", ""),
                        },
                    )
                )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="health_bench_professional",
            display_name="HealthBenchProfessionalScenario",
            description="HealthBenchProfessional: a new benchmark designed to better measure the capabilities of AI systems for health. Built in partnership with 262 physicians who have practiced in 60 countries",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Verify whether answers to questions from LLMs are correct according to a rubric",
                when="Any",
                who="Researcher",
                language="Any",
            ),
            main_metric="health_bench_score",
            main_split="test",
        )
