import os
from typing import List
import pandas as pd

from datasets import DatasetDict, load_dataset

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists, ensure_file_downloaded

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
)

class MedXpertQATextScenario(Scenario):
    """
    The MedXpertQA dataset introduced in the MedXpert paper by Li et al:
    @article{zuo2025medxpertqa,
    title={Medxpertqa: Benchmarking expert-level medical reasoning and understanding},
    author={Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
    journal={arXiv preprint arXiv:2501.18362},
    year={2025}
    }
    """

    HF_DATASET_NAME = "TsinghuaC3I/MedXpertQA"

    name = "medxpert_qa"
    description = (
        "MedXpertQA is a benchmark designed to evaluate the medical reasoning and understanding capabilities of"
        "language models. Each instance in the dataset consists of a medical question and its corresponding "
        "expert-level answer. The benchmark assesses a model's ability to comprehend complex medical information, reason through clinical scenarios, and provide accurate and informative responses that align with expert knowledge in the field of medicine."
    )
    tags = ["knowledge", "generation", "question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(
            self.HF_DATASET_NAME,
            "Text",
            cache_dir=data_path)
        
        # split the dataset into train, validation, and test splits
        splits = {TEST_SPLIT: ["test"]}
        instances: List[Instance] = []
        for (
            helm_split_name,
            dataset_splits_name,
        ) in splits.items():  # Iterate over the splits
            for dataset_split_name in dataset_splits_name:
                split_data = dataset[dataset_split_name]

                for example in split_data:
                    question = example["question"]
                    answer = example["label"]

                    instance = Instance(
                        input=Input(text=question),
                        references=[
                            Reference(
                                Output(text=option),
                                tags=[CORRECT_TAG] if alpha == answer else [],
                            )
                            for alpha, option in example['options'].items()
                        ],
                        split=helm_split_name,
                        extra_data={
                            "id": example["id"],
                            "medical_task": example["medical_task"],
                            "body_system": example["body_system"],
                            "question_type": example["question_type"],
                        },
                    )
                    instances.append(instance)
        return instances