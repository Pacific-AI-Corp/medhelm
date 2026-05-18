import os
from typing import List

from datasets import DatasetDict, load_dataset

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists

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
    From "MedXpertQA: Benchmarking Expert-Level Medical Knowledge and Reasoning" (2025),
    MedXpertQA is a highly challenging benchmark designed to evaluate expert-level medical knowledge,
    clinical reasoning, and advanced problem-solving abilities in large language models.
    The benchmark contains 4,460 questions spanning 17 medical specialties and 11 body systems,
    with a dedicated Text subset for text-only evaluation and an MM subset for multimodal clinical reasoning.

    The dataset includes rigorously curated specialty board-style questions enriched with detailed clinical contexts,
    patient records, and examination findings. MedXpertQA applies filtering, augmentation, and data synthesis
    techniques to improve difficulty, reduce data leakage risks, and ensure strong clinical relevance through
    multiple rounds of expert review.

    HuggingFace Dataset: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA
    ArXiv Paper: https://arxiv.org/abs/2501.18362

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
        "MedXpertQA Text is a text-only benchmark designed to evaluate expert-level medical knowledge, clinical reasoning,"
        " and advanced problem-solving capabilities in large language models across diverse medical specialties and body systems."
        " It features rigorously curated and clinically relevant board-style questions, enhanced through expert review and data synthesis "
        "techniques to ensure high difficulty, reliability, and minimal data leakage."
    )
    tags = ["knowledge", "generation", "question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        dataset: DatasetDict = load_dataset(self.HF_DATASET_NAME, "Text", cache_dir=data_path)

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
                            for alpha, option in example["options"].items()
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

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="MedXpertQA Text",
            description=self.description,
            taxonomy=TaxonomyInfo(
                task="Question answering",
                what="Answer expert-level medical questions across diverse specialties and body systems",
                when="Any",
                who="Medical professionals, Medical students",
                language="English",
            ),
            main_metric="exact_match",
            main_split=TEST_SPLIT,
        )
