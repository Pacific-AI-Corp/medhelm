import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from helm.benchmark.scenarios.health_bench_scenario import (
    HealthBenchProfessionalScenario,
    HealthBenchScenario,
)
from helm.benchmark.scenarios.scenario import TEST_SPLIT


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_health_bench_scenario_get_instances_parses_prompt_and_extra_data():
    rows = [
        {
            "prompt": [{"role": "user", "content": "What is aspirin?"}],
            "rubrics": [{"criterion": "accuracy"}],
            "example_tags": ["cardiology"],
            "prompt_id": "hb-1",
        },
        {
            "prompt": [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Pain"}],
        },
    ]
    with TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "healthbench.jsonl")
        _write_jsonl(data_path, rows)
        with patch("helm.benchmark.scenarios.health_bench_scenario.ensure_file_downloaded"):
            scenario = HealthBenchScenario()
            instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert instances[0].id == "0"
    assert instances[1].id == "1"
    assert instances[0].split == TEST_SPLIT
    assert instances[1].split == TEST_SPLIT
    assert instances[0].references == []
    assert instances[0].input.messages == rows[0]["prompt"]
    assert instances[0].extra_data == {
        "rubrics": [{"criterion": "accuracy"}],
        "example_tags": ["cardiology"],
        "prompt_id": "hb-1",
    }
    assert instances[1].extra_data["rubrics"] == []
    assert instances[1].extra_data["example_tags"] == []
    assert instances[1].extra_data["prompt_id"] is None


def test_health_bench_professional_scenario_get_instances_parses_conversation_and_extra_data():
    rows = [
        {
            "conversation": {"messages": [{"role": "user", "content": "Symptoms?"}]},
            "rubric_items": [{"id": "r1"}],
            "id": "prof-1",
            "physician_response": "Consider differential X.",
        },
        {
            "conversation": {},
            "id": None,
        },
    ]

    def _mock_download(source_url: str, target_path: str, unpack: bool = False, **_kwargs: object) -> None:
        assert source_url == HealthBenchProfessionalScenario.DATASET_DOWNLOAD_URL
        assert unpack is True
        os.makedirs(target_path, exist_ok=True)
        inner = os.path.join(target_path, "healthbench_professional_eval.jsonl")
        _write_jsonl(inner, rows)

    with TemporaryDirectory() as tmpdir:
        with patch(
            "helm.benchmark.scenarios.health_bench_scenario.ensure_file_downloaded",
            side_effect=_mock_download,
        ):
            scenario = HealthBenchProfessionalScenario()
            instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert instances[0].split == TEST_SPLIT
    assert instances[0].references == []
    assert instances[0].input.messages == [{"role": "user", "content": "Symptoms?"}]
    assert instances[0].extra_data == {
        "rubrics": [{"id": "r1"}],
        "prompt_id": "prof-1",
        "physician_response": "Consider differential X.",
    }
    assert instances[1].input.messages == []
    assert instances[1].extra_data["rubrics"] == []
    assert instances[1].extra_data["prompt_id"] is None
    assert instances[1].extra_data["physician_response"] == ""


def test_health_bench_scenario_metadata():
    meta = HealthBenchScenario().get_metadata()
    assert meta.name == "health_bench"
    assert meta.display_name == "HealthBench"
    assert meta.main_metric == "medhelm_health_score"
    assert meta.main_split == "test"
    assert meta.taxonomy.task == "Classification"


def test_health_bench_professional_scenario_metadata():
    meta = HealthBenchProfessionalScenario().get_metadata()
    assert meta.name == "health_bench_professional"
    assert meta.display_name == "HealthBench Professional"
    assert meta.main_metric == "health_bench_professional_score"
    assert meta.main_split == "test"


# ---------------------------------------------------------------------------
# Integration tests (network download; run via `pytest -m scenarios` / CI scenario job).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_health_bench_scenario_get_instances():
    scenario = HealthBenchScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) >= 1
    assert all(inst.split == TEST_SPLIT for inst in instances)
    assert instances[0].references == []
    assert isinstance(instances[0].input.messages, list)
    assert instances[0].extra_data is not None
    assert "rubrics" in instances[0].extra_data
    assert "example_tags" in instances[0].extra_data
    assert "prompt_id" in instances[0].extra_data


@pytest.mark.scenarios
def test_health_bench_professional_scenario_get_instances():
    scenario = HealthBenchProfessionalScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) >= 1
    assert all(inst.split == TEST_SPLIT for inst in instances)
    assert instances[0].references == []
    assert isinstance(instances[0].input.messages, list)
    assert instances[0].extra_data is not None
    assert "rubrics" in instances[0].extra_data
    assert "prompt_id" in instances[0].extra_data
    assert "physician_response" in instances[0].extra_data
