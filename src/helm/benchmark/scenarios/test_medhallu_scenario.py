import pytest
from collections import Counter
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medhallu_scenario import MedHalluScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _row(
    question="What is X?",
    knowledge="Some PubMed-derived knowledge snippet.",
    ground_truth="Factual answer.",
    hallucinated="Hallucinated answer.",
) -> dict:
    """Build a row matching the MedHallu HuggingFace schema."""
    return {
        "Question": question,
        "Knowledge": knowledge,
        "Ground Truth": ground_truth,
        "Hallucinated Answer": hallucinated,
    }


# ---------------------------------------------------------------------------
# Unit tests for `create_instance` (pure prompt construction).
# ---------------------------------------------------------------------------


def test_create_instance_builds_expected_prompt():
    scenario = MedHalluScenario()
    instance = scenario.create_instance(
        question="Q?",
        knowledge="K.",
        answer="A.",
        label="0",
        split=TEST_SPLIT,
    )

    assert instance.input.text == "World Knowledge: K.\n\nQuestion: Q?\n\nAnswer: A.\n"


def test_create_instance_reference_is_label_string():
    scenario = MedHalluScenario()
    instance = scenario.create_instance("Q?", "K.", "A.", label="1", split=TEST_SPLIT)

    assert instance.references == [
        Reference(output=Output(text="1"), tags=[CORRECT_TAG]),
    ]


@pytest.mark.parametrize("split", [TEST_SPLIT, "train"])
def test_create_instance_propagates_split(split):
    scenario = MedHalluScenario()
    instance = scenario.create_instance("Q?", "K.", "A.", label="0", split=split)

    assert instance.split == split


def test_create_instance_handles_multiline_inputs():
    """Real PubMed knowledge snippets often span paragraphs with embedded newlines."""
    scenario = MedHalluScenario()
    knowledge = "Sentence 1.\nSentence 2.\nSentence 3."
    instance = scenario.create_instance("Q?", knowledge, "A.", label="0", split=TEST_SPLIT)

    assert instance.input.text == f"World Knowledge: {knowledge}\n\nQuestion: Q?\n\nAnswer: A.\n"


# ---------------------------------------------------------------------------
# Mocked end-to-end tests for `get_instances`.
# ---------------------------------------------------------------------------


def test_get_instances_single_row_produces_two_instances(monkeypatch):
    """Every dataset row yields TWO instances: one for the factual answer (label=0) and one for
    the hallucinated answer (label=1)."""
    fake_dataset = [
        _row(
            question="What is fever?",
            knowledge="Body temperature elevation.",
            ground_truth="An elevated body temperature.",
            hallucinated="A drop in body temperature.",
        )
    ]
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medhallu_scenario.load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert instances[0].references[0].output.text == "0"
    assert "An elevated body temperature." in instances[0].input.text
    assert instances[1].references[0].output.text == "1"
    assert "A drop in body temperature." in instances[1].input.text
    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_pairs_alternate_label_0_and_1(monkeypatch):
    """For each row, the gt instance (label=0) is emitted immediately before the hallucinated one
    (label=1). The label pattern must therefore alternate 0, 1, 0, 1, ..."""
    fake_dataset = [_row(question=f"Q{i}") for i in range(4)]
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medhallu_scenario.load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    labels = [i.references[0].output.text for i in instances]
    assert labels == ["0", "1", "0", "1", "0", "1", "0", "1"]


def test_get_instances_keeps_same_question_for_both_labels(monkeypatch):
    """The gt/hallu pair for a single row must share the same Question and Knowledge —
    only the Answer differs."""
    fake_dataset = [
        _row(
            question="What is the cause of X?",
            knowledge="X is caused by Y.",
            ground_truth="Y causes X.",
            hallucinated="Z causes X.",
        )
    ]
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medhallu_scenario.load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    gt, hallu = instances
    assert "Question: What is the cause of X?" in gt.input.text
    assert "Question: What is the cause of X?" in hallu.input.text
    assert "World Knowledge: X is caused by Y." in gt.input.text
    assert "World Knowledge: X is caused by Y." in hallu.input.text
    assert "Answer: Y causes X." in gt.input.text
    assert "Answer: Z causes X." in hallu.input.text


def test_get_instances_label_distribution_is_balanced(monkeypatch):
    """Because each row yields exactly one gt and one hallucinated instance, the label
    distribution must be exactly 50/50."""
    fake_dataset = [_row(question=f"Q{i}") for i in range(10)]
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medhallu_scenario.load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    distribution = Counter(i.references[0].output.text for i in instances)
    assert distribution["0"] == 10
    assert distribution["1"] == 10


def test_get_instances_empty_dataset_returns_no_instances(monkeypatch):
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medhallu_scenario.load_dataset",
        lambda *args, **kwargs: [],
    )

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


def test_get_instances_load_dataset_pinned_arguments(monkeypatch):
    """The scenario pins a HuggingFace dataset path, subset, split and revision. All four
    matter for reproducibility — a regression in any of them breaks the benchmark."""
    recorded: dict = {}

    def _fake_load_dataset(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return []

    monkeypatch.setattr("helm.benchmark.scenarios.medhallu_scenario.load_dataset", _fake_load_dataset)

    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        scenario.get_instances(tmpdir)

    # The dataset path and subset are positional; the split + revision are keyword.
    assert recorded["args"] == ("UTAustin-AIHealth/MedHallu", "pqa_labeled")
    assert recorded["kwargs"]["split"] == "train"
    assert recorded["kwargs"]["revision"] == "515060458a945c633debc6fd5baac7764416b724"


# ---------------------------------------------------------------------------
# Real-network integration test.
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_medhallu_scenario_integration():
    """End-to-end smoke test against the real `UTAustin-AIHealth/MedHallu` dataset (~1000 rows)."""
    scenario = MedHalluScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    # 1000 rows × 2 (gt + hallucinated) = 2000 instances.
    assert len(instances) == 2000
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)

    distribution = Counter(i.references[0].output.text for i in instances)
    assert distribution["0"] == 1000
    assert distribution["1"] == 1000


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MedHalluScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medhallu"
    assert metadata.display_name == "MedHallu"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MedHalluScenario()

    assert scenario.name == "medhallu"
    assert "knowledge" in scenario.tags
    assert "reasoning" in scenario.tags
    assert "biomedical" in scenario.tags
