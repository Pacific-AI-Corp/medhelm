import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.headqa_scenario import HeadQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


VALID_CATEGORIES = {"biology", "chemistry", "medicine", "nursery", "pharmacology", "psychology"}


@pytest.fixture(autouse=True)
def _trust_remote_code(monkeypatch):
    """The `dvilares/head_qa` dataset uses a custom loading script, so we must opt in."""
    monkeypatch.setenv("HF_DATASETS_TRUST_REMOTE_CODE", "1")


@pytest.mark.scenarios
def test_headqa_scenario_english_default():
    scenario = HeadQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2675
    assert instances[0].split == TEST_SPLIT
    assert instances[0].input.text == "Form extracellular fibers with high tensile strength:"
    assert instances[0].references == [
        Reference(output=Output(text="Fibronectin"), tags=[]),
        Reference(output=Output(text="Collagen"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="Integrins"), tags=[]),
        Reference(output=Output(text="Proteoglycans"), tags=[]),
    ]
    assert instances[0].references[1].is_correct
    assert instances[0].extra_data == {
        "id": 1,
        "name": "Cuaderno_2016_1_B",
        "category": "biology",
        "year": "2016",
    }


@pytest.mark.scenarios
def test_headqa_scenario_spanish():
    scenario = HeadQAScenario(language="es")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2675
    assert instances[0].split == TEST_SPLIT
    assert instances[0].input.text == "Forma fibras extracelulares con gran resistencia a la tensión:"
    correct_refs = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert len(correct_refs) == 1
    assert correct_refs[0].output.text == "Colágeno."


@pytest.mark.scenarios
@pytest.mark.parametrize(
    "category,expected_count",
    [
        ("medicine", 396),
        ("biology", 454),
    ],
)
def test_headqa_scenario_filter_by_category(category, expected_count):
    scenario = HeadQAScenario(category=category)
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == expected_count
    assert all(
        instance.extra_data is not None and instance.extra_data["category"] == category for instance in instances
    )


@pytest.mark.scenarios
def test_headqa_scenario_instance_structure():
    scenario = HeadQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(instance.input.text for instance in instances)
    assert all(len(instance.references) >= 2 for instance in instances)
    assert all(sum(1 for ref in instance.references if CORRECT_TAG in ref.tags) == 1 for instance in instances)
    assert all(
        instance.extra_data is not None and {"id", "name", "category", "year"} <= instance.extra_data.keys()
        for instance in instances
    )
    assert all(
        instance.extra_data is not None and instance.extra_data["category"] in VALID_CATEGORIES
        for instance in instances
    )


def test_headqa_scenario_metadata():
    scenario = HeadQAScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "head_qa"
    assert metadata.display_name == "HeadQA"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Question answering"
    assert metadata.taxonomy.language == "English"


def test_headqa_scenario_init_defaults():
    scenario = HeadQAScenario()

    assert scenario.language == "en"
    assert scenario.category is None
    assert scenario.SKIP_VQA is True
    assert scenario.SKIP_TEXTQA is False


def test_headqa_scenario_init_invalid_skip_flags():
    """Both SKIP_VQA and SKIP_TEXTQA True would skip every example, so __init__ asserts at least one is True."""

    class BothFalseHeadQA(HeadQAScenario):
        SKIP_VQA = False
        SKIP_TEXTQA = False

    with pytest.raises(AssertionError):
        BothFalseHeadQA()
