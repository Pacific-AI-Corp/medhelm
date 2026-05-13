import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medbullets_scenario import MedBulletsScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


@pytest.mark.scenarios
def test_medbullets_scenario_get_instances():
    scenario = MedBulletsScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 308
    assert instances[0].split == TEST_SPLIT
    assert instances[0].input.text.startswith(
        "A 64-year-old man presents to the emergency room with a headache and nausea."
    )
    assert instances[0].references == [
        Reference(output=Output(text="Acetazolamide"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="Amitriptyline"), tags=[]),
        Reference(output=Output(text="Clopidogrel"), tags=[]),
        Reference(output=Output(text="Epinephrine"), tags=[]),
        Reference(output=Output(text="Verapamil"), tags=[]),
    ]
    assert instances[0].references[0].is_correct


@pytest.mark.scenarios
def test_medbullets_scenario_instance_structure():
    scenario = MedBulletsScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(instance.input.text for instance in instances)
    assert all(len(instance.references) == len(MedBulletsScenario.POSSIBLE_ANSWER_CHOICES) for instance in instances)
    assert all(sum(1 for ref in instance.references if CORRECT_TAG in ref.tags) == 1 for instance in instances)


def test_medbullets_scenario_metadata():
    scenario = MedBulletsScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medbullets"
    assert metadata.display_name == "Medbullets"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Question answering"
    assert metadata.taxonomy.language == "English"
