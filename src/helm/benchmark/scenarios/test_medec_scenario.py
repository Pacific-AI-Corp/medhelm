import os
import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medec_scenario import MedecScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


@pytest.mark.scenarios
def test_medec_scenario_get_instances():
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 597
    assert instances[0].split == TEST_SPLIT
    assert instances[0].input.text.startswith(
        "0 A 29-year-old internal medicine resident presents to the emergency department"
    )
    assert instances[0].references == [
        Reference(
            output=Output(text="10 Patient's symptoms are suspected to be due to Schistosoma mansoni."),
            tags=[CORRECT_TAG],
        ),
    ]
    assert instances[0].references[0].is_correct


@pytest.mark.scenarios
def test_medec_scenario_instance_structure():
    """Every MEDEC instance has exactly one reference, marked correct, with text either 'CORRECT'
    or '<sentence_id> <corrected_sentence>'."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(instance.input.text for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(
        instance.references[0].output.text == "CORRECT" or instance.references[0].output.text.split(" ", 1)[0].isdigit()
        for instance in instances
    )


@pytest.mark.scenarios
def test_medec_scenario_label_distribution():
    """The MEDEC test set contains a mix of clean and erroneous narratives; both buckets must be present."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    correct_only = [i for i in instances if i.references[0].output.text == "CORRECT"]
    with_error = [i for i in instances if i.references[0].output.text != "CORRECT"]

    assert len(correct_only) == 286
    assert len(with_error) == 311
    assert len(correct_only) + len(with_error) == len(instances)


def _write_csv(path: str, rows: list[dict]) -> None:
    """Helper: write a list of dict rows as CSV, matching the column names MEDEC uses."""
    import csv

    fieldnames = ["Sentences", "Error Flag", "Error Sentence ID", "Corrected Sentence"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_medec_process_csv_no_error_flag_zero():
    """Rows with Error Flag = 0 must produce a single 'CORRECT' reference."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Sentences": "0 The patient is healthy.",
                    "Error Flag": "0",
                    "Error Sentence ID": "-1",
                    "Corrected Sentence": "NA",
                }
            ],
        )
        instances = scenario.process_csv(csv_path, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].input.text == "0 The patient is healthy."
    assert instances[0].references == [
        Reference(output=Output(text="CORRECT"), tags=[CORRECT_TAG]),
    ]


def test_medec_process_csv_with_error():
    """Rows with Error Flag = 1 and a valid correction must produce '<id> <corrected>'."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Sentences": "0 First sentence. 1 Second sentence with a wrong drug.",
                    "Error Flag": "1",
                    "Error Sentence ID": "1",
                    "Corrected Sentence": "Second sentence with the right drug.",
                }
            ],
        )
        instances = scenario.process_csv(csv_path, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].references == [
        Reference(
            output=Output(text="1 Second sentence with the right drug."),
            tags=[CORRECT_TAG],
        ),
    ]


@pytest.mark.parametrize(
    "error_sentence_id,corrected_sentence",
    [
        ("-1", "Some correction"),
        ("3", "NA"),
    ],
)
def test_medec_process_csv_error_flag_one_but_invalid_fields(error_sentence_id, corrected_sentence):
    """If Error Flag is 1 but the correction or sentence ID is missing, the row falls back to 'CORRECT'."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Sentences": "0 Some clinical note.",
                    "Error Flag": "1",
                    "Error Sentence ID": error_sentence_id,
                    "Corrected Sentence": corrected_sentence,
                }
            ],
        )
        instances = scenario.process_csv(csv_path, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].references[0].output.text == "CORRECT"


def test_medec_process_csv_skips_empty_sentences():
    """Rows with an empty Sentences column must be skipped entirely."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Sentences": "",
                    "Error Flag": "0",
                    "Error Sentence ID": "-1",
                    "Corrected Sentence": "NA",
                },
                {
                    "Sentences": "0 Valid note.",
                    "Error Flag": "0",
                    "Error Sentence ID": "-1",
                    "Corrected Sentence": "NA",
                },
            ],
        )
        instances = scenario.process_csv(csv_path, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].input.text == "0 Valid note."


def test_medec_process_csv_strips_whitespace():
    """Leading/trailing whitespace in Sentences must be stripped from input.text."""
    scenario = MedecScenario()
    with TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic.csv")
        _write_csv(
            csv_path,
            [
                {
                    "Sentences": "   0 Padded note.   ",
                    "Error Flag": "0",
                    "Error Sentence ID": "-1",
                    "Corrected Sentence": "NA",
                }
            ],
        )
        instances = scenario.process_csv(csv_path, TEST_SPLIT)

    assert instances[0].input.text == "0 Padded note."


def test_medec_scenario_metadata():
    scenario = MedecScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medec"
    assert metadata.display_name == "Medec"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "medec_error_flag_accuracy"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"


def test_medec_scenario_basic_attributes():
    scenario = MedecScenario()

    assert scenario.name == "medec"
    assert "biomedical" in scenario.tags
    assert "error_detection" in scenario.tags
    assert "error_correction" in scenario.tags
    assert MedecScenario.TEST_URL.endswith(".csv")
    assert MedecScenario.GIT_HASH in MedecScenario.TEST_URL
