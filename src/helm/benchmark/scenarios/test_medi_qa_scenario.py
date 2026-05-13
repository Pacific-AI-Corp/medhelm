import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medi_qa_scenario import MediQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _answer(rank: int, text: str) -> dict:
    """Build the nested {Answer: {ReferenceRank, AnswerText}} structure used in MEDIQA."""
    return {"Answer": {"ReferenceRank": rank, "AnswerText": text}}


def _row(qid="42", question="What is X?", answers=None) -> dict:
    """Build a HuggingFace MEDIQA row: each row wraps its content under a QUESTION key."""
    if answers is None:
        answers = [_answer(1, "Default answer.")]
    return {
        "QUESTION": {
            "QID": qid,
            "QuestionText": question,
            "AnswerList": answers,
        }
    }


# ---------------------------------------------------------------------------
# Unit tests for `_get_highest_ranked_answer`.
# ---------------------------------------------------------------------------


def test_get_highest_ranked_answer_returns_rank_one_answer():
    scenario = MediQAScenario()
    answers = [
        _answer(rank=3, text="Lower ranked."),
        _answer(rank=1, text="Best answer."),
        _answer(rank=2, text="Second best."),
    ]

    assert scenario._get_highest_ranked_answer(answers) == "Best answer."


def test_get_highest_ranked_answer_returns_empty_string_when_no_rank_one():
    """The helper iterates without a fallback; if no answer has ReferenceRank == 1,
    `best_answer` stays as the empty string it was initialized with."""
    scenario = MediQAScenario()
    answers = [
        _answer(rank=2, text="Second."),
        _answer(rank=3, text="Third."),
    ]

    assert scenario._get_highest_ranked_answer(answers) == ""


def test_get_highest_ranked_answer_returns_empty_string_for_empty_list():
    scenario = MediQAScenario()
    assert scenario._get_highest_ranked_answer([]) == ""


def test_get_highest_ranked_answer_breaks_on_first_match():
    """If multiple answers share ReferenceRank == 1 (malformed data), only the first one is kept
    because the loop hits `break` immediately after the assignment."""
    scenario = MediQAScenario()
    answers = [
        _answer(rank=1, text="First."),
        _answer(rank=1, text="Duplicate top-rank."),
    ]

    assert scenario._get_highest_ranked_answer(answers) == "First."


def test_get_highest_ranked_answer_picks_rank_one_regardless_of_position():
    scenario = MediQAScenario()
    answers = [
        _answer(rank=5, text="Way lower."),
        _answer(rank=2, text="Second."),
        _answer(rank=1, text="The chosen one."),
    ]

    assert scenario._get_highest_ranked_answer(answers) == "The chosen one."


# ---------------------------------------------------------------------------
# Unit tests for `process_csv`.
# ---------------------------------------------------------------------------


def test_process_csv_single_row_produces_full_instance():
    scenario = MediQAScenario()
    data = [
        _row(
            qid="Q-1",
            question="What is Bassen-Kornzweig syndrome?",
            answers=[
                _answer(rank=2, text="Lower ranked."),
                _answer(rank=1, text="Correct best answer."),
            ],
        )
    ]

    instances = scenario.process_csv(data, TEST_SPLIT)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.split == TEST_SPLIT
    assert instance.id == "Q-1"
    assert instance.input.text == "What is Bassen-Kornzweig syndrome?"
    assert instance.references == [
        Reference(output=Output(text="Correct best answer."), tags=[CORRECT_TAG]),
    ]


def test_process_csv_multiple_rows_keep_order_and_ids():
    scenario = MediQAScenario()
    data = [
        _row(qid="A", question="Q-A"),
        _row(qid="B", question="Q-B"),
        _row(qid="C", question="Q-C"),
    ]

    instances = scenario.process_csv(data, TEST_SPLIT)

    assert [i.id for i in instances] == ["A", "B", "C"]
    assert [i.input.text for i in instances] == ["Q-A", "Q-B", "Q-C"]


def test_process_csv_empty_data_returns_no_instances():
    scenario = MediQAScenario()

    assert scenario.process_csv([], TEST_SPLIT) == []


def test_process_csv_row_without_rank_one_still_creates_instance_with_empty_reference():
    """Per the helper's behavior, a row without a rank-1 answer becomes an Instance whose only
    Reference has empty text. The instance is *not* skipped."""
    scenario = MediQAScenario()
    data = [
        _row(
            qid="missing-top-rank",
            question="A question with no rank-1 answer.",
            answers=[_answer(rank=2, text="Not top."), _answer(rank=3, text="Even lower.")],
        )
    ]

    instances = scenario.process_csv(data, TEST_SPLIT)

    assert len(instances) == 1
    assert instances[0].references == [
        Reference(output=Output(text=""), tags=[CORRECT_TAG]),
    ]


@pytest.mark.parametrize("split", [TEST_SPLIT, "train"])
def test_process_csv_propagates_split(split):
    scenario = MediQAScenario()
    data = [_row()]

    instances = scenario.process_csv(data, split)

    assert instances[0].split == split


# ---------------------------------------------------------------------------
# End-to-end mocked test for `get_instances` (replaces `load_dataset`).
# ---------------------------------------------------------------------------


def test_get_instances_with_mocked_load_dataset(monkeypatch):
    """`get_instances` only consumes the `test` split; the mock provides several splits to make
    sure the scenario picks the right one."""
    fake_dataset = {
        "train_live_qa_med": [_row(qid="train-1", question="Train Q")],  # ignored
        "validation": [_row(qid="val-1", question="Validation Q")],  # ignored
        "test": [
            _row(qid="t-1", question="Test Q-1", answers=[_answer(rank=1, text="Top test answer.")]),
            _row(
                qid="t-2",
                question="Test Q-2",
                answers=[_answer(rank=1, text="Another."), _answer(rank=2, text="Lower.")],
            ),
        ],
    }

    def _fake_load_dataset(name, **kwargs):
        assert name == "bigbio/mediqa_qa"
        # The scenario pins the dataset revision; surface it in the assertion to detect drift.
        assert kwargs.get("revision") == "9288641f4c785c95dc9079fa526dabb12efdb041"
        assert kwargs.get("trust_remote_code") is True
        return fake_dataset

    monkeypatch.setattr("helm.benchmark.scenarios.medi_qa_scenario.load_dataset", _fake_load_dataset)

    scenario = MediQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 2
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert [i.id for i in instances] == ["t-1", "t-2"]
    assert instances[0].references[0].output.text == "Top test answer."
    assert instances[1].references[0].output.text == "Another."


def test_get_instances_returns_empty_when_test_split_is_empty(monkeypatch):
    monkeypatch.setattr(
        "helm.benchmark.scenarios.medi_qa_scenario.load_dataset",
        lambda *args, **kwargs: {"test": []},
    )

    scenario = MediQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


# ---------------------------------------------------------------------------
# Integration test against the real HuggingFace dataset (slow, opt-in, network flaky).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_medi_qa_scenario_get_instances_integration():
    scenario = MediQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 150
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(instance.input.text for instance in instances)
    assert all(instance.id is not None for instance in instances)
    # IDs must be unique so they can be used as stable identifiers downstream.
    assert len({instance.id for instance in instances}) == len(instances)


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MediQAScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "medi_qa"
    assert metadata.display_name == "MEDIQA"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "medi_qa_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MediQAScenario()

    assert scenario.name == "medi_qa"
    assert "biomedical" in scenario.tags
    assert "knowledge" in scenario.tags
