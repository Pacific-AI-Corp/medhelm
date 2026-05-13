import json
import os
import pytest
from collections import Counter
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.pubmed_qa_scenario import PubMedQAScenario
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Output,
    PassageQuestionInput,
    Reference,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _entry(
    question: str = "Is X helpful?",
    labels=None,
    contexts=None,
    final_decision: str = "yes",
):
    """Build a JSON entry matching the PubMedQA schema, with overridable defaults."""
    return {
        "QUESTION": question,
        "LABELS": labels if labels is not None else ["BACKGROUND", "METHODS"],
        "CONTEXTS": contexts if contexts is not None else ["Some background.", "Some methods."],
        "MESHES": [],
        "YEAR": "2020",
        "reasoning_required_pred": final_decision,
        "reasoning_free_pred": final_decision,
        "final_decision": final_decision,
    }


def _patch_with_entries(monkeypatch, entries: dict) -> None:
    """Replace `ensure_file_downloaded` so it materialises a JSON file containing `entries`."""

    def _fake(source_url, target_path, **kwargs):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    monkeypatch.setattr("helm.benchmark.scenarios.pubmed_qa_scenario.ensure_file_downloaded", _fake)


# ---------------------------------------------------------------------------
# Integration test against the real PubMedQA labelled subset (PQA-L, 1000 items).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_pubmed_qa_scenario_get_instances():
    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1000
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(isinstance(instance.input, PassageQuestionInput) for instance in instances)
    assert all(len(instance.references) == 3 for instance in instances)
    # Exactly one correct reference per instance, and its text is yes/no/maybe.
    correct_texts = [
        ref.output.text for instance in instances for ref in instance.references if CORRECT_TAG in ref.tags
    ]
    assert len(correct_texts) == len(instances)
    assert set(correct_texts) <= set(PubMedQAScenario.POSSIBLE_ANSWER_CHOICES)


@pytest.mark.scenarios
def test_pubmed_qa_answer_distribution_matches_dataset():
    """The PQA-L test split has a stable label distribution worth pinning so a future change to
    the data source or processing is noticed immediately."""
    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    correct_answers = [
        ref.output.text for instance in instances for ref in instance.references if CORRECT_TAG in ref.tags
    ]
    distribution = Counter(correct_answers)

    assert distribution["yes"] == 552
    assert distribution["no"] == 338
    assert distribution["maybe"] == 110


# ---------------------------------------------------------------------------
# Mocked end-to-end tests for `get_instances`.
# ---------------------------------------------------------------------------


def test_get_instances_basic_single_entry(monkeypatch):
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(
                question="Is anorectal endosonography valuable?",
                labels=["AIMS", "METHODS"],
                contexts=["Some aims.", "Some methods."],
                final_decision="yes",
            ),
        },
    )

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1
    assert instances[0].split == TEST_SPLIT
    assert instances[0].input == PassageQuestionInput(
        passage="Aims. Some aims.\nMethods. Some methods.",
        question="Is anorectal endosonography valuable?\n",
        passage_prefix="Context: ",
        separator="\n\n",
    )
    assert instances[0].references == [
        Reference(output=Output(text="yes"), tags=[CORRECT_TAG]),
        Reference(output=Output(text="no"), tags=[]),
        Reference(output=Output(text="maybe"), tags=[]),
    ]


@pytest.mark.parametrize("decision", ["yes", "no", "maybe"])
def test_get_instances_marks_correct_answer_for_every_decision(monkeypatch, decision):
    _patch_with_entries(monkeypatch, {"id-1": _entry(final_decision=decision)})

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    correct = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
    assert len(correct) == 1
    assert correct[0].output.text == decision


def test_get_instances_titlecases_labels(monkeypatch):
    """Labels arrive UPPERCASE in the source JSON; the scenario should render them title-cased."""
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(
                labels=["BACKGROUND", "RESULTS", "CONCLUSIONS"],
                contexts=["bg.", "res.", "concl."],
            )
        },
    )

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    text = instances[0].input.text
    assert "Background. bg." in text
    assert "Results. res." in text
    assert "Conclusions. concl." in text
    # Original uppercase labels must not leak through.
    assert "BACKGROUND" not in text
    assert "RESULTS" not in text


def test_get_instances_joins_passages_with_newline(monkeypatch):
    """Each label/context pair must be on its own line within the passage portion of the input."""
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(
                labels=["AIMS", "RESULTS"],
                contexts=["Aim text.", "Result text."],
            )
        },
    )

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert "Aims. Aim text.\nResults. Result text." in instances[0].input.text


def test_get_instances_handles_multiple_entries(monkeypatch):
    """Every entry in the JSON dict becomes its own instance, in dict-iteration order."""
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(question="Q1", final_decision="yes"),
            "id-2": _entry(question="Q2", final_decision="no"),
            "id-3": _entry(question="Q3", final_decision="maybe"),
        },
    )

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 3
    texts = [instance.input.text for instance in instances]
    for i, text in enumerate(texts, start=1):
        assert f"Question: Q{i}\n" in text
    correct_answers = [
        ref.output.text for instance in instances for ref in instance.references if CORRECT_TAG in ref.tags
    ]
    assert correct_answers == ["yes", "no", "maybe"]


def test_get_instances_input_contains_question_with_trailing_newline(monkeypatch):
    """The scenario passes `question + "\\n"` so prompts can append the multiple-choice block
    cleanly on the next line. The final `.text` therefore ends with the question + newline."""
    _patch_with_entries(monkeypatch, {"id-1": _entry(question="Does X work?")})

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances[0].input.text.endswith("Question: Does X work?\n")


def test_get_instances_uses_passage_question_input_with_expected_format(monkeypatch):
    """`PassageQuestionInput` flattens into `<passage_prefix><passage><separator><question_prefix><question>`.
    Re-derive the expected `.text` from the synthetic entry and compare end-to-end."""
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(
                question="Is X helpful?",
                labels=["AIMS", "METHODS"],
                contexts=["a.", "m."],
            )
        },
    )

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert isinstance(instances[0].input, PassageQuestionInput)
    expected = "Context: Aims. a.\nMethods. m.\n\nQuestion: Is X helpful?\n"
    assert instances[0].input.text == expected


# ---------------------------------------------------------------------------
# The scenario hardcodes two `assert` invariants. Document them with explicit tests so any
# refactor that loosens/tightens the schema is caught.
# ---------------------------------------------------------------------------


def test_get_instances_raises_when_labels_and_contexts_length_mismatch(monkeypatch):
    """`assert len(contexts) == len(context_labels)` guards a corrupt JSON entry."""
    _patch_with_entries(
        monkeypatch,
        {
            "id-1": _entry(
                labels=["AIMS"],
                contexts=["Aim text.", "Stray extra context."],  # mismatched
            )
        },
    )

    scenario = PubMedQAScenario()
    with pytest.raises(AssertionError):
        with TemporaryDirectory() as tmpdir:
            scenario.get_instances(tmpdir)


def test_get_instances_raises_on_invalid_final_decision(monkeypatch):
    """`final_decision` must be one of yes/no/maybe; anything else trips the inner assert."""
    _patch_with_entries(
        monkeypatch,
        {"id-1": _entry(final_decision="unknown")},
    )

    scenario = PubMedQAScenario()
    with pytest.raises(AssertionError):
        with TemporaryDirectory() as tmpdir:
            scenario.get_instances(tmpdir)


def test_get_instances_returns_empty_for_empty_input(monkeypatch):
    """An empty dict in the source file must yield zero instances without errors."""
    _patch_with_entries(monkeypatch, {})

    scenario = PubMedQAScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert instances == []


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = PubMedQAScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "pubmed_qa"
    assert metadata.display_name == "PubMedQA"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Question answering"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = PubMedQAScenario()

    assert scenario.name == "pubmed_qa"
    assert "biomedical" in scenario.tags
    assert "question_answering" in scenario.tags
    assert PubMedQAScenario.POSSIBLE_ANSWER_CHOICES == ["yes", "no", "maybe"]
