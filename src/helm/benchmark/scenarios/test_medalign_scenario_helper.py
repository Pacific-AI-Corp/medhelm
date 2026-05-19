import os
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest

from helm.benchmark.scenarios.medalign_scenario_helper import (  # type: ignore[attr-defined]
    add_reference_responses,
    extract_patient_id_from_fname,
    get_ehrs,
    get_instructions,
    get_tokenizer,
    pack_and_trim_prompts,
    preprocess_prompts,
    return_dataset_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic TSV / EHR datasets on disk.
# ---------------------------------------------------------------------------


def _write_tsv(path: str, rows: List[dict], columns: List[str]) -> None:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, sep="\t", index=False)


def _write_instructions_tsv(path: str, rows: List[dict]) -> None:
    """`get_instructions` requires the columns instruction_id, question, person_id."""
    _write_tsv(path, rows, ["instruction_id", "question", "person_id", "is_selected_ehr"])


# ---------------------------------------------------------------------------
# `get_instructions` — TSV reading + validation.
# ---------------------------------------------------------------------------


def test_get_instructions_returns_mapping_from_id_to_instruction_and_patient():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "instr.tsv")
        _write_instructions_tsv(
            path,
            [
                {"instruction_id": 1, "question": "What is X?", "person_id": 10, "is_selected_ehr": "yes"},
                {"instruction_id": 2, "question": "What is Y?", "person_id": 20, "is_selected_ehr": "no"},
            ],
        )

        result = get_instructions(path)

    assert result == {
        1: {"instruction": "What is X?", "patient_id": 10},
        2: {"instruction": "What is Y?", "patient_id": 20},
    }


def test_get_instructions_keeps_all_rows_regardless_of_is_selected_ehr():
    """The `is_selected_ehr` filter is commented out in the source. This test pins the
    current behaviour (both `yes` and `no` rows are kept) so a future re-enable is intentional."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "instr.tsv")
        _write_instructions_tsv(
            path,
            [
                {"instruction_id": 1, "question": "Q1", "person_id": 1, "is_selected_ehr": "no"},
                {"instruction_id": 2, "question": "Q2", "person_id": 2, "is_selected_ehr": "no"},
            ],
        )

        result = get_instructions(path)

    assert len(result) == 2


def test_get_instructions_raises_for_missing_file():
    with TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_instructions(os.path.join(tmp, "no_such_file.tsv"))


def test_get_instructions_raises_when_required_columns_missing():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broken.tsv")
        # Missing `person_id`.
        _write_tsv(path, [{"instruction_id": 1, "question": "Q"}], ["instruction_id", "question"])

        with pytest.raises(ValueError, match="missing one or more of the required columns"):
            get_instructions(path)


def test_get_instructions_handles_duplicate_instruction_ids_by_keeping_last():
    """The dict comprehension overwrites earlier rows when the same instruction_id appears
    twice. Pin this so any future de-duping logic is explicit."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dupes.tsv")
        _write_instructions_tsv(
            path,
            [
                {"instruction_id": 1, "question": "first", "person_id": 1, "is_selected_ehr": "yes"},
                {"instruction_id": 1, "question": "second", "person_id": 2, "is_selected_ehr": "yes"},
            ],
        )

        result = get_instructions(path)

    assert result == {1: {"instruction": "second", "patient_id": 2}}


# ---------------------------------------------------------------------------
# `extract_patient_id_from_fname`.
# ---------------------------------------------------------------------------


def test_extract_patient_id_from_fname_basic_xml():
    assert extract_patient_id_from_fname("12345.xml") == 12345


def test_extract_patient_id_from_fname_no_extension():
    """The function splits on '.' and takes [0], so a bare numeric name still works."""
    assert extract_patient_id_from_fname("987") == 987


def test_extract_patient_id_from_fname_multi_dot_filename():
    """For `12345.foo.xml`, the leading component is "12345" so the int parse still succeeds."""
    assert extract_patient_id_from_fname("12345.foo.xml") == 12345


def test_extract_patient_id_from_fname_documented_discrepancy_with_docstring():
    """KNOWN DISCREPANCY:
    The docstring promises the function handles the format 'EHR_<patient_id>.xml', but the
    implementation calls `int(name)` on the part before the first '.'. For 'EHR_12345.xml',
    that's `int('EHR_12345')` which raises `ValueError`. Pin this behaviour so any future
    docstring-implementation reconciliation is intentional."""
    with pytest.raises(ValueError):
        extract_patient_id_from_fname("EHR_12345.xml")


def test_extract_patient_id_from_fname_raises_on_non_numeric():
    with pytest.raises(ValueError):
        extract_patient_id_from_fname("not_a_number.xml")


# ---------------------------------------------------------------------------
# `get_ehrs`.
# ---------------------------------------------------------------------------


def test_get_ehrs_reads_every_xml_in_directory():
    with TemporaryDirectory() as tmp:
        for pt_id, content in [(1, "<ehr>1</ehr>"), (2, "<ehr>2</ehr>")]:
            with open(os.path.join(tmp, f"{pt_id}.xml"), "w", encoding="utf-8") as f:
                f.write(content)

        ehrs = get_ehrs(tmp)

    assert ehrs == {1: "<ehr>1</ehr>", 2: "<ehr>2</ehr>"}


def test_get_ehrs_supports_utf8_content():
    with TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "5.xml"), "w", encoding="utf-8") as f:
            f.write("Migrăña — paciente número uno")

        ehrs = get_ehrs(tmp)

    assert ehrs[5] == "Migrăña — paciente número uno"


def test_get_ehrs_raises_for_missing_directory():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_ehrs("/no/such/directory/exists/here")


def test_get_ehrs_crashes_on_non_numeric_filenames_documented_limitation():
    """KNOWN LIMITATION:
    The body has a `if pt_id is None` skip branch, but `extract_patient_id_from_fname` *never*
    returns None — it returns an int or raises `ValueError`. So a file like 'EHR_1.xml' (the
    format the docstring claims is supported) actually crashes `get_ehrs`. Pin the current
    behaviour."""
    with TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "EHR_42.xml"), "w", encoding="utf-8") as f:
            f.write("noop")

        with pytest.raises(ValueError):
            get_ehrs(tmp)


# ---------------------------------------------------------------------------
# `get_tokenizer` — routes between tiktoken and HF transformers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["tiktoken", "chatgpt", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "cl100k_base"],
)
def test_get_tokenizer_returns_tiktoken_for_known_aliases(name):
    import tiktoken

    tokenizer = get_tokenizer(name)
    # Pyright cannot statically know the concrete subclass; just confirm the API surface and
    # that encoding round-trips a simple ASCII string.
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")
    assert isinstance(tokenizer, tiktoken.Encoding)


@pytest.mark.parametrize("name", ["TIKTOKEN", "GPT-4", "Cl100K_Base"])
def test_get_tokenizer_case_insensitive_for_aliases(name):
    """The match uses `tokenizer_name.lower()`, so any casing of a known alias must work."""
    tokenizer = get_tokenizer(name)
    assert hasattr(tokenizer, "encode")


def test_get_tokenizer_falls_back_to_huggingface_for_unknown_name(monkeypatch):
    """For an unknown name, the helper must call `transformers.AutoTokenizer.from_pretrained`
    with `legacy=False`. We mock the call so we never actually hit HF Hub."""
    recorded = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            recorded["name"] = name
            recorded["kwargs"] = kwargs
            return "FAKE_HF_TOKENIZER"

    monkeypatch.setattr(
        "helm.benchmark.scenarios.medalign_scenario_helper.transformers.AutoTokenizer",
        _FakeAutoTokenizer,
    )

    result = get_tokenizer("some-org/some-model")
    assert result == "FAKE_HF_TOKENIZER"
    assert recorded == {"name": "some-org/some-model", "kwargs": {"legacy": False}}


# ---------------------------------------------------------------------------
# `pack_and_trim_prompts` — verified against a deterministic fake tokenizer
# that treats every character as a token.
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """A trivial deterministic tokenizer where each token is a single character. This lets us
    reason exactly about token counts in tests without depending on tiktoken's behaviour."""

    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


PROMPT_TEMPLATE = "Instruction: {question}\n\nEHR: {ehr}\n\nAnswer:"


def test_pack_and_trim_returns_empty_ehr_when_include_ehr_is_false():
    """With `include_ehr=False`, `target_ehr_length` is forced to 0, so the EHR section in the
    prompt becomes empty regardless of how long the patient record actually is."""
    instructions = {1: {"instruction": "Q?", "patient_id": 7}}
    ehrs = {7: "X" * 1000}

    prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_string=PROMPT_TEMPLATE,
        context_length=10_000,
        generation_length=256,
        tokenizer=_CharTokenizer(),
        include_ehr=False,
    )

    assert prompts[1] == "Instruction: Q?\n\nEHR: \n\nAnswer:"


def test_pack_and_trim_returns_empty_ehr_when_budget_is_exhausted_by_instruction():
    """If `target_ehr_length <= 0` (template + instruction already exceed the context budget),
    the EHR is dropped entirely."""
    instructions = {1: {"instruction": "Q" * 200, "patient_id": 7}}
    ehrs = {7: "Patient EHR content"}

    prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_string=PROMPT_TEMPLATE,
        context_length=10,
        generation_length=5,
        tokenizer=_CharTokenizer(),
        include_ehr=True,
    )

    assert prompts[1] == f"Instruction: {'Q' * 200}\n\nEHR: \n\nAnswer:"


def test_pack_and_trim_produces_one_prompt_per_instruction():
    instructions = {
        1: {"instruction": "Q1", "patient_id": 7},
        2: {"instruction": "Q2", "patient_id": 7},
    }
    ehrs = {7: "Note about patient 7"}

    prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_string=PROMPT_TEMPLATE,
        context_length=200,
        generation_length=10,
        tokenizer=_CharTokenizer(),
        include_ehr=True,
    )

    assert set(prompts.keys()) == {1, 2}
    for prompt in prompts.values():
        assert prompt.startswith("Instruction: ")
        assert "Answer:" in prompt


def test_pack_and_trim_handles_int_string_patient_ids():
    """`pack_and_trim_prompts` calls `int(instructions[id]["patient_id"])`. A string-encoded ID
    must therefore be looked up via the int form."""
    instructions = {1: {"instruction": "Q", "patient_id": "42"}}
    ehrs = {42: "ehr text"}

    prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_string=PROMPT_TEMPLATE,
        context_length=10_000,
        generation_length=10,
        tokenizer=_CharTokenizer(),
        include_ehr=False,
    )

    assert 1 in prompts


# ---------------------------------------------------------------------------
# `preprocess_prompts` — end-to-end orchestration over disk artifacts.
# ---------------------------------------------------------------------------


def _setup_minimal_medalign_layout(data_dir: str) -> dict:
    """Lay down the on-disk files that `preprocess_prompts` / `return_dataset_dataframe` read.
    Returns the relevant paths for use in test assertions."""
    ehr_dir = os.path.join(data_dir, "medalign_ehr_xml")
    os.makedirs(ehr_dir, exist_ok=True)
    for pt_id, body in [(10, "<ehr>patient 10</ehr>"), (20, "<ehr>patient 20</ehr>")]:
        with open(os.path.join(ehr_dir, f"{pt_id}.xml"), "w", encoding="utf-8") as f:
            f.write(body)

    instr_path = os.path.join(data_dir, "clinician-reviewed-model-responses.tsv")
    _write_instructions_tsv(
        instr_path,
        [
            {"instruction_id": 1, "question": "Q1?", "person_id": 10, "is_selected_ehr": "yes"},
            {"instruction_id": 2, "question": "Q2?", "person_id": 20, "is_selected_ehr": "yes"},
        ],
    )

    refs_path = os.path.join(data_dir, "clinician-instruction-responses.tsv")
    _write_tsv(
        refs_path,
        [
            {"instruction_id": 1, "annotator_num": "Annotator_1", "clinician_response": "Ref A1"},
            {"instruction_id": 1, "annotator_num": "Annotator_2", "clinician_response": "Ref A2"},
            {"instruction_id": 2, "annotator_num": "Annotator_1", "clinician_response": "Ref B1"},
            {"instruction_id": 3, "annotator_num": "Annotator_1", "clinician_response": "orphan"},
        ],
        ["instruction_id", "annotator_num", "clinician_response"],
    )

    return {"instr_path": instr_path, "refs_path": refs_path, "ehr_dir": ehr_dir}


def test_preprocess_prompts_returns_one_row_per_instruction(monkeypatch):
    """End-to-end: builds a fake on-disk dataset and verifies that `preprocess_prompts`
    produces one row per instruction with a non-empty prompt string. The tokenizer is forced to
    the char-level fake to keep the test deterministic and free of network."""

    def _fake_get_tokenizer(name):
        return _CharTokenizer()

    monkeypatch.setattr("helm.benchmark.scenarios.medalign_scenario_helper.get_tokenizer", _fake_get_tokenizer)

    with TemporaryDirectory() as tmp:
        paths = _setup_minimal_medalign_layout(tmp)

        df = preprocess_prompts(
            target_context_length=2048,
            generation_length=128,
            path_to_instructions=paths["instr_path"],
            path_to_ehrs=paths["ehr_dir"],
            include_ehr=True,
            tokenizer="tiktoken",
        )

    assert set(df.columns) == {"instruction_id", "prompt"}
    assert set(df["instruction_id"]) == {1, 2}
    assert all(isinstance(p, str) and "Answer:" in p for p in df["prompt"])


def test_preprocess_prompts_asserts_non_empty_when_context_length_is_zero(monkeypatch):
    """If the context-length budget is so small that *every* prompt drops out, `pack_and_trim`
    still produces one truncated entry per input; what would actually cause the inner
    `assert filled_prompts` to fire is an *empty* instructions input. We exercise that path."""

    def _fake_get_tokenizer(name):
        return _CharTokenizer()

    monkeypatch.setattr("helm.benchmark.scenarios.medalign_scenario_helper.get_tokenizer", _fake_get_tokenizer)

    with TemporaryDirectory() as tmp:
        ehr_dir = os.path.join(tmp, "ehrs")
        os.makedirs(ehr_dir, exist_ok=True)

        instr_path = os.path.join(tmp, "instr.tsv")
        _write_instructions_tsv(instr_path, [])  # header only, no rows

        with pytest.raises(AssertionError, match="No prompts were found"):
            preprocess_prompts(
                target_context_length=2048,
                generation_length=128,
                path_to_instructions=instr_path,
                path_to_ehrs=ehr_dir,
                include_ehr=True,
                tokenizer="tiktoken",
            )


# ---------------------------------------------------------------------------
# `add_reference_responses` — filter + merge on instruction_id.
# ---------------------------------------------------------------------------


def test_add_reference_responses_filters_to_first_annotator_and_inner_joins():
    prompts_df = pd.DataFrame(
        [
            {"instruction_id": 1, "prompt": "P1"},
            {"instruction_id": 2, "prompt": "P2"},
        ]
    )

    with TemporaryDirectory() as tmp:
        refs_path = os.path.join(tmp, "refs.tsv")
        _write_tsv(
            refs_path,
            [
                {"instruction_id": 1, "annotator_num": "Annotator_1", "clinician_response": "A1"},
                {"instruction_id": 1, "annotator_num": "Annotator_2", "clinician_response": "should not appear"},
                {"instruction_id": 2, "annotator_num": "Annotator_1", "clinician_response": "B1"},
                {"instruction_id": 9, "annotator_num": "Annotator_1", "clinician_response": "orphan"},
            ],
            ["instruction_id", "annotator_num", "clinician_response"],
        )

        merged = add_reference_responses(prompts_df, refs_path)

    assert set(merged["instruction_id"]) == {1, 2}
    assert "should not appear" not in merged["clinician_response"].tolist()
    assert "orphan" not in merged["clinician_response"].tolist()  # inner join drops it
    assert set(merged.columns) >= {"instruction_id", "clinician_response", "prompt"}


def test_add_reference_responses_returns_empty_when_no_annotator_one_rows():
    prompts_df = pd.DataFrame([{"instruction_id": 1, "prompt": "P1"}])

    with TemporaryDirectory() as tmp:
        refs_path = os.path.join(tmp, "refs.tsv")
        _write_tsv(
            refs_path,
            [
                {"instruction_id": 1, "annotator_num": "Annotator_2", "clinician_response": "A2"},
            ],
            ["instruction_id", "annotator_num", "clinician_response"],
        )

        merged = add_reference_responses(prompts_df, refs_path)

    assert merged.empty


# ---------------------------------------------------------------------------
# `return_dataset_dataframe` — top-level orchestrator.
# ---------------------------------------------------------------------------


def test_return_dataset_dataframe_end_to_end(monkeypatch):
    """End-to-end test of the public entry point. The tokenizer is mocked to the char-level
    fake, so no network or heavy model load happens."""

    def _fake_get_tokenizer(name):
        return _CharTokenizer()

    monkeypatch.setattr("helm.benchmark.scenarios.medalign_scenario_helper.get_tokenizer", _fake_get_tokenizer)

    with TemporaryDirectory() as tmp:
        _setup_minimal_medalign_layout(tmp)
        df = return_dataset_dataframe(max_length=2048, data_path=tmp)

    assert set(df["instruction_id"]) == {1, 2}
    assert set(df["clinician_response"]) == {"Ref A1", "Ref B1"}
    assert all("Answer:" in p for p in df["prompt"])


def test_return_dataset_dataframe_raises_when_instructions_tsv_missing():
    """`check_file_exists` should raise (typically `FileNotFoundError`) before any preprocessing
    is attempted, giving the user a clear error message."""
    with TemporaryDirectory() as tmp:
        with pytest.raises(Exception):
            return_dataset_dataframe(max_length=2048, data_path=tmp)


def test_return_dataset_dataframe_raises_when_references_tsv_missing(monkeypatch):
    """If only the references file is missing (instructions is present), the helper should
    still fail loudly."""
    with TemporaryDirectory() as tmp:
        # Only write the instructions TSV and the EHR dir, but not the references TSV.
        ehr_dir = os.path.join(tmp, "medalign_ehr_xml")
        os.makedirs(ehr_dir, exist_ok=True)
        with open(os.path.join(ehr_dir, "1.xml"), "w", encoding="utf-8") as f:
            f.write("ehr")
        _write_instructions_tsv(
            os.path.join(tmp, "clinician-reviewed-model-responses.tsv"),
            [{"instruction_id": 1, "question": "Q?", "person_id": 1, "is_selected_ehr": "yes"}],
        )

        with pytest.raises(Exception):
            return_dataset_dataframe(max_length=2048, data_path=tmp)
