import json
import os
import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.ehr_sql_scenario import EhrSqlScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Output, Reference


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Unit tests for `extract_schema` (pure SQL parsing).
# ---------------------------------------------------------------------------


def test_extract_schema_single_create_table():
    """The parser requires CREATE TABLE and the closing `;` on separate lines."""
    scenario = EhrSqlScenario()
    sql = "CREATE TABLE patient\n( id INT NOT NULL\n);\n"

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert schema == "CREATE TABLE patient\n( id INT NOT NULL\n);"


def test_extract_schema_multiple_create_tables():
    scenario = EhrSqlScenario()
    sql = "CREATE TABLE patient\n( id INT\n);\n" "\n" "CREATE TABLE diagnosis\n( code TEXT\n);\n"

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert "CREATE TABLE patient" in schema
    assert "CREATE TABLE diagnosis" in schema
    assert schema.count("CREATE TABLE") == 2
    # Tables are joined by a blank line.
    assert "\n\n" in schema


def test_extract_schema_drops_single_line_table_due_to_parser_limitation():
    """When CREATE TABLE and the closing `;` are on the same line, the parser never flushes the
    statement because the `endswith(";")` check is only run in the `elif` branch (i.e. on later
    lines). Real `eicu.sql` works around this by always putting CREATE TABLE on its own line."""
    scenario = EhrSqlScenario()
    sql = "CREATE TABLE patient ( id INT NOT NULL );\n"

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert schema == ""


def test_extract_schema_multiline_create_table():
    """Real EHR schemas span many lines and the parser must keep collecting until the closing `;`."""
    scenario = EhrSqlScenario()
    sql = (
        "CREATE TABLE patient\n"
        "(\n"
        "    uniquepid VARCHAR(10) NOT NULL,\n"
        "    age VARCHAR(10) NOT NULL,\n"
        "    gender VARCHAR(25)\n"
        ");\n"
    )

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert schema.startswith("CREATE TABLE patient")
    assert "uniquepid VARCHAR(10) NOT NULL" in schema
    assert "gender VARCHAR(25)" in schema
    assert schema.endswith(");")


def test_extract_schema_ignores_drop_and_other_statements():
    """Only CREATE TABLE statements should be captured; DROP/CREATE INDEX are ignored."""
    scenario = EhrSqlScenario()
    sql = (
        "DROP TABLE IF EXISTS patient;\n"
        "CREATE TABLE patient\n( id INT\n);\n"
        "CREATE INDEX idx_pat ON patient(id);\n"
    )

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert "DROP TABLE" not in schema
    assert "CREATE INDEX" not in schema
    assert "CREATE TABLE patient" in schema


def test_extract_schema_no_create_table_returns_empty_string():
    scenario = EhrSqlScenario()
    sql = "-- A comment only.\nSELECT 1;\n"

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert schema == ""


def test_extract_schema_drops_unterminated_table():
    """Tables that never close with `;` are not flushed into the output.

    This documents the current parser behavior: `inside_create_table` is only released when a
    line ending with `;` is encountered.
    """
    scenario = EhrSqlScenario()
    sql = "CREATE TABLE patient ( id INT\n-- no closing semicolon"

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "schema.sql")
        _write_text(path, sql)
        schema = scenario.extract_schema(path)

    assert schema == ""


# ---------------------------------------------------------------------------
# Unit tests for `process_json`.
# ---------------------------------------------------------------------------


def _entry(question="What is the patient age?", query="SELECT age FROM patient;", **overrides):
    """Build a JSON entry with sensible defaults, overridable per test."""
    entry = {
        "question": question,
        "query": query,
        "value": {"foo": "bar"},
        "is_impossible": False,
    }
    entry.update(overrides)
    return entry


def test_process_json_basic_entry_creates_instance():
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(path, [_entry()])

        instances = scenario.process_json(path, schema_prompt="CREATE TABLE patient (id INT);", split=TRAIN_SPLIT)

    assert len(instances) == 1
    assert instances[0].split == TRAIN_SPLIT
    assert "-- Database Schema:\nCREATE TABLE patient (id INT);" in instances[0].input.text
    assert "What is the patient age?" in instances[0].input.text
    assert instances[0].references == [
        Reference(output=Output(text="SELECT age FROM patient;"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].extra_data["db_path"] == "eicu.sqlite"
    assert instances[0].extra_data["value"] == {"foo": "bar"}
    assert instances[0].extra_data["is_impossible"] is False


def test_process_json_filters_out_null_string_queries():
    """Entries with `query == "null"` (literal string, not Python None) must be dropped."""
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(
            path,
            [
                _entry(query="SELECT 1;"),
                _entry(query="null"),  # filtered
                _entry(query="SELECT 2;"),
            ],
        )

        instances = scenario.process_json(path, schema_prompt="schema", split=TEST_SPLIT)

    assert len(instances) == 2
    assert [i.references[0].output.text for i in instances] == ["SELECT 1;", "SELECT 2;"]


def test_process_json_skips_entries_missing_question_or_query():
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(
            path,
            [
                {"query": "SELECT 1;"},  # missing question
                {"question": "Q?"},  # missing query
                _entry(),  # valid
            ],
        )

        instances = scenario.process_json(path, schema_prompt="schema", split=TEST_SPLIT)

    assert len(instances) == 1


def test_process_json_value_defaults_to_empty_dict_when_missing():
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(path, [{"question": "Q?", "query": "SELECT 1;"}])  # no `value`, no `is_impossible`

        instances = scenario.process_json(path, schema_prompt="schema", split=TEST_SPLIT)

    assert instances[0].extra_data["value"] == {}
    assert instances[0].extra_data["is_impossible"] is False


def test_process_json_propagates_is_impossible_flag():
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(
            path,
            [
                _entry(is_impossible=True),
                _entry(is_impossible=False),
            ],
        )

        instances = scenario.process_json(path, schema_prompt="schema", split=TEST_SPLIT)

    assert instances[0].extra_data["is_impossible"] is True
    assert instances[1].extra_data["is_impossible"] is False


def test_process_json_empty_list_returns_no_instances():
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(path, [])

        instances = scenario.process_json(path, schema_prompt="schema", split=TEST_SPLIT)

    assert instances == []


def test_process_json_includes_schema_prompt_in_every_input():
    """The same schema prompt must be prefixed to every generated instance."""
    scenario = EhrSqlScenario()
    schema_prompt = "CREATE TABLE patient ( id INT );\n\nCREATE TABLE diagnosis ( code TEXT );"
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(path, [_entry(question=f"Q{i}", query=f"SELECT {i};") for i in range(3)])

        instances = scenario.process_json(path, schema_prompt=schema_prompt, split=TRAIN_SPLIT)

    assert len(instances) == 3
    for i, instance in enumerate(instances):
        assert instance.input.text.startswith(f"-- Database Schema:\n{schema_prompt}")
        assert f"Q{i}" in instance.input.text


@pytest.mark.parametrize("split", [TRAIN_SPLIT, TEST_SPLIT])
def test_process_json_propagates_split(split):
    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.json")
        _write_json(path, [_entry()])

        instances = scenario.process_json(path, schema_prompt="schema", split=split)

    assert instances[0].split == split


# ---------------------------------------------------------------------------
# Mocked tests for the `download_*` / `setup_database` wrappers.
# ---------------------------------------------------------------------------


def _patch_ensure_file_downloaded(monkeypatch):
    """Replace `ensure_file_downloaded` with a recorder; returns the call log."""
    calls: list = []

    def _fake(source_url, target_path, **kwargs):
        calls.append({"source_url": source_url, "target_path": target_path, "kwargs": kwargs})
        # Touch the file so callers that read it later won't crash.
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("")

    monkeypatch.setattr("helm.benchmark.scenarios.ehr_sql_scenario.ensure_file_downloaded", _fake)
    return calls


def test_setup_database_targets_correct_url_and_path(monkeypatch):
    scenario = EhrSqlScenario()
    calls = _patch_ensure_file_downloaded(monkeypatch)

    with TemporaryDirectory() as tmpdir:
        db_path = scenario.setup_database(tmpdir)

    assert db_path == os.path.join(tmpdir, "eicu.sqlite")
    assert len(calls) == 1
    assert calls[0]["source_url"] == EhrSqlScenario.DB_URL
    assert calls[0]["target_path"] == db_path


def test_download_sql_schema_targets_correct_url_and_path(monkeypatch):
    scenario = EhrSqlScenario()
    calls = _patch_ensure_file_downloaded(monkeypatch)

    with TemporaryDirectory() as tmpdir:
        schema_path = scenario.download_sql_schema(tmpdir)

    assert schema_path == os.path.join(tmpdir, "eicu.sql")
    assert calls[0]["source_url"] == EhrSqlScenario.SQL_SCHEMA_URL


@pytest.mark.parametrize("split_name", ["train", "valid", "test"])
def test_download_json_builds_split_specific_url_and_filename(monkeypatch, split_name):
    scenario = EhrSqlScenario()
    calls = _patch_ensure_file_downloaded(monkeypatch)

    with TemporaryDirectory() as tmpdir:
        json_path = scenario.download_json(split_name, tmpdir)

    assert json_path == os.path.join(tmpdir, f"ehrsql_{split_name}.json")
    assert calls[0]["source_url"] == f"{EhrSqlScenario.BASE_URL}/{split_name}.json"


# ---------------------------------------------------------------------------
# Fully-mocked end-to-end test for `get_instances`.
# ---------------------------------------------------------------------------


def test_get_instances_end_to_end_with_mocks(monkeypatch):
    """Verifies the full orchestration without touching the network: SQL schema parsing,
    JSON ingestion across all three split files, and HELM-split remapping
    (`train -> TRAIN_SPLIT`, `valid -> TEST_SPLIT`, `test -> TEST_SPLIT`)."""

    train_entries = [
        {"question": "Q-train-1", "query": "SELECT 1;", "value": {}, "is_impossible": False},
        {"question": "Q-train-2", "query": "SELECT 2;", "value": {}, "is_impossible": False},
    ]
    valid_entries = [
        {"question": "Q-valid-1", "query": "SELECT 3;", "value": {}, "is_impossible": False},
        {"question": "Q-valid-2", "query": "null", "value": {}, "is_impossible": True},  # filtered
    ]
    test_entries = [
        {"question": "Q-test-1", "query": "SELECT 4;", "value": {}, "is_impossible": False},
    ]
    schema_sql = "CREATE TABLE patient\n( id INT\n);\n"

    def _fake_ensure_file_downloaded(source_url, target_path, **kwargs):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if target_path.endswith("eicu.sqlite"):
            with open(target_path, "wb") as f:
                f.write(b"")  # placeholder; not used in get_instances
        elif target_path.endswith("eicu.sql"):
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(schema_sql)
        elif target_path.endswith("ehrsql_train.json"):
            _write_json(target_path, train_entries)
        elif target_path.endswith("ehrsql_valid.json"):
            _write_json(target_path, valid_entries)
        elif target_path.endswith("ehrsql_test.json"):
            _write_json(target_path, test_entries)
        else:
            raise AssertionError(f"Unexpected download target: {target_path}")

    monkeypatch.setattr(
        "helm.benchmark.scenarios.ehr_sql_scenario.ensure_file_downloaded",
        _fake_ensure_file_downloaded,
    )

    scenario = EhrSqlScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    # 2 train + 1 valid (one filtered) + 1 test = 4
    assert len(instances) == 4

    by_split: dict = {TRAIN_SPLIT: [], TEST_SPLIT: []}
    for instance in instances:
        by_split[instance.split].append(instance)

    assert len(by_split[TRAIN_SPLIT]) == 2
    assert len(by_split[TEST_SPLIT]) == 2  # 1 from valid + 1 from test

    # Schema content propagates into every instance's input text.
    assert all("CREATE TABLE patient" in instance.input.text for instance in instances)
    # db_path is consistent.
    assert all(instance.extra_data["db_path"] == "eicu.sqlite" for instance in instances)


# ---------------------------------------------------------------------------
# Static attributes.
# ---------------------------------------------------------------------------


def test_basic_attributes():
    scenario = EhrSqlScenario()

    assert scenario.name == "ehr_sql"
    assert "sql" in scenario.tags
    assert "medical" in scenario.tags
    assert "reasoning" in scenario.tags
    assert EhrSqlScenario.BASE_URL.endswith("/eicu")
    assert "drive.usercontent.google.com" in EhrSqlScenario.DB_URL
    assert EhrSqlScenario.SQL_SCHEMA_URL.endswith("/eicu.sql")
    # The schema URL pins a specific git commit for reproducibility.
    assert "9eb39b5e1fd0e4e2ec7bc31208a768dcdf873c50" in EhrSqlScenario.SQL_SCHEMA_URL


def test_scenario_has_no_metadata_method():
    """Unlike other MedHELM scenarios, EHR SQL does not implement get_metadata. Document this
    so that anyone refactoring is forced to consciously add it (and update this test)."""
    scenario = EhrSqlScenario()

    assert "get_metadata" not in EhrSqlScenario.__dict__
    # The base class provides a default that raises, so calling it is a runtime error.
    with pytest.raises(NotImplementedError):
        scenario.get_metadata()
