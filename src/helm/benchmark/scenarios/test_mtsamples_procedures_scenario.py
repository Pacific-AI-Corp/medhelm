import os
import pytest
from tempfile import TemporaryDirectory
from typing import List

from helm.benchmark.scenarios.mtsamples_procedures_scenario import MTSamplesProceduresScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# `fetch_file_list` queries this exact GitHub API endpoint. Mirroring it as a constant
# keeps the mock dispatch table in sync with the scenario.
# ---------------------------------------------------------------------------

FETCH_API_URL = "https://api.github.com/repos/raulista1997/benchmarkdata/contents/mtsample_procedure"


# ---------------------------------------------------------------------------
# Helpers for mocking `requests.get`.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


def _make_requests_get(responses_by_url, recorded_calls=None):
    """Build a fake `requests.get` that dispatches by URL and optionally records call kwargs."""

    def _fake_get(url, *args, **kwargs):
        if recorded_calls is not None:
            recorded_calls.append({"url": url, "kwargs": kwargs})
        if url not in responses_by_url:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        return responses_by_url[url]

    return _fake_get


# ---------------------------------------------------------------------------
# Unit tests for `extract_sections` (pure text logic).
# ---------------------------------------------------------------------------


def test_extract_sections_all_present():
    scenario = MTSamplesProceduresScenario()
    text = "PATIENT: John Doe\n" "FINDINGS: Healthy heart.\n" "SUMMARY: Discharge.\n" "PLAN: Aspirin daily.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan == "Aspirin daily."
    assert summary == "Discharge."
    assert findings == "Healthy heart."


def test_extract_sections_only_findings():
    scenario = MTSamplesProceduresScenario()
    text = "FINDINGS: Normal scan.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan is None
    assert summary is None
    assert findings == "Normal scan."


def test_extract_sections_none_present():
    scenario = MTSamplesProceduresScenario()
    text = "OPERATION: Routine procedure.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan is None
    assert summary is None
    assert findings is None


def test_extract_sections_takes_only_first_line():
    scenario = MTSamplesProceduresScenario()
    text = "PLAN: Step one.\nStep two should not be captured."

    plan, _, _ = scenario.extract_sections(text)

    assert plan == "Step one."


# ---------------------------------------------------------------------------
# Unit tests for `remove_sections` (this scenario removes ALL three sections, not just PLAN).
# ---------------------------------------------------------------------------


def test_remove_sections_strips_all_three():
    """Unlike mtsamples_replicate (which only strips PLAN), this scenario strips PLAN/SUMMARY/FINDINGS.
    The implementation iterates over the three keywords and keeps only the content before the first
    one encountered, so the final string contains none of them."""
    scenario = MTSamplesProceduresScenario()
    text = "HISTORY: Surgery.\nFINDINGS: F\nSUMMARY: S\nPLAN: P"

    cleaned = scenario.remove_sections(text)

    assert "PLAN:" not in cleaned
    assert "SUMMARY:" not in cleaned
    assert "FINDINGS:" not in cleaned
    assert cleaned.startswith("HISTORY:")


def test_remove_sections_without_any_keyword():
    scenario = MTSamplesProceduresScenario()
    text = "Just a regular note with no special sections."

    cleaned = scenario.remove_sections(text)

    assert cleaned == text


def test_remove_sections_only_plan_present():
    scenario = MTSamplesProceduresScenario()
    text = "BEFORE\nPLAN: take meds"

    cleaned = scenario.remove_sections(text)

    assert cleaned == "BEFORE"


def test_remove_sections_only_summary_present():
    scenario = MTSamplesProceduresScenario()
    text = "BEFORE\nSUMMARY: ok"

    cleaned = scenario.remove_sections(text)

    assert cleaned == "BEFORE"


def test_remove_sections_only_findings_present():
    scenario = MTSamplesProceduresScenario()
    text = "BEFORE\nFINDINGS: normal"

    cleaned = scenario.remove_sections(text)

    assert cleaned == "BEFORE"


# ---------------------------------------------------------------------------
# Unit tests for `fetch_file_list` (mocked).
# ---------------------------------------------------------------------------


def test_fetch_file_list_returns_only_txt_files(monkeypatch):
    """Mixed file types should be filtered down to .txt entries."""
    scenario = MTSamplesProceduresScenario()
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get(
            {
                FETCH_API_URL: _FakeResponse(
                    status_code=200,
                    json_data=[
                        {"name": "procedure_a.txt"},
                        {"name": "procedure_b.txt"},
                        {"name": "README.md"},
                        {"name": "thumbnail.png"},
                    ],
                )
            }
        ),
    )

    files = scenario.fetch_file_list()

    assert files == ["procedure_a.txt", "procedure_b.txt"]


def test_fetch_file_list_sends_github_accept_header(monkeypatch):
    """The scenario advertises GitHub's JSON content type via the Accept header; the contract
    matters because GitHub may change its default response format."""
    scenario = MTSamplesProceduresScenario()
    recorded: list = []
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get(
            {FETCH_API_URL: _FakeResponse(status_code=200, json_data=[])},
            recorded_calls=recorded,
        ),
    )

    scenario.fetch_file_list()

    assert len(recorded) == 1
    assert recorded[0]["url"] == FETCH_API_URL
    assert recorded[0]["kwargs"]["headers"] == {"Accept": "application/vnd.github+json"}


def test_fetch_file_list_raises_on_http_error(monkeypatch):
    scenario = MTSamplesProceduresScenario()
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get({FETCH_API_URL: _FakeResponse(status_code=403)}),
    )

    with pytest.raises(Exception, match="Failed to fetch file list"):
        scenario.fetch_file_list()


# ---------------------------------------------------------------------------
# Unit tests for `download_file` (mocked).
# ---------------------------------------------------------------------------


def test_download_file_writes_content(monkeypatch):
    scenario = MTSamplesProceduresScenario()
    file_name = "proc.txt"
    raw_url = MTSamplesProceduresScenario.RAW_BASE_URL + file_name
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get({raw_url: _FakeResponse(status_code=200, text="procedure content")}),
    )

    with TemporaryDirectory() as tmpdir:
        path = scenario.download_file(file_name, tmpdir)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "procedure content"


def test_download_file_skips_existing_file(monkeypatch):
    """If the file is cached on disk, no HTTP request should be issued."""
    scenario = MTSamplesProceduresScenario()

    def _explode(*args, **kwargs):
        raise AssertionError("requests.get must not be called when the file already exists.")

    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _explode,
    )

    with TemporaryDirectory() as tmpdir:
        cached = os.path.join(tmpdir, "cached.txt")
        with open(cached, "w", encoding="utf-8") as f:
            f.write("already here")

        path = scenario.download_file("cached.txt", tmpdir)

        assert path == cached


def test_download_file_raises_on_http_error(monkeypatch):
    scenario = MTSamplesProceduresScenario()
    file_name = "missing.txt"
    raw_url = MTSamplesProceduresScenario.RAW_BASE_URL + file_name
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get({raw_url: _FakeResponse(status_code=404)}),
    )

    with TemporaryDirectory() as tmpdir:
        with pytest.raises(Exception, match="Failed to download"):
            scenario.download_file(file_name, tmpdir)


# ---------------------------------------------------------------------------
# End-to-end tests for `get_instances` with mocked GitHub (fast, no real network).
# ---------------------------------------------------------------------------


def _patch_scenario_with_fake_files(monkeypatch, files: List[tuple]):
    """Patch `requests.get` so the scenario sees only the supplied (filename, content) pairs."""
    responses = {
        FETCH_API_URL: _FakeResponse(
            status_code=200,
            json_data=[{"name": name} for name, _ in files],
        )
    }
    for name, content in files:
        responses[MTSamplesProceduresScenario.RAW_BASE_URL + name] = _FakeResponse(status_code=200, text=content)
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_procedures_scenario.requests.get",
        _make_requests_get(responses),
    )


def test_get_instances_with_mocked_files(monkeypatch):
    files = [
        ("proc_with_plan.txt", "OPERATION: Knee replacement.\nPLAN: Physiotherapy.\n"),
        ("proc_with_findings.txt", "FINDINGS: Healthy postop.\n"),
        ("proc_with_summary.txt", "SUMMARY: Routine outcome.\n"),
        ("proc_skipped.txt", "Just an unstructured note."),
    ]
    _patch_scenario_with_fake_files(monkeypatch, files)

    scenario = MTSamplesProceduresScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 3
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    # All three section keywords must be removed from any returned input.
    for instance in instances:
        assert "PLAN:" not in instance.input.text
        assert "SUMMARY:" not in instance.input.text
        assert "FINDINGS:" not in instance.input.text


def test_get_instances_prefers_plan_then_summary_then_findings(monkeypatch):
    """`plan or summary or findings` short-circuits in that order, so PLAN wins when all are present."""
    _patch_scenario_with_fake_files(
        monkeypatch,
        [
            ("all.txt", "FINDINGS: F.\nSUMMARY: S.\nPLAN: chosen.\n"),
            ("summary_and_findings.txt", "FINDINGS: F.\nSUMMARY: chosen.\n"),
            ("findings_only.txt", "FINDINGS: chosen.\n"),
        ],
    )

    scenario = MTSamplesProceduresScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    refs = sorted(instance.references[0].output.text for instance in instances)
    assert refs == ["chosen.", "chosen.", "chosen."]


def test_get_instances_swallows_per_file_errors(monkeypatch):
    """Per-file failures (caught by the try/except in get_instances) must not abort the run."""
    files = [
        ("good.txt", "OPERATION: ok.\nPLAN: post-op care.\n"),
        # Lowercase 'plan:' triggers IndexError in extract_sections because the implementation
        # detects via text.upper() but then splits on the literal uppercase token.
        ("triggers_index_error.txt", "note mentioning plan: in lowercase only"),
    ]
    _patch_scenario_with_fake_files(monkeypatch, files)

    scenario = MTSamplesProceduresScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1
    assert instances[0].references[0].output.text == "post-op care."


# ---------------------------------------------------------------------------
# Real-network integration test (slow, opt-in).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_get_instances_integration():
    """End-to-end smoke test against the real GitHub repo. Slow (~2 minutes for ~429 files)."""
    scenario = MTSamplesProceduresScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) > 0
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(instance.references[0].output.text for instance in instances)
    for instance in instances:
        assert "PLAN:" not in instance.input.text
        assert "SUMMARY:" not in instance.input.text
        assert "FINDINGS:" not in instance.input.text


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MTSamplesProceduresScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "mtsamples_procedures"
    assert metadata.display_name == "MTSamples Procedures"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "mtsamples_procedures_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MTSamplesProceduresScenario()

    assert scenario.name == "mtsamples_procedures"
    assert "medical" in scenario.tags
    assert "transcription" in scenario.tags
    assert "plan_generation" in scenario.tags
    assert MTSamplesProceduresScenario.GIT_HASH in MTSamplesProceduresScenario.RAW_BASE_URL
    assert MTSamplesProceduresScenario.GIT_HASH in MTSamplesProceduresScenario.GITHUB_DIR_URL
    assert MTSamplesProceduresScenario.RAW_BASE_URL.endswith("/mtsample_procedure/")
