import os
import pytest
from tempfile import TemporaryDirectory
from typing import List

from helm.benchmark.scenarios.mtsamples_replicate_scenario import MTSamplesReplicateScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers for mocking `requests.get` without hitting the network.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for `requests.Response` covering the attributes the scenario uses."""

    def __init__(self, status_code: int, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


def _make_requests_get(responses_by_url):
    """Return a function that mimics `requests.get` and dispatches by URL."""

    def _fake_get(url, *args, **kwargs):
        if url not in responses_by_url:
            raise AssertionError(f"Unexpected URL requested in test: {url}")
        return responses_by_url[url]

    return _fake_get


# ---------------------------------------------------------------------------
# Unit tests for `extract_sections` (pure text logic, no I/O).
# ---------------------------------------------------------------------------


def test_extract_sections_all_present():
    scenario = MTSamplesReplicateScenario()
    text = (
        "PATIENT: John Doe\n"
        "FINDINGS: Mild swelling on the left ankle.\n"
        "SUMMARY: Follow-up in 2 weeks.\n"
        "PLAN: Ibuprofen 400mg every 6 hours as needed.\n"
    )

    plan, summary, findings = scenario.extract_sections(text)

    assert plan == "Ibuprofen 400mg every 6 hours as needed."
    assert summary == "Follow-up in 2 weeks."
    assert findings == "Mild swelling on the left ankle."


def test_extract_sections_only_plan():
    scenario = MTSamplesReplicateScenario()
    text = "PATIENT: Jane\nPLAN: Schedule MRI.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan == "Schedule MRI."
    assert summary is None
    assert findings is None


def test_extract_sections_only_summary():
    scenario = MTSamplesReplicateScenario()
    text = "SUMMARY: Patient is stable.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan is None
    assert summary == "Patient is stable."
    assert findings is None


def test_extract_sections_only_findings():
    scenario = MTSamplesReplicateScenario()
    text = "FINDINGS: Normal echocardiogram.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan is None
    assert summary is None
    assert findings == "Normal echocardiogram."


def test_extract_sections_none_present():
    scenario = MTSamplesReplicateScenario()
    text = "Some random clinical narrative without the required keywords.\n"

    plan, summary, findings = scenario.extract_sections(text)

    assert plan is None
    assert summary is None
    assert findings is None


def test_extract_sections_takes_only_first_line():
    """The implementation only captures the first line after the keyword (split on '\\n', 1)."""
    scenario = MTSamplesReplicateScenario()
    text = "PLAN: Step one.\nStep two should not be captured."

    plan, _, _ = scenario.extract_sections(text)

    assert plan == "Step one."


def test_extract_sections_strips_whitespace():
    scenario = MTSamplesReplicateScenario()
    text = "PLAN:    Lots of leading whitespace.   \n"

    plan, _, _ = scenario.extract_sections(text)

    assert plan == "Lots of leading whitespace."


# ---------------------------------------------------------------------------
# Unit tests for `remove_plan_section` (pure text logic).
# ---------------------------------------------------------------------------


def test_remove_plan_section_strips_plan():
    scenario = MTSamplesReplicateScenario()
    text = "HISTORY: Patient has X.\nPLAN: Take medication."

    cleaned = scenario.remove_plan_section(text)

    assert cleaned == "HISTORY: Patient has X."
    assert "PLAN:" not in cleaned


def test_remove_plan_section_no_plan_keyword():
    """If PLAN: is missing, the input is returned unchanged."""
    scenario = MTSamplesReplicateScenario()
    text = "HISTORY: Patient has X.\nSUMMARY: All good."

    cleaned = scenario.remove_plan_section(text)

    assert cleaned == text


def test_remove_plan_section_preserves_leading_content():
    scenario = MTSamplesReplicateScenario()
    text = "Line 1\nLine 2\nPLAN: cut here\nLine after"

    cleaned = scenario.remove_plan_section(text)

    assert cleaned.startswith("Line 1")
    assert "Line after" not in cleaned


# ---------------------------------------------------------------------------
# Unit tests for `fetch_file_list` (mocked HTTP).
# ---------------------------------------------------------------------------


def test_fetch_file_list_returns_only_txt_files(monkeypatch):
    """The GitHub API may return mixed file types; only .txt files should be kept."""
    scenario = MTSamplesReplicateScenario()
    api_response = _FakeResponse(
        status_code=200,
        json_data=[
            {"name": "note_a.txt"},
            {"name": "note_b.txt"},
            {"name": "README.md"},
            {"name": "image.png"},
            {"name": "subfolder"},
        ],
    )
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _make_requests_get({MTSamplesReplicateScenario.API_BASE_URL: api_response}),
    )

    files = scenario.fetch_file_list()

    assert files == ["note_a.txt", "note_b.txt"]


def test_fetch_file_list_raises_on_http_error(monkeypatch):
    scenario = MTSamplesReplicateScenario()
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _make_requests_get(
            {MTSamplesReplicateScenario.API_BASE_URL: _FakeResponse(status_code=404, text="Not Found")}
        ),
    )

    with pytest.raises(Exception, match="Failed to fetch file list"):
        scenario.fetch_file_list()


# ---------------------------------------------------------------------------
# Unit tests for `download_file` (mocked HTTP).
# ---------------------------------------------------------------------------


def test_download_file_writes_content_when_missing(monkeypatch):
    scenario = MTSamplesReplicateScenario()
    file_name = "sample.txt"
    raw_url = MTSamplesReplicateScenario.RAW_BASE_URL + file_name

    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _make_requests_get({raw_url: _FakeResponse(status_code=200, text="hello world")}),
    )

    with TemporaryDirectory() as tmpdir:
        path = scenario.download_file(file_name, tmpdir)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "hello world"


def test_download_file_skips_existing_file(monkeypatch):
    """If the file is already on disk, no HTTP request should be made."""
    scenario = MTSamplesReplicateScenario()

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("requests.get must not be called when the file already exists.")

    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _should_not_be_called,
    )

    with TemporaryDirectory() as tmpdir:
        existing = os.path.join(tmpdir, "cached.txt")
        with open(existing, "w", encoding="utf-8") as f:
            f.write("already here")
        path = scenario.download_file("cached.txt", tmpdir)

        assert path == existing
        with open(path, encoding="utf-8") as f:
            assert f.read() == "already here"


def test_download_file_raises_on_http_error(monkeypatch):
    scenario = MTSamplesReplicateScenario()
    file_name = "missing.txt"
    raw_url = MTSamplesReplicateScenario.RAW_BASE_URL + file_name
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _make_requests_get({raw_url: _FakeResponse(status_code=404)}),
    )

    with TemporaryDirectory() as tmpdir:
        with pytest.raises(Exception, match="Failed to download"):
            scenario.download_file(file_name, tmpdir)


# ---------------------------------------------------------------------------
# Mocked end-to-end tests for `get_instances` (no real network).
# ---------------------------------------------------------------------------


def _patch_scenario_with_fake_files(monkeypatch, files: List[tuple]):
    """Patch `requests.get` so the scenario sees only the supplied (filename, content) pairs."""
    api_response = _FakeResponse(
        status_code=200,
        json_data=[{"name": name} for name, _ in files],
    )
    responses = {MTSamplesReplicateScenario.API_BASE_URL: api_response}
    for name, content in files:
        responses[MTSamplesReplicateScenario.RAW_BASE_URL + name] = _FakeResponse(
            status_code=200, text=content
        )
    monkeypatch.setattr(
        "helm.benchmark.scenarios.mtsamples_replicate_scenario.requests.get",
        _make_requests_get(responses),
    )


def test_get_instances_with_mocked_files(monkeypatch):
    files = [
        (
            "note_with_plan.txt",
            "HISTORY: Knee pain.\nPLAN: Physical therapy 2x per week.\n",
        ),
        (
            "note_with_summary_only.txt",
            "HISTORY: Cough.\nSUMMARY: Likely viral infection.\n",
        ),
        (
            "note_with_findings_only.txt",
            "FINDINGS: Mild lung opacities.\n",
        ),
        (
            "note_without_any_section.txt",
            "Just a free-form clinical note.\n",
        ),
    ]
    _patch_scenario_with_fake_files(monkeypatch, files)

    scenario = MTSamplesReplicateScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 3
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)

    refs = sorted(instance.references[0].output.text for instance in instances)
    assert refs == [
        "Likely viral infection.",
        "Mild lung opacities.",
        "Physical therapy 2x per week.",
    ]


def test_get_instances_prefers_plan_over_summary_and_findings(monkeypatch):
    """When a note contains all three sections, PLAN must win."""
    _patch_scenario_with_fake_files(
        monkeypatch,
        [
            (
                "all_sections.txt",
                "FINDINGS: F.\nSUMMARY: S.\nPLAN: This is the chosen reference.\n",
            )
        ],
    )

    scenario = MTSamplesReplicateScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1
    assert instances[0].references[0].output.text == "This is the chosen reference."
    assert "PLAN:" not in instances[0].input.text


def test_get_instances_swallows_per_file_errors(monkeypatch):
    """A single broken file must not stop the rest of the dataset from being returned.

    The scenario wraps per-file processing in a try/except, so passing in a file that triggers
    a failure (e.g. PLAN keyword inside text but with no following content) should not crash.
    """
    files = [
        ("good.txt", "HISTORY: ok.\nPLAN: a treatment plan.\n"),
        # Surface a real bug observed in production data: lower-case "plan:" makes the
        # implementation try `text.split("PLAN:")[1]` and raise IndexError, which is then
        # caught by the try/except in `get_instances`.
        ("triggers_index_error.txt", "Some narrative mentioning the word plan: in lowercase only."),
    ]
    _patch_scenario_with_fake_files(monkeypatch, files)

    scenario = MTSamplesReplicateScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) == 1
    assert instances[0].references[0].output.text == "a treatment plan."


# ---------------------------------------------------------------------------
# Real-network integration test (slow; opt-in only).
# ---------------------------------------------------------------------------


@pytest.mark.scenarios
def test_mtsamples_scenario_get_instances_integration():
    """End-to-end smoke test against the real GitHub repo. ~2 minutes due to 1000 file downloads."""
    scenario = MTSamplesReplicateScenario()
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)

    assert len(instances) > 0
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 1 for instance in instances)
    assert all(CORRECT_TAG in instance.references[0].tags for instance in instances)
    assert all(instance.references[0].output.text for instance in instances)
    assert all("PLAN:" not in instance.input.text for instance in instances)


# ---------------------------------------------------------------------------
# Metadata + static attributes.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = MTSamplesReplicateScenario()
    metadata = scenario.get_metadata()

    assert metadata.name == "mtsamples_replicate"
    assert metadata.display_name == "MTSamples"
    assert metadata.short_display_name == "MTSamples"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "mtsamples_replicate_accuracy"
    assert metadata.taxonomy.task == "Text generation"
    assert metadata.taxonomy.language == "English"


def test_basic_attributes():
    scenario = MTSamplesReplicateScenario()

    assert scenario.name == "mtsamples_replicate"
    assert "medical" in scenario.tags
    assert "transcription" in scenario.tags
    assert "plan_generation" in scenario.tags
    assert MTSamplesReplicateScenario.GIT_HASH in MTSamplesReplicateScenario.API_BASE_URL
    assert MTSamplesReplicateScenario.GIT_HASH in MTSamplesReplicateScenario.RAW_BASE_URL
