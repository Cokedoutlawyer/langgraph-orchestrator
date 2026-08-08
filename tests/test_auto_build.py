"""Integration tests for the auto_build package.

All tests are hermetic: no network or LLM access required. HTTP calls are
mocked via unittest.mock. SQLite persistence uses a temp directory.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure src is importable (mirrors conftest, which only inserts for collection
# of this module if run standalone).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_build.state import (
    BuildPlan,
    BuildTask,
    PipelineState,
    PipelineStatus,
    StateStore,
)
from auto_build.qa import QAChecker
from auto_build.llm import LLMClient
from auto_build.web_fetcher import WebFetcher
from auto_build.oxylabs_client import OxylabsClient, OxylabsError
from auto_build.planner import Planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> PipelineState:
    defaults = dict(
        repo_owner="octo",
        repo_name="cat",
        issue_number=42,
        issue_title="Add FastAPI endpoint",
        requirements="Build a FastAPI endpoint",
    )
    defaults.update(overrides)
    return PipelineState(**defaults)


# ---------------------------------------------------------------------------
# 1) StateStore save / load / delete / find_by_issue
# ---------------------------------------------------------------------------

class TestStateStore:
    """StateStore persistence round-trips using a temp SQLite DB."""

    def test_save_and_load(self, tmp_path):
        db = tmp_path / "state.db"
        store = StateStore(db_path=db)
        state = _make_state(status=PipelineStatus.PLANNING)

        store.save(state)

        loaded = store.load(state.run_id)
        assert loaded is not None
        assert loaded.run_id == state.run_id
        assert loaded.status == PipelineStatus.PLANNING
        assert loaded.repo_owner == "octo"
        assert loaded.repo_name == "cat"
        assert loaded.issue_number == 42

    def test_load_missing_returns_none(self, tmp_path):
        store = StateStore(db_path=tmp_path / "state.db")
        assert store.load("does-not-exist") is None

    def test_delete(self, tmp_path):
        db = tmp_path / "state.db"
        store = StateStore(db_path=db)
        state = _make_state()
        store.save(state)
        assert store.load(state.run_id) is not None

        store.delete(state.run_id)

        assert store.load(state.run_id) is None

    def test_find_by_issue(self, tmp_path):
        db = tmp_path / "state.db"
        store = StateStore(db_path=db)

        # Two runs for issue 42, one for issue 7
        s1 = _make_state(issue_number=42)
        s2 = _make_state(issue_number=42)
        s3 = _make_state(issue_number=7)
        store.save(s1)
        store.save(s2)
        store.save(s3)

        found = store.find_by_issue(42)
        assert len(found) == 2
        assert all(r.issue_number == 42 for r in found)
        # No results for unknown issue
        assert store.find_by_issue(999) == []

    def test_save_updates_existing(self, tmp_path):
        db = tmp_path / "state.db"
        store = StateStore(db_path=db)
        state = _make_state(status=PipelineStatus.PLANNING)
        store.save(state)

        # Transition to CODING and save again
        state.status = PipelineStatus.CODING
        store.save(state)

        loaded = store.load(state.run_id)
        assert loaded is not None
        assert loaded.status == PipelineStatus.CODING
        # Still only one row
        assert len(store.list_runs()) == 1


# ---------------------------------------------------------------------------
# 2) QAChecker Python syntax (valid + invalid)
# ---------------------------------------------------------------------------

class TestQACheckerPythonSyntax:
    def test_valid_python_passes(self):
        qa = QAChecker()
        files = {"main.py": "def add(a, b):\n    return a + b\n"}
        errors = qa._check_syntax(files)
        assert errors == []

    def test_invalid_python_detected(self):
        qa = QAChecker()
        files = {"broken.py": "def add(a, b)\n    return a + b\n"}  # missing colon
        errors = qa._check_syntax(files)
        assert len(errors) == 1
        assert "broken.py" in errors[0]
        assert "syntax error" in errors[0].lower()

    def test_run_passes_for_valid_files(self):
        qa = QAChecker()
        state = _make_state()
        state.files_committed = {"good.py": "x = 1\n"}
        result = qa.run(state)
        assert result.passed is True
        assert result.lint_passed is True
        assert result.lint_errors == []

    def test_run_fails_for_invalid_files(self):
        qa = QAChecker()
        state = _make_state()
        state.files_committed = {"bad.py": "def (\n"}
        result = qa.run(state)
        assert result.passed is False
        assert result.lint_passed is False
        assert len(result.lint_errors) == 1


# ---------------------------------------------------------------------------
# 3) QAChecker security scan detects hardcoded API keys
# ---------------------------------------------------------------------------

class TestQACheckerSecurity:
    def test_detects_hardcoded_api_key(self):
        qa = QAChecker()
        files = {
            "config.py": 'api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"\n',
        }
        findings = qa._security_scan(files)
        assert len(findings) >= 1
        # Should mention either hardcoded API key or OpenAI-style key
        joined = " ".join(findings).lower()
        assert "api key" in joined or "openai" in joined

    def test_detects_openai_style_key(self):
        qa = QAChecker()
        files = {"client.py": 'token = "sk-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx"\n'}
        findings = qa._security_scan(files)
        assert len(findings) >= 1

    def test_skips_test_files(self):
        qa = QAChecker()
        files = {
            "tests/test_config.py": 'api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"\n',
        }
        findings = qa._security_scan(files)
        assert findings == []

    def test_clean_file_no_findings(self):
        qa = QAChecker()
        files = {"main.py": "def run():\n    pass\n"}
        assert qa._security_scan(files) == []


# ---------------------------------------------------------------------------
# 4) QAChecker skips TypeScript files in JS syntax check
# ---------------------------------------------------------------------------

class TestQACheckerTypeScriptSkip:
    def test_ts_file_skipped(self):
        qa = QAChecker()
        # Even with invalid JS content, .ts should be skipped (returns None)
        err = qa._check_js_syntax("app.ts", "const x = ; invalid ts {{{")
        assert err is None

    def test_tsx_file_skipped(self):
        qa = QAChecker()
        err = qa._check_js_syntax("component.tsx", "return <div>")
        assert err is None

    def test_full_check_syntax_skips_ts(self):
        qa = QAChecker()
        files = {"app.ts": "this is not valid javascript at all !!!"}
        # Should produce no errors because TS is skipped
        errors = qa._check_syntax(files)
        assert errors == []


# ---------------------------------------------------------------------------
# 5) LLMClient construction from env vars
# ---------------------------------------------------------------------------

class TestLLMClientConstruction:
    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key-12345")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        # Clear fallbacks
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        client = LLMClient()
        assert client.api_key == "test-key-12345"
        assert client.base_url == "https://api.example.com/v1"
        assert client.model == "test-model"
        assert client.temperature == 0.2
        assert client.max_tokens == 4096

    def test_falls_back_to_openai_env(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v2")

        client = LLMClient()
        assert client.api_key == "openai-key"
        assert client.base_url == "https://api.openai.com/v2"

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        client = LLMClient()
        assert client.model == "gpt-4o"
        assert client.base_url == "https://api.openai.com/v1"
        assert client.api_key == ""


# ---------------------------------------------------------------------------
# 6) WebFetcher search returns results (mock httpx)
# ---------------------------------------------------------------------------

class TestWebFetcherSearch:
    def test_search_returns_parsed_results(self):
        fetcher = WebFetcher()

        # Craft DuckDuckGo-style HTML with two results
        html = """
        <html><body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffastapi.tiangolo.com%2F">FastAPI</a>
        <a class="result__snippet">FastAPI is a modern web framework.</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ffastapi-tutorial">FastAPI Tutorial</a>
        <a class="result__snippet">Learn FastAPI in this tutorial.</a>
        </body></html>
        """

        mock_resp = mock.MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = mock.MagicMock()

        mock_client = mock.MagicMock()
        mock_client.get = mock.MagicMock(return_value=mock_resp)

        # Inject the mock client so no real HTTP is made
        fetcher._client = mock_client

        results = fetcher.search("FastAPI documentation", max_results=5)

        assert len(results) == 2
        assert results[0]["url"] == "https://fastapi.tiangolo.com/"
        assert results[0]["title"] == "FastAPI"
        assert "FastAPI is a modern web framework" in results[0]["snippet"]
        assert results[1]["url"] == "https://example.com/fastapi-tutorial"
        mock_client.get.assert_called_once()

    def test_search_empty_on_failure(self):
        fetcher = WebFetcher()
        mock_client = mock.MagicMock()
        mock_client.get = mock.MagicMock(side_effect=Exception("network down"))
        fetcher._client = mock_client

        results = fetcher.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
# 7) OxylabsClient construction and health check (mock httpx)
# ---------------------------------------------------------------------------

class TestOxylabsClient:
    def test_construction_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OXYLABS_AI_STUDIO_API_KEY", raising=False)
        with pytest.raises(OxylabsError):
            OxylabsClient()

    def test_construction_with_explicit_key(self):
        client = OxylabsClient(api_key="oxylabs-test-key")
        assert client.api_key == "oxylabs-test-key"
        assert client.timeout == 120.0

    def test_health_check_ok(self):
        client = OxylabsClient(api_key="oxylabs-test-key")

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = mock.MagicMock(return_value={
            "credits_remaining": 500,
            "plan": "free",
        })

        mock_client = mock.MagicMock()
        mock_client.get = mock.MagicMock(return_value=mock_resp)
        client._client = mock_client

        result = client.health_check()
        assert result["ok"] is True
        assert result["credits_remaining"] == 500
        assert result["plan"] == "free"
        mock_client.get.assert_called_once_with("/account/info")

    def test_health_check_invalid_key(self):
        client = OxylabsClient(api_key="bad-key")

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 401

        mock_client = mock.MagicMock()
        mock_client.get = mock.MagicMock(return_value=mock_resp)
        client._client = mock_client

        result = client.health_check()
        assert result["ok"] is False
        assert "Invalid API key" in result["error"]


# ---------------------------------------------------------------------------
# 8) Planner._extract_research_topics finds library names
# ---------------------------------------------------------------------------

class TestPlannerResearchTopics:
    def _make_planner(self) -> Planner:
        # Pass mocks to avoid constructing real LLM/GitHub/WebFetcher clients
        # that might touch env or network.
        return Planner(
            llm=mock.MagicMock(spec=LLMClient),
            github=mock.MagicMock(),
            web_fetcher=mock.MagicMock(spec=WebFetcher),
        )

    def test_finds_known_library_names(self):
        planner = self._make_planner()
        text = "We need to build a FastAPI endpoint and use httpx for requests."
        topics = planner._extract_research_topics(text)
        # Should find FastAPI and httpx (and possibly "requests")
        joined = " ".join(topics).lower()
        assert "fastapi" in joined
        assert "httpx" in joined

    def test_finds_multiple_libraries(self):
        planner = self._make_planner()
        text = "Integrate LangChain, LangGraph, and Pydantic into the pipeline."
        topics = planner._extract_research_topics(text)
        joined = " ".join(topics).lower()
        assert "langchain" in joined
        assert "langgraph" in joined
        assert "pydantic" in joined

    def test_finds_react_and_next(self):
        planner = self._make_planner()
        text = "Use React and Next.js for the frontend."
        topics = planner._extract_research_topics(text)
        joined = " ".join(topics).lower()
        assert "react" in joined
        assert "next.js" in joined

    def test_empty_text_returns_empty(self):
        planner = self._make_planner()
        assert planner._extract_research_topics("") == []
        assert planner._extract_research_topics("just some plain words") == []

    def test_deduplicates(self):
        planner = self._make_planner()
        text = "FastAPI is great. FastAPI is fast. Use FastAPI."
        topics = planner._extract_research_topics(text)
        # FastAPI should appear only once (deduplicated)
        fastapi_count = sum(1 for t in topics if "fastapi" in t.lower())
        assert fastapi_count == 1

    def test_limits_to_five(self):
        planner = self._make_planner()
        text = (
            "Use FastAPI Flask Django Pydantic LangChain LangGraph React Vue "
            "Angular Next.js PyTorch TensorFlow"
        )
        topics = planner._extract_research_topics(text)
        assert len(topics) <= 5
