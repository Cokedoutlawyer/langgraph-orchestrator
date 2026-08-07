"""Tests for vendor adapters — research, model, browser, MCP, GitHub."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline_schemas.adapters import (
    BrightDataResearchAdapter,
    OxylabsResearchAdapter,
    VeniceResearchAdapter,
    OpenAICompatModelAdapter,
    CamofoxBrowserAdapter,
    BrightDataBrowserAdapter,
    OxylabsBrowserAdapter,
    MCPResearchAdapter,
    GitHubShipAdapter,
)
from pipeline_schemas import (
    EvidenceSource, Evidence, ProviderName, ContentType,
    ModelResult, ModelPhase, BrowserResult, BrowserTask,
    FailoverEvent, FailoverReason,
    ShipResult, PipelineFailure,
)


class TestBrightDataAdapter:
    def test_from_vendor_response(self):
        src = BrightDataResearchAdapter.from_vendor_response("# Title\n\nContent", url="https://example.com")
        assert isinstance(src, EvidenceSource)
        assert src.provider == ProviderName.BRIGHTDATA
        assert src.content == "# Title\n\nContent"
        assert src.url == "https://example.com"

    def test_from_vendor_error_blocked(self):
        event = BrightDataResearchAdapter.from_vendor_error(Exception("blocked by CAPTCHA"))
        assert event.from_provider == ProviderName.BRIGHTDATA
        assert event.to_provider == ProviderName.OXYLABS
        assert event.reason == FailoverReason.BLOCKED

    def test_from_vendor_error_generic(self):
        event = BrightDataResearchAdapter.from_vendor_error(Exception("timeout"))
        assert event.reason == FailoverReason.ERROR


class TestOxylabsAdapter:
    def test_from_vendor_response_list(self):
        response = [
            {"url": "https://a.com", "title": "A", "content": "content A"},
            {"url": "https://b.com", "title": "B", "content": "content B"},
        ]
        sources = OxylabsResearchAdapter.from_vendor_response(response)
        assert len(sources) == 2
        assert all(s.provider == ProviderName.OXYLABS for s in sources)
        assert sources[0].url == "https://a.com"

    def test_from_vendor_response_string(self):
        sources = OxylabsResearchAdapter.from_vendor_response("raw content")
        assert len(sources) == 1
        assert sources[0].content == "raw content"

    def test_from_vendor_error_rate_limit(self):
        event = OxylabsResearchAdapter.from_vendor_error(Exception("429 Too Many Requests"))
        assert event.reason == FailoverReason.RATE_LIMIT
        assert event.to_provider == ProviderName.VENICE


class TestVeniceAdapter:
    def test_from_vendor_response(self):
        src = VeniceResearchAdapter.from_vendor_response("venice result text")
        assert src.provider == ProviderName.VENICE
        assert src.content == "venice result text"

    def test_from_vendor_error_no_failover(self):
        event = VeniceResearchAdapter.from_vendor_error(Exception("failed"))
        assert event.to_provider is None  # Last resort — no failover


class TestModelAdapter:
    def test_from_vendor_response(self):
        response = {
            "choices": [{"message": {"content": "plan output"}}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        result = OpenAICompatModelAdapter.from_vendor_response(
            response, run_id="r1", phase=ModelPhase.PLAN, provider="openai"
        )
        assert isinstance(result, ModelResult)
        assert result.content == "plan output"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.tokens_input == 100
        assert result.tokens_output == 50

    def test_from_vendor_error_retryable(self):
        fail = OpenAICompatModelAdapter.from_vendor_error(Exception("429 rate limited"))
        assert fail.retryable is True

    def test_from_vendor_error_terminal_auth(self):
        fail = OpenAICompatModelAdapter.from_vendor_error(Exception("401 unauthorized"))
        assert fail.retryable is False

    def test_to_vendor_request_json_format(self):
        req = OpenAICompatModelAdapter.to_vendor_request(
            "system", "user", 0.2, 4096, "json_object"
        )
        assert req["response_format"] == {"type": "json_object"}


class TestBrowserAdapters:
    def test_camofox_adapter(self):
        result = CamofoxBrowserAdapter.from_vendor_response(
            "<tree>content</tree>", url="https://x.com", run_id="r"
        )
        assert result.backend == "camofox"
        assert result.content_type == ContentType.ACCESSIBILITY_TREE

    def test_brightdata_browser_adapter(self):
        result = BrightDataBrowserAdapter.from_vendor_response("markdown content", url="https://x.com")
        assert result.backend == "brightdata"
        assert result.content_type == ContentType.MARKDOWN

    def test_oxylabs_browser_adapter(self):
        result = OxylabsBrowserAdapter.from_vendor_response("content", url="https://x.com")
        assert result.backend == "oxylabs"


class TestMCPAdapter:
    def test_call_search_normalizes_response(self):
        class MockMCP:
            def call_tool(self, name, args):
                return "mocked content"

        adapter = MCPResearchAdapter(mcp_client=MockMCP())
        src = adapter.call_search("search_engine", "test query", run_id="r", provider=ProviderName.BRIGHTDATA)
        assert isinstance(src, EvidenceSource)
        assert src.provider == ProviderName.BRIGHTDATA
        assert src.content == "mocked content"

    def test_no_client_raises(self):
        adapter = MCPResearchAdapter()
        with pytest.raises(RuntimeError, match="No MCP client"):
            adapter.call_search("t", "q")


class TestGitHubAdapter:
    def test_from_vendor_pr_response(self):
        response = {"number": 42, "html_url": "https://github.com/owner/repo/pull/42"}
        result = GitHubShipAdapter.from_vendor_pr_response(response, run_id="r", branch="auto-build/test")
        assert isinstance(result, ShipResult)
        assert result.pr_number == 42
        assert result.pr_url == "https://github.com/owner/repo/pull/42"

    def test_from_vendor_error_422(self):
        fail = GitHubShipAdapter.from_vendor_error(Exception("422 Validation Failed"))
        assert fail.category.value == "shipping_error"
        assert "already exists" in fail.message

    def test_from_vendor_error_409(self):
        fail = GitHubShipAdapter.from_vendor_error(Exception("409 Merge Conflict"))
        assert "conflict" in fail.message.lower()
