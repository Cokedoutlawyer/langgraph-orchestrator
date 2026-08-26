"""Vendor adapters — bidirectional canonical ↔ vendor schema mapping.

Each adapter isolates vendor-native responses from the pipeline. No vendor
object escapes its adapter. The adapters implement:
  to_vendor_request(canonical_model) → vendor-compatible request
  from_vendor_response(response) → canonical result schema
  from_vendor_error(error) → FailoverEvent or PipelineFailure
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from . import (
    Evidence,
    EvidenceSource,
    FailoverEvent,
    FailoverReason,
    ProviderName,
    ModelResult,
    BrowserResult,
    BrowserTask,
    ContentType,
    ModelPhase,
    PipelineFailure,
    FailureCategory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Research Adapters
# ---------------------------------------------------------------------------

class BrightDataResearchAdapter:
    """BrightData MCP → EvidenceSchema adapter."""

    @staticmethod
    def to_vendor_request(topic: str, max_results: int = 3, geo: str | None = None) -> dict:
        """Convert research topic to MCP tool call args."""
        return {"url": topic, "country": geo} if geo else {"url": topic}

    @staticmethod
    def to_vendor_search_request(query: str, geo: str | None = None) -> dict:
        return {"query": query, "country": geo} if geo else {"query": query}

    @staticmethod
    def from_vendor_response(
        response_text: str,
        url: str = "",
        run_id: str = "",
    ) -> EvidenceSource:
        return EvidenceSource(
            url=url or "brightdata://result",
            title=f"BrightData scrape: {url[:60]}" if url else "BrightData result",
            content=response_text,
            provider=ProviderName.BRIGHTDATA,
            content_type=ContentType.MARKDOWN,
            chars=len(response_text),
        )

    @staticmethod
    def from_vendor_error(error: Exception) -> FailoverEvent:
        reason = FailoverReason.BLOCKED if "blocked" in str(error).lower() else FailoverReason.ERROR
        return FailoverEvent(
            from_provider=ProviderName.BRIGHTDATA,
            to_provider=ProviderName.OXYLABS,
            reason=reason,
        )


class OxylabsResearchAdapter:
    """Oxylabs AI Studio → EvidenceSchema adapter."""

    @staticmethod
    def to_vendor_request(topic: str, max_results: int = 3, geo: str | None = None) -> dict:
        return {
            "query": topic,
            "limit": max_results,
            "return_content": True,
            **({"geo_location": geo} if geo else {}),
        }

    @staticmethod
    def to_vendor_scrape_request(url: str, output_format: str = "markdown") -> dict:
        return {"url": url, "output_format": output_format}

    @staticmethod
    def from_vendor_response(
        response: dict | list | str,
        run_id: str = "",
    ) -> list[EvidenceSource]:
        sources: list[EvidenceSource] = []
        if isinstance(response, list):
            for item in response:
                sources.append(EvidenceSource(
                    url=item.get("url", ""),
                    title=item.get("title", "Oxylabs result"),
                    content=item.get("content", item.get("snippet", "")),
                    provider=ProviderName.OXYLABS,
                    content_type=ContentType.MARKDOWN,
                    chars=len(item.get("content", "")),
                ))
        elif isinstance(response, dict):
            results = response.get("results", [response] if response.get("content") else [])
            for item in results:
                sources.append(EvidenceSource(
                    url=item.get("url", ""),
                    title=item.get("title", "Oxylabs result"),
                    content=item.get("content", item.get("snippet", "")),
                    provider=ProviderName.OXYLABS,
                    content_type=ContentType.MARKDOWN,
                    chars=len(item.get("content", "")),
                ))
        elif isinstance(response, str):
            sources.append(EvidenceSource(
                url="oxylabs://result",
                title="Oxylabs result",
                content=response,
                provider=ProviderName.OXYLABS,
                content_type=ContentType.MARKDOWN,
                chars=len(response),
            ))
        return sources

    @staticmethod
    def from_vendor_error(error: Exception) -> FailoverEvent:
        reason = FailoverReason.RATE_LIMIT if "429" in str(error) else FailoverReason.ERROR
        return FailoverEvent(
            from_provider=ProviderName.OXYLABS,
            to_provider=ProviderName.VENICE,
            reason=reason,
        )


class VeniceResearchAdapter:
    """Venice API → EvidenceSchema adapter (last resort)."""

    @staticmethod
    def to_vendor_request(topic: str) -> dict:
        return {
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": topic}],
            "enable_web_scraping": True,
        }

    @staticmethod
    def from_vendor_response(
        response_text: str,
        run_id: str = "",
    ) -> EvidenceSource:
        return EvidenceSource(
            url="venice://result",
            title="Venice result",
            content=response_text,
            provider=ProviderName.VENICE,
            content_type=ContentType.TEXT,
            chars=len(response_text),
        )

    @staticmethod
    def from_vendor_error(error: Exception) -> FailoverEvent:
        return FailoverEvent(
            from_provider=ProviderName.VENICE,
            to_provider=None,
            reason=FailoverReason.ERROR,
        )


# ---------------------------------------------------------------------------
# Model Adapters (OpenAI-compatible: OpenRouter, OpenAI, Grok, Nous)
# ---------------------------------------------------------------------------

class OpenAICompatModelAdapter:
    """Adapter for OpenAI-compatible chat completions APIs.

    Works with OpenRouter, OpenAI, Grok/xAI, and Nous Portal since they all
    implement the same /chat/completions endpoint shape.
    """

    @staticmethod
    def to_vendor_request(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: str = "text",
    ) -> dict:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def from_vendor_response(
        response: dict,
        run_id: str = "",
        phase: ModelPhase = ModelPhase.PLAN,
        provider: str = "openai",
    ) -> ModelResult:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        model = response.get("model", "")
        usage = response.get("usage", {})
        return ModelResult(
            run_id=run_id,
            phase=phase,
            content=content,
            provider=provider,
            model=model,
            tokens_input=usage.get("prompt_tokens"),
            tokens_output=usage.get("completion_tokens"),
        )

    @staticmethod
    def from_vendor_error(error: Exception) -> PipelineFailure:
        msg = str(error)
        if "429" in msg or "rate" in msg.lower():
            return PipelineFailure(
                stage="model",
                category=FailureCategory.PROVIDER_RETRYABLE,
                provider="unknown",
                retryable=True,
                message=msg,
            )
        if "401" in msg or "auth" in msg.lower():
            return PipelineFailure(
                stage="model",
                category=FailureCategory.PROVIDER_TERMINAL,
                provider="unknown",
                retryable=False,
                message=msg,
            )
        if "5" in msg[0:3] if msg else False:
            return PipelineFailure(
                stage="model",
                category=FailureCategory.PROVIDER_RETRYABLE,
                provider="unknown",
                retryable=True,
                message=msg,
            )
        return PipelineFailure(
            stage="model",
            category=FailureCategory.PROVIDER_TERMINAL,
            provider="unknown",
            retryable=False,
            message=msg,
        )


# ---------------------------------------------------------------------------
# Browser Adapters
# ---------------------------------------------------------------------------

class CamofoxBrowserAdapter:
    """Camofox → BrowserResultSchema adapter."""

    @staticmethod
    def to_vendor_request(task: BrowserTask) -> dict:
        return {
            "url": task.url,
            "userId": task.run_id,
            "sessionKey": task.session_id or "",
        }

    @staticmethod
    def from_vendor_response(
        content: str,
        url: str = "",
        session_id: str | None = None,
        run_id: str = "",
    ) -> BrowserResult:
        return BrowserResult(
            run_id=run_id,
            content=content,
            content_type=ContentType.ACCESSIBILITY_TREE,
            backend="camofox",
            url=url,
            session_id=session_id,
        )

    @staticmethod
    def from_vendor_error(error: Exception) -> FailoverEvent:
        return FailoverEvent(
            from_provider=ProviderName.CAMOFOX,
            to_provider=ProviderName.BRIGHTDATA,
            reason=FailoverReason.ERROR,
        )


class BrightDataBrowserAdapter:
    """BrightData browser → BrowserResultSchema adapter."""

    @staticmethod
    def to_vendor_request(task: BrowserTask) -> dict:
        return {"url": task.url}

    @staticmethod
    def from_vendor_response(content: str, url: str = "", run_id: str = "") -> BrowserResult:
        return BrowserResult(
            run_id=run_id,
            content=content,
            content_type=ContentType.MARKDOWN,
            backend="brightdata",
            url=url,
        )

    @staticmethod
    def from_vendor_error(error: Exception) -> FailoverEvent:
        return FailoverEvent(
            from_provider=ProviderName.BRIGHTDATA,
            to_provider=ProviderName.OXYLABS,
            reason=FailoverReason.ERROR,
        )


class OxylabsBrowserAdapter:
    """Oxylabs browser agent → BrowserResultSchema adapter."""

    @staticmethod
    def to_vendor_request(task: BrowserTask) -> dict:
        return {
            "url": task.url,
            "task_prompt": task.task_prompt or f"Extract content from {task.url}",
            "output_format": task.output_format,
        }

    @staticmethod
    def from_vendor_response(content: str, url: str = "", run_id: str = "") -> BrowserResult:
        return BrowserResult(
            run_id=run_id,
            content=content,
            content_type=ContentType.MARKDOWN,
            backend="oxylabs",
            url=url,
        )


# ---------------------------------------------------------------------------
# MCP Adapter — wraps MCP tool calls in validated schema boundaries
# ---------------------------------------------------------------------------

class MCPResearchAdapter:
    """MCP tool boundary adapter.

    Ensures MCP tool calls pass through the schema system:
    ResearchRequest → MCP request adapter → MCP invocation → response adapter → EvidenceSchema
    """

    def __init__(self, mcp_client: Any = None):
        """Initialize with an optional MCPStdioClient instance."""
        self._mcp_client = mcp_client

    def call_search(
        self,
        tool_name: str,
        query: str,
        run_id: str = "",
        provider: ProviderName = ProviderName.BRIGHTDATA,
    ) -> EvidenceSource:
        """Call an MCP search tool and normalize to EvidenceSource."""
        if self._mcp_client is None:
            raise RuntimeError("No MCP client configured")

        # Adapt canonical request to MCP inputSchema
        mcp_args = {"query": query}

        # Invoke MCP tool
        raw_response = self._mcp_client.call_tool(tool_name, mcp_args)

        # Normalize vendor response to canonical schema
        if provider == ProviderName.BRIGHTDATA:
            return BrightDataResearchAdapter.from_vendor_response(raw_response, url=query, run_id=run_id)
        elif provider == ProviderName.OXYLABS:
            sources = OxylabsResearchAdapter.from_vendor_response(raw_response, run_id=run_id)
            return sources[0] if sources else EvidenceSource(
                url="oxylabs://empty", title="Oxylabs empty", content="",
                provider=ProviderName.OXYLABS,
            )
        else:
            return EvidenceSource(
                url=f"{provider.value}://result",
                title=f"{provider.value} result",
                content=str(raw_response),
                provider=provider,
                content_type=ContentType.TEXT,
                chars=len(str(raw_response)),
            )

    def call_scrape(
        self,
        tool_name: str,
        url: str,
        run_id: str = "",
        provider: ProviderName = ProviderName.BRIGHTDATA,
    ) -> EvidenceSource:
        """Call an MCP scrape tool and normalize to EvidenceSource."""
        if self._mcp_client is None:
            raise RuntimeError("No MCP client configured")

        mcp_args = {"url": url}
        raw_response = self._mcp_client.call_tool(tool_name, mcp_args)

        if provider == ProviderName.BRIGHTDATA:
            return BrightDataResearchAdapter.from_vendor_response(raw_response, url=url, run_id=run_id)
        else:
            return EvidenceSource(
                url=url,
                title=f"{provider.value} scrape",
                content=str(raw_response),
                provider=provider,
                chars=len(str(raw_response)),
            )


# ---------------------------------------------------------------------------
# GitHub Adapter
# ---------------------------------------------------------------------------

class GitHubShipAdapter:
    """GitHub Git Data API adapter.

    Consumes ChangeSetSchema + ShipAuthorizationSchema → GitHub operations → ShipResultSchema
    """

    @staticmethod
    def to_vendor_create_branch(owner: str, repo: str, branch: str, sha: str) -> dict:
        return {"ref": f"refs/heads/{branch}", "sha": sha}

    @staticmethod
    def to_vendor_create_blob(owner: str, repo: str, content: str) -> dict:
        return {"content": content, "encoding": "utf-8"}

    @staticmethod
    def to_vendor_create_pr(
        owner: str, repo: str, title: str, head: str, base: str, body: str,
    ) -> dict:
        return {"title": title, "head": head, "base": base, "body": body}

    @staticmethod
    def to_vendor_merge(
        owner: str, repo: str, number: int, method: str, title: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"merge_method": method}
        if title:
            payload["commit_title"] = title
        return payload

    @staticmethod
    def from_vendor_pr_response(response: dict, run_id: str = "", branch: str = "") -> "ShipResult":
        from . import ShipResult
        return ShipResult(
            run_id=run_id,
            pr_number=response.get("number"),
            pr_url=response.get("html_url"),
            branch_name=branch,
        )

    @staticmethod
    def from_vendor_merge_response(response: dict, run_id: str = "", branch: str = "") -> "ShipResult":
        from . import ShipResult
        return ShipResult(
            run_id=run_id,
            merged=True,
            merge_message=response.get("message", "Merged"),
            branch_name=branch,
        )

    @staticmethod
    def from_vendor_error(error: Exception, run_id: str = "") -> PipelineFailure:
        msg = str(error)
        if "422" in msg:
            return PipelineFailure(
                stage="shipping", run_id=run_id,
                category=FailureCategory.SHIPPING_ERROR,
                message="PR already exists or validation failed",
                details=msg,
            )
        if "409" in msg:
            return PipelineFailure(
                stage="shipping", run_id=run_id,
                category=FailureCategory.SHIPPING_ERROR,
                message="Merge conflict",
                details=msg,
            )
        return PipelineFailure(
            stage="shipping", run_id=run_id,
            category=FailureCategory.SHIPPING_ERROR,
            message=msg,
        )
