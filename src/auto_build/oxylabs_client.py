"""Oxylabs AI Studio client — stealthy real-time web scraping.

Provides three scraping modes:
1. ai_scraper: Scrape a single URL with AI-powered content extraction
2. ai_search: Search the web and return structured results with content
3. ai_browser_agent: Browser automation with natural-language task prompts

All calls go through Oxylabs' proxy infrastructure, making them stealthy
(rotating IPs, anti-detection, geo-rotation). This replaces the basic
WebFetcher's DuckDuckGo path when an Oxylabs API key is configured.

Setup:
    Set OXYLABS_AI_STUDIO_API_KEY in your environment or .env file.

Usage in the auto-build pipeline:
    from auto_build.oxylabs_client import OxylabsClient
    client = OxylabsClient(api_key="...")
    content = client.ai_scraper("https://example.com", output_format="markdown")
    results = client.ai_search("FastAPI dependency injection", limit=5)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Oxylabs AI Studio API endpoints
_BASE_URL = "https://api.oxylabs.io/ai-studio/v1"


class OxylabsError(Exception):
    """Raised when an Oxylabs API call fails."""


class OxylabsClient:
    """HTTP client for Oxylabs AI Studio.

    Provides stealthy web scraping with AI-powered content extraction.
    All requests go through Oxylabs' proxy infrastructure (rotating IPs,
    anti-detection headers, geo-rotation).

    Three modes:
    - ai_scraper: Scrape a URL → clean markdown/JSON/HTML
    - ai_search: Web search → structured results with page content
    - ai_browser_agent: Browser automation via natural-language prompts
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 3.0,
        max_poll: float = 90.0,
    ):
        self.api_key = api_key or os.environ.get("OXYLABS_AI_STUDIO_API_KEY", "")
        if not self.api_key:
            raise OxylabsError(
                "OXYLABS_AI_STUDIO_API_KEY not set. Get a key at "
                "https://dashboard.oxylabs.io/ (1000 free credits on AI Studio)."
            )
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll = max_poll
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "hermes-auto-build/1.0",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> OxylabsClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------
    # AI Scraper — scrape a single URL
    # -----------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )
    def ai_scraper(
        self,
        url: str,
        output_format: str = "markdown",
        render_javascript: bool = False,
        geo: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Scrape a URL using Oxylabs AI Studio.

        Args:
            url: URL to scrape.
            output_format: "markdown", "json", or "html".
            render_javascript: Whether to render JS (slower, needed for SPAs).
            geo: ISO country code for geo-located scraping (e.g., "us", "gb", "de").
            timeout: Per-request timeout override (seconds).

        Returns:
            Scraped content as a string (markdown/JSON/HTML).
        """
        payload: dict[str, Any] = {
            "url": url,
            "output_format": output_format,
            "render_javascript": render_javascript,
        }
        if geo:
            payload["geo_location"] = geo

        logger.info("Oxylabs.ai_scraper: %s (format=%s, js=%s, geo=%s)", url[:80], output_format, render_javascript, geo)

        # AI Studio uses async jobs — submit, then poll for result
        response = self._submit_job("/ai-scraper", payload)
        job_id: str | None = response.get("job_id") or response.get("id")
        if not job_id:
            # Some endpoints return content directly (sync mode)
            content = response.get("content") or response.get("result") or ""
            if content:
                return content
            raise OxylabsError(f"No job_id or content in response: {response}")

        result = self._poll_job(job_id, timeout or self.max_poll)
        return result.get("content", "")

    # -----------------------------------------------------------------
    # AI Search — web search with content extraction
    # -----------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )
    def ai_search(
        self,
        query: str,
        limit: int = 5,
        return_content: bool = True,
        geo: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, str]]:
        """Search the web using Oxylabs AI Studio.

        Args:
            query: Search query.
            limit: Max results to return.
            return_content: If True, fetch full page content for each result.
            geo: ISO country code for geo-located search.
            timeout: Per-request timeout override (seconds).

        Returns:
            List of dicts with 'url', 'title', 'content' keys.
        """
        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "return_content": return_content,
        }
        if geo:
            payload["geo_location"] = geo

        logger.info("Oxylabs.ai_search: '%s' (limit=%d, content=%s)", query[:60], limit, return_content)

        response = self._submit_job("/ai-search", payload)
        job_id: str | None = response.get("job_id") or response.get("id")
        if not job_id:
            # Sync response
            results = response.get("results", [])
            if results:
                return self._parse_search_results(results)

        result = self._poll_job(job_id, timeout or self.max_poll)
        results = result.get("results", [])
        return self._parse_search_results(results)

    def _parse_search_results(self, raw: list[dict] | dict) -> list[dict[str, str]]:
        """Normalize search results to a consistent format."""
        if isinstance(raw, dict):
            raw = [raw]

        parsed: list[dict[str, str]] = []
        for r in raw:
            parsed.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "") or r.get("snippet", ""),
                "snippet": r.get("snippet", "")[:500],
            })
        return parsed

    # -----------------------------------------------------------------
    # AI Browser Agent — browser automation with natural language
    # -----------------------------------------------------------------

    def ai_browser_agent(
        self,
        url: str,
        task_prompt: str,
        output_format: str = "markdown",
        geo: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a browser automation task on a URL.

        Args:
            url: Starting URL.
            task_prompt: Natural-language task (e.g., "Find the pricing table
                and extract all plan names and prices").
            output_format: "markdown", "json", or "html".
            geo: ISO country code for geo-located browsing.
            timeout: Per-request timeout override (seconds).

        Returns:
            Extracted content as a string.
        """
        payload: dict[str, Any] = {
            "url": url,
            "task_prompt": task_prompt,
            "output_format": output_format,
        }
        if geo:
            payload["geo_location"] = geo

        logger.info("Oxylabs.ai_browser_agent: %s — %s", url[:60], task_prompt[:60])

        response = self._submit_job("/ai-browser-agent", payload)
        job_id = response.get("job_id") or response.get("id")
        if not job_id:
            content = response.get("content") or response.get("result") or ""
            if content:
                return content
            raise OxylabsError(f"No job_id or content in response: {response}")

        result = self._poll_job(job_id, timeout or self.max_poll)
        return result.get("content", "")

    # -----------------------------------------------------------------
    # Job submission and polling (AI Studio uses async jobs)
    # -----------------------------------------------------------------

    def _submit_job(self, endpoint: str, payload: dict) -> dict:
        """Submit a job to the AI Studio API. Returns the response JSON."""
        try:
            resp = self.client.post(endpoint, json=payload)
        except httpx.NetworkError as e:
            raise OxylabsError(f"Network error submitting job to {endpoint}: {e}")
        except httpx.TimeoutException as e:
            raise OxylabsError(f"Timeout submitting job to {endpoint}: {e}")

        if resp.status_code == 401:
            raise OxylabsError("Authentication failed. Check your OXYLABS_AI_STUDIO_API_KEY.")
        if resp.status_code == 429:
            raise OxylabsError("Rate limit exceeded. Try again later.")
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("message", resp.text)
            except Exception:
                msg = resp.text
            raise OxylabsError(f"API error {resp.status_code}: {msg}")

        try:
            return resp.json()
        except Exception as e:
            raise OxylabsError(f"Failed to parse response as JSON: {e}")

    def _poll_job(self, job_id: str, timeout: float) -> dict:
        """Poll for job completion. Returns the result dict."""
        elapsed = 0.0
        while elapsed < timeout:
            try:
                resp = self.client.get(f"/jobs/{job_id}")
            except httpx.NetworkError as e:
                logger.warning("Oxylabs poll error: %s", e)
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                continue

            if resp.status_code == 404:
                raise OxylabsError(f"Job {job_id} not found")

            if resp.status_code >= 400:
                raise OxylabsError(f"Poll error {resp.status_code}: {resp.text}")

            data = resp.json()
            status = data.get("status", "").lower()

            if status in ("done", "completed", "finished"):
                return data.get("result", data)
            elif status in ("failed", "error"):
                error = data.get("error", "Unknown error")
                raise OxylabsError(f"Job {job_id} failed: {error}")
            elif status in ("pending", "running", "in_progress", "queued"):
                logger.debug("Oxylabs job %s: %s (%.0fs)", job_id[:8], status, elapsed)
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                continue
            else:
                # Unknown status — check if there's content anyway
                if data.get("content") or data.get("result"):
                    return data
                logger.debug("Oxylabs job %s: unknown status '%s' (%.0fs)", job_id[:8], status, elapsed)
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval

        raise OxylabsError(f"Job {job_id} timed out after {timeout:.0f}s")

    # -----------------------------------------------------------------
    # High-level research method (drop-in replacement for WebFetcher.research)
    # -----------------------------------------------------------------

    def research(
        self,
        query: str,
        max_results: int = 3,
        max_chars_per_page: int = 3000,
        geo: str | None = None,
    ) -> str:
        """Search the web and return combined content from top results.

        This is the primary method for the auto-build pipeline. It searches,
        fetches content for each result, and returns a combined text block
        suitable for LLM context.

        Args:
            query: Search query.
            max_results: Number of results to fetch content for.
            max_chars_per_page: Max chars per page.
            geo: ISO country code for geo-located search.

        Returns:
            Combined research text with source URLs.
        """
        try:
            results = self.ai_search(query, limit=max_results, return_content=True, geo=geo)
        except OxylabsError as e:
            logger.warning("Oxylabs research: search failed: %s", e)
            return ""

        if not results:
            return ""

        parts: list[str] = [f"Web research for: '{query}' (via Oxylabs AI Studio)\n"]

        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            if not content and result.get("url"):
                # Fall back to scraping the URL directly
                try:
                    content = self.ai_scraper(
                        result["url"],
                        output_format="markdown",
                        geo=geo,
                    )
                except OxylabsError:
                    content = result.get("snippet", "")

            if content:
                parts.append(f"---\nSource {i}: {result.get('title', 'Untitled')}\nURL: {result.get('url', '')}\n")
                parts.append(content[:max_chars_per_page])
                parts.append("")
            elif result.get("snippet"):
                parts.append(f"---\nSource {i}: {result.get('title', 'Untitled')}\nURL: {result.get('url', '')}")
                parts.append(f"(snippet only) {result['snippet']}")
                parts.append("")

        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Check if the Oxylabs API is reachable and the key is valid.

        Returns:
            Dict with 'ok' (bool), 'credits_remaining' (int, if available),
            and 'error' (str, if not ok).
        """
        try:
            resp = self.client.get("/account/info")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "ok": True,
                    "credits_remaining": data.get("credits_remaining"),
                    "plan": data.get("plan", ""),
                }
            elif resp.status_code == 401:
                return {"ok": False, "error": "Invalid API key"}
            else:
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
