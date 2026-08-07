"""WebFetcher — research fallback for the auto-build pipeline.

When the planner or coder encounters something it can't figure out from
the repo context alone (an unfamiliar API, a library pattern, a GitHub
Actions syntax question), the WebFetcher can:
1. Search the web for relevant documentation
2. Fetch and extract the content of a URL
3. Provide the result as additional context to the LLM

This makes the pipeline self-correcting: instead of guessing when it
doesn't know something, it can look it up.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class WebFetcher:
    """Web search and content extraction for the auto-build pipeline.

    Uses a pluggable search backend (default: web search via httpx) and
    extracts clean text from HTML pages. Designed to be called from the
    Planner and Coder when they need additional context.

    Usage:
        fetcher = WebFetcher()
        results = fetcher.search("FastAPI dependency injection pattern")
        content = fetcher.fetch_url(results[0]["url"])
    """

    # Search backend — can be overridden for different providers
    def __init__(
        self,
        search_api_key: str | None = None,
        search_base_url: str = "https://www.googleapis.com/customsearch/v1",
        timeout: float = 30.0,
    ):
        self.search_api_key = search_api_key
        self.search_base_url = search_base_url
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "hermes-auto-build/1.0"},
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> WebFetcher:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Web search (DuckDuckGo HTML — no API key needed)
    # -----------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web and return a list of results.

        Uses DuckDuckGo HTML endpoint (no API key required).

        Returns:
            List of dicts with 'url', 'title', and 'snippet' keys.
        """
        logger.info("WebFetcher: searching for: %s", query[:100])

        try:
            resp = self.client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "s": max_results},
            )
            resp.raise_for_status()
            return self._parse_ddg_results(resp.text, max_results)
        except Exception as e:
            logger.warning("WebFetcher: search failed: %s", e)
            return []

    def _parse_ddg_results(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML search results."""
        results: list[dict[str, str]] = []

        # Extract result links and snippets from DuckDuckGo HTML
        # Results are in <a class="result__a" href="...">title</a>
        # Snippets are in <a class="result__snippet">...</a>
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(links[:max_results]):
            # Clean up the URL (DuckDuckGo wraps URLs in a redirect)
            if "uddg=" in url:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                url = unquote(params.get("uddg", [url])[0])

            snippet = ""
            if i < len(snippets):
                # Strip HTML tags from snippet
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

            # Strip HTML tags from title
            clean_title = re.sub(r"<[^>]+>", "", title).strip()

            results.append({
                "url": url,
                "title": clean_title,
                "snippet": snippet[:500],
            })

        logger.info("WebFetcher: found %d result(s)", len(results))
        return results

    # -----------------------------------------------------------------
    # URL fetching and content extraction
    # -----------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )
    def fetch_url(self, url: str, max_chars: int = 5000) -> str:
        """Fetch a URL and extract readable text content.

        Strips HTML tags, scripts, and styles. Returns the first
        max_chars of readable text.

        Args:
            url: URL to fetch.
            max_chars: Maximum characters to return.
        Returns:
            Extracted text content, or empty string on failure.
        """
        logger.info("WebFetcher: fetching %s", url[:100])

        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type:
                text = self._extract_text_from_html(resp.text)
            elif "text/plain" in content_type or "application/json" in content_type:
                text = resp.text
            else:
                logger.debug("WebFetcher: skipping non-text content-type: %s", content_type)
                return ""

            return text[:max_chars]
        except Exception as e:
            logger.warning("WebFetcher: fetch failed for %s: %s", url[:50], e)
            return ""

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML.

        Removes scripts, styles, and HTML tags. Preserves structure
        with newlines between block elements.
        """
        # Remove script and style elements entirely
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Convert block elements to newlines
        html = re.sub(r"</?(?:p|div|h[1-6]|li|ul|ol|br|tr|td|th|table|pre|code)[^>]*>", "\n", html, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", html)

        # Decode common HTML entities
        entities = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
            "&#39;": "'", "&nbsp;": " ", "&copy;": "(c)", "&trade;": "(TM)",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Normalize whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Research — search + fetch in one call
    # -----------------------------------------------------------------

    def research(self, query: str, max_results: int = 3, max_chars_per_page: int = 3000) -> str:
        """Search the web and fetch content from top results.

        This is the primary method for the planner/coder to use when
        they need additional context. It searches, fetches the top
        results, and returns a combined text block.

        Args:
            query: Search query.
            max_results: Number of search results to fetch.
            max_chars_per_page: Max chars per fetched page.
        Returns:
            Combined research text, formatted with source URLs.
        """
        results = self.search(query, max_results=max_results)
        if not results:
            return "(no web results found)"

        parts: list[str] = [f"Web research for: '{query}'\n"]

        for i, result in enumerate(results, 1):
            content = self.fetch_url(result["url"], max_chars=max_chars_per_page)
            if content:
                parts.append(f"---\nSource {i}: {result['title']}\nURL: {result['url']}\n")
                parts.append(content[:max_chars_per_page])
                parts.append("")
            else:
                # Fall back to snippet if full page fetch fails
                if result["snippet"]:
                    parts.append(f"---\nSource {i}: {result['title']}\nURL: {result['url']}")
                    parts.append(f"(snippet only) {result['snippet']}")
                    parts.append("")

        return "\n".join(parts)

    # -----------------------------------------------------------------
    # GitHub-specific research
    # -----------------------------------------------------------------

    def fetch_github_file(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Fetch a raw file from GitHub's raw content endpoint.

        This is faster than the API for fetching file content because
        it doesn't consume API rate limits.

        Args:
            owner: Repo owner.
            repo: Repo name.
            path: File path within the repo.
            ref: Branch or commit SHA.
        Returns:
            Raw file content, or empty string on failure.
        """
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        return self.fetch_url(url, max_chars=10000)

    def fetch_github_readme(self, owner: str, repo: str) -> str:
        """Fetch a repo's README from raw.githubusercontent.com."""
        for readme_name in ("README.md", "readme.md", "README.rst", "README.txt"):
            content = self.fetch_github_file(owner, repo, readme_name)
            if content:
                return content
        return ""
