"""Oxylabs AI Studio tool wrapper — tier-2 failover. Uses the stdio MCP server.

Param is output_format (NOT return_format). AI Studio calls are async jobs
(30-60s). Auth via OXYLABS_AI_STUDIO_API_KEY (1000 free credits).
"""
from __future__ import annotations

from ..mcp_client import get_oxylabs_client


class OxylabsError(Exception):
    pass


def ai_scraper(url: str, output_format: str = "markdown", render_javascript: bool = False, geo: str | None = None) -> str:
    """Scrape a URL via Oxylabs AI Studio."""
    args: dict = {
        "url": url,
        "output_format": output_format,
        "render_javascript": render_javascript,
    }
    if geo:
        args["geo_location"] = geo
    try:
        return get_oxylabs_client().call_tool("ai_scraper", args)
    except (RuntimeError, TimeoutError) as e:
        raise OxylabsError(str(e))


def ai_search(query: str, limit: int = 5, return_content: bool = False, geo: str | None = None) -> str:
    args: dict = {"query": query, "limit": limit, "return_content": return_content}
    if geo:
        args["geo_location"] = geo
    try:
        return get_oxylabs_client().call_tool("ai_search", args)
    except (RuntimeError, TimeoutError) as e:
        raise OxylabsError(str(e))


def ai_browser_agent(url: str, task_prompt: str, output_format: str = "markdown", geo: str | None = None) -> str:
    """Browser automation with a natural-language task prompt."""
    args: dict = {"url": url, "task_prompt": task_prompt, "output_format": output_format}
    if geo:
        args["geo_location"] = geo
    try:
        return get_oxylabs_client().call_tool("ai_browser_agent", args)
    except (RuntimeError, TimeoutError) as e:
        raise OxylabsError(str(e))
