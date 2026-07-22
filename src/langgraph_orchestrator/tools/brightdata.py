"""BrightData tool wrapper — tier-1. Uses the stdio MCP server (@brightdata/mcp).

The hosted HTTP MCP endpoint doesn't accept bearer tokens directly; the stdio
server we verified in Hermes is the working integration. PRO_MODE = 74 tools.
"""
from __future__ import annotations

from ..mcp_client import get_brightdata_client


class BrightDataBlocked(Exception):
    """BrightData returned a block / CAPTCHA / 5xx — trigger failover."""


def scrape_as_markdown(url: str, geo: str | None = None) -> str:
    """Scrape a single URL and return markdown."""
    args: dict = {"url": url}
    if geo:
        args["country"] = geo
    try:
        return get_brightdata_client().call_tool("scrape_as_markdown", args)
    except RuntimeError as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("403", "429", "block", "captcha", "5")):
            raise BrightDataBlocked(msg)
        raise


def search_engine(query: str, engine: str = "google", geo: str | None = None) -> str:
    args: dict = {"query": query, "engine": engine}
    if geo:
        args["geo_location"] = geo
    try:
        return get_brightdata_client().call_tool("search_engine", args)
    except RuntimeError as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("403", "429", "block", "captcha", "5")):
            raise BrightDataBlocked(msg)
        raise


def discover(query: str, num_results: int = 5, geo: str | None = None) -> str:
    args: dict = {"query": query, "num_results": num_results}
    if geo:
        args["country"] = geo
    return get_brightdata_client().call_tool("discover", args)


def web_data(target: str, url: str, **extra) -> str:
    """Structured dataset extraction, e.g. web_data_amazon_product."""
    args = {"url": url, **extra}
    return get_brightdata_client().call_tool(f"web_data_{target}", args)


def scraping_browser_navigate(url: str) -> str:
    """Full browser automation — navigate to URL (rendering, JS-heavy pages)."""
    return get_brightdata_client().call_tool("scraping_browser_navigate", {"url": url})
