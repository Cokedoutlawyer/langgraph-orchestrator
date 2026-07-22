"""Routing layer — BrightData → Oxylabs → Venice failover with provider logging.

Priority order (per user spec):
1. BrightData via MCP scrape_as_markdown or browser_navigate (rendering/geo)
2. Oxylabs via ai_scraper or ai_search on BrightData block/CAPTCHA/5xx
3. Venice enable_web_scraping:true for <3 simple pages needing LLM answer

Always logs which provider served which request.
"""
from __future__ import annotations

import structlog
from typing import Literal

from .tools.brightdata import (
    BrightDataBlocked, scrape_as_markdown as bd_scrape, search_engine as bd_search,
    scraping_browser_navigate as bd_navigate, discover as bd_discover,
)
from .tools.oxylabs import (
    OxylabsError, ai_scraper as ox_scrape, ai_search as ox_search, ai_browser_agent as ox_browser,
)
from .tools.venice import VeniceError, chat_with_scraping as venice_chat
from .tools.camofox import CamofoxError, navigate as camo_navigate, snapshot as camo_snapshot, health as camo_health

log = structlog.get_logger("router")

Provider = Literal["brightdata", "oxylabs", "venice", "camofox"]


def _log(url: str, provider: Provider, tool: str, status: str, chars: int, extra: str = "") -> None:
    log.info("scrape-served", url=url, provider=provider, tool=tool, status=status, chars=chars, extra=extra)


def scrape(url: str, *, render: bool = False, geo: str | None = None) -> tuple[str, Provider]:
    """Scrape a URL with 3-tier failover. Returns (content, provider_used)."""
    # Tier 1: BrightData
    try:
        if render:
            content = bd_scrape(url, geo=geo)
            # bd_scrape uses scrape_as_markdown which handles rendering via the MCP
        else:
            content = bd_scrape(url, geo=geo)
        _log(url, "brightdata", "scrape_as_markdown", "success", len(content))
        return content, "brightdata"
    except BrightDataBlocked as e:
        _log(url, "brightdata", "scrape_as_markdown", "blocked", 0, str(e)[:120])
        # Tier 2: Oxylabs
        try:
            content = ox_scrape(url, geo=geo, render_javascript=render)
            _log(url, "brightdata", "scrape_as_markdown", "blocked→failover", 0, f"→ oxylabs ({e})")
            _log(url, "oxylabs", "ai_scraper", "success", len(content))
            return content, "oxylabs"
        except OxylabsError as e2:
            _log(url, "oxylabs", "ai_scraper", "error", 0, str(e2)[:120])
            # Tier 3: Venice
            content = venice_chat(f"Extract the main content from this page: {url}", urls=[url])
            _log(url, "venice", "chat_with_scraping", "success", len(content))
            return content, "venice"

    # Any other exception from BrightData -> Oxylabs -> Venice
    except Exception as e:
        _log(url, "brightdata", "scrape_as_markdown", "error", 0, str(e)[:120])
        try:
            content = ox_scrape(url, geo=geo, render_javascript=render)
            _log(url, "oxylabs", "ai_scraper", "success", len(content))
            return content, "oxylabs"
        except Exception as e2:
            _log(url, "oxylabs", "ai_scraper", "error", 0, str(e2)[:120])
            content = venice_chat(f"Extract the main content from this page: {url}", urls=[url])
            _log(url, "venice", "chat_with_scraping", "success", len(content))
            return content, "venice"


def search(query: str, *, geo: str | None = None) -> tuple[str, Provider]:
    """Web search with failover."""
    try:
        content = bd_search(query, geo=geo)
        _log(query, "brightdata", "search_engine", "success", len(content))
        return content, "brightdata"
    except Exception as e:
        _log(query, "brightdata", "search_engine", "error", 0, str(e)[:120])
        try:
            content = ox_search(query, geo=geo)
            _log(query, "oxylabs", "ai_search", "success", len(content))
            return content, "oxylabs"
        except Exception as e2:
            _log(query, "oxylabs", "ai_search", "error", 0, str(e2)[:120])
            content = venice_chat(query)
            _log(query, "venice", "chat_with_scraping", "success", len(content))
            return content, "venice"


def render(url: str, session_id: str | None = None) -> tuple[str, Provider]:
    """Full browser rendering via Camofox (local stealth) with BrightData browser fallback."""
    # Try Camofox first if healthy (local, free, stealthy)
    if camo_health():
        try:
            info = camo_navigate(url, session_id=session_id)
            snap = info.get("snapshot", "")
            _log(url, "camofox", "navigate+snapshot", "success", len(snap))
            return snap, "camofox"
        except CamofoxError as e:
            _log(url, "camofox", "navigate", "error", 0, str(e)[:120])

    # Fallback to BrightData scraping_browser
    try:
        content = bd_navigate(url)
        _log(url, "brightdata", "scraping_browser_navigate", "success", len(content))
        return content, "brightdata"
    except Exception as e:
        _log(url, "brightdata", "scraping_browser_navigate", "error", 0, str(e)[:120])
        # Final fallback: Oxylabs browser agent
        content = ox_browser(url, task_prompt=f"Navigate to {url} and extract the main content")
        _log(url, "oxylabs", "ai_browser_agent", "success", len(content))
        return content, "oxylabs"
