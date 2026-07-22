"""Plan node — analyzes the query and produces a plan + tool selection.

Uses the LLM to decide:
- needs_rendering (JS-heavy pages -> camofox/brightdata browser)
- needs_geo + geo (geo-targeted scraping)
- ordered sub-steps
- selected provider + tool + args for the first step
"""
from __future__ import annotations

import structlog
from .llm import llm_json
from .models import resolve
from ..state import OrchestratorState

log = structlog.get_logger("plan")

PLAN_SYSTEM = """You are a web-research planning agent. Analyze the user query and produce a JSON plan.

Available tools (in priority order):
1. brightdata.scrape_as_markdown(url, geo) — single page, markdown (PRIMARY)
2. brightdata.search_engine(query, geo) — Google/Bing/Yandex SERP
3. brightdata.scraping_browser_navigate(url) — full browser rendering
4. oxylabs.ai_scraper(url, geo) — FAILOVER on brightdata block/CAPTCHA/5xx
5. oxylabs.ai_search(query, geo) — FAILOVER search
6. venice.chat_with_scraping(prompt, urls) — last resort, <3 pages, needs LLM answer
7. camofox.navigate+snapshot — local stealth browser for JS-heavy pages

Decide:
- needs_rendering: true ONLY if the target is a JS-heavy SPA (React/Vue) or requires login/interaction. Default false.
- needs_geo: true ONLY if the user EXPLICITLY names a country, region, city, or locale (e.g. "in the UK", "Japan", "en-US"). Do NOT infer geo from retailer names (Amazon, Walmart, Best Buy), brand names, currency symbols, or domain TLDs. When unsure, set false.
- geo: ISO country code (e.g. "US", "GB") if needs_geo is true, else null.
- steps: ordered list of 1-5 short sub-steps. Prefer ONE broad search when comparing across multiple sites, then scrape the discovered pages.
- For the FIRST step, pick the best provider+tool+args. Prefer brightdata. Use search_engine when no specific URL is given; use scrape_as_markdown when a URL is known.

Respond with ONLY this JSON shape:
{
  "needs_rendering": false,
  "needs_geo": false,
  "geo": null,
  "steps": ["step 1", "step 2"],
  "first_step": {
    "provider": "brightdata",
    "tool": "search_engine",
    "args": {"query": "...", "geo": null}
  }
}"""


def plan_node(state: OrchestratorState) -> dict:
    """Analyze query -> produce plan. Never raises (graceful degradation)."""
    query = state.get("query", "")
    log.info("plan.start", query=query, source=state.get("source"))

    try:
        result = llm_json(PLAN_SYSTEM, query, model=resolve("plan"))
    except Exception as e:
        log.error("plan.llm_failed", error=str(e)[:120])
        # Graceful fallback: treat as a simple search
        result = {
            "needs_rendering": False,
            "needs_geo": False,
            "geo": None,
            "steps": [f"Search for: {query}"],
            "first_step": {"provider": "brightdata", "tool": "search_engine", "args": {"query": query}},
        }

    first = result.get("first_step", {})
    # On retry (retries > 0), force failover to next provider tier
    retries = state.get("retries", 0)
    if retries > 0:
        current = first.get("provider", "brightdata")
        if current == "brightdata":
            first = {"provider": "oxylabs", "tool": "ai_search" if "search" in str(first.get("tool", "")) else "ai_scraper",
                     "args": _swap_args(first.get("args", {"query": query}))}
            log.info("plan.failover", retry=retries, new_provider="oxylabs")
        elif current == "oxylabs":
            first = {"provider": "venice", "tool": "chat_with_scraping",
                     "args": {"prompt": query, "urls": []}}
            log.info("plan.failover", retry=retries, new_provider="venice")

    log.info("plan.done", steps=result.get("steps", []),
             provider=first.get("provider"), tool=first.get("tool"),
             rendering=result.get("needs_rendering"), geo=result.get("geo"), retries=retries)

    return {
        "plan": str(result),
        "needs_rendering": result.get("needs_rendering", False),
        "needs_geo": result.get("needs_geo", False),
        "geo": result.get("geo"),
        "steps": result.get("steps", []),
        "current_step_idx": 0,
        # IMPORTANT: do NOT reset retries here — it's managed by retry_bump node
        "selected_provider": first.get("provider", "brightdata"),
        "selected_tool": first.get("tool", "search_engine"),
        "tool_args": first.get("args", {"query": query}),
        "logs": [{"level": "info", "message": f"Plan: {len(result.get('steps', []))} steps, provider={first.get('provider')}, retry={retries}",
                  "timestamp": _now(), "node": "plan"}],
    }


def _swap_args(args: dict) -> dict:
    """Convert brightdata args to oxylabs args (query stays, url stays)."""
    return args


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
