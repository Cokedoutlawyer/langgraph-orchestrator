"""Run node — executes the selected tool via the routing layer.

Handles the 3-tier failover: BrightData -> Oxylabs -> Venice.
Records the tool call in state.tool_calls with provider logging.
"""
from __future__ import annotations

import structlog
from ..state import OrchestratorState, ToolCall
from ..router import scrape, search, render

log = structlog.get_logger("run")


def run_node(state: OrchestratorState) -> dict:
    """Execute the tool call. Never raises — records errors for the summary node."""
    tool = state.get("selected_tool", "search_engine")
    args = state.get("tool_args", {})
    geo = state.get("geo") if state.get("needs_geo") else None
    query = state.get("query", "")

    log.info("run.start", tool=tool, args=args, geo=geo)

    tool_call = ToolCall(
        provider="", tool=tool, args=args, result_preview="", chars=0,
        status="success", timestamp=_now(),
    )

    try:
        if tool in ("search_engine", "ai_search"):
            q = args.get("query", query)
            content, provider = search(q, geo=geo)
        elif tool in ("scrape_as_markdown", "ai_scraper"):
            url = args.get("url", "")
            render_flag = state.get("needs_rendering", False)
            content, provider = scrape(url, render=render_flag, geo=geo)
        elif tool == "render":
            url = args.get("url", "")
            content, provider = render(url, session_id=state.get("thread_id"))
        elif tool in ("chat_with_scraping", "venice"):
            from ..tools.venice import chat_with_scraping
            urls = args.get("urls", [])
            content = chat_with_scraping(args.get("prompt", query), urls=urls)
            provider = "venice"
        else:
            # Default: treat as search
            content, provider = search(query, geo=geo)

        tool_call["provider"] = provider
        tool_call["chars"] = len(content)
        tool_call["result_preview"] = content[:300]
        tool_call["status"] = "success"

        log.info("run.success", provider=provider, chars=len(content))
        return {
            "raw_result": content,
            "result_chars": len(content),
            "tool_calls": [tool_call],
            "logs": [{"level": "info", "message": f"run: {provider} returned {len(content)} chars",
                      "timestamp": _now(), "node": "run"}],
        }

    except Exception as e:
        tool_call["status"] = "error"
        tool_call["result_preview"] = str(e)[:300]
        log.error("run.failed", error=str(e)[:200])
        return {
            "raw_result": "",
            "result_chars": 0,
            "tool_calls": [tool_call],
            "errors": [str(e)[:300]],
            "logs": [{"level": "error", "message": f"run failed: {str(e)[:120]}",
                      "timestamp": _now(), "node": "run"}],
        }


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
