"""Get node — prepares the tool call based on plan output.

If needs_rendering, route to the render path (camofox/brightdata browser).
Otherwise, use the selected provider+tool from the plan.
"""
from __future__ import annotations

import structlog
from ..state import OrchestratorState

log = structlog.get_logger("get")


def get_node(state: OrchestratorState) -> dict:
    """Prepare tool call. May adjust provider if rendering is needed."""
    provider = state.get("selected_provider", "brightdata")
    tool = state.get("selected_tool", "search_engine")
    args = state.get("tool_args", {})
    needs_rendering = state.get("needs_rendering", False)

    # If rendering needed and not already a browser tool, switch to render path
    if needs_rendering and tool not in ("scraping_browser_navigate", "navigate", "ai_browser_agent"):
        log.info("get.reroute_render", original_tool=tool)
        return {
            "selected_tool": "render",
            "tool_args": {"url": args.get("url", "")},
            "logs": [{"level": "info", "message": f"Rerouted to render path (needs_rendering=True)",
                      "timestamp": _now(), "node": "get"}],
        }

    log.info("get.ready", provider=provider, tool=tool, args_keys=list(args.keys()))
    return {
        "logs": [{"level": "info", "message": f"Tool ready: {provider}.{tool}",
                  "timestamp": _now(), "node": "get"}],
    }


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
