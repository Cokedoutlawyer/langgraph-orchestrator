#!/usr/bin/env python3
"""CLI entry point for the LangGraph orchestrator.

Usage:
  orchestrator "search query"            # one-shot query
  orchestrator --serve                   # start FastAPI server
  orchestrator --health                  # check dependencies
  orchestrator --stream "query"          # stream events
"""
from __future__ import annotations

import sys
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="LangGraph Orchestrator")
    parser.add_argument("query", nargs="?", help="Query to run")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--health", action="store_true", help="Check dependencies")
    parser.add_argument("--stream", action="store_true", help="Stream events")
    parser.add_argument("--source", default="cli", help="Source plane (cli/web/telegram/cron)")
    parser.add_argument("--thread-id", default=None, help="Checkpoint thread ID")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    if args.health:
        from langgraph_orchestrator.tools.camofox import health as camofox_health
        from langgraph_orchestrator.config import settings
        deps = {
            "camofox": camofox_health(),
            "brightdata_token": bool(settings.brightdata_api_token),
            "oxylabs_key": bool(settings.oxylabs_ai_studio_api_key),
            "venice_key": bool(settings.venice_api_key),
            "blackbox_key": bool(settings.blackbox_api_key),
        }
        print(json.dumps({"ok": all(deps.values()), "dependencies": deps}, indent=2))
        return

    if args.serve:
        from langgraph_orchestrator.server import serve
        serve(host=args.host, port=args.port)
        return

    if not args.query:
        parser.print_help()
        return

    from langgraph_orchestrator.graph import run_query

    if args.stream:
        # Streaming via the graph directly
        from langgraph_orchestrator.graph import get_graph
        import uuid
        graph = get_graph()
        tid = args.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        from datetime import datetime, timezone
        initial = {
            "query": args.query, "source": args.source, "thread_id": tid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tool_calls": [], "logs": [], "errors": [], "retries": 0, "current_step_idx": 0,
        }
        for event in graph.stream(initial, config=config, stream_mode="updates"):
            for node, state in event.items():
                print(f"[{node}] → {list(state.keys())}", file=sys.stderr)
        # Get final state
        final = graph.get_state(config)
        print(final.values.get("summary", "(no summary)"))
        return

    # One-shot
    result = run_query(args.query, source=args.source, thread_id=args.thread_id)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result.get("summary", "(no summary)"))
    print("\n" + "-" * 60)
    providers = [c.get("provider") for c in result.get("tool_calls", [])]
    print(f"Providers: {' → '.join(providers) if providers else 'none'}")
    print(f"Chars: {result.get('result_chars', 0)}")
    print(f"Thread: {result.get('thread_id')}")
    if result.get("errors"):
        print(f"Errors: {result['errors'][:3]}")


if __name__ == "__main__":
    main()
