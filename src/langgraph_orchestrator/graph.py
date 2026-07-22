"""Compiled LangGraph StateGraph — Plan -> Get -> Run -> Summary.

Features:
- StateGraph with TypedDict state + append reducers
- SqliteSaver checkpointer (persistence, resumes, time travel)
- Branching edge: Run -> Summary on success, Run -> retry path on failure (up to 2 retries)
- Error boundaries: every node degrades gracefully (no exceptions propagate)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import OrchestratorState
from .nodes import plan_node, get_node, run_node, rag_node, summary_node


def _ensure_checkpointer() -> SqliteSaver:
    """Set up SQLite checkpointer. Creates DB dir if needed."""
    from .config import settings
    uri = settings.checkpointer_uri
    # sqlite:///~/.hermes/cache/orchestrator-checkpoints.db
    if uri.startswith("sqlite:///"):
        db_path = uri.replace("sqlite:///", "", 1)
        if db_path.startswith("~"):
            db_path = os.path.expanduser(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
    else:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteSaver(conn)


def _route_after_run(state: OrchestratorState) -> str:
    """Branching edge: after Run, go to RAG on success or retry on failure."""
    errors = state.get("errors", [])
    raw = state.get("raw_result", "")

    # Success path -> rag (condensed: vectorize + memory retrieval in one node)
    if raw and not errors:
        return "rag"

    # Retry path (up to 2 retries with failover to next provider)
    retries = state.get("retries", 0)
    if retries < 2 and errors:
        return "plan"  # re-plan with failover

    # Exhausted retries -> summary with degraded info
    return "summary"


def _retry_bump(state: OrchestratorState) -> dict:
    """On retry, bump retry counter and clear errors so the next plan run is fresh."""
    return {"retries": state.get("retries", 0) + 1, "errors": []}


def build_graph():
    """Build and compile the orchestrator graph with checkpointer."""
    builder = StateGraph(OrchestratorState)

    # Add nodes
    builder.add_node("plan", plan_node)
    builder.add_node("get", get_node)
    builder.add_node("run", run_node)
    builder.add_node("rag", rag_node)
    builder.add_node("summary", summary_node)
    builder.add_node("retry_bump", _retry_bump)

    # Linear edges: START -> plan -> get -> run
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "get")
    builder.add_edge("get", "run")

    # Branching edge: run -> rag (success) | run -> retry_bump (failure)
    builder.add_conditional_edges(
        "run",
        _route_after_run,
        {"rag": "rag", "retry_bump": "retry_bump"},
    )
    builder.add_edge("retry_bump", "plan")
    # Condensed RAG: vectorize + memory retrieval in one node -> summary
    builder.add_edge("rag", "summary")

    # Summary -> END
    builder.add_edge("summary", END)

    # Compile with checkpointer for persistence
    checkpointer = _ensure_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# Module-level singleton (built once, reused)
_graph: StateGraph | None = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, source: str = "cli", thread_id: str | None = None) -> dict:
    """Run a query through the full graph. Returns final state."""
    import uuid
    graph = get_graph()
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    initial_state = {
        "query": query,
        "source": source,
        "thread_id": tid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tool_calls": [],
        "logs": [],
        "errors": [],
        "retries": 0,
        "current_step_idx": 0,
    }
    final = graph.invoke(initial_state, config=config)
    return final
