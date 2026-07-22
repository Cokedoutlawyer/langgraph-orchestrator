"""Graph state schema + reducers."""
from __future__ import annotations

from typing import TypedDict, Annotated, Literal
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Reducer: append-only list for logs and messages
def append_list(left: list, right: list | None) -> list:
    if right is None:
        return left
    return (left or []) + (right if isinstance(right, list) else [right])


# Reducer: overwrite (last-write-wins) but keep history
def overwrite(left, right):
    return right if right is not None else left


class ToolCall(TypedDict):
    provider: str        # brightdata | oxylabs | venice | camofox
    tool: str
    args: dict
    result_preview: str  # first N chars
    chars: int
    status: str          # success | blocked | error
    timestamp: str


class LogEntry(TypedDict):
    level: str           # info | warn | error
    message: str
    timestamp: str
    node: str


class OrchestratorState(TypedDict, total=False):
    """Shared state across Plan -> Get -> Run -> Summary nodes."""

    # Input plane
    source: str                # cli | web | telegram | cron
    query: str                 # user request
    thread_id: str             # checkpoint thread

    # Plan node
    plan: str                  # LLM-produced plan
    needs_rendering: bool      # True -> route to browser (camofox)
    needs_geo: bool            # True -> pass geo_location to scrapers
    geo: str | None            # ISO country code
    steps: list[str]           # ordered sub-steps
    retries: int               # per-step retry counter
    current_step_idx: int

    # Get node (tool selection)
    selected_provider: str     # brightdata | oxylabs | venice
    selected_tool: str
    tool_args: dict

    # Run node (execution)
    raw_result: str            # scraped/extracted content
    result_chars: int
    tool_calls: Annotated[list[ToolCall], append_list]

    # Routing / failover
    failover: bool             # True if primary failed, move to next tier
    failover_reason: str

    # Summary node
    summary: str               # final LLM answer
    summary_model: str

    # RAG (Engram)
    rag_context: str           # retrieved memory context string
    rag_memories: list         # raw memory dicts

    # Layer 4.5 — Vectorize (chunk + embed + store)
    vectorized_chunks: int     # number of chunks stored this run
    vectorize_backend: str     # which backend was used

    # Logging
    logs: Annotated[list[LogEntry], append_list]
    errors: Annotated[list[str], append_list]
    started_at: str
    finished_at: str
