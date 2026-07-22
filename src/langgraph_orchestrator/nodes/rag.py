"""RAG node (condensed) — Layer 4.5 + memory retrieval in one node.

Merges the old vectorize_node + rag_node into a single pass:
  1. Chunk + embed + store the scraped content (VectorStore backend).
  2. Retrieve user-scoped long-term memory (Engram) for personalization.
Both are non-blocking (try/except) — RAG augments, never breaks the pipeline.

Graph: run → rag → summary (was run → vectorize → rag → summary).
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone

from ..state import OrchestratorState
from ..tools.vectorize import get_store, DEFAULT_BACKEND
from ..tools.engram import search_memories
from ..config import settings

log = structlog.get_logger("rag")


def rag_node(state: OrchestratorState) -> dict:
    """Store scraped content as vectors + retrieve memory. One pass, never raises."""
    raw = state.get("raw_result", "")
    query = state.get("query", "")
    thread_id = state.get("thread_id", "")
    user_id = settings.engram_default_user
    logs: list[dict] = []
    vectorized_chunks = 0
    vectorize_backend = DEFAULT_BACKEND

    # --- 1. Vectorize: chunk + embed + store the scraped content ---
    if raw:
        tool_calls = state.get("tool_calls", [])
        url = ""
        if tool_calls:
            last = tool_calls[-1]
            url = (last.get("args") or {}).get("url", "") if isinstance(last.get("args"), dict) else ""
        provider = tool_calls[-1]["provider"] if tool_calls else "unknown"
        try:
            store = get_store()
            vectorized_chunks = store.ingest(
                query=query, url=url or f"thread:{thread_id}",
                content=raw, source=provider,
            )
            vectorize_backend = store.name
            logs.append(_log(f"vectorize: stored {vectorized_chunks} chunks via {store.name}", "info"))
            log.info("rag.vectorized", chunks=vectorized_chunks, backend=store.name)
        except Exception as e:
            logs.append(_log(f"vectorize failed (non-fatal): {str(e)[:100]}", "error"))
            log.error("rag.vectorize_failed", error=str(e)[:200])

    # --- 2. Retrieve: pull user-scoped memory for personalization ---
    rag_context = ""
    rag_memories: list = []
    try:
        memories = search_memories(query, user_id=user_id, retrieval="hybrid", auto_limit=5)
        if memories:
            lines = [f"[{i+1}] (score={m['score']:.2f}) {m['content']}" for i, m in enumerate(memories)]
            rag_context = "\n".join(lines)
            rag_memories = memories
            logs.append(_log(f"memory: {len(memories)} retrieved (top={memories[0]['score']:.2f})", "info"))
            log.info("rag.retrieved", count=len(memories), top_score=memories[0]["score"])
        else:
            logs.append(_log("memory: 0 retrieved", "info"))
    except Exception as e:
        logs.append(_log(f"memory failed (non-fatal): {str(e)[:100]}", "error"))
        log.warn("rag.memory_failed", error=str(e)[:200])

    return {
        "rag_context": rag_context,
        "rag_memories": rag_memories,
        "vectorized_chunks": vectorized_chunks,
        "vectorize_backend": vectorize_backend,
        "logs": logs,
    }


def _log(message: str, level: str) -> dict:
    return {
        "level": level,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "rag",
    }
