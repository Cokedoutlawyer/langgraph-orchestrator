"""Engram RAG tool wrapper — Weaviate Engram memory service.

User-scoped long-term memory with async ingestion + hybrid/vector/BM25 retrieval.
Ingest scraped content into Engram; retrieve relevant memories to augment answers.

API (verified against weaviate-engram 1.0.0):
  client.memories.add(messages, user_id=) -> Run (async, returns run_id)
  client.memories.search(query=, user_id=, retrieval_config=) -> SearchResults
  client.memories.get(memory_id, user_id=) -> Memory
  client.memories.delete(memory_id, user_id=) -> None
  client.runs.wait(run_id, timeout=) -> RunStatus  # block until async ingestion done
"""
from __future__ import annotations

import os
from typing import Any

from engram import EngramClient, HybridRetrieval, VectorRetrieval, BM25Retrieval
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings


class EngramError(Exception):
    pass


def _client() -> EngramClient:
    key = settings.engram_api_key
    if not key:
        raise EngramError("ENGRAM_API_KEY not set")
    return EngramClient(api_key=key)


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def add_memories(
    messages: list[dict[str, str]],
    user_id: str = "matts@hermes",
    group: str | None = None,
    wait: bool = True,
) -> str:
    """Ingest a conversation into Engram memory. Returns run_id.

    add() returns immediately — memory processing happens async. If wait=True
    (default), blocks via runs.wait() until ingestion completes so the memory
    is immediately searchable.
    """
    client = _client()
    try:
        run = client.memories.add(messages, user_id=user_id, group=group)
        if wait:
            try:
                client.runs.wait(run.run_id, timeout=60.0)
            except Exception:
                pass  # non-fatal: memory will be searchable shortly
        return run.run_id
    except Exception as e:
        raise EngramError(f"add_memories failed: {e}")


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def search_memories(
    query: str,
    user_id: str = "matts@hermes",
    retrieval: str = "hybrid",
    auto_limit: int = 5,
    group: str | None = None,
) -> list[dict]:
    """Search Engram memory. Returns list of {content, score, memory_id} dicts.

    retrieval: "hybrid" (default, vector+BM25), "vector" (semantic), "bm25" (keyword).
    """
    client = _client()
    # Build retrieval config
    if retrieval == "vector":
        rc: Any = VectorRetrieval()
    elif retrieval == "bm25":
        rc = BM25Retrieval()
    else:
        rc = HybridRetrieval()
    try:
        results = client.memories.search(
            query=query, user_id=user_id, retrieval_config=rc, group=group,
        )
        # SearchResults is a Sequence of Memory objects (content, id, score, topic, tags)
        out = []
        for m in list(results)[:auto_limit]:
            out.append({
                "content": getattr(m, "content", "") or "",
                "score": float(getattr(m, "score", 0.0) or 0.0),
                "memory_id": str(getattr(m, "id", "") or ""),
                "topic": getattr(m, "topic", None),
                "tags": getattr(m, "tags", None),
            })
        return out
    except Exception as e:
        raise EngramError(f"search_memories failed: {e}")


def get_memory(memory_id: str, user_id: str = "matts@hermes") -> dict:
    """Fetch a single memory by ID."""
    client = _client()
    try:
        m = client.memories.get(memory_id, user_id=user_id)
        return {"content": getattr(m, "content", str(m))}
    except Exception as e:
        raise EngramError(f"get_memory failed: {e}")


def delete_memory(memory_id: str, user_id: str = "matts@hermes") -> None:
    """Delete a memory by ID."""
    client = _client()
    try:
        client.memories.delete(memory_id, user_id=user_id)
    except Exception as e:
        raise EngramError(f"delete_memory failed: {e}")
