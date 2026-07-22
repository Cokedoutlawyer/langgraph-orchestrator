"""Summary node — synthesizes the final answer from scraped content + query.

Uses the LLM to produce a grounded summary. Includes provider attribution
and graceful degradation on partial results.
"""
from __future__ import annotations

import structlog
from .llm import llm_chat
from .models import resolve
from ..state import OrchestratorState
from ..config import settings

log = structlog.get_logger("summary")

SUMMARY_SYSTEM = """You are a research summarization agent. Given a user query and scraped web content, produce a clear, accurate, grounded answer.

Rules:
- Base your answer ONLY on the provided content. If the content is empty or insufficient, say so.
- Cite which provider served the data if available.
- Be concise but complete. Use bullet points for lists.
- If the content is raw HTML or JSON, extract the relevant facts.
- Match the user's tone (direct, no fluff)."""


def summary_node(state: OrchestratorState) -> dict:
    """Synthesize final answer from raw_result + query."""
    query = state.get("query", "")
    raw = state.get("raw_result", "")
    calls = state.get("tool_calls", [])
    errors = state.get("errors", [])
    model = resolve("summary")

    # Provider attribution
    providers = [c.get("provider", "?") for c in calls if c.get("provider")]
    provider_str = " → ".join(dict.fromkeys(providers)) if providers else "none"

    log.info("summary.start", query=query, chars=len(raw), providers=provider_str, errors=len(errors))

    if not raw and errors:
        # Graceful degradation: all tools failed
        summary = f"Unable to complete the request. All scraping providers failed:\n" + "\n".join(f"- {e}" for e in errors[:5])
        log.warn("summary.degraded", errors=errors[:3])
    elif not raw:
        summary = "No content was retrieved for this query."
    else:
        # Truncate very large content to stay in context
        content = raw[:20000] if len(raw) > 20000 else raw
        # RAG augmentation: inject retrieved long-term memory context if present
        rag_ctx = state.get("rag_context", "")
        rag_block = ""
        if rag_ctx:
            rag_block = f"""
Relevant long-term memory (user context from prior sessions):
{rag_ctx}
---
Use retrieved memory to personalize the answer: explicitly anchor recommendations to stated user preferences using phrases like "since you...", and treat any disliked or irrelevant items as hard exclusions—never mention them even if they appear in scraped content. Let positive preferences guide emphasis and negative preferences govern exclusion. Ground factual claims in scraped content for currency, but always filter scraped options through the user's preferences before including them.
"""
        user_msg = f"""User query: {query}

Data providers used: {provider_str}
{rag_block}
Scraped content ({len(raw)} chars):
---
{content}
---

Produce a grounded answer to the user query based on this content."""
        try:
            summary = llm_chat(SUMMARY_SYSTEM, user_msg, model=model)
        except Exception as e:
            log.error("summary.llm_failed", error=str(e)[:120])
            # Graceful: return raw content if LLM fails
            summary = f"(LLM summary failed: {str(e)[:80]})\n\nRaw content ({len(raw)} chars):\n{content[:3000]}"

    log.info("summary.done", summary_chars=len(summary))
    from datetime import datetime, timezone
    return {
        "summary": summary,
        "summary_model": model,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "logs": [{"level": "info", "message": f"summary: {len(summary)} chars via {model}",
                  "timestamp": datetime.now(timezone.utc).isoformat(), "node": "summary"}],
    }
