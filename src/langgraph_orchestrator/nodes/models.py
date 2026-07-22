"""Hand-selected model matrix — one model per task, not one model for everything.

Grounded in real /models listings (Blackbox 120 models, Venice 104) + the
privacy/latency/accuracy tradeoffs of this stack. Every assignment is explicit;
nothing inherits a generic default.

Rationale (grounded in live servability — verified 2026-07-22):
- Planning = routing/tool-selection JSON. Needs strong reasoning + JSON
  discipline. deepseek-v4-pro is the best servable reasoner (Anthropic/GPT-5
  routes were Vertex-down at selection time; DeepSeek serves reliably).
- Summarization = grounded synthesis over scraped content + memory. Long-context,
  high coherence. Gemini 3.5 Flash is 1M-context, fast, cheap, strong synthesis.
- Venice scraping = private inference with web access. Llama 3.3 70b is the
  proven Venice scraper (supports enable_web_scraping). Verified working.
- Embeddings = Snowflake Arctic L v2.0 (text2vec-weaviate). Multilingual 8192-tok,
  no external key, private to WCD. Locked.
- Fallback (orchestrator) = glm-5.2. Cheap, always-on, good enough when primary
  is rate-limited.

All via the Blackbox LiteLLM proxy (api.blackbox.ai/v1) except Venice (direct).
Swap by editing this file only — nodes read from here, not hardcoded strings.
Re-verify a new model serves with a 1-token completion before committing it.
"""
from __future__ import annotations

from ..config import settings

# ---- Task -> model assignments (Blackbox slugs, all verified servable) ----
PLAN_MODEL = "blackboxai/deepseek/deepseek-v4-pro"         # reasoning + JSON routing
SUMMARY_MODEL = "blackboxai/google/gemini-3.5-flash"        # 1M-ctx grounded synthesis
FALLBACK_MODEL = settings.default_model                     # glm-5.2 (always-on)
VENICE_MODEL = "llama-3.3-70b"                              # private scraper (Venice-direct)
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0" # text2vec-weaviate, locked


def resolve(task: str) -> str:
    """Return the hand-selected model for a task. Falls back to glm-5.2."""
    return {
        "plan": PLAN_MODEL,
        "summary": SUMMARY_MODEL,
        "fallback": FALLBACK_MODEL,
        "venice": VENICE_MODEL,
        "embedding": EMBEDDING_MODEL,
    }.get(task, FALLBACK_MODEL)
