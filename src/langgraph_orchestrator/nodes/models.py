"""Hand-selected model matrix — one model per task, not one model for everything.

Grounded in real /models listings (Blackbox 120 models, Venice 104) + the
privacy/latency/accuracy tradeoffs of this stack. Every assignment is explicit;
nothing inherits a generic default.

Rationale (grounded in live servability — verified 2026-07-22):
- Planning = routing/tool-selection JSON. Needs strong reasoning + JSON
  discipline. deepseek-v4-pro is the best servable reasoner (Anthropic/GPT-5
  routes were Vertex-down at selection time; DeepSeek serves reliably).
  Routed via Blackbox (api.blackbox.ai) — that chain is healthy.
- Summarization = grounded synthesis over scraped content + memory. Long-context,
  high coherence. Routed DIRECT to Google Gemini (gemini-3-flash-preview, 1M-ctx)
  via generativelanguage.googleapis.com — cuts the Blackbox + Vercel AI Gateway
  hops from the critical path (was 3 hops: Blackbox→Vercel→Gemini; now 1 hop:
  orchestrator→Google). GEMINI_API_KEY is billing-verified. Falls back to
  Blackbox glm-5.2 if the direct path errors.
- Venice scraping = private inference with web access. Llama 3.3 70b is the
  proven Venice scraper (supports enable_web_scraping). Verified working.
- Embeddings = Snowflake Arctic L v2.0 (text2vec-weaviate). Multilingual 8192-tok,
  no external key, private to WCD. Locked.
- Fallback (orchestrator) = glm-5.2 via Blackbox. Cheap, always-on.

Two LLM transport paths (see nodes/llm.py):
- llm_chat()              -> Blackbox LiteLLM proxy (api.blackbox.ai/v1)
- llm_chat_gemini_direct() -> Google direct (generativelanguage.googleapis.com)
Summary uses the direct path; plan/fallback use Blackbox. Swap by editing this
file only — nodes read from here, not hardcoded strings.
Re-verify a new model serves with a 1-token completion before committing it.
"""
from __future__ import annotations

from ..config import settings

# ---- Task -> model assignments (all verified servable on their transport) ----
# plan/fallback: Blackbox slugs (api.blackbox.ai). summary: bare Google slug.
PLAN_MODEL = "blackboxai/deepseek/deepseek-v4-pro"         # reasoning + JSON routing (Blackbox)
SUMMARY_MODEL = "gemini-3-flash-preview"                    # 1M-ctx synthesis (DIRECT Google, 1 hop)
FALLBACK_MODEL = settings.default_model                     # glm-5.2 (always-on, Blackbox)
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
