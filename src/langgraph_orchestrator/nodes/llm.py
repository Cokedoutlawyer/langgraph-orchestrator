"""LLM client — multi-path: Blackbox LiteLLM proxy (default) + direct Google Gemini.

Routing rule (no-failures optimization):
- plan  -> Blackbox -> DeepSeek v4-pro   (Blackbox chain healthy, reasoning model)
- summary -> DIRECT Google Gemini         (cuts Blackbox + Vercel hops from critical path;
                                           1 hop vs 3; GEMINI_API_KEY is billing-verified)
- fallback -> Blackbox glm-5.2

Use llm_chat() for Blackbox-routed calls, llm_chat_gemini_direct() for the direct
Google path. Nodes pick the function via resolve(); summary calls the direct one.
"""
from __future__ import annotations

import httpx
import json
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_chat(system: str, user: str, model: str | None = None, temperature: float = 0.3) -> str:
    """Call the Blackbox LiteLLM proxy (api.blackbox.ai/v1) — OpenAI-compatible."""
    model = model or settings.default_model
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.blackbox_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(
            f"{settings.blackbox_base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_chat_gemini_direct(system: str, user: str, model: str, temperature: float = 0.3) -> str:
    """Call Google Gemini DIRECTLY — bypasses Blackbox + Vercel hops.

    One hop: orchestrator -> generativelanguage.googleapis.com -> Gemini.
    Uses GEMINI_API_KEY (billing-verified). Cuts the 3-hop chain to 1.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set for direct Gemini path")
    # Gemini generateContent API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature},
    }
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini direct HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def llm_json(system: str, user: str, model: str | None = None) -> dict:
    """LLM call that expects JSON output. Strips markdown fences defensively."""
    raw = llm_chat(system, user, model=model, temperature=0.1)
    # Defensive fence stripping (Gemini-style; Blackbox usually clean)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    # Find first { ... last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1:
        text = text[first : last + 1]
    return json.loads(text)
