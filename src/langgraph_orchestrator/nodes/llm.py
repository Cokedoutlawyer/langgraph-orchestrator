"""LLM client — uses Blackbox LiteLLM proxy (OpenAI-compatible) for all graph nodes."""
from __future__ import annotations

import httpx
import json
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
