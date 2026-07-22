"""Venice tool wrapper — tier-3 last resort.

For simple public pages (<3 URLs) where the agent needs BOTH page content AND an LLM answer.
Venice handles the fetch internally via enable_web_scraping: true — no proxy needed.
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings


TIMEOUT = 60.0
VENICE_MODEL = "llama-3.3-70b"  # Venice-hosted, supports web scraping


class VeniceError(Exception):
    pass


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.venice_api_key}",
        "Content-Type": "application/json",
    }


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def chat_with_scraping(prompt: str, urls: list[str] | None = None) -> str:
    """Send a chat completion with enable_web_scraping=true.

    Venice fetches the URL(s) internally and includes the content in the LLM's
    context, then returns an answer grounded in that content.
    """
    if urls:
        url_text = "\n".join(f"- {u}" for u in urls[:3])
        full_prompt = f"Reference these pages (fetched via web scraping):\n{url_text}\n\nTask: {prompt}"
    else:
        full_prompt = prompt

    payload = {
        "model": VENICE_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "venice_parameters": {"enable_web_scraping": True},
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{settings.venice_base_url}/chat/completions",
            json=payload,
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise VeniceError(f"Venice HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def scrape_and_summarize(url: str, prompt: str = "Summarize this page") -> str:
    """Convenience: scrape one URL and get an LLM-grounded summary."""
    return chat_with_scraping(prompt, urls=[url])
