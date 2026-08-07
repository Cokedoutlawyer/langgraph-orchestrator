"""LLM interface for the auto-build pipeline.

Uses OpenAI-compatible chat completions API for plan generation and code
generation. Works with any provider that supports the OpenAI format
(OpenAI, OpenRouter, Nous, xAI, etc.).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client for plan/code generation.

    Reads configuration from environment:
    - LLM_API_KEY or OPENAI_API_KEY: API key
    - LLM_BASE_URL or OPENAI_BASE_URL: API base URL
    - LLM_MODEL: model name (default: gpt-4o)
    - LLM_TEMPERATURE: temperature (default: 0.2 for code gen)
    - LLM_MAX_TOKENS: max output tokens (default: 4096)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant message content.

        Args:
            messages: List of {role, content} dicts.
            response_format: Optional response format (e.g. {"type": "json_object"}).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
        Returns:
            The text content of the assistant's response.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = self.client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            logger.error("LLM API error %d: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.debug("LLM response (%d chars)", len(content))
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict:
        """Send a chat completion and parse the response as JSON.

        Includes retry for JSON parse failures with a corrective prompt.
        """
        # First attempt with json_object format if supported
        try:
            content = self.chat(
                messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(content)
        except (json.JSONDecodeError, Exception):
            # Fallback: try without response_format (some providers don't support it)
            pass

        # Second attempt: ask for JSON explicitly
        messages = list(messages)
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no code fences, just a JSON object."

        content = self.chat(messages, temperature=temperature)

        # Try to extract JSON from the response (handles markdown code fences)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Could not parse LLM response as JSON: {content[:200]}")

    def generate_code(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate code from a prompt. Returns the code string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.chat(messages, temperature=temperature, max_tokens=8192)

        # Strip markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (fence) and last line (fence)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        return response
