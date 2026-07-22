"""Configuration loader — reads from Hermes .env and environment."""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_hermes_env() -> None:
    """Load Hermes .env into os.environ if not already set (don't override)."""
    candidates = [
        Path.home() / ".hermes" / ".env",
        Path("/mnt/c/Users/peter/AppData/Local/hermes/.env"),
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_hermes_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(Path.home() / ".hermes" / ".env"), extra="ignore")

    # LLM (Blackbox LiteLLM proxy — OpenAI-compatible)
    blackbox_api_key: str = Field(alias="BLACKBOX_API_KEY", default="")
    blackbox_base_url: str = "https://api.blackbox.ai/v1"
    default_model: str = "z-ai/glm-5.2"

    # BrightData MCP (API token + proxy)
    brightdata_api_token: str = Field(alias="BRIGHTDATA_API_TOKEN", default="")
    brightdata_proxy_password: str = Field(alias="BRIGHTDATA_PROXY_PASSWORD", default="")

    # Oxylabs AI Studio
    oxylabs_ai_studio_api_key: str = Field(alias="OXYLABS_AI_STUDIO_API_KEY", default="")

    # Venice
    venice_api_key: str = Field(alias="VENICE_API_KEY", default="")
    venice_base_url: str = "https://api.venice.ai/api/v1"

    # Camofox browser
    camofox_url: str = Field(alias="CAMOFOX_URL", default="http://localhost:9377")
    camofox_api_key: str = "change-me"

    # Engram RAG (Weaviate)
    engram_api_key: str = Field(alias="ENGRAM_API_KEY", default="")
    engram_default_user: str = "matts@hermes"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8100

    # Checkpointer
    checkpointer_uri: str = "sqlite:///~/.hermes/cache/orchestrator-checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
