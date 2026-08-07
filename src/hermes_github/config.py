"""Configuration for hermes_github."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class GitHubConfig(BaseModel):
    """Configuration for GitHub API client and workspace.

    All fields can be overridden via environment variables with the GITHUB_ prefix.
    """

    token: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("GITHUB_TOKEN", ""))
    )
    """GitHub personal access token (classic or fine-grained)."""

    api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub REST API base URL (override for GitHub Enterprise)."
    )

    api_version: str = Field(
        default="2022-11-28",
        description="X-GitHub-Api-Version header value."
    )

    default_owner: str = Field(
        default_factory=lambda: os.environ.get("GITHUB_DEFAULT_OWNER", "Nietzsche-Ubermensch"),
        description="Default repo owner for operations that don't specify one."
    )

    workspace_root: Path = Field(
        default_factory=lambda: Path(os.environ.get("GITHUB_WORKSPACE_ROOT", str(Path.home() / "github-workspace"))),
        description="Root directory for cloned repos."
    )

    default_branch: str = Field(
        default="main",
        description="Default branch name to target for merges."
    )

    request_timeout: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds."
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for transient failures."
    )

    retry_base_delay: float = Field(
        default=1.0,
        description="Base delay (seconds) for exponential backoff."
    )

    rate_limit_buffer: int = Field(
        default=100,
        description="Minimum remaining requests before proactively waiting for rate limit reset."
    )

    trusted_actors: list[str] = Field(
        default_factory=lambda: os.environ.get(
            "GITHUB_TRUSTED_ACTORS", "Nietzsche-Ubermensch"
        ).split(","),
        description="GitHub usernames allowed to trigger auto-merge."
    )

    @property
    def token_str(self) -> str:
        """Get the token as a plain string (for internal use)."""
        return self.token.get_secret_value()

    def headers(self) -> dict[str, str]:
        """Build standard HTTP headers for GitHub API requests."""
        return {
            "Authorization": f"Bearer {self.token_str}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "hermes-github/1.0",
        }

    @classmethod
    def from_env(cls) -> GitHubConfig:
        """Create config from environment variables."""
        return cls(
            token=os.environ.get("GITHUB_TOKEN", ""),
            default_owner=os.environ.get("GITHUB_DEFAULT_OWNER", "Nietzsche-Ubermensch"),
            workspace_root=Path(os.environ.get("GITHUB_WORKSPACE_ROOT", str(Path.home() / "github-workspace"))),
        )
