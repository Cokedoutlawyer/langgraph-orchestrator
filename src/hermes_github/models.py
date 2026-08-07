"""Pydantic models for GitHub API entities."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    """Status of a check run or commit status."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckConclusion(str, Enum):
    """Conclusion of a completed check run."""
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    STALE = "stale"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


class MergeMethod(str, Enum):
    """Method for merging a pull request."""
    MERGE = "merge"
    SQUASH = "squash"
    REBASE = "rebase"


class PRState(str, Enum):
    """State of a pull request."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class IssueState(str, Enum):
    """State of an issue."""
    OPEN = "open"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Repo(BaseModel):
    """GitHub repository."""
    id: int
    name: str
    full_name: str
    owner: str
    private: bool = False
    default_branch: str = "main"
    clone_url: str = ""
    html_url: str = ""
    description: str | None = None
    archived: bool = False
    disabled: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Repo:
        """Create from GitHub API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            owner=data["owner"]["login"],
            private=data.get("private", False),
            default_branch=data.get("default_branch", "main"),
            clone_url=data.get("clone_url", ""),
            html_url=data.get("html_url", ""),
            description=data.get("description"),
            archived=data.get("archived", False),
            disabled=data.get("disabled", False),
        )


class Branch(BaseModel):
    """Git branch."""
    name: str
    commit_sha: str
    protected: bool = False
    repo_full_name: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any], repo_full_name: str = "") -> Branch:
        return cls(
            name=data["name"],
            commit_sha=data["commit"]["sha"],
            protected=data.get("protected", False),
            repo_full_name=repo_full_name,
        )


class Label(BaseModel):
    """Issue/PR label."""
    id: int
    name: str
    color: str
    description: str | None = None


class Issue(BaseModel):
    """GitHub issue."""
    number: int
    title: str
    body: str | None = None
    state: IssueState = IssueState.OPEN
    author: str = ""
    labels: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    repo_full_name: str = ""
    html_url: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any], repo_full_name: str = "") -> Issue:
        return cls(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=IssueState(data.get("state", "open")),
            author=data.get("user", {}).get("login", ""),
            labels=[l["name"] for l in data.get("labels", [])],
            assignees=[a["login"] for a in data.get("assignees", [])],
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            repo_full_name=repo_full_name,
            html_url=data.get("html_url", ""),
        )


class Review(BaseModel):
    """PR review."""
    id: int
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, PENDING, DISMISSED
    author: str
    body: str | None = None
    submitted_at: datetime | None = None


class PullRequest(BaseModel):
    """GitHub pull request."""
    number: int
    title: str
    body: str | None = None
    state: PRState = PRState.OPEN
    draft: bool = False
    merged: bool = False
    mergeable: bool | None = None
    head_ref: str = ""
    head_sha: str = ""
    base_ref: str = ""
    author: str = ""
    labels: list[str] = Field(default_factory=list)
    requested_reviewers: list[str] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    repo_full_name: str = ""
    html_url: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any], repo_full_name: str = "") -> PullRequest:
        return cls(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=PRState(data.get("state", "open")),
            draft=data.get("draft", False),
            merged=data.get("merged", False),
            mergeable=data.get("mergeable"),
            head_ref=data.get("head", {}).get("ref", ""),
            head_sha=data.get("head", {}).get("sha", ""),
            base_ref=data.get("base", {}).get("ref", ""),
            author=data.get("user", {}).get("login", ""),
            labels=[l["name"] for l in data.get("labels", [])],
            requested_reviewers=[r["login"] for r in data.get("requested_reviewers", [])],
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            repo_full_name=repo_full_name,
            html_url=data.get("html_url", ""),
        )


class CheckRun(BaseModel):
    """GitHub check run (modern status check)."""
    id: int
    name: str
    status: CheckStatus = CheckStatus.QUEUED
    conclusion: CheckConclusion | None = None
    head_sha: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    html_url: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> CheckRun:
        return cls(
            id=data["id"],
            name=data["name"],
            status=CheckStatus(data.get("status", "queued")),
            conclusion=CheckConclusion(data["conclusion"]) if data.get("conclusion") else None,
            head_sha=data.get("head_sha", ""),
            started_at=_parse_dt(data.get("started_at")),
            completed_at=_parse_dt(data.get("completed_at")),
            html_url=data.get("html_url", ""),
        )


class CommitStatus(BaseModel):
    """Legacy commit status (from the Statuses API)."""
    state: str  # error, failure, pending, success
    context: str
    description: str | None = None
    target_url: str | None = None

    @property
    def is_success(self) -> bool:
        return self.state == "success"

    @property
    def is_pending(self) -> bool:
        return self.state == "pending"


class MergeResult(BaseModel):
    """Result of a merge operation."""
    merged: bool
    sha: str | None = None
    message: str = ""


class CreateBranchResult(BaseModel):
    """Result of creating a branch."""
    branch_name: str
    sha: str
    repo_full_name: str


class CreatePRResult(BaseModel):
    """Result of creating a pull request."""
    number: int
    url: str
    head_ref: str
    repo_full_name: str


class CommentResult(BaseModel):
    """Result of posting a comment."""
    id: int
    url: str


class FileChange(BaseModel):
    """A file change in a commit or PR."""
    filename: str
    status: str  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    sha: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | None) -> datetime | None:
    """Parse ISO 8601 datetime string from GitHub API."""
    if not value:
        return None
    try:
        # GitHub returns "2025-01-15T12:34:56Z" — replace Z with +00:00
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)
