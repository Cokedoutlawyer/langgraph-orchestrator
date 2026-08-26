"""hermes_github — Seamless GitHub integration for Hermes Agent.

Provides a typed GitHub API client, multi-repo workspace management,
and git operations for automated multi-agent build workflows.
"""
from __future__ import annotations

from .client import GitHubClient
from .config import GitHubConfig
from .exceptions import (
    GitHubError,
    GitHubAPIError,
    GitHubAuthError,
    GitHubRateLimitError,
    GitHubNotFoundError,
    GitError,
    GitConflictError,
)
from .models import (
    Repo,
    Branch,
    Issue,
    PullRequest,
    CheckRun,
    CommitStatus,
    Review,
    MergeResult,
    CreateBranchResult,
    CreatePRResult,
)
from .workspace import Workspace, WorkspaceRepo
from .git_ops import GitOps

__version__ = "1.0.0"

__all__ = [
    "GitHubClient",
    "GitHubConfig",
    "GitHubError",
    "GitHubAPIError",
    "GitHubAuthError",
    "GitHubRateLimitError",
    "GitHubNotFoundError",
    "GitError",
    "GitConflictError",
    "Repo",
    "Branch",
    "Issue",
    "PullRequest",
    "CheckRun",
    "CommitStatus",
    "Review",
    "MergeResult",
    "CreateBranchResult",
    "CreatePRResult",
    "Workspace",
    "WorkspaceRepo",
    "GitOps",
]
