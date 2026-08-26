"""Exception hierarchy for hermes_github."""
from __future__ import annotations


class GitHubError(Exception):
    """Base exception for all hermes_github errors."""


class GitHubAPIError(GitHubError):
    """Raised when the GitHub API returns an error response."""

    def __init__(self, status_code: int, message: str, response_body: str | None = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"GitHub API error {status_code}: {message}")


class GitHubAuthError(GitHubAPIError):
    """Raised when authentication fails (401/403)."""

    def __init__(self, message: str = "Authentication failed. Check your GITHUB_TOKEN."):
        super().__init__(401, message)


class GitHubRateLimitError(GitHubAPIError):
    """Raised when the GitHub API rate limit is exceeded."""

    def __init__(self, reset_at: str | None = None):
        self.reset_at = reset_at
        msg = "GitHub API rate limit exceeded."
        if reset_at:
            msg += f" Resets at: {reset_at}"
        super().__init__(403, msg)


class GitHubNotFoundError(GitHubAPIError):
    """Raised when a resource is not found (404)."""

    def __init__(self, resource: str = "Resource"):
        super().__init__(404, f"{resource} not found")


class GitError(Exception):
    """Base exception for git operation errors."""


class GitConflictError(GitError):
    """Raised when a git operation results in merge conflicts."""

    def __init__(self, branch: str, conflicting_files: list[str] | None = None):
        self.branch = branch
        self.conflicting_files = conflicting_files or []
        msg = f"Merge conflicts on branch '{branch}'"
        if conflicting_files:
            msg += f": {', '.join(conflicting_files)}"
        super().__init__(msg)
