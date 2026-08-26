"""GitHub REST API client with retries, rate-limit handling, and pagination.

Designed for use in automated build pipelines — every call has proper error
handling, exponential backoff on transient failures, and proactive rate-limit
awareness. No stubs, no placeholders.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import httpx

from .config import GitHubConfig
from .exceptions import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)
from .models import (
    Branch,
    CheckRun,
    CommitStatus,
    CreateBranchResult,
    CreatePRResult,
    FileChange,
    Issue,
    Label,
    MergeMethod,
    MergeResult,
    PullRequest,
    Repo,
    Review,
    CommentResult,
)

logger = logging.getLogger(__name__)


class GitHubClient:
    """Typed GitHub REST API client.

    Wraps httpx with:
    - Exponential backoff retry on 5xx and network errors
    - Proactive rate-limit handling (waits before hitting limit)
    - Automatic pagination via paginate() / paginate_iterator()
    - Typed responses via Pydantic models
    """

    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig.from_env()
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-init httpx client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.api_base_url,
                headers=self.config.headers(),
                timeout=httpx.Timeout(self.config.request_timeout),
            )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Core request method with retries
    # -----------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        raw_response: bool = False,
    ) -> Any:
        """Execute an API request with retry and rate-limit handling.

        Returns the parsed JSON body (or raw httpx.Response if raw_response=True).
        """
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            # Check rate limit before making the request
            self._check_rate_limit()

            try:
                resp = self.client.request(method, url, params=params, json=json)
            except httpx.NetworkError as e:
                last_error = e
                logger.warning("Network error (attempt %d/%d): %s", attempt, self.config.max_retries, e)
                self._backoff(attempt)
                continue
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning("Timeout (attempt %d/%d): %s", attempt, self.config.max_retries, e)
                self._backoff(attempt)
                continue

            # Update rate limit tracking from response headers
            self._update_rate_limit(resp)

            # Success
            if resp.status_code < 400:
                if raw_response:
                    return resp
                if resp.status_code == 204 or not resp.content:
                    return None
                return resp.json()

            # Handle specific error codes
            if resp.status_code == 401 or resp.status_code == 403:
                # Could be auth error or rate limit
                remaining = resp.headers.get("x-ratelimit-remaining", "")
                if resp.status_code == 403 and remaining == "0":
                    reset_at = resp.headers.get("x-ratelimit-reset")
                    raise GitHubRateLimitError(
                        reset_at=datetime.fromtimestamp(int(reset_at), tz=timezone.utc).isoformat()
                        if reset_at
                        else None
                    )
                raise GitHubAuthError()

            if resp.status_code == 404:
                raise GitHubNotFoundError(url)

            if resp.status_code >= 500:
                last_error = GitHubAPIError(resp.status_code, resp.text)
                logger.warning(
                    "Server error %d (attempt %d/%d): %s",
                    resp.status_code, attempt, self.config.max_retries, resp.text[:200],
                )
                self._backoff(attempt)
                continue

            # 4xx (non-retryable client error)
            try:
                body = resp.json()
                msg = body.get("message", resp.text)
            except Exception:
                msg = resp.text
            raise GitHubAPIError(resp.status_code, msg, resp.text)

        # Exhausted retries
        raise GitHubAPIError(503, f"Request failed after {self.config.max_retries} retries: {last_error}")

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff sleep."""
        delay = self.config.retry_base_delay * (2 ** (attempt - 1))
        logger.debug("Backing off for %.1fs (attempt %d)", delay, attempt)
        time.sleep(delay)

    # -----------------------------------------------------------------
    # Rate limit handling
    # -----------------------------------------------------------------

    _rate_limit_remaining: int = 5000
    _rate_limit_reset: float = 0.0

    def _update_rate_limit(self, resp: httpx.Response) -> None:
        """Track rate limit from response headers."""
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if remaining is not None:
            try:
                self._rate_limit_remaining = int(remaining)
            except ValueError:
                pass
        if reset is not None:
            try:
                self._rate_limit_reset = float(reset)
            except ValueError:
                pass

    def _check_rate_limit(self) -> None:
        """Proactively wait if we're close to the rate limit."""
        if self._rate_limit_remaining > self.config.rate_limit_buffer:
            return
        if self._rate_limit_reset <= 0:
            return

        now = time.time()
        wait_seconds = self._rate_limit_reset - now
        if wait_seconds > 0 and wait_seconds < 600:  # Don't wait more than 10 minutes
            logger.info(
                "Rate limit near (%d remaining). Waiting %.0fs for reset.",
                self._rate_limit_remaining, wait_seconds,
            )
            time.sleep(wait_seconds + 1)

    def get_rate_limit(self) -> dict[str, Any]:
        """Check current rate limit status."""
        return self._request("GET", "/rate_limit")

    # -----------------------------------------------------------------
    # Pagination
    # -----------------------------------------------------------------

    def paginate(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        per_page: int = 100,
    ) -> list[Any]:
        """Fetch all pages and return combined list."""
        results: list[Any] = []
        for item in self.paginate_iterator(method, url, params=params, per_page=per_page):
            results.append(item)
        return results

    def paginate_iterator(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        per_page: int = 100,
    ) -> Iterator[Any]:
        """Yield items from all pages of a paginated endpoint."""
        page = 1
        params = dict(params or {})
        params.setdefault("per_page", per_page)

        while True:
            params["page"] = page
            data = self._request(method, url, params=params)

            if not data:
                break

            if isinstance(data, list):
                for item in data:
                    yield item
                if len(data) < per_page:
                    break
            else:
                # Non-list response, yield once and stop
                yield data
                break

            page += 1

    # -----------------------------------------------------------------
    # Repository operations
    # -----------------------------------------------------------------

    def get_repo(self, owner: str, repo: str) -> Repo:
        """Get repository info."""
        data = self._request("GET", f"/repos/{owner}/{repo}")
        return Repo.from_api(data)

    def list_repos(self, owner: str | None = None) -> list[Repo]:
        """List all repos for an owner (user or org)."""
        owner = owner or self.config.default_owner
        data = self.paginate("GET", f"/users/{owner}/repos", params={"type": "all", "sort": "updated"})
        return [Repo.from_api(d) for d in data]

    def get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch name for a repo."""
        r = self.get_repo(owner, repo)
        return r.default_branch

    # -----------------------------------------------------------------
    # Branch operations
    # -----------------------------------------------------------------

    def get_branch(self, owner: str, repo: str, branch: str) -> Branch:
        """Get info about a specific branch."""
        data = self._request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
        return Branch.from_api(data, f"{owner}/{repo}")

    def create_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        from_branch: str = "main",
    ) -> CreateBranchResult:
        """Create a new branch from an existing branch's HEAD."""
        # Get the SHA of the source branch
        src = self.get_branch(owner, repo, from_branch)
        sha = src.commit_sha

        self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
        )
        return CreateBranchResult(
            branch_name=branch_name,
            sha=sha,
            repo_full_name=f"{owner}/{repo}",
        )

    def delete_branch(self, owner: str, repo: str, branch: str) -> None:
        """Delete a branch (e.g., after merge)."""
        self._request("DELETE", f"/repos/{owner}/{repo}/git/refs/heads/{branch}")

    # -----------------------------------------------------------------
    # File operations
    # -----------------------------------------------------------------

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> str | None:
        """Get the decoded content of a file at a given path/ref."""
        try:
            data = self._request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{path}",
                params={"ref": ref},
            )
            if data is None:
                return None
            import base64
            content = data.get("content", "")
            if data.get("encoding") == "base64" and content:
                return base64.b64decode(content).decode("utf-8")
            return content
        except GitHubNotFoundError:
            return None

    def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
        sha: str | None = None,
    ) -> dict:
        """Create or update a file via the Contents API."""
        import base64
        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha  # Required for updates
        return self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", json=body)

    # -----------------------------------------------------------------
    # Issue operations
    # -----------------------------------------------------------------

    def get_issue(self, owner: str, repo: str, number: int) -> Issue:
        """Get a single issue."""
        data = self._request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        return Issue.from_api(data, f"{owner}/{repo}")

    def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: str = "",
    ) -> list[Issue]:
        """List issues in a repo."""
        params: dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = labels
        data = self.paginate("GET", f"/repos/{owner}/{repo}/issues", params=params)
        # Filter out PRs (the issues endpoint returns both)
        return [Issue.from_api(d, f"{owner}/{repo}") for d in data if "pull_request" not in d]

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> Issue:
        """Create a new issue."""
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        data = self._request("POST", f"/repos/{owner}/{repo}/issues", json=payload)
        return Issue.from_api(data, f"{owner}/{repo}")

    def add_issue_labels(self, owner: str, repo: str, number: int, labels: list[str]) -> None:
        """Add labels to an issue."""
        self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/labels",
            json={"labels": labels},
        )

    # -----------------------------------------------------------------
    # Comment operations
    # -----------------------------------------------------------------

    def create_issue_comment(
        self,
        owner: str,
        repo: str,
        number: int,
        body: str,
    ) -> CommentResult:
        """Create a comment on an issue or PR."""
        data = self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        return CommentResult(id=data["id"], url=data["html_url"])

    def list_issue_comments(self, owner: str, repo: str, number: int) -> list[dict]:
        """List comments on an issue or PR."""
        return self.paginate(
            "GET",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
        )

    def update_issue_comment(self, owner: str, repo: str, comment_id: int, body: str) -> None:
        """Edit an existing issue/PR comment."""
        self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}",
            json={"body": body},
        )

    # -----------------------------------------------------------------
    # Pull Request operations
    # -----------------------------------------------------------------

    def get_pr(self, owner: str, repo: str, number: int) -> PullRequest:
        """Get a single pull request."""
        data = self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}")
        return PullRequest.from_api(data, f"{owner}/{repo}")

    def list_prs(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        sort: str = "updated",
        direction: str = "desc",
    ) -> list[PullRequest]:
        """List pull requests in a repo."""
        data = self.paginate(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state, "sort": sort, "direction": direction},
        )
        return [PullRequest.from_api(d, f"{owner}/{repo}") for d in data]

    def create_pr(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        draft: bool = False,
    ) -> CreatePRResult:
        """Create a pull request."""
        payload: dict[str, Any] = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        }
        data = self._request("POST", f"/repos/{owner}/{repo}/pulls", json=payload)
        return CreatePRResult(
            number=data["number"],
            url=data["html_url"],
            head_ref=head,
            repo_full_name=f"{owner}/{repo}",
        )

    def merge_pr(
        self,
        owner: str,
        repo: str,
        number: int,
        method: MergeMethod = MergeMethod.SQUASH,
        commit_title: str | None = None,
        commit_message: str | None = None,
    ) -> MergeResult:
        """Merge a pull request."""
        payload: dict[str, Any] = {"merge_method": method.value}
        if commit_title:
            payload["commit_title"] = commit_title
        if commit_message:
            payload["commit_message"] = commit_message

        try:
            data = self._request(
                "PUT",
                f"/repos/{owner}/{repo}/pulls/{number}/merge",
                json=payload,
            )
            return MergeResult(
                merged=True,
                sha=data.get("sha", ""),
                message=data.get("message", "Merged"),
            )
        except GitHubAPIError as e:
            return MergeResult(merged=False, message=e.args[0])

    def get_pr_reviews(self, owner: str, repo: str, number: int) -> list[Review]:
        """List reviews on a PR."""
        data = self.paginate("GET", f"/repos/{owner}/{repo}/pulls/{number}/reviews")
        return [
            Review(
                id=d["id"],
                state=d.get("state", ""),
                author=d.get("user", {}).get("login", ""),
                body=d.get("body"),
                submitted_at=_parse_dt(d.get("submitted_at")) if d.get("submitted_at") else None,
            )
            for d in data
        ]

    def get_pr_files(self, owner: str, repo: str, number: int) -> list[FileChange]:
        """List files changed in a PR."""
        data = self.paginate("GET", f"/repos/{owner}/{repo}/pulls/{number}/files")
        return [
            FileChange(
                filename=d["filename"],
                status=d["status"],
                additions=d.get("additions", 0),
                deletions=d.get("deletions", 0),
                sha=d.get("sha", ""),
            )
            for d in data
        ]

    # -----------------------------------------------------------------
    # Check Run / Status operations
    # -----------------------------------------------------------------

    def list_check_runs(self, owner: str, repo: str, ref: str) -> list[CheckRun]:
        """List all check runs for a given ref (commit SHA or branch)."""
        data = self.paginate(
            "GET",
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs",
        )
        return [CheckRun.from_api(d) for d in data]

    def list_commit_statuses(self, owner: str, repo: str, ref: str) -> list[CommitStatus]:
        """List legacy commit statuses for a ref."""
        data = self.paginate(
            "GET",
            f"/repos/{owner}/{repo}/commits/{ref}/statuses",
        )
        return [
            CommitStatus(
                state=d.get("state", "pending"),
                context=d.get("context", ""),
                description=d.get("description"),
                target_url=d.get("target_url"),
            )
            for d in data
        ]

    def get_combined_status(self, owner: str, repo: str, ref: str) -> dict[str, Any]:
        """Get the combined status for a ref (aggregates all statuses)."""
        return self._request("GET", f"/repos/{owner}/{repo}/commits/{ref}/status")

    def get_combined_check_status(self, owner: str, repo: str, ref: str) -> dict[str, Any]:
        """Get combined check runs + statuses, deduplicated.

        Returns a dict with:
        - 'total': number of unique checks
        - 'completed': number completed
        - 'passed': number with success/skipped/neutral
        - 'failed': number with failure/cancelled/timed_out
        - 'pending': number in_progress/queued
        - 'all_passed': True if all completed and all passed
        - 'details': list of {name, status, conclusion} dicts
        """
        seen: dict[str, dict[str, str]] = {}

        # Check runs (modern)
        for run in self.list_check_runs(owner, repo, ref):
            if run.name == "Auto-merge on CI success":
                continue  # Skip our own workflow
            seen[run.name] = {
                "name": run.name,
                "status": run.status.value,
                "conclusion": run.conclusion.value if run.conclusion else "",
            }

        # Commit statuses (legacy)
        for status in self.list_commit_statuses(owner, repo, ref):
            if status.context in seen:
                continue
            seen[status.context] = {
                "name": status.context,
                "status": "completed" if status.state != "pending" else "in_progress",
                "conclusion": _map_status_to_conclusion(status.state),
            }

        all_details = list(seen.values())
        total = len(all_details)
        completed = [d for d in all_details if d["status"] == "completed"]
        passed = [d for d in completed if d["conclusion"] in ("success", "skipped", "neutral")]
        failed = [d for d in completed if d["conclusion"] in ("failure", "cancelled", "timed_out", "action_required")]
        pending = [d for d in all_details if d["status"] != "completed"]

        return {
            "total": total,
            "completed": len(completed),
            "passed": len(passed),
            "failed": len(failed),
            "pending": len(pending),
            "all_passed": total > 0 and len(completed) == total and len(failed) == 0,
            "details": all_details,
        }

    # -----------------------------------------------------------------
    # Branch Protection
    # -----------------------------------------------------------------

    def get_branch_protection(self, owner: str, repo: str, branch: str) -> dict | None:
        """Get branch protection rules. Returns None if no rules configured."""
        try:
            return self._request("GET", f"/repos/{owner}/{repo}/branches/{branch}/protection")
        except GitHubNotFoundError:
            return None

    # -----------------------------------------------------------------
    # Raw git operations via Blob API
    # -----------------------------------------------------------------

    def create_blob(self, owner: str, repo: str, content: str, encoding: str = "utf-8") -> str:
        """Create a git blob and return its SHA."""
        data = self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/blobs",
            json={"content": content, "encoding": encoding},
        )
        return data["sha"]

    def get_tree(self, owner: str, repo: str, sha: str) -> dict:
        """Get a tree object."""
        return self._request("GET", f"/repos/{owner}/{repo}/git/trees/{sha}")

    def create_tree(
        self,
        owner: str,
        repo: str,
        base_tree: str | None,
        tree_entries: list[dict],
    ) -> str:
        """Create a tree object with the given entries. Returns SHA."""
        payload: dict[str, Any] = {"tree": tree_entries}
        if base_tree:
            payload["base_tree"] = base_tree
        data = self._request("POST", f"/repos/{owner}/{repo}/git/trees", json=payload)
        return data["sha"]

    def create_commit(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        parent_shas: list[str],
        message: str,
    ) -> str:
        """Create a commit object. Returns SHA."""
        data = self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/commits",
            json={
                "tree": tree_sha,
                "parents": parent_shas,
                "message": message,
            },
        )
        return data["sha"]

    def update_ref(self, owner: str, repo: str, ref: str, sha: str) -> None:
        """Update a ref to point to a new SHA."""
        self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/git/refs/heads/{ref}",
            json={"sha": sha},
        )

    def commit_multiple_files(
        self,
        owner: str,
        repo: str,
        branch: str,
        files: dict[str, str],
        message: str,
    ) -> str:
        """Commit multiple files in a single commit via the Git Data API.

        Args:
            files: dict mapping file path → file content
        Returns:
            The SHA of the new commit.
        """
        # 1. Get the current branch HEAD commit and its tree
        branch_data = self.get_branch(owner, repo, branch)
        parent_sha = branch_data.commit_sha
        tree_data = self.get_tree(owner, repo, parent_sha)
        base_tree = tree_data.get("sha")

        # 2. Create blobs for each file
        tree_entries: list[dict] = []
        for path, content in files.items():
            blob_sha = self.create_blob(owner, repo, content)
            tree_entries.append({
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            })

        # 3. Create a new tree based on the base tree
        new_tree_sha = self.create_tree(owner, repo, base_tree, tree_entries)

        # 4. Create the commit
        new_commit_sha = self.create_commit(
            owner, repo, new_tree_sha, [parent_sha], message
        )

        # 5. Update the branch ref
        self.update_ref(owner, repo, branch, new_commit_sha)

        return new_commit_sha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | None) -> datetime:
    """Parse ISO 8601 datetime from GitHub API."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _map_status_to_conclusion(state: str) -> str:
    """Map legacy status state to check conclusion equivalent."""
    return {
        "success": "success",
        "failure": "failure",
        "error": "failure",
        "neutral": "neutral",
    }.get(state, "")
