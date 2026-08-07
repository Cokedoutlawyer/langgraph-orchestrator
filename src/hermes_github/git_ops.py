"""High-level git operations for automated build pipelines.

Wraps the GitHub API client and local git commands into composable
operations: commit files, push branches, create PRs, monitor CI, merge.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .client import GitHubClient
from .config import GitHubConfig
from .exceptions import GitError, GitConflictError, GitHubAPIError
from .models import (
    CheckRun,
    CommitStatus,
    CreateBranchResult,
    CreatePRResult,
    MergeMethod,
    MergeResult,
    PullRequest,
)

logger = logging.getLogger(__name__)


class GitOps:
    """High-level git operations combining API + local git.

    This class bridges the GitHub API client (for PR/merge/check operations)
    with local git commands (for commit/push operations) to provide a unified
    interface for automated build pipelines.
    """

    def __init__(
        self,
        client: GitHubClient | None = None,
        config: GitHubConfig | None = None,
    ):
        self.config = config or GitHubConfig.from_env()
        self.client = client or GitHubClient(self.config)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> GitOps:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Branch operations
    # -----------------------------------------------------------------

    def create_feature_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        base: str = "main",
    ) -> CreateBranchResult:
        """Create a feature branch via the API (no local clone needed)."""
        return self.client.create_branch(owner, repo, branch_name, from_branch=base)

    def delete_feature_branch(self, owner: str, repo: str, branch: str) -> None:
        """Delete a feature branch after merge."""
        try:
            self.client.delete_branch(owner, repo, branch)
        except GitHubAPIError as e:
            logger.warning("Failed to delete branch %s: %s", branch, e)

    # -----------------------------------------------------------------
    # Commit operations (via Git Data API — no local clone needed)
    # -----------------------------------------------------------------

    def commit_files(
        self,
        owner: str,
        repo: str,
        branch: str,
        files: dict[str, str],
        message: str,
    ) -> str:
        """Commit multiple files to a branch via the Git Data API.

        Args:
            files: Dict mapping file path → file content (UTF-8 strings).
            message: Commit message.
        Returns:
            The SHA of the new commit.
        """
        return self.client.commit_multiple_files(owner, repo, branch, files, message)

    def commit_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
    ) -> str:
        """Create or update a single file via the Contents API."""
        # Check if file exists (need SHA for update)
        existing = self.client.get_file_content(owner, repo, path, ref=branch)
        sha = None
        if existing is not None:
            # File exists — need its blob SHA for update
            try:
                data = self.client._request(
                    "GET",
                    f"/repos/{owner}/{repo}/contents/{path}",
                    params={"ref": branch},
                )
                sha = data.get("sha") if data else None
            except GitHubAPIError:
                pass

        self.client.create_or_update_file(owner, repo, path, content, message, branch, sha)
        return branch

    # -----------------------------------------------------------------
    # PR operations
    # -----------------------------------------------------------------

    def create_pr(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
    ) -> CreatePRResult:
        """Create a pull request."""
        return self.client.create_pr(owner, repo, title, head, base, body)

    def get_pr(self, owner: str, repo: str, number: int) -> PullRequest:
        """Get a pull request."""
        return self.client.get_pr(owner, repo, number)

    def merge_pr(
        self,
        owner: str,
        repo: str,
        number: int,
        method: MergeMethod = MergeMethod.SQUASH,
        title: str | None = None,
    ) -> MergeResult:
        """Merge a pull request."""
        pr = self.client.get_pr(owner, repo, number)

        # Pre-merge checks
        if pr.draft:
            return MergeResult(merged=False, message="PR is a draft")
        if pr.merged:
            return MergeResult(merged=False, message="PR is already merged")
        if pr.mergeable is False:
            return MergeResult(
                merged=False,
                message="PR has merge conflicts — resolve before merging",
            )

        commit_title = title or f"{pr.title} (#{pr.number})"
        return self.client.merge_pr(owner, repo, number, method, commit_title)

    def merge_pr_on_green(
        self,
        owner: str,
        repo: str,
        number: int,
        method: MergeMethod = MergeMethod.SQUASH,
        poll_interval: float = 30.0,
        max_wait: float = 600.0,
    ) -> MergeResult:
        """Wait for all CI checks to pass, then merge.

        Polls check runs and commit statuses until all are complete.
        If all pass, merges the PR. If any fail, returns without merging.
        Times out after max_wait seconds.
        """
        pr = self.client.get_pr(owner, repo, number)
        if pr.draft or pr.merged:
            return MergeResult(merged=False, message="PR is draft or already merged")

        logger.info("Waiting for CI checks on PR #%d (max %ds)...", number, int(max_wait))
        elapsed = 0.0

        while elapsed < max_wait:
            status = self.client.get_combined_check_status(owner, repo, pr.head_sha)

            if status["total"] == 0:
                logger.debug("No checks found yet (elapsed %.0fs)...", elapsed)
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue

            if status["pending"] > 0:
                logger.debug(
                    "Checks in progress: %d/%d complete (elapsed %.0fs)",
                    status["completed"], status["total"], elapsed,
                )
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue

            # All checks complete
            if status["all_passed"]:
                logger.info("All %d checks passed. Merging PR #%d.", status["total"], number)
                return self.merge_pr(owner, repo, number, method)

            # Some checks failed
            failed_names = [
                d["name"] for d in status["details"]
                if d["conclusion"] in ("failure", "cancelled", "timed_out")
            ]
            logger.warning("Checks failed: %s. Not merging.", ", ".join(failed_names))
            return MergeResult(
                merged=False,
                message=f"CI checks failed: {', '.join(failed_names)}",
            )

        logger.warning("Timed out waiting for CI after %.0fs", max_wait)
        return MergeResult(merged=False, message=f"Timed out after {int(max_wait)}s waiting for CI")

    # -----------------------------------------------------------------
    # Review operations
    # -----------------------------------------------------------------

    def check_reviews(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        """Check PR review status.

        Returns dict with:
        - 'approved': count of APPROVED reviews
        - 'changes_requested': count of CHANGES_REQUESTED reviews
        - 'pending': count of PENDING reviews
        - 'ready': True if no changes requested and no pending reviews
        """
        reviews = self.client.get_pr_reviews(owner, repo, number)
        approved = [r for r in reviews if r.state == "APPROVED"]
        changes_requested = [r for r in reviews if r.state == "CHANGES_REQUESTED"]
        pending = [r for r in reviews if r.state == "PENDING"]

        return {
            "approved": len(approved),
            "changes_requested": len(changes_requested),
            "pending": len(pending),
            "ready": len(changes_requested) == 0 and len(pending) == 0,
            "reviews": reviews,
        }

    # -----------------------------------------------------------------
    # Branch protection
    # -----------------------------------------------------------------

    def check_branch_protection(
        self,
        owner: str,
        repo: str,
        branch: str,
    ) -> dict[str, Any]:
        """Check branch protection rules.

        Returns dict with:
        - 'protected': bool
        - 'required_reviews': int (0 if none)
        - 'required_status_checks': list of check names
        - 'enforce_admins': bool
        """
        protection = self.client.get_branch_protection(owner, repo, branch)
        if protection is None:
            return {
                "protected": False,
                "required_reviews": 0,
                "required_status_checks": [],
                "enforce_admins": False,
            }

        rpr = protection.get("required_pull_request_reviews", {}) or {}
        rsc = protection.get("required_status_checks", {}) or {}

        return {
            "protected": True,
            "required_reviews": rpr.get("required_approving_review_count", 0),
            "required_status_checks": rsc.get("contexts", []),
            "enforce_admins": protection.get("enforce_admins", {}).get("enabled", False),
        }

    # -----------------------------------------------------------------
    # Full pipeline: commit → PR → merge-on-green
    # -----------------------------------------------------------------

    def ship(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        files: dict[str, str],
        commit_message: str,
        pr_title: str,
        pr_body: str = "",
        base: str = "main",
        merge_method: MergeMethod = MergeMethod.SQUASH,
        wait_for_ci: bool = True,
        ci_timeout: float = 600.0,
    ) -> dict[str, Any]:
        """End-to-end ship: create branch, commit files, create PR, merge on green.

        This is the primary entry point for the shipper stage of the
        auto-build pipeline.

        Returns a dict with:
        - 'branch': branch name
        - 'commit_sha': commit SHA
        - 'pr_number': PR number
        - 'pr_url': PR URL
        - 'merged': bool
        - 'merge_message': str
        """
        result: dict[str, Any] = {
            "branch": branch_name,
            "commit_sha": None,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "merge_message": "",
        }

        # 1. Create feature branch
        logger.info("Creating branch %s on %s/%s...", branch_name, owner, repo)
        self.create_feature_branch(owner, repo, branch_name, base)

        # 2. Commit files
        logger.info("Committing %d file(s) to %s...", len(files), branch_name)
        commit_sha = self.commit_files(owner, repo, branch_name, files, commit_message)
        result["commit_sha"] = commit_sha

        # 3. Create PR
        logger.info("Creating PR: %s", pr_title)
        pr = self.create_pr(owner, repo, pr_title, branch_name, base, pr_body)
        result["pr_number"] = pr.number
        result["pr_url"] = pr.url

        # 4. Wait for CI and merge (optional)
        if wait_for_ci:
            logger.info("Waiting for CI checks to pass (timeout: %.0fs)...", ci_timeout)
            merge_result = self.merge_pr_on_green(
                owner, repo, pr.number, merge_method,
                max_wait=ci_timeout,
            )
            result["merged"] = merge_result.merged
            result["merge_message"] = merge_result.message

            # Clean up branch after successful merge
            if merge_result.merged:
                self.delete_feature_branch(owner, repo, branch_name)
        else:
            result["merge_message"] = "PR created, CI wait skipped"

        return result
