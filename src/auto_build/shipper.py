"""Shipper — creates PRs and merges them when CI passes.

The shipper takes the pipeline state (with committed files on a feature branch)
and:
1. Creates a pull request with the plan's title and body
2. Optionally waits for CI checks to pass
3. Auto-merges the PR if all checks pass
4. Cleans up the feature branch after merge
5. Reports the ship result back to the pipeline state
"""
from __future__ import annotations

import logging
from typing import Any

from hermes_github.client import GitHubClient
from hermes_github.git_ops import GitOps
from hermes_github.models import MergeMethod, MergeResult

from .state import PipelineState, ShipResult

logger = logging.getLogger(__name__)


class Shipper:
    """Creates PRs and ships them (merge on green).

    Wraps GitOps.ship() with pipeline-specific logic:
    - Builds the PR body from the plan and QA results
    - Reports progress via the notifier
    - Handles merge failures gracefully
    """

    def __init__(
        self,
        github: GitHubClient | None = None,
        git_ops: GitOps | None = None,
        wait_for_ci: bool = True,
        ci_timeout: float = 600.0,
        merge_method: MergeMethod = MergeMethod.SQUASH,
    ):
        self.github = github or GitHubClient()
        self.git_ops = git_ops or GitOps(self.github)
        self._owns_github = github is None
        self.wait_for_ci = wait_for_ci
        self.ci_timeout = ci_timeout
        self.merge_method = merge_method

    def close(self) -> None:
        if self._owns_github:
            self.github.close()

    def __enter__(self) -> Shipper:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def ship(self, state: PipelineState) -> ShipResult:
        """Create a PR from the pipeline state and optionally merge on green.

        Args:
            state: Pipeline state with plan, commit_sha, and files_committed populated.
        Returns:
            ShipResult with PR number, URL, merge status.
        """
        if state.plan is None:
            raise ValueError("Cannot ship: no plan in pipeline state")
        if not state.commit_sha:
            raise ValueError("Cannot ship: no commit SHA in pipeline state")

        plan = state.plan
        owner = state.repo_owner
        repo = state.repo_name

        logger.info("Shipper: shipping %s/%s branch=%s", owner, repo, plan.branch_name)

        # Build the PR body from plan + QA results
        pr_body = self._build_pr_body(state)

        # Check if a PR already exists for this branch
        existing_prs = self.github.list_prs(owner, repo, state="open")
        existing = None
        for p in existing_prs:
            if p.head_ref == plan.branch_name:
                existing = p
                break

        if existing:
            logger.info("Shipper: PR #%d already exists for %s — reusing", existing.number, plan.branch_name)
            result = ShipResult(
                pr_number=existing.number,
                pr_url=existing.html_url,
                branch_name=plan.branch_name,
                commit_sha=state.commit_sha,
            )
        else:
            # Create the PR
            logger.info("Shipper: creating PR '%s' on %s/%s", plan.pr_title, owner, repo)
            pr = self.git_ops.create_pr(
                owner=owner,
                repo=repo,
                title=plan.pr_title,
                head=plan.branch_name,
                base="main",
                body=pr_body,
            )
            result = ShipResult(
                pr_number=pr.number,
                pr_url=pr.url,
                branch_name=plan.branch_name,
                commit_sha=state.commit_sha,
            )
            logger.info("Shipper: PR created #%d — %s", pr.number, pr.url)

        # Wait for CI and merge
        if self.wait_for_ci:
            logger.info("Shipper: waiting for CI (timeout: %.0fs)...", self.ci_timeout)
            merge_result = self.git_ops.merge_pr_on_green(
                owner=owner,
                repo=repo,
                number=result.pr_number,
                method=self.merge_method,
                max_wait=self.ci_timeout,
            )
            result.merged = merge_result.merged
            result.merge_message = merge_result.message

            if merge_result.merged:
                logger.info("Shipper: ✅ PR #%d merged!", result.pr_number)
                # Clean up the feature branch
                self.git_ops.delete_feature_branch(owner, repo, plan.branch_name)
            else:
                logger.warning("Shipper: ⚠️ PR not merged: %s", merge_result.message)

        return result

    def ship_without_merge(self, state: PipelineState) -> ShipResult:
        """Create a PR without waiting for CI or merging.

        Useful when CI checks need to be reviewed manually.
        """
        self.wait_for_ci = False
        return self.ship(state)

    def _build_pr_body(self, state: PipelineState) -> str:
        """Build the PR body from the plan and QA results."""
        plan = state.plan
        parts: list[str] = []

        parts.append(plan.pr_body)
        parts.append("")
        parts.append("---")
        parts.append("*This PR was created by the Hermes Auto-Build pipeline.*")
        parts.append(f"*Run ID: `{state.run_id}`*")

        if state.trigger == "issue" and state.issue_number:
            parts.append(f"*Triggered by: #{state.issue_number}*")
        parts.append("")

        # Add task summary
        if plan.tasks:
            parts.append("## Changes")
            parts.append("")
            for task in plan.tasks:
                icon = "✅" if task.status == "completed" else "❌" if task.status == "failed" else "⏳"
                parts.append(f"- {icon} {task.description} (`{task.file_path}`)")
            parts.append("")

        # Add QA results
        if state.qa_result:
            qa = state.qa_result
            parts.append("## QA Results")
            parts.append("")
            parts.append(f"- **Status:** {'✅ Passed' if qa.passed else '❌ Failed'}")
            if qa.lint_passed is not None:
                parts.append(f"- **Syntax/Lint:** {'✅' if qa.lint_passed else '❌'}")
            if qa.security_passed is not None:
                parts.append(f"- **Security:** {'✅' if qa.security_passed else '⚠️ Findings'}")
            if qa.tests_run:
                parts.append(f"- **Tests:** {qa.tests_passed} passed, {qa.tests_failed} failed")
            parts.append("")
            if qa.overall_message:
                parts.append(f"```\n{qa.overall_message}\n```")

        return "\n".join(parts)
