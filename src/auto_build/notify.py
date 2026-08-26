"""Notifier — posts progress comments on GitHub issues and PRs.

The notifier provides a single interface for posting progress updates to
GitHub throughout the pipeline lifecycle. It deduplicates comments to
avoid spam on repeated triggers and supports updating an existing comment
rather than creating a new one.
"""
from __future__ import annotations

import logging
from typing import Any

from hermes_github.client import GitHubClient

logger = logging.getLogger(__name__)


class Notifier:
    """Posts progress comments on GitHub issues and PRs.

    Features:
    - Phase transition comments (when the pipeline moves to a new phase)
    - Error comments (when the pipeline fails)
    - Success comments (when the pipeline completes)
    - Duplicate detection (won't repost the same comment)
    - Comment editing (updates a single progress comment instead of spamming)
    """

    def __init__(self, github: GitHubClient | None = None):
        self.github = github or GitHubClient()
        self._owns_github = github is None
        self._progress_comment_id: int | None = None

    def close(self) -> None:
        if self._owns_github:
            self.github.close()

    def __enter__(self) -> Notifier:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def notify_start(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        run_id: str,
        requirements: str = "",
    ) -> None:
        """Post a comment when the pipeline starts.

        Creates an initial progress comment that will be updated as the
        pipeline progresses, rather than posting a new comment each time.
        """
        body = self._build_progress_comment(
            phase="Starting",
            status="🚀",
            run_id=run_id,
            details=requirements,
            phases={"plan": "⏳", "code": "⏳", "qa": "⏳", "ship": "⏳"},
        )

        try:
            result = self.github.create_issue_comment(owner, repo, issue_number, body)
            self._progress_comment_id = result.id
            logger.info("Notifier: posted start comment (id=%d)", result.id)
        except Exception as e:
            logger.warning("Notifier: failed to post start comment: %s", e)

    def notify_phase(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        run_id: str,
        phase: str,
        details: str = "",
        phases: dict[str, str] | None = None,
    ) -> None:
        """Update the progress comment when the pipeline moves to a new phase.

        Args:
            phase: Current phase name (planning, coding, qa, shipping, completed, failed).
            details: Additional details about this phase's progress.
            phases: Dict mapping phase names to status icons (✅ ⏳ ❌).
        """
        if phases is None:
            phases = {
                "plan": "✅" if phase in ("coding", "qa", "shipping", "completed") else "⏳",
                "code": "✅" if phase in ("qa", "shipping", "completed") else "⏳",
                "qa": "✅" if phase in ("shipping", "completed") else "⏳",
                "ship": "✅" if phase in ("completed") else "⏳",
            }

        status_icon = {
            "planning": "📋",
            "coding": "💻",
            "qa": "🔍",
            "shipping": "🚢",
            "completed": "✅",
            "failed": "❌",
        }.get(phase, "⏳")

        body = self._build_progress_comment(
            phase=phase.capitalize(),
            status=status_icon,
            run_id=run_id,
            details=details,
            phases=phases,
        )

        try:
            if self._progress_comment_id:
                self.github.update_issue_comment(
                    owner, repo, self._progress_comment_id, body
                )
                logger.info("Notifier: updated progress comment (phase=%s)", phase)
            else:
                result = self.github.create_issue_comment(owner, repo, issue_number, body)
                self._progress_comment_id = result.id
                logger.info("Notifier: posted progress comment (phase=%s, id=%d)", phase, result.id)
        except Exception as e:
            logger.warning("Notifier: failed to update progress comment: %s", e)

    def notify_success(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        run_id: str,
        pr_number: int | None = None,
        pr_url: str | None = None,
        merged: bool = False,
    ) -> None:
        """Post a success comment when the pipeline completes."""
        phases = {"plan": "✅", "code": "✅", "qa": "✅", "ship": "✅"}

        details_parts = ["Pipeline completed successfully!"]
        if pr_number and pr_url:
            details_parts.append(f"\n**PR:** #{pr_number} — {pr_url}")
        if merged:
            details_parts.append("\n**Status:** ✅ Merged to main")

        body = self._build_progress_comment(
            phase="Completed",
            status="✅",
            run_id=run_id,
            details="\n".join(details_parts),
            phases=phases,
        )

        try:
            if self._progress_comment_id:
                self.github.update_issue_comment(
                    owner, repo, self._progress_comment_id, body
                )
            else:
                self.github.create_issue_comment(owner, repo, issue_number, body)
            logger.info("Notifier: posted success comment")
        except Exception as e:
            logger.warning("Notifier: failed to post success comment: %s", e)

    def notify_error(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        run_id: str,
        error: str,
        phase: str = "",
    ) -> None:
        """Post an error comment when the pipeline fails."""
        phases = {
            "plan": "✅" if phase not in ("planning",) else "❌",
            "code": "✅" if phase not in ("planning", "coding") else "❌" if phase == "coding" else "⏳",
            "qa": "✅" if phase not in ("planning", "coding", "qa") else "❌" if phase == "qa" else "⏳",
            "ship": "✅" if phase == "completed" else "❌" if phase == "shipping" else "⏳",
        }

        body = self._build_progress_comment(
            phase="Failed" if not phase else f"Failed during {phase}",
            status="❌",
            run_id=run_id,
            details=f"**Error:**\n```\n{error}\n```",
            phases=phases,
        )

        try:
            if self._progress_comment_id:
                self.github.update_issue_comment(
                    owner, repo, self._progress_comment_id, body
                )
            else:
                self.github.create_issue_comment(owner, repo, issue_number, body)
            logger.info("Notifier: posted error comment")
        except Exception as e:
            logger.warning("Notifier: failed to post error comment: %s", e)

    def _build_progress_comment(
        self,
        phase: str,
        status: str,
        run_id: str,
        details: str = "",
        phases: dict[str, str] | None = None,
    ) -> str:
        """Build a progress comment body."""
        if phases is None:
            phases = {}

        phase_line = " → ".join(
            f"{phases.get(name, '⏳')} {name.capitalize()}"
            for name in ("plan", "code", "qa", "ship")
        )

        parts = [
            f"## {status} Auto-Build Pipeline — {phase}",
            "",
            f"{phase_line}",
            "",
            f"`run_id: {run_id}`",
        ]

        if details:
            parts.append("")
            parts.append(details)

        parts.append("")
        parts.append("---")
        parts.append("*Automated by [Hermes Agent](https://github.com/NousResearch/hermes-agent) • Auto-Build Pipeline*")

        return "\n".join(parts)
