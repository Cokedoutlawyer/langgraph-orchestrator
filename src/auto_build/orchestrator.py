"""Orchestrator — the PLAN → CODE → QA → SHIP state machine.

The orchestrator coordinates the four stages of the auto-build pipeline:
1. PLAN: The Planner analyzes the requirement and produces a BuildPlan
2. CODE: The Coder generates code and commits it to a feature branch
3. QA: The QAChecker runs syntax, security, and (optionally) lint/type/test checks
4. SHIP: The Shipper creates a PR and merges it when CI passes

Each stage updates the PipelineState, which is persisted to SQLite after
every transition. This enables resume-after-crash and audit trails.

The orchestrator also posts progress comments on the triggering issue
via the Notifier, so the user can see the pipeline's progress in real time.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

from hermes_github.client import GitHubClient
from hermes_github.config import GitHubConfig
from hermes_github.git_ops import GitOps
from hermes_github.models import MergeMethod

from .coder import Coder
from .llm import LLMClient
from .notify import Notifier
from .planner import Planner
from .qa import QAChecker
from .shipper import Shipper
from .web_fetcher import WebFetcher
from .oxylabs_client import OxylabsClient, OxylabsError
from .state import (
    BuildPlan,
    PipelineState,
    PipelineStatus,
    StateStore,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the PLAN → CODE → QA → SHIP pipeline.

    Usage:
        orch = Orchestrator()
        result = orch.run_from_issue("Nietzsche-Ubermensch", "goofy", 42)
        print(result.status, result.ship_result)

    Or for manual requirements:
        result = orch.run_from_requirement(
            owner="Nietzsche-Ubermensch",
            repo="goofy",
            requirements="Add a dark mode toggle to the sidebar",
        )
    """

    def __init__(
        self,
        config: GitHubConfig | None = None,
        llm: LLMClient | None = None,
        github: GitHubClient | None = None,
        git_ops: GitOps | None = None,
        planner: Planner | None = None,
        coder: Coder | None = None,
        qa_checker: QAChecker | None = None,
        shipper: Shipper | None = None,
        notifier: Notifier | None = None,
        state_store: StateStore | None = None,
        web_fetcher: WebFetcher | None = None,
        wait_for_ci: bool = True,
        ci_timeout: float = 600.0,
        merge_method: MergeMethod = MergeMethod.SQUASH,
        workspace: str | None = None,
    ):
        self.config = config or GitHubConfig.from_env()
        self.github = github or GitHubClient(self.config)
        self.llm = llm or LLMClient()
        self.git_ops = git_ops or GitOps(self.github, self.config)

        # Auto-detect Oxylabs from env — enables stealthy scraping
        import os as _os
        _oxylabs = None
        if _os.environ.get("OXYLABS_AI_STUDIO_API_KEY"):
            try:
                _oxylabs = OxylabsClient()
            except OxylabsError:
                pass

        self.web_fetcher = web_fetcher or WebFetcher(oxylabs_client=_oxylabs)
        self.planner = planner or Planner(self.llm, self.github, self.web_fetcher)
        self.coder = coder or Coder(self.llm, self.github, self.git_ops)
        self.qa_checker = qa_checker or QAChecker(workspace=workspace)
        self.shipper = shipper or Shipper(
            self.github, self.git_ops,
            wait_for_ci=wait_for_ci,
            ci_timeout=ci_timeout,
            merge_method=merge_method,
        )
        self.notifier = notifier or Notifier(self.github)
        self.state_store = state_store or StateStore()

        self._owns_resources = all(x is None for x in [llm, github, git_ops, planner, coder, shipper, notifier, web_fetcher])

    def close(self) -> None:
        """Close all owned resources."""
        if self._owns_resources:
            self.planner.close()
            self.coder.close()
            self.shipper.close()
            self.notifier.close()
            self.git_ops.close()

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Entry points
    # -----------------------------------------------------------------

    def run_from_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        wait_for_ci: bool | None = None,
    ) -> PipelineState:
        """Run the full pipeline triggered by a GitHub issue.

        Reads the issue, plans, codes, checks, and ships.
        """
        logger.info("Orchestrator: starting pipeline from issue %s/%s#%d", owner, repo, issue_number)

        # Fetch the issue
        issue = self.github.get_issue(owner, repo, issue_number)
        state = PipelineState(
            trigger="issue",
            issue_number=issue_number,
            issue_title=issue.title,
            issue_body=issue.body or "",
            repo_owner=owner,
            repo_name=repo,
            requirements=issue.body or issue.title,
        )

        # Notify start
        self.notifier.notify_start(
            owner, repo, issue_number, state.run_id,
            f"Issue: {issue.title}",
        )

        # Save initial state
        self.state_store.save(state)

        # Run the pipeline
        return self._run_pipeline(state, owner, repo, issue_number, wait_for_ci)

    def run_from_requirement(
        self,
        owner: str,
        repo: str,
        requirements: str,
        title: str = "Manual build request",
        wait_for_ci: bool | None = None,
    ) -> PipelineState:
        """Run the full pipeline from a manual text requirement.

        No GitHub issue is required — the pipeline runs directly from
        the provided requirements string.
        """
        logger.info("Orchestrator: starting pipeline from requirement: %s", title)

        state = PipelineState(
            trigger="manual",
            issue_title=title,
            issue_body=requirements,
            repo_owner=owner,
            repo_name=repo,
            requirements=requirements,
        )

        self.state_store.save(state)
        return self._run_pipeline(state, owner, repo, None, wait_for_ci)

    def resume(self, run_id: str) -> PipelineState | None:
        """Resume an interrupted pipeline run.

        Loads the state from the store and continues from the last
        completed phase.
        """
        state = self.state_store.load(run_id)
        if state is None:
            logger.warning("Orchestrator: run %s not found", run_id)
            return None

        logger.info(
            "Orchestrator: resuming run %s from phase %s (status=%s)",
            run_id, state.current_phase, state.status,
        )

        if state.status == PipelineStatus.COMPLETED:
            logger.info("Orchestrator: run already COMPLETED — nothing to resume")
            return state

        if state.status == PipelineStatus.CANCELLED:
            logger.info("Orchestrator: run was CANCELLED — nothing to resume")
            return state

        # FAILED runs can be resumed — reset status to PENDING so the
        # pipeline picks up from the last completed phase
        if state.status == PipelineStatus.FAILED:
            logger.info("Orchestrator: resuming FAILED run — retrying from last completed phase")
            # Determine which phase to resume from based on what's in state
            if state.commit_sha and state.files_committed:
                # Code completed — resume from QA
                state.status = PipelineStatus.QA
                state.current_phase = "qa"
                state.errors = []  # Clear previous errors
            elif state.plan:
                # Plan completed but no code — resume from CODE
                state.status = PipelineStatus.CODING
                state.current_phase = "coding"
                state.errors = []
            else:
                # No plan — start fresh
                state.status = PipelineStatus.PENDING
                state.current_phase = ""
                state.errors = []
            self.state_store.save(state)

        return self._run_pipeline(
            state,
            state.repo_owner,
            state.repo_name,
            state.issue_number,
            None,
            resume=True,
        )

    # -----------------------------------------------------------------
    # Core pipeline logic
    # -----------------------------------------------------------------

    def _run_pipeline(
        self,
        state: PipelineState,
        owner: str,
        repo: str,
        issue_number: int | None,
        wait_for_ci: bool | None,
        resume: bool = False,
    ) -> PipelineState:
        """Execute the pipeline stages sequentially.

        Each stage:
        1. Updates the state status
        2. Persists the state
        3. Notifies the issue (if applicable)
        4. Executes the stage
        5. Records the result in the state
        """
        try:
            # ────────────── PLAN ──────────────
            if state.status in (PipelineStatus.PENDING, PipelineStatus.PLANNING):
                state.status = PipelineStatus.PLANNING
                state.current_phase = "planning"
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "planning",
                        "Analyzing requirements and producing build plan...",
                    )

                plan = self.planner.plan(state)
                state.plan = plan
                state.status = PipelineStatus.CODING  # Ready for next phase
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "planning",
                        f"Plan ready: {plan.summary[:200]}\nBranch: {plan.branch_name}\nFiles: {len(plan.tasks)}",
                    )

            # ────────────── CODE ──────────────
            if state.status in (PipelineStatus.CODING,) or (resume and state.commit_sha is None):
                state.status = PipelineStatus.CODING
                state.current_phase = "coding"
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "coding",
                        f"Generating code for {len(state.plan.tasks)} file(s)...",
                    )

                commit_sha, files = self.coder.execute(state)
                state.commit_sha = commit_sha
                state.files_committed = files
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "coding",
                        f"Committed {len(files)} file(s) as {commit_sha[:7]}",
                    )

            # ────────────── QA ──────────────
            if state.status in (PipelineStatus.CODING,) or (resume and state.qa_result is None):
                state.status = PipelineStatus.QA
                state.current_phase = "qa"
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "qa",
                        "Running quality checks...",
                    )

                qa_result = self.qa_checker.run(state)
                state.qa_result = qa_result
                self.state_store.save(state)

                if issue_number:
                    self.notifier.notify_phase(
                        owner, repo, issue_number, state.run_id, "qa",
                        qa_result.overall_message,
                    )

                if not qa_result.passed:
                    state.status = PipelineStatus.FAILED
                    state.current_phase = "qa"
                    state.errors.append(f"QA failed: {qa_result.overall_message}")
                    self.state_store.save(state)

                    if issue_number:
                        self.notifier.notify_error(
                            owner, repo, issue_number, state.run_id,
                            qa_result.overall_message, "qa",
                        )
                    return state

            # ────────────── SHIP ──────────────
            state.status = PipelineStatus.SHIPPING
            state.current_phase = "shipping"
            self.state_store.save(state)

            if issue_number:
                self.notifier.notify_phase(
                    owner, repo, issue_number, state.run_id, "shipping",
                    "Creating pull request...",
                )

            # Override wait_for_ci if specified
            if wait_for_ci is not None:
                self.shipper.wait_for_ci = wait_for_ci

            ship_result = self.shipper.ship(state)
            state.ship_result = ship_result
            state.status = PipelineStatus.COMPLETED if ship_result.merged else PipelineStatus.FAILED
            if not ship_result.merged and ship_result.merge_message:
                state.errors.append(ship_result.merge_message)

            state.current_phase = "completed" if ship_result.merged else "shipping"
            self.state_store.save(state)

            if issue_number:
                if ship_result.merged:
                    self.notifier.notify_success(
                        owner, repo, issue_number, state.run_id,
                        ship_result.pr_number, ship_result.pr_url,
                        merged=True,
                    )
                else:
                    self.notifier.notify_error(
                        owner, repo, issue_number, state.run_id,
                        f"PR created but not merged: {ship_result.merge_message}",
                        "shipping",
                    )

            logger.info("Orchestrator: pipeline %s — %s", state.run_id, state.status)
            return state

        except Exception as e:
            logger.error("Orchestrator: pipeline failed: %s\n%s", e, traceback.format_exc())
            state.status = PipelineStatus.FAILED
            state.errors.append(f"{e.__class__.__name__}: {e}")
            state.current_phase = state.current_phase or "unknown"
            self.state_store.save(state)

            if issue_number:
                self.notifier.notify_error(
                    owner, repo, issue_number, state.run_id,
                    f"{e.__class__.__name__}: {e}",
                    state.current_phase,
                )

            return state
