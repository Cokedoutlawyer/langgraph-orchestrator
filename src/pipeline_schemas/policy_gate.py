"""PolicyGate — enforces policy rules between QA and shipping.

No code ships without ShipAuthorizationSchema.authorized=true. The PolicyGate
independently validates:
  1. qa_passed — QAReportSchema.passed must be true
  2. no_critical_security — SecurityResult.critical_count must be 0
  3. trusted_actor — triggering actor must be in trusted_actors list
  4. branch_naming — BuildPlanSchema.branch_name must match "auto-build/" prefix
  5. file_count_limit — ChangeSetSchema.files must not exceed 50 files
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from . import (
    BuildPlan,
    ChangeSet,
    PolicyCheck,
    QAReport,
    ShipAuthorization,
    PipelineFailure,
    FailureCategory,
)

logger = logging.getLogger(__name__)

MAX_FILE_COUNT = 50
BRANCH_PATTERN = re.compile(r"^auto-build/[a-z0-9/-]+$")


class PolicyGate:
    """Policy enforcement point between QA and GitHub shipping.

    The only legal path from QA to GitHub is:
        QA → QAReportSchema → PolicyGate → ShipAuthorizationSchema → GitHub
    """

    def __init__(self, trusted_actors: list[str] | None = None, max_files: int = MAX_FILE_COUNT):
        self.trusted_actors = set(trusted_actors or ["Nietzsche-Ubermensch"])
        self.max_files = max_files

    def evaluate(
        self,
        run_id: str,
        qa_report: QAReport,
        plan: BuildPlan | None = None,
        change_set: ChangeSet | None = None,
        trusted_actor: str = "",
        merge_method: str = "squash",
    ) -> ShipAuthorization:
        """Evaluate all policy rules and produce ShipAuthorizationSchema.

        Returns ShipAuthorization with authorized=true only if all checks pass.
        """
        checks: list[PolicyCheck] = []

        # Rule 1: QA passed
        qa_ok = qa_report.passed
        checks.append(PolicyCheck(
            rule="qa_passed",
            passed=qa_ok,
            message="QA report passed" if qa_ok else "QA report failed",
        ))

        # Rule 2: No critical security findings
        no_critical = qa_report.security_scan.critical_count == 0
        checks.append(PolicyCheck(
            rule="no_critical_security",
            passed=no_critical,
            message=f"{qa_report.security_scan.critical_count} critical finding(s)"
            if not no_critical else "No critical security findings",
        ))

        # Rule 3: Trusted actor
        actor_ok = trusted_actor in self.trusted_actors if trusted_actor else False
        checks.append(PolicyCheck(
            rule="trusted_actor",
            passed=actor_ok,
            message=f"Actor '{trusted_actor}' is trusted" if actor_ok
            else f"Actor '{trusted_actor}' is NOT in trusted list",
        ))

        # Rule 4: Branch naming convention
        branch_name = plan.branch_name if plan else ""
        branch_ok = bool(branch_name and BRANCH_PATTERN.match(branch_name))
        checks.append(PolicyCheck(
            rule="branch_naming",
            passed=branch_ok,
            message=f"Branch '{branch_name}' matches auto-build/ pattern" if branch_ok
            else f"Branch '{branch_name}' does NOT match auto-build/ pattern",
        ))

        # Rule 5: File count limit
        file_count = len(change_set.files) if change_set else 0
        file_count_ok = file_count <= self.max_files
        checks.append(PolicyCheck(
            rule="file_count_limit",
            passed=file_count_ok,
            message=f"{file_count} files (limit {self.max_files})" if file_count_ok
            else f"{file_count} files exceeds limit of {self.max_files}",
        ))

        # Determine authorization
        all_passed = all(c.passed for c in checks)
        failed_checks = [c for c in checks if not c.passed]
        denial_reason = "; ".join(f"{c.rule}: {c.message}" for c in failed_checks) if failed_checks else ""

        if not all_passed:
            logger.warning("PolicyGate DENIED run %s: %s", run_id, denial_reason)
        else:
            logger.info("PolicyGate APPROVED run %s", run_id)

        return ShipAuthorization(
            run_id=run_id,
            authorized=all_passed,
            qa_report=qa_report,
            policy_checks=checks,
            denial_reason=denial_reason,
            merge_method=merge_method,
        )
