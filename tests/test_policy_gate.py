"""Tests for PolicyGate enforcement."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline_schemas.policy_gate import PolicyGate
from pipeline_schemas import (
    QAReport, CheckResult, SecurityResult,
    BuildPlan, BuildTask, ChangeSet,
    ShipAuthorization, PipelineFailure,
)


def _make_qa(passed: bool = True, critical: int = 0) -> QAReport:
    return QAReport(
        run_id="test",
        passed=passed,
        syntax_check=CheckResult(passed=passed),
        security_scan=SecurityResult(passed=critical == 0, critical_count=critical),
        overall_message="test",
    )

def _make_plan(branch: str = "auto-build/test") -> BuildPlan:
    return BuildPlan(
        run_id="test", summary="t", repo="t", branch_name=branch,
        commit_message="t", pr_title="t", pr_body="t",
        tasks=[BuildTask(description="t", file_path="t.py")],
    )

def _make_cs(n_files: int = 1) -> ChangeSet:
    files = {f"file_{i}.py": "x = 1\n" for i in range(n_files)}
    return ChangeSet(run_id="test", branch_name="auto-build/test", files=files,
                     commit_sha="a"*40, commit_message="t", syntax_validated=True)


class TestPolicyGate:

    def test_fully_approved(self):
        """All checks pass → authorized=true."""
        gate = PolicyGate(trusted_actors=["alice"])
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(True),
            plan=_make_plan(), change_set=_make_cs(1),
            trusted_actor="alice",
        )
        assert auth.authorized is True
        assert len(auth.policy_checks) == 5
        assert all(c.passed for c in auth.policy_checks)
        assert auth.denial_reason == ""

    def test_qa_failed(self):
        """QA not passed → denied."""
        gate = PolicyGate()
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(False),
            plan=_make_plan(), change_set=_make_cs(),
            trusted_actor="Nietzsche-Ubermensch",
        )
        assert auth.authorized is False
        assert any(c.rule == "qa_passed" and not c.passed for c in auth.policy_checks)

    def test_critical_security_issue(self):
        """Critical security finding → denied."""
        gate = PolicyGate()
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(True, critical=2),
            plan=_make_plan(), change_set=_make_cs(),
            trusted_actor="Nietzsche-Ubermensch",
        )
        assert auth.authorized is False
        assert any(c.rule == "no_critical_security" and not c.passed for c in auth.policy_checks)

    def test_untrusted_actor(self):
        """Untrusted actor → denied."""
        gate = PolicyGate(trusted_actors=["alice"])
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(True),
            plan=_make_plan(), change_set=_make_cs(),
            trusted_actor="eve",
        )
        assert auth.authorized is False
        assert any(c.rule == "trusted_actor" and not c.passed for c in auth.policy_checks)

    def test_invalid_branch_name(self):
        """Branch not matching auto-build/ prefix → denied."""
        gate = PolicyGate()
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(True),
            plan=_make_plan(branch="feat/wrong-prefix"),
            change_set=_make_cs(),
            trusted_actor="Nietzsche-Ubermensch",
        )
        assert auth.authorized is False
        assert any(c.rule == "branch_naming" and not c.passed for c in auth.policy_checks)

    def test_file_limit_exceeded(self):
        """Too many files → denied."""
        gate = PolicyGate(max_files=5)
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(True),
            plan=_make_plan(), change_set=_make_cs(10),
            trusted_actor="Nietzsche-Ubermensch",
        )
        assert auth.authorized is False
        assert any(c.rule == "file_count_limit" and not c.passed for c in auth.policy_checks)

    def test_denial_reason_populated(self):
        """When denied, denial_reason lists the failed checks."""
        gate = PolicyGate(trusted_actors=["alice"])
        auth = gate.evaluate(
            run_id="r", qa_report=_make_qa(False),
            plan=_make_plan(), change_set=_make_cs(),
            trusted_actor="eve",
        )
        assert auth.authorized is False
        assert "qa_passed" in auth.denial_reason
        assert "trusted_actor" in auth.denial_reason
