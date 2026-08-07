"""Tests for GitHub shipping authorization enforcement."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline_schemas.orchestrator import _github_ship
from pipeline_schemas import (
    ShipAuthorization, ShipResult, BuildPlan, BuildTask,
    ChangeSet, HermesRunRequest, PipelineFailure,
)
from pipeline_schemas.graph_state import initial_state


def _make_auth(authorized: bool = True) -> ShipAuthorization:
    return ShipAuthorization(run_id="test-ship", authorized=authorized, merge_method="squash")

def _make_plan() -> BuildPlan:
    return BuildPlan(
        run_id="test-ship", summary="t", repo="r", branch_name="auto-build/test",
        commit_message="t", pr_title="t", pr_body="t",
        tasks=[BuildTask(description="t", file_path="t.py")],
    )

def _make_cs() -> ChangeSet:
    return ChangeSet(
        run_id="test-ship", branch_name="auto-build/test",
        files={"t.py": "x = 1\n"}, commit_sha="a"*40,
        commit_message="t", syntax_validated=True,
    )


class TestShippingEnforcement:

    def test_approved_authorization_ships(self):
        """Approved authorization → ship_fn called → ShipResultSchema produced."""
        called = [False]
        def ship_fn(auth, plan, cs, req):
            called[0] = True
            return ShipResult(run_id="test-ship", pr_number=1, pr_url="http://x", merged=True, merge_message="Merged")

        state = initial_state(run_id="test-ship", repo_owner="o", repo_name="r", trusted_actor="alice")
        state["ship_authorization"] = _make_auth(True)
        state["build_plan"] = _make_plan()
        state["change_set"] = _make_cs()

        result = _github_ship(state, ship_fn=ship_fn)
        assert called[0] is True
        assert result["ship_result"].merged is True
        assert result["ship_result"].pr_number == 1

    def test_missing_authorization_rejected(self):
        """No authorization → ship_fn NOT called, failure recorded."""
        called = [False]
        def ship_fn(auth, plan, cs, req):
            called[0] = True
            return ShipResult(run_id="test-ship", merged=True)

        state = initial_state(run_id="test-ship", repo_owner="o", repo_name="r", trusted_actor="alice")
        state["ship_authorization"] = None  # No authorization

        result = _github_ship(state, ship_fn=ship_fn)
        assert called[0] is False  # Ship fn was NOT called
        assert result["ship_result"].merged is False
        assert len(result.get("pipeline_failures", [])) > 0

    def test_denied_authorization_rejected(self):
        """Denied authorization → ship_fn NOT called."""
        called = [False]
        def ship_fn(auth, plan, cs, req):
            called[0] = True
            return ShipResult(run_id="test-ship", merged=True)

        state = initial_state(run_id="test-ship", repo_owner="o", repo_name="r", trusted_actor="alice")
        state["ship_authorization"] = _make_auth(False)  # Denied

        result = _github_ship(state, ship_fn=ship_fn)
        assert called[0] is False
        assert result["ship_result"].merged is False

    def test_ship_result_validated(self):
        """Ship result is validated through the schema boundary."""
        def ship_fn(auth, plan, cs, req):
            return ShipResult(run_id="test-ship", pr_number=1, pr_url="http://x", merged=True)

        state = initial_state(run_id="test-ship", repo_owner="o", repo_name="r", trusted_actor="alice")
        state["ship_authorization"] = _make_auth(True)
        state["build_plan"] = _make_plan()
        state["change_set"] = _make_cs()

        result = _github_ship(state, ship_fn=ship_fn)
        assert result["ship_result"] is not None
        # Should be a validated ShipResult
        assert hasattr(result["ship_result"], "schema_version")
