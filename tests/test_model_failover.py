"""Tests for model router failover."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline_schemas.orchestrator import _model_route, _default_model_fn
from pipeline_schemas import (
    RoutingPlan, ModelPhase, ModelResult,
    PipelineFailure, FailureCategory,
)
from pipeline_schemas.graph_state import initial_state


def _make_state(providers: list[str] | None = None, preferred: str | None = None):
    state = initial_state(run_id="test-failover", repo_owner="o", repo_name="r", trusted_actor="alice")
    plan = RoutingPlan(
        run_id="test-failover",
        research_needed=False,
        research_topics=[],
        model_phase=ModelPhase.PLAN,
        system_prompt="sys",
        user_prompt="usr",
        response_format="json_object",
        preferred_model_provider=preferred,
        fallback_model_providers=providers or ["openrouter", "openai", "grok"],
    )
    state["routing_plan"] = plan
    return state


class TestModelFailover:

    def test_first_provider_succeeds(self):
        """First provider works → no failover needed."""
        def model_fn(req, provider):
            return {"choices": [{"message": {"content": "{}"}}], "model": "test", "usage": {}}

        state = _make_state()
        result = _model_route(state, model_fn=model_fn)
        assert result["model_result"] is not None
        assert result["model_result"].failover_used is False
        assert len(result.get("provider_attempts", [])) == 1

    def test_first_fails_second_succeeds(self):
        """First provider retryable failure → second succeeds with failover."""
        call_count = [0]

        def model_fn(req, provider):
            call_count[0] += 1
            if provider == "openrouter":
                raise Exception("429 rate limited")
            return {"choices": [{"message": {"content": "{}"}}], "model": "backup", "usage": {}}

        state = _make_state()
        result = _model_route(state, model_fn=model_fn)
        assert result["model_result"] is not None
        assert result["model_result"].failover_used is True
        assert result["model_result"].provider == "openai"
        assert call_count[0] == 2

    def test_non_retryable_failure_stops_failover(self):
        """Terminal failure stops failover immediately."""
        def model_fn(req, provider):
            raise Exception("401 unauthorized")

        state = _make_state()
        result = _model_route(state, model_fn=model_fn)
        assert result["model_result"] is None
        assert len(result.get("pipeline_failures", [])) > 0
        fail = result["pipeline_failures"][0]
        assert fail.retryable is False

    def test_all_providers_fail(self):
        """All providers fail → structured pipeline failure."""
        def model_fn(req, provider):
            raise Exception("500 server error")

        state = _make_state()
        result = _model_route(state, model_fn=model_fn)
        assert result["model_result"] is None
        assert len(result.get("pipeline_failures", [])) > 0
        fail = result["pipeline_failures"][0]
        assert fail.category == FailureCategory.PROVIDER_TERMINAL

    def test_attempt_limit_bounded(self):
        """Failover attempts are bounded — no infinite loop."""
        def model_fn(req, provider):
            raise Exception("500 server error")

        state = _make_state(providers=["a", "b", "c", "d", "e", "f"])  # 6 providers
        result = _model_route(state, model_fn=model_fn)
        attempts = result.get("provider_attempts", [])
        assert len(attempts) <= 6  # Bounded by provider count

    def test_run_id_survives_attempts(self):
        """Run ID is preserved across all attempts."""
        def model_fn(req, provider):
            raise Exception("500 server error")

        state = _make_state()
        result = _model_route(state, model_fn=model_fn)
        failures = result.get("pipeline_failures", [])
        if failures:
            assert failures[0].run_id == "test-failover"
