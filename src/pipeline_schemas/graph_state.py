"""Pipeline graph state — one authoritative typed state representation.

Uses TypedDict with append reducers for list fields so LangGraph merges
state correctly across nodes instead of replacing the entire dict.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


def append_list(left: list, right: list | None) -> list:
    """Reducer: append-only list."""
    if right is None:
        return left
    return (left or []) + (right if isinstance(right, list) else [right])


def overwrite(left: Any, right: Any) -> Any:
    """Reducer: last-write-wins."""
    return right if right is not None else left


class PipelineGraphState(TypedDict, total=False):
    """Authoritative pipeline state flowing through LangGraph.

    Each key uses the overwrite reducer (default) so nodes only need to
    return the keys they modify. The append_list reducer is used for
    list fields that accumulate across nodes.
    """
    # Correlation
    run_id: str
    correlation_id: str
    actor: str
    current_stage: str
    completed_stages: Annotated[list[str], append_list]

    # Request
    hermes_request: Any  # HermesRunRequest

    # Routing
    routing_plan: Any  # RoutingPlan

    # Research
    research_request: Any  # ResearchRequest
    evidence: Any  # Evidence

    # Model
    model_task: Any  # ModelTask
    model_result: Any  # ModelResult

    # Browser
    browser_task: Any  # BrowserTask
    browser_result: Any  # BrowserResult

    # Provider tracking
    provider_attempts: Annotated[list[dict], append_list]

    # Pipeline core
    build_plan: Any  # BuildPlan
    change_set: Any  # ChangeSet
    qa_report: Any  # QAReport
    ship_authorization: Any  # ShipAuthorization
    ship_result: Any  # ShipResult

    # Failures
    validation_failures: Annotated[list, append_list]
    pipeline_failures: Annotated[list, append_list]

    # Misc
    retry_counters: dict
    pipeline_result: Any  # PipelineResult
    logs: Annotated[list[dict], append_list]


def initial_state(
    run_id: str,
    trigger: str = "manual",
    repo_owner: str = "",
    repo_name: str = "",
    trusted_actor: str = "",
    requirements: str = "",
    issue_number: int | None = None,
    issue_title: str = "",
    issue_body: str = "",
    wait_for_ci: bool = True,
) -> dict:
    """Create initial graph state from a Hermes run request."""
    from . import HermesRunRequest, TriggerType

    request = HermesRunRequest(
        run_id=run_id,
        trigger=TriggerType(trigger),
        repo_owner=repo_owner,
        repo_name=repo_name,
        trusted_actor=trusted_actor,
        requirements=requirements,
        issue_number=issue_number,
        issue_title=issue_title,
        issue_body=issue_body,
        wait_for_ci=wait_for_ci,
    )

    return {
        "run_id": run_id,
        "correlation_id": run_id,
        "actor": trusted_actor,
        "current_stage": "start",
        "completed_stages": [],
        "hermes_request": request,
        "routing_plan": None,
        "research_request": None,
        "evidence": None,
        "model_task": None,
        "model_result": None,
        "browser_task": None,
        "browser_result": None,
        "provider_attempts": [],
        "build_plan": None,
        "change_set": None,
        "qa_report": None,
        "ship_authorization": None,
        "ship_result": None,
        "validation_failures": [],
        "pipeline_failures": [],
        "retry_counters": {},
        "pipeline_result": None,
        "logs": [],
    }
