"""Canonical pipeline schemas with versioning.

Every component boundary uses one of these schemas. Each carries an explicit
schema_version so serialized payloads can be migrated across versions.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TriggerType(str, Enum):
    MANUAL = "manual"
    ISSUE = "issue"
    PR = "pr"
    WORKFLOW_DISPATCH = "workflow_dispatch"
    WEBHOOK = "webhook"
    CRON = "cron"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    CODING = "coding"
    QA = "qa"
    SHIPPING = "shipping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProviderName(str, Enum):
    BRIGHTDATA = "brightdata"
    OXYLABS = "oxylabs"
    VENICE = "venice"
    CAMOFOX = "camofox"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GROK = "grok"
    NOUS = "nous"


class ModelPhase(str, Enum):
    PLAN = "plan"
    CODE = "code"


class ContentType(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    TEXT = "text"
    ACCESSIBILITY_TREE = "accessibility_tree"


class FailureCategory(str, Enum):
    SCHEMA_VALIDATION = "schema_validation"
    PROVIDER_RETRYABLE = "provider_retryable"
    PROVIDER_TERMINAL = "provider_terminal"
    MCP_ERROR = "mcp_error"
    BROWSER_ERROR = "browser_error"
    POLICY_DENIED = "policy_denied"
    QA_FAILED = "qa_failed"
    SHIPPING_ERROR = "shipping_error"
    ORCHESTRATION_ERROR = "orchestration_error"


class FailoverReason(str, Enum):
    BLOCKED = "blocked"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    HEALTH_FAIL = "health_fail"


# ---------------------------------------------------------------------------
# Versioned base
# ---------------------------------------------------------------------------

class VersionedModel(BaseModel):
    """Base class for all canonical schemas. Carries schema_version."""
    schema_version: str = Field(default="1.0", description="Schema version (semver)")


# ---------------------------------------------------------------------------
# 1. HermesRunRequestSchema
# ---------------------------------------------------------------------------

class HermesRunRequest(VersionedModel):
    run_id: str = Field(default_factory=_uuid)
    trigger: TriggerType = TriggerType.MANUAL
    source: str = "cli"
    repo_owner: str
    repo_name: str
    issue_number: Optional[int] = None
    issue_title: str = ""
    issue_body: str = ""
    requirements: str = ""
    base_branch: str = "main"
    wait_for_ci: bool = True
    ci_timeout: float = 600.0
    trusted_actor: str = ""
    idempotency_key: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. RoutingPlanSchema
# ---------------------------------------------------------------------------

class BrowserTaskItem(VersionedModel):
    url: str
    task_prompt: str = ""
    output_format: str = "markdown"
    render_javascript: bool = True
    geo: Optional[str] = None
    preferred_backend: Optional[str] = None


class RoutingPlan(VersionedModel):
    run_id: str
    research_needed: bool = True
    research_topics: list[str] = Field(default_factory=list)
    model_phase: ModelPhase = ModelPhase.PLAN
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    response_format: str = "json_object"
    preferred_model_provider: Optional[str] = None
    fallback_model_providers: list[str] = Field(default_factory=list)
    browser_needed: bool = False
    browser_tasks: list[BrowserTaskItem] = Field(default_factory=list)
    geo: Optional[str] = None


# ---------------------------------------------------------------------------
# 3. ResearchRequestSchema
# ---------------------------------------------------------------------------

class ResearchRequest(VersionedModel):
    run_id: str
    topics: list[str]
    max_results_per_topic: int = 3
    max_chars_per_page: int = 3000
    geo: Optional[str] = None
    render_javascript: bool = False
    failover_tier: int = 1


# ---------------------------------------------------------------------------
# 4. EvidenceSchema
# ---------------------------------------------------------------------------

class FailoverEvent(BaseModel):
    from_provider: ProviderName
    to_provider: Optional[ProviderName] = None
    reason: FailoverReason
    timestamp: datetime = Field(default_factory=_utcnow)


class EvidenceSource(BaseModel):
    url: str
    title: str
    content: str
    provider: ProviderName
    content_type: ContentType = ContentType.MARKDOWN
    chars: int = 0


class Evidence(VersionedModel):
    run_id: str
    sources: list[EvidenceSource] = Field(default_factory=list)
    combined_context: str = ""
    providers_used: list[ProviderName] = Field(default_factory=list)
    total_chars: int = 0
    failover_events: list[FailoverEvent] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 5. ModelTaskSchema
# ---------------------------------------------------------------------------

class ModelTask(VersionedModel):
    run_id: str
    phase: ModelPhase
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 4096
    response_format: str = "text"
    preferred_provider: Optional[str] = None
    fallback_providers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 6. ModelResultSchema
# ---------------------------------------------------------------------------

class ModelResult(VersionedModel):
    run_id: str
    phase: ModelPhase
    content: str
    parsed: Optional[dict] = None
    provider: str = ""
    model: str = ""
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    latency_ms: Optional[int] = None
    failover_used: bool = False
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 7. BrowserTaskSchema (reuses BrowserTaskItem + run_id)
# ---------------------------------------------------------------------------

class BrowserTask(VersionedModel):
    run_id: str
    url: str
    task_prompt: str = ""
    output_format: str = "markdown"
    render_javascript: bool = True
    geo: Optional[str] = None
    session_id: Optional[str] = None
    preferred_backend: Optional[str] = None


# ---------------------------------------------------------------------------
# 8. BrowserResultSchema
# ---------------------------------------------------------------------------

class BrowserResult(VersionedModel):
    run_id: str
    content: str
    content_type: ContentType = ContentType.MARKDOWN
    backend: str = ""
    url: str = ""
    screenshot_path: Optional[str] = None
    session_id: Optional[str] = None
    latency_ms: Optional[int] = None
    failover_used: bool = False
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 9. BuildPlanSchema (BuildTask + BuildPlan)
# ---------------------------------------------------------------------------

class BuildTask(VersionedModel):
    id: str = Field(default_factory=lambda: _uuid()[:8])
    description: str
    file_path: str
    action: str = "create"
    content_preview: str = ""
    status: str = "pending"
    error: Optional[str] = None


class BuildPlan(VersionedModel):
    run_id: str
    summary: str
    repo: str
    branch_name: str
    commit_message: str
    pr_title: str
    pr_body: str
    tasks: list[BuildTask] = Field(default_factory=list)
    research_context: str = ""
    estimated_files: int = 0
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 10. ChangeSetSchema
# ---------------------------------------------------------------------------

class ChangeSet(VersionedModel):
    run_id: str
    branch_name: str
    files: dict[str, str] = Field(default_factory=dict)
    commit_sha: str = ""
    commit_message: str = ""
    syntax_validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 11. QAReportSchema
# ---------------------------------------------------------------------------

class CheckResult(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)


class SecurityResult(BaseModel):
    passed: bool
    findings: list[str] = Field(default_factory=list)
    critical_count: int = 0


class TestResult(BaseModel):
    run: bool = False
    passed: int = 0
    failed: int = 0


class QAReport(VersionedModel):
    run_id: str
    passed: bool
    syntax_check: CheckResult = Field(default_factory=lambda: CheckResult(passed=True))
    security_scan: SecurityResult = Field(default_factory=lambda: SecurityResult(passed=True))
    lint_check: Optional[CheckResult] = None
    type_check: Optional[CheckResult] = None
    test_results: Optional[TestResult] = None
    overall_message: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 12. ShipAuthorizationSchema
# ---------------------------------------------------------------------------

class PolicyCheck(BaseModel):
    rule: str
    passed: bool
    message: str = ""


class ShipAuthorization(VersionedModel):
    run_id: str
    authorized: bool
    qa_report: Optional[QAReport] = None
    policy_checks: list[PolicyCheck] = Field(default_factory=list)
    denial_reason: str = ""
    merge_method: str = "squash"
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 13. ShipResultSchema
# ---------------------------------------------------------------------------

class ShipResult(VersionedModel):
    run_id: str
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    merged: bool = False
    merge_message: str = ""
    branch_name: str = ""
    commit_sha: str = ""
    branch_deleted: bool = False
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 14. PipelineResultSchema
# ---------------------------------------------------------------------------

class PipelineResult(VersionedModel):
    run_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    trigger: str = "manual"
    repo_owner: str = ""
    repo_name: str = ""
    plan: Optional[BuildPlan] = None
    change_set: Optional[ChangeSet] = None
    qa_report: Optional[QAReport] = None
    ship_result: Optional[ShipResult] = None
    errors: list[str] = Field(default_factory=list)
    current_phase: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    duration_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Structured Failure
# ---------------------------------------------------------------------------

class PipelineFailure(BaseModel):
    stage: str
    run_id: str = ""
    category: FailureCategory
    provider: Optional[str] = None
    retryable: bool = False
    message: str = ""
    details: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
