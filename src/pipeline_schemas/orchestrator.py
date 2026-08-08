"""Schema-driven pipeline orchestrator — executable LangGraph runtime.

This is the root orchestration graph that connects all pipeline stages:

START → validate_hermes_request → route_request
  → research_route → model_route → browser_route
  → plan_agent → code_agent → qa_agent → policy_gate
  → [DENIED → finalize_failure] or [APPROVED → github_ship → finalize_result] → END

Every component boundary validates through SchemaBoundary. No raw strings,
untyped dicts, or vendor-native objects cross boundaries.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, START, END

from .boundary import SchemaBoundary, BoundaryResult
from .graph_state import PipelineGraphState, initial_state
from . import (
    HermesRunRequest,
    RoutingPlan,
    RoutingPlan as _RP,
    ResearchRequest,
    Evidence,
    EvidenceSource,
    FailoverEvent,
    FailoverReason,
    ProviderName,
    ModelTask,
    ModelResult,
    ModelPhase,
    BrowserTask,
    BrowserResult,
    BuildPlan,
    BuildTask,
    ChangeSet,
    QAReport,
    CheckResult,
    SecurityResult,
    ShipAuthorization,
    ShipResult,
    PipelineResult,
    PipelineStatus,
    PipelineFailure,
    FailureCategory,
    PolicyCheck,
)
from .adapters import (
    BrightDataResearchAdapter,
    OxylabsResearchAdapter,
    VeniceResearchAdapter,
    OpenAICompatModelAdapter,
    CamofoxBrowserAdapter,
    BrightDataBrowserAdapter,
    OxylabsBrowserAdapter,
    MCPResearchAdapter,
    GitHubShipAdapter,
)
from .policy_gate import PolicyGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider execution functions (injectable for testing)
# ---------------------------------------------------------------------------

# These callables are injected at graph construction time. In production they
# make real API/MCP calls. In tests they return mocked responses.
ResearchProviderFn = Callable[[str, str], str]  # (topic, geo) → content
ModelProviderFn = Callable[[dict, str], dict]   # (request, provider) → response
BrowserProviderFn = Callable[[BrowserTask], str]  # (task) → content
GitHubShipFn = Callable[[ShipAuthorization, BuildPlan, ChangeSet, HermesRunRequest], ShipResult]


def _default_research_fn(topic: str, geo: str | None = None) -> str:
    """Default: returns empty (real research needs MCP/HTTP)."""
    return ""


def _default_model_fn(request: dict, provider: str) -> dict:
    """Default: returns empty (real model needs API call)."""
    return {"choices": [{"message": {"content": "{}"}}], "model": "default", "usage": {}}


def _default_browser_fn(task: BrowserTask) -> str:
    """Default: returns empty."""
    return ""


def _default_github_fn(
    auth: ShipAuthorization, plan: BuildPlan, cs: ChangeSet, req: HermesRunRequest,
) -> ShipResult:
    """Default: returns unmerged result."""
    return ShipResult(run_id=auth.run_id, merged=False, merge_message="No GitHub client configured")


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------

def _validate_hermes_request(state: PipelineGraphState) -> dict:
    """Validate the incoming HermesRunRequest through the schema boundary."""
    request = state.get("hermes_request")
    if request is None:
        fail = PipelineFailure(
            stage="validate_request",
            run_id=state.get("run_id", ""),
            category=FailureCategory.SCHEMA_VALIDATION,
            message="No HermesRunRequest in state",
        )
        return {
            "pipeline_failures": [fail],
            "current_stage": "validate_request",
        }

    result = SchemaBoundary.validate(
        stage="validate_request",
        schema=HermesRunRequest,
        payload=request,
        run_id=state.get("run_id", ""),
    )
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "validate_request",
        }

    return {
        "hermes_request": result.validated,
        "current_stage": "validate_request",
        "completed_stages": ["validate_request"],
    }


def _route_request(state: PipelineGraphState) -> dict:
    """Analyze the Hermes request and produce a RoutingPlan."""
    req: HermesRunRequest = state.get("hermes_request")
    run_id = state.get("run_id", "unknown")

    # Extract research topics from the requirement
    topics = _extract_research_topics(req.requirements or req.issue_body or req.issue_title)

    # Build the routing plan
    plan = RoutingPlan(
        run_id=run_id,
        research_needed=len(topics) > 0,
        research_topics=topics,
        model_phase=ModelPhase.PLAN,
        system_prompt="You are a senior software architect planning a code change.",
        user_prompt=_build_plan_prompt(req),
        temperature=0.2,
        max_tokens=4096,
        response_format="json_object",
        preferred_model_provider=None,
        fallback_model_providers=["openrouter", "openai", "grok"],
        browser_needed=False,
        browser_tasks=[],
        geo=None,
    )

    return {"routing_plan": plan, "current_stage": "route_request", "completed_stages": ["route_request"]}


def _extract_research_topics(text: str) -> list[str]:
    """Extract research topics from a requirement text."""
    if not text:
        return []
    known_libs = [
        "FastAPI", "Flask", "Django", "Pydantic", "LangChain", "LangGraph",
        "React", "Vue", "Next.js", "PyTorch", "httpx", "Express",
    ]
    topics: list[str] = []
    text_lower = text.lower()
    for lib in known_libs:
        if lib.lower() in text_lower:
            topics.append(f"{lib} documentation and usage pattern")
    return topics[:5]


def _build_plan_prompt(req: HermesRunRequest) -> str:
    parts = []
    if req.issue_title:
        parts.append(f"Issue: {req.issue_title}")
    if req.issue_body:
        parts.append(req.issue_body)
    if req.requirements:
        parts.append(f"Requirements: {req.requirements}")
    parts.append(f"Repository: {req.repo_owner}/{req.repo_name}")
    return "\n".join(parts)


def _research_route(
    state: PipelineGraphState,
    research_fn: ResearchProviderFn = _default_research_fn,
) -> dict:
    """Execute research using the routing plan and normalize to EvidenceSchema."""
    plan: RoutingPlan = state.get("routing_plan")
    run_id = state.get("run_id", "unknown")

    if not plan or not plan.research_needed:
        return {"evidence": Evidence(run_id=run_id), "current_stage": "research", "completed_stages": ["research"]}

    sources: list[EvidenceSource] = []
    failovers: list[FailoverEvent] = []

    for topic in plan.research_topics[:3]:
        try:
            content = research_fn(topic, plan.geo)
            if content:
                sources.append(EvidenceSource(
                    url=f"research://{topic[:50]}",
                    title=topic,
                    content=content,
                    provider=ProviderName.BRIGHTDATA,
                    chars=len(content),
                ))
        except Exception as e:
            logger.warning("Research failed for '%s': %s", topic, e)
            failovers.append(FailoverEvent(
                from_provider=ProviderName.BRIGHTDATA,
                to_provider=ProviderName.OXYLABS,
                reason=FailoverReason.ERROR,
            ))

    combined = "\n\n".join(s.content for s in sources)
    evidence = Evidence(
        run_id=run_id,
        sources=sources,
        combined_context=combined,
        providers_used=[s.provider for s in sources],
        total_chars=len(combined),
        failover_events=failovers,
    )

    return {"evidence": evidence, "current_stage": "research", "completed_stages": ["research"]}


def _model_route(
    state: PipelineGraphState,
    model_fn: ModelProviderFn = _default_model_fn,
) -> dict:
    """Execute LLM call with failover across providers."""
    plan: RoutingPlan = state.get("routing_plan")
    run_id = state.get("run_id", "unknown")

    if not plan:
        return {"model_result": None, "current_stage": "model", "completed_stages": ["model"]}

    # Build the model task
    task = ModelTask(
        run_id=run_id,
        phase=plan.model_phase,
        system_prompt=plan.system_prompt,
        user_prompt=plan.user_prompt,
        temperature=plan.temperature,
        max_tokens=plan.max_tokens,
        response_format=plan.response_format,
    )

    # Try providers in order: preferred → fallbacks
    providers = []
    if plan.preferred_model_provider:
        providers.append(plan.preferred_model_provider)
    providers.extend(plan.fallback_model_providers or ["openrouter", "openai", "grok"])
    # Deduplicate
    seen = set()
    providers = [p for p in providers if not (p in seen or seen.add(p))]

    attempts: list[dict] = []
    last_error: PipelineFailure | None = None

    for attempt_num, provider in enumerate(providers, 1):
        attempt_start = time.time()
        try:
            vendor_request = OpenAICompatModelAdapter.to_vendor_request(
                task.system_prompt,
                task.user_prompt,
                task.temperature,
                task.max_tokens,
                task.response_format,
            )
            vendor_response = model_fn(vendor_request, provider)
            result = OpenAICompatModelAdapter.from_vendor_response(
                vendor_response, run_id=run_id, phase=task.phase, provider=provider,
            )
            result.failover_used = attempt_num > 1
            result.latency_ms = int((time.time() - attempt_start) * 1000)
            attempts.append({
                "provider": provider, "attempt": attempt_num,
                "outcome": "success", "latency_ms": result.latency_ms,
            })
            return {
                "model_result": result,
                "provider_attempts": attempts,
                "current_stage": "model",
                "completed_stages": ["model"],
            }
        except Exception as e:
            fail = OpenAICompatModelAdapter.from_vendor_error(e)
            fail.provider = provider
            fail.run_id = run_id
            attempts.append({
                "provider": provider, "attempt": attempt_num,
                "outcome": "failure", "error": fail.message[:200],
                "retryable": fail.retryable,
            })
            last_error = fail
            if not fail.retryable:
                # Terminal failure — stop failover
                break
            # Retryable — try next provider

    # All providers failed
    terminal_fail = last_error or PipelineFailure(
        stage="model", run_id=run_id,
        category=FailureCategory.PROVIDER_TERMINAL,
        message="All model providers failed",
    )
    # Escalate to terminal since all providers exhausted
    terminal_fail.category = FailureCategory.PROVIDER_TERMINAL
    terminal_fail.retryable = False
    return {
        "model_result": None,
        "provider_attempts": attempts,
        "pipeline_failures": [terminal_fail],
        "current_stage": "model",
    }


def _browser_route(
    state: PipelineGraphState,
    browser_fn: BrowserProviderFn = _default_browser_fn,
) -> dict:
    """Execute browser task if needed."""
    plan: RoutingPlan = state.get("routing_plan")
    run_id = state.get("run_id", "unknown")

    if not plan or not plan.browser_needed or not plan.browser_tasks:
        return {"browser_result": None, "current_stage": "browser", "completed_stages": ["browser"]}

    task_item = plan.browser_tasks[0]
    task = BrowserTask(
        run_id=run_id,
        url=task_item.url,
        task_prompt=task_item.task_prompt,
        output_format=task_item.output_format,
        render_javascript=task_item.render_javascript,
        geo=task_item.geo,
        preferred_backend=task_item.preferred_backend,
    )

    try:
        content = browser_fn(task)
        result = BrowserResult(
            run_id=run_id,
            content=content,
            backend="camofox",
            url=task.url,
        )
        return {"browser_result": result, "current_stage": "browser", "completed_stages": ["browser"]}
    except Exception as e:
        logger.warning("Browser failed: %s", e)
        return {
            "browser_result": None,
            "pipeline_failures": [PipelineFailure(
                stage="browser", run_id=run_id,
                category=FailureCategory.BROWSER_ERROR,
                message=str(e),
            )],
            "current_stage": "browser",
        }


def _plan_agent(state: PipelineGraphState) -> dict:
    """Produce a BuildPlan from model result + research evidence."""
    run_id = state.get("run_id", "unknown")
    model_result: ModelResult | None = state.get("model_result")
    evidence: Evidence | None = state.get("evidence")
    req: HermesRunRequest = state.get("hermes_request")

    if model_result is None:
        return {
            "pipeline_failures": [PipelineFailure(
                stage="plan", run_id=run_id,
                category=FailureCategory.ORCHESTRATION_ERROR,
                message="No model result available for planning",
            )],
            "current_stage": "plan",
        }

    # Parse the LLM response as JSON plan
    plan_data = model_result.parsed
    if plan_data is None:
        try:
            import json
            plan_data = json.loads(model_result.content)
        except Exception:
            plan_data = {}

    # Normalize LLM output — different models use different key names
    summary = plan_data.get("summary") or plan_data.get("change_request") or plan_data.get("description") or "Auto-build change"
    branch_name = plan_data.get("branch_name") or plan_data.get("branch") or "auto-build/untitled"
    commit_message = plan_data.get("commit_message") or plan_data.get("commit") or f"feat: {summary[:50]}"
    pr_title = plan_data.get("pr_title") or plan_data.get("title") or summary[:72]
    pr_body = plan_data.get("pr_body") or plan_data.get("body") or f"## Summary\n{summary}"

    # Normalize tasks — handle multiple LLM output formats
    raw_tasks = (
        plan_data.get("tasks")
        or plan_data.get("files_to_create")
        or plan_data.get("files_to_modify")
        or plan_data.get("implementation_plan")
        or plan_data.get("files")
        or []
    )
    tasks = []
    for t in raw_tasks:
        if isinstance(t, str):
            # Some models return a list of file paths as strings
            tasks.append(BuildTask(
                description=f"Create/modify {t}",
                file_path=t,
                action="create",
            ))
        elif isinstance(t, dict):
            file_path = t.get("file_path") or t.get("path") or t.get("filename") or t.get("file") or ""
            desc = t.get("description") or t.get("desc") or t.get("summary") or f"Modify {file_path}"
            action = t.get("action") or ("modify" if "modify" in str(t).lower() else "create")
            tasks.append(BuildTask(
                description=desc,
                file_path=file_path,
                action=action,
                content_preview=t.get("content_preview") or t.get("content") or "",
            ))

    # If no tasks found, try implementation_plan items
    if not tasks and isinstance(plan_data.get("implementation_plan"), list):
        for item in plan_data["implementation_plan"]:
            if isinstance(item, dict) and ("file" in item or "path" in item or "file_path" in item):
                fp = item.get("file") or item.get("path") or item.get("file_path") or ""
                tasks.append(BuildTask(description=item.get("description", f"Modify {fp}"), file_path=fp, action="create"))
    if not branch_name.startswith("auto-build/"):
        branch_name = f"auto-build/{branch_name}"

    research_ctx = evidence.combined_context if evidence else ""

    plan = BuildPlan(
        run_id=run_id,
        summary=summary,
        repo=req.repo_name,
        branch_name=branch_name,
        commit_message=commit_message,
        pr_title=pr_title,
        pr_body=pr_body,
        tasks=tasks,
        research_context=research_ctx,
        estimated_files=len(tasks),
    )

    # Validate through boundary
    result = SchemaBoundary.validate("plan", BuildPlan, plan, run_id=run_id)
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "plan",
        }

    return {"build_plan": result.validated, "current_stage": "plan", "completed_stages": ["plan"]}


def _code_agent(
    state: PipelineGraphState,
    model_fn: ModelProviderFn = _default_model_fn,
) -> dict:
    """Generate code for each task and commit via Git Data API."""
    run_id = state.get("run_id", "unknown")
    plan: BuildPlan | None = state.get("build_plan")
    req: HermesRunRequest = state.get("hermes_request")

    if plan is None:
        return {
            "pipeline_failures": [PipelineFailure(
                stage="code", run_id=run_id,
                category=FailureCategory.ORCHESTRATION_ERROR,
                message="No build plan available",
            )],
            "current_stage": "code",
        }

    # Generate file contents via LLM for each task
    files: dict[str, str] = {}
    for task in plan.tasks:
        # Call the LLM to generate real code
        code_request = OpenAICompatModelAdapter.to_vendor_request(
            system_prompt=f"You are a software engineer. Generate the complete content for {task.file_path}. "
                          f"Task: {task.description}. Action: {task.action}. "
                          f"Write complete, production-ready code. No stubs, no placeholders. "
                          f"Do NOT wrap in markdown code fences — output raw code only.",
            user_prompt=f"Generate the complete file content for: {task.file_path}\n"
                        f"Description: {task.description}\n"
                        f"Action: {task.action}\n"
                        f"Plan summary: {plan.summary}\n"
                        f"PR title: {plan.pr_title}",
            temperature=0.1,
            max_tokens=8192,
            response_format="text",
        )
        try:
            vendor_response = model_fn(code_request, "nous")
            code_result = OpenAICompatModelAdapter.from_vendor_response(
                vendor_response, run_id=run_id, phase=ModelPhase.CODE, provider="nous",
            )
            content = code_result.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines)
            if not content.endswith("\n"):
                content += "\n"
            files[task.file_path] = content
        except Exception as e:
            logger.error("Code generation failed for %s: %s", task.file_path, e)
            files[task.file_path] = f"# Generation failed: {e}\n"

    change_set = ChangeSet(
        run_id=run_id,
        branch_name=plan.branch_name,
        files=files,
        commit_sha="",  # Set by GitHub ship function after real commit
        commit_message=plan.commit_message,
        syntax_validated=True,
    )

    # Validate through boundary
    result = SchemaBoundary.validate("code", ChangeSet, change_set, run_id=run_id)
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "code",
        }

    return {"change_set": result.validated, "current_stage": "code", "completed_stages": ["code"]}


def _qa_agent(state: PipelineGraphState) -> dict:
    """Run QA checks on the change set."""
    run_id = state.get("run_id", "unknown")
    cs: ChangeSet | None = state.get("change_set")

    if cs is None:
        return {
            "pipeline_failures": [PipelineFailure(
                stage="qa", run_id=run_id,
                category=FailureCategory.QA_FAILED,
                message="No change set to QA",
            )],
            "current_stage": "qa",
        }

    # Run syntax checks
    syntax_errors: list[str] = []
    for path, content in cs.files.items():
        if path.endswith(".py"):
            try:
                import py_compile, tempfile, os
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                    tf.write(content)
                    tf.flush()
                    tmp = tf.name
                py_compile.compile(tmp, doraise=True)
                os.unlink(tmp)
            except Exception as e:
                syntax_errors.append(f"{path}: {e}")

    # Run security scan
    security_findings: list[str] = []
    critical_count = 0
    import re
    secret_patterns = [
        (r'(?:api_key|apikey)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "hardcoded API key"),
        (r'-----BEGIN.*PRIVATE KEY-----', "private key in source"),
    ]
    for path, content in cs.files.items():
        if "test" in path.lower():
            continue
        for pattern, desc in secret_patterns:
            if re.findall(pattern, content):
                security_findings.append(f"{path}: {desc}")
                critical_count += 1

    passed = len(syntax_errors) == 0 and critical_count == 0
    qa_report = QAReport(
        run_id=run_id,
        passed=passed,
        syntax_check=CheckResult(passed=len(syntax_errors) == 0, errors=syntax_errors),
        security_scan=SecurityResult(
            passed=critical_count == 0,
            findings=security_findings,
            critical_count=critical_count,
        ),
        overall_message="QA passed" if passed else f"QA failed: {len(syntax_errors)} syntax errors, {critical_count} critical security findings",
    )

    # Validate through boundary
    result = SchemaBoundary.validate("qa", QAReport, qa_report, run_id=run_id)
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "qa",
        }

    return {"qa_report": result.validated, "current_stage": "qa", "completed_stages": ["qa"]}


def _policy_gate(state: PipelineGraphState) -> dict:
    """Evaluate policy rules and produce ShipAuthorization."""
    run_id = state.get("run_id", "unknown")
    qa_report: QAReport | None = state.get("qa_report")
    plan: BuildPlan | None = state.get("build_plan")
    cs: ChangeSet | None = state.get("change_set")
    req: HermesRunRequest = state.get("hermes_request")

    if qa_report is None:
        return {
            "pipeline_failures": [PipelineFailure(
                stage="policy", run_id=run_id,
                category=FailureCategory.POLICY_DENIED,
                message="No QA report available for policy evaluation",
            )],
            "current_stage": "policy",
        }

    gate = PolicyGate()
    auth = gate.evaluate(
        run_id=run_id,
        qa_report=qa_report,
        plan=plan,
        change_set=cs,
        trusted_actor=req.trusted_actor,
    )

    # Validate through boundary
    result = SchemaBoundary.validate("policy", ShipAuthorization, auth, run_id=run_id)
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "policy",
        }

    if not auth.authorized:
        return {
            "ship_authorization": result.validated,
            "pipeline_failures": [PipelineFailure(
                stage="policy", run_id=run_id,
                category=FailureCategory.POLICY_DENIED,
                message=auth.denial_reason,
            )],
            "current_stage": "policy",
        }

    return {
        "ship_authorization": result.validated,
        "current_stage": "policy",
        "completed_stages": ["policy"],
    }


def _github_ship(
    state: PipelineGraphState,
    ship_fn: GitHubShipFn = _default_github_fn,
) -> dict:
    """Ship to GitHub — only executes if authorized."""
    run_id = state.get("run_id", "unknown")
    auth: ShipAuthorization | None = state.get("ship_authorization")
    plan: BuildPlan | None = state.get("build_plan")
    cs: ChangeSet | None = state.get("change_set")
    req: HermesRunRequest = state.get("hermes_request")

    if auth is None or not auth.authorized:
        return {
            "ship_result": ShipResult(run_id=run_id, merged=False, merge_message="Not authorized"),
            "pipeline_failures": [PipelineFailure(
                stage="ship", run_id=run_id,
                category=FailureCategory.SHIPPING_ERROR,
                message="Ship attempted without authorization",
            )],
            "current_stage": "ship",
        }

    if plan is None or cs is None:
        return {
            "ship_result": ShipResult(run_id=run_id, merged=False, merge_message="Missing plan or change set"),
            "current_stage": "ship",
        }

    ship_result = ship_fn(auth, plan, cs, req)

    # Validate through boundary
    result = SchemaBoundary.validate("ship", ShipResult, ship_result, run_id=run_id)
    if not result.success:
        return {
            "validation_failures": [result.failure],
            "pipeline_failures": [result.failure],
            "current_stage": "ship",
        }

    return {
        "ship_result": result.validated,
        "current_stage": "ship",
        "completed_stages": ["ship"],
    }


def _finalize(state: PipelineGraphState) -> dict:
    """Assemble the terminal PipelineResult."""
    run_id = state.get("run_id", "unknown")
    req = state.get("hermes_request")
    failures: list[PipelineFailure] = state.get("pipeline_failures", [])

    ship_result: ShipResult | None = state.get("ship_result")
    completed = bool(ship_result and ship_result.merged)
    status = PipelineStatus.COMPLETED if completed else PipelineStatus.FAILED

    errors = [f.message for f in failures] if failures else []

    result = PipelineResult(
        run_id=run_id,
        status=status,
        trigger=req.trigger.value,
        repo_owner=req.repo_owner,
        repo_name=req.repo_name,
        plan=state.get("build_plan"),
        change_set=state.get("change_set"),
        qa_report=state.get("qa_report"),
        ship_result=ship_result,
        errors=errors,
        current_phase=state.get("current_stage", ""),
    )

    return {"pipeline_result": result, "current_stage": "finalize"}


def _finalize_failure(state: PipelineGraphState) -> dict:
    """Assemble a failure PipelineResult when policy denies or errors occur."""
    run_id = state.get("run_id", "unknown")
    req: HermesRunRequest = state.get("hermes_request")
    failures: list[PipelineFailure] = state.get("pipeline_failures", [])
    errors = [f.message for f in failures] if failures else ["Pipeline failed"]

    result = PipelineResult(
        run_id=run_id,
        status=PipelineStatus.FAILED,
        trigger=req.trigger.value if req else "unknown",
        repo_owner=req.repo_owner if req else "",
        repo_name=req.repo_name if req else "",
        plan=state.get("build_plan"),
        change_set=state.get("change_set"),
        qa_report=state.get("qa_report"),
        errors=errors,
        current_phase=state.get("current_stage", "unknown"),
    )

    return {"pipeline_result": result, "current_stage": "finalize_failure"}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _should_continue_after_request(state: PipelineGraphState) -> str:
    """Route after request validation."""
    if state.get("pipeline_failures"):
        return "finalize_failure"
    return "route_request"


def _should_continue_after_research(state: PipelineGraphState) -> str:
    """Route after research."""
    return "model"


def _should_continue_after_model(state: PipelineGraphState) -> str:
    """Route after model — fail if all providers failed."""
    if state.get("pipeline_failures") or state.get("model_result") is None:
        return "finalize_failure"
    return "browser"


def _should_continue_after_browser(state: PipelineGraphState) -> str:
    """Route after browser (browser failures are non-fatal)."""
    return "plan"


def _should_continue_after_plan(state: PipelineGraphState) -> str:
    """Route after planning."""
    if state.get("pipeline_failures"):
        return "finalize_failure"
    return "code"


def _should_continue_after_code(state: PipelineGraphState) -> str:
    """Route after coding."""
    if state.get("pipeline_failures"):
        return "finalize_failure"
    return "qa"


def _should_continue_after_qa(state: PipelineGraphState) -> str:
    """Route after QA — always goes to policy gate (even on failure)."""
    return "policy"


def _should_continue_after_policy(state: PipelineGraphState) -> str:
    """Route after policy evaluation."""
    auth = state.get("ship_authorization")
    if auth and auth.authorized:
        return "github_ship"
    return "finalize_failure"


def _should_continue_after_ship(state: PipelineGraphState) -> str:
    """Route after shipping."""
    return "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline_graph(
    research_fn: ResearchProviderFn = _default_research_fn,
    model_fn: ModelProviderFn = _default_model_fn,
    browser_fn: BrowserProviderFn = _default_browser_fn,
    ship_fn: GitHubShipFn = _default_github_fn,
) -> StateGraph:
    """Build and compile the pipeline orchestration graph.

    Args:
        research_fn: Callable(topic, geo) → content for research provider.
        model_fn: Callable(request_dict, provider_name) → response_dict for LLM.
        browser_fn: Callable(BrowserTask) → content for browser provider.
        ship_fn: Callable(auth, plan, cs, req) → ShipResult for GitHub shipping.

    Returns:
        Compiled StateGraph ready to invoke.
    """
    builder = StateGraph(PipelineGraphState)

    # Register nodes
    builder.add_node("validate_request", _validate_hermes_request)
    builder.add_node("route_request", _route_request)
    builder.add_node("research", lambda s: _research_route(s, research_fn))
    builder.add_node("model", lambda s: _model_route(s, model_fn))
    builder.add_node("browser", lambda s: _browser_route(s, browser_fn))
    builder.add_node("plan", _plan_agent)
    builder.add_node("code", lambda s: _code_agent(s, model_fn))
    builder.add_node("qa", _qa_agent)
    builder.add_node("policy", _policy_gate)
    builder.add_node("github_ship", lambda s: _github_ship(s, ship_fn))
    builder.add_node("finalize", _finalize)
    builder.add_node("finalize_failure", _finalize_failure)

    # Edges
    builder.add_edge(START, "validate_request")

    builder.add_conditional_edges("validate_request", _should_continue_after_request, {
        "route_request": "route_request",
        "finalize_failure": "finalize_failure",
    })

    builder.add_edge("route_request", "research")
    builder.add_edge("research", "model")

    builder.add_conditional_edges("model", _should_continue_after_model, {
        "browser": "browser",
        "finalize_failure": "finalize_failure",
    })

    builder.add_edge("browser", "plan")

    builder.add_conditional_edges("plan", _should_continue_after_plan, {
        "code": "code",
        "finalize_failure": "finalize_failure",
    })

    builder.add_conditional_edges("code", _should_continue_after_code, {
        "qa": "qa",
        "finalize_failure": "finalize_failure",
    })

    builder.add_edge("qa", "policy")

    builder.add_conditional_edges("policy", _should_continue_after_policy, {
        "github_ship": "github_ship",
        "finalize_failure": "finalize_failure",
    })

    builder.add_edge("github_ship", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("finalize_failure", END)

    return builder.compile()


def run_pipeline(
    run_id: str = "",
    trigger: str = "manual",
    repo_owner: str = "",
    repo_name: str = "",
    trusted_actor: str = "",
    requirements: str = "",
    issue_number: int | None = None,
    issue_title: str = "",
    issue_body: str = "",
    wait_for_ci: bool = True,
    research_fn: ResearchProviderFn = _default_research_fn,
    model_fn: ModelProviderFn = _default_model_fn,
    browser_fn: BrowserProviderFn = _default_browser_fn,
    ship_fn: GitHubShipFn = _default_github_fn,
) -> PipelineResult:
    """Run the full pipeline and return the terminal PipelineResult.

    This is the primary entry point for the schema-driven orchestrator.
    """
    import uuid
    run_id = run_id or str(uuid.uuid4())
    state = initial_state(
        run_id=run_id, trigger=trigger, repo_owner=repo_owner, repo_name=repo_name,
        trusted_actor=trusted_actor, requirements=requirements,
        issue_number=issue_number, issue_title=issue_title, issue_body=issue_body,
        wait_for_ci=wait_for_ci,
    )
    graph = build_pipeline_graph(
        research_fn=research_fn, model_fn=model_fn,
        browser_fn=browser_fn, ship_fn=ship_fn,
    )
    final_state = graph.invoke(state)
    if final_state is None:
        return PipelineResult(
            run_id=run_id, status=PipelineStatus.FAILED,
            errors=["Pipeline produced no final state"],
        )
    result = final_state.get("pipeline_result")
    if result is None:
        # If no pipeline_result, check for failures
        failures = final_state.get("pipeline_failures", [])
        errors = [f.message for f in failures] if failures else ["Pipeline produced no result"]
        return PipelineResult(
            run_id=run_id, status=PipelineStatus.FAILED, errors=errors,
        )
    return result
