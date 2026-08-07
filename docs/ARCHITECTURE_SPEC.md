# Chief Software Architect Specification
## Multi-Agent Schema-Driven Auto-Build Pipeline (PLAN → CODE → QA → SHIP)

**Document Version:** 1.0
**Date:** 2026-08-07
**Status:** Implementation-Ready
**Repository:** Nietzsche-Ubermensch/langgraph-orchestrator

---

## 1. Architectural Invariant

Every component boundary is a validated schema. No component communicates
with another through natural language, untyped dicts, or implicit
conventions. The system is a typed graph:

```
Agent_i —[validated schema]—> Agent_{i+1}
```

For every vendor-specific interface:

```
vendor schema ↔ canonical pipeline schema (bidirectional adapter)
```

This invariant is enforced at runtime: each schema is a Pydantic model that
must validate before the downstream component is allowed to execute. A
validation failure halts the pipeline and records the error in the state
store — it does not propagate as a Python exception.

---

## 2. System Topology

```
                         ┌───────────────┐
                         │    HERMES     │
                         └───────┬───────┘
                                 │
                     HermesRunRequestSchema v1.0
                                 │
                                 ▼
                     ┌────────────────────┐
                     │ LANGGRAPH ROUTER   │
                     └─────────┬──────────┘
                               │
                       RoutingPlanSchema v1.0
                  ┌────────────┼─────────────┐
                  │            │             │
                  ▼            ▼             ▼
             RESEARCH        MODEL        BROWSER
              ROUTER         ROUTER        ROUTER
                  │            │             │
        ResearchRequest   ModelTask     BrowserTask
             Schema v1.0    Schema v1.0    Schema v1.0
                  │            │             │
       ┌──────────┼────┐       │       ┌─────┴────────┐
       ▼          ▼    ▼       ▼       ▼              ▼
 BrightData  Oxylabs Venice  OpenRouter Camofox   BrightData
    MCP        MCP/API       /OpenAI/     Browser     Browser
                              /Grok
       └──────────┬────┘       │       └─────┬────────┘
                  │            │             │
            EvidenceSchema  ModelResult   BrowserResult
              v1.0          Schema v1.0    Schema v1.0
                  │            │             │
                  └────────────┼─────────────┘
                               ▼
                         PLAN AGENT
                               │
                         BuildPlanSchema v1.0
                               ▼
                         CODE AGENT
                               │
                         ChangeSetSchema v1.0
                               ▼
                          QA AGENT
                               │
                          QAReportSchema v1.0
                               ▼
                         POLICY GATE
                               │
                    ShipAuthorizationSchema v1.0
                               ▼
                         GITHUB AGENT
                               │
                         ShipResultSchema v1.0
                               ▼
                       PipelineResultSchema v1.0
                               │
                               ▼
                             HERMES
```

---

## 3. Schema Catalog

### 3.1 Ingress Schemas

#### HermesRunRequestSchema v1.0

```xml
<schema name="HermesRunRequestSchema" version="1.0" owner="Hermes">
  <fields>
    <field name="run_id" type="str" required="true" description="UUID4 pipeline run identifier" />
    <field name="trigger" type="TriggerType" required="true" enum="manual|issue|pr|workflow_dispatch|webhook|cron" />
    <field name="source" type="str" required="true" default="cli" description="Originating surface" />
    <field name="repo_owner" type="str" required="true" />
    <field name="repo_name" type="str" required="true" />
    <field name="issue_number" type="int" required="false" description="GitHub issue that triggered the run" />
    <field name="issue_title" type="str" required="false" />
    <field name="issue_body" type="str" required="false" />
    <field name="requirements" type="str" required="false" description="Free-text requirements for manual triggers" />
    <field name="base_branch" type="str" required="false" default="main" />
    <field name="wait_for_ci" type="bool" required="false" default="true" />
    <field name="ci_timeout" type="float" required="false" default="600.0" unit="seconds" />
    <field name="trusted_actor" type="str" required="true" description="GitHub username of the triggering actor" />
    <field name="idempotency_key" type="str" required="false" description="Deduplication key for webhook triggers" />
  </fields>
  <validation>
    - At least one of (issue_number, requirements) must be present.
    - trusted_actor must be in the configured trusted_actors list.
    - repo_owner/repo_name must resolve to an existing, non-archived repository.
    - idempotency_key, if present, is checked against the state store; duplicate runs are rejected with HTTP 409 semantics.
  </validation>
  <produced_by>Hermes</produced_by>
  <consumed_by>LangGraphRouter</consumed_by>
</schema>
```

#### RoutingPlanSchema v1.0

```xml
<schema name="RoutingPlanSchema" version="1.0" owner="LangGraphRouter">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="research_needed" type="bool" required="true" description="Whether web research is needed before planning" />
    <field name="research_topics" type="list[str]" required="false" description="Extracted topics from the requirement" />
    <field name="model_task" type="ModelTaskSchema" required="true" description="LLM task for plan/code generation" />
    <field name="browser_needed" type="bool" required="false" description="Whether browser rendering is needed for research" />
    <field name="browser_tasks" type="list[BrowserTaskSchema]" required="false" />
    <field name="geo" type="str" required="false" description="ISO country code for geo-located research" />
  </fields>
  <validation>
    - research_needed=false implies research_topics is empty.
    - If browser_needed=true, browser_tasks must have at least one entry.
  </validation>
  <produced_by>LangGraphRouter</produced_by>
  <consumed_by>ResearchRouter, ModelRouter, BrowserRouter</consumed_by>
</schema>
```

### 3.2 Research Layer Schemas

#### ResearchRequestSchema v1.0

```xml
<schema name="ResearchRequestSchema" version="1.0" owner="ResearchRouter">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="topics" type="list[str]" required="true" description="Search queries to execute" />
    <field name="max_results_per_topic" type="int" required="false" default="3" />
    <field name="max_chars_per_page" type="int" required="false" default="3000" />
    <field name="geo" type="str" required="false" description="ISO country code" />
    <field name="render_javascript" type="bool" required="false" default="false" />
    <field name="failover_tier" type="int" required="false" default="1" enum="1|2|3" description="Which provider tier to start at (1=BrightData, 2=Oxylabs, 3=Venice)" />
  </fields>
  <validation>
    - topics must be non-empty.
    - failover_tier determines starting provider; failover proceeds 1→2→3.
  </validation>
  <produced_by>ResearchRouter</produced_by>
  <consumed_by>BrightDataAdapter, OxylabsAdapter, VeniceAdapter</consumed_by>
</schema>
```

#### EvidenceSchema v1.0

```xml
<schema name="EvidenceSchema" version="1.0" owner="ResearchLayer">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="sources" type="list[EvidenceSource]" required="true" />
    <field name="combined_context" type="str" required="true" description="Merged text suitable for LLM context injection" />
    <field name="providers_used" type="list[ProviderName]" required="true" enum="brightdata|oxylabs|venice|camofox" />
    <field name="total_chars" type="int" required="true" />
    <field name="failover_events" type="list[FailoverEvent]" required="false" description="Record of provider failovers" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <sub_schema name="EvidenceSource">
    <field name="url" type="str" required="true" />
    <field name="title" type="str" required="true" />
    <field name="content" type="str" required="true" />
    <field name="provider" type="ProviderName" required="true" />
    <field name="content_type" type="str" required="true" enum="markdown|json|html|text" />
    <field name="chars" type="int" required="true" />
  </sub_schema>
  <sub_schema name="FailoverEvent">
    <field name="from_provider" type="ProviderName" required="true" />
    <field name="to_provider" type="ProviderName" required="true" />
    <field name="reason" type="str" required="true" enum="blocked|error|timeout|rate_limit" />
    <field name="timestamp" type="datetime" required="true" />
  </sub_schema>
  <validation>
    - combined_context must be non-empty if sources is non-empty.
    - Each source's content must not exceed max_chars_per_page from the request.
  </validation>
  <produced_by>BrightDataAdapter, OxylabsAdapter, VeniceAdapter</produced_by>
  <consumed_by>PLAN</consumed_by>
</schema>
```

### 3.3 Model Layer Schemas

#### ModelTaskSchema v1.0

```xml
<schema name="ModelTaskSchema" version="1.0" owner="ModelRouter">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="phase" type="ModelPhase" required="true" enum="plan|code" description="Which pipeline phase needs the LLM" />
    <field name="system_prompt" type="str" required="true" />
    <field name="user_prompt" type="str" required="true" />
    <field name="temperature" type="float" required="false" default="0.2" />
    <field name="max_tokens" type="int" required="false" default="4096" />
    <field name="response_format" type="str" required="false" enum="text|json_object" default="text" />
    <field name="preferred_provider" type="str" required="false" enum="openrouter|openai|grok|nous" description="Preferred model provider" />
    <field name="fallback_providers" type="list[str]" required="false" default="[]" />
  </fields>
  <validation>
    - If response_format=json_object, the response must parse as valid JSON.
    - If the preferred provider fails, fallback_providers are tried in order.
  </validation>
  <produced_by>ModelRouter</produced_by>
  <consumed_by>OpenRouterAdapter, OpenAIAdapter, GrokAdapter, NousAdapter</consumed_by>
</schema>
```

#### ModelResultSchema v1.0

```xml
<schema name="ModelResultSchema" version="1.0" owner="ModelLayer">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="phase" type="ModelPhase" required="true" enum="plan|code" />
    <field name="content" type="str" required="true" description="Raw LLM response text" />
    <field name="parsed" type="dict" required="false" description="Parsed JSON if response_format=json_object" />
    <field name="provider" type="str" required="true" enum="openrouter|openai|grok|nous" />
    <field name="model" type="str" required="true" description="Model name used" />
    <field name="tokens_input" type="int" required="false" />
    <field name="tokens_output" type="int" required="false" />
    <field name="latency_ms" type="int" required="false" />
    <field name="failover_used" type="bool" required="false" default="false" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <validation>
    - content must be non-empty.
    - If response_format was json_object, parsed must be present and valid.
  </validation>
  <produced_by>OpenRouterAdapter, OpenAIAdapter, GrokAdapter, NousAdapter</produced_by>
  <consumed_by>PLAN (phase=plan), CODE (phase=code)</consumed_by>
</schema>
```

### 3.4 Browser Layer Schemas

#### BrowserTaskSchema v1.0

```xml
<schema name="BrowserTaskSchema" version="1.0" owner="BrowserRouter">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="url" type="str" required="true" />
    <field name="task_prompt" type="str" required="false" description="Natural-language task for browser agent" />
    <field name="output_format" type="str" required="false" default="markdown" enum="markdown|json|html" />
    <field name="render_javascript" type="bool" required="false" default="true" />
    <field name="geo" type="str" required="false" />
    <field name="session_id" type="str" required="false" description="Camofox session ID for reuse" />
    <field name="preferred_backend" type="str" required="false" enum="camofox|brightdata|oxylabs" />
  </fields>
  <validation>
    - url must be a valid HTTP(S) URL.
    - If task_prompt is absent, the task is a simple page extraction.
  </validation>
  <produced_by>BrowserRouter</produced_by>
  <consumed_by>CamofoxAdapter, BrightDataBrowserAdapter, OxylabsBrowserAdapter</consumed_by>
</schema>
```

#### BrowserResultSchema v1.0

```xml
<schema name="BrowserResultSchema" version="1.0" owner="BrowserLayer">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="content" type="str" required="true" description="Extracted page content" />
    <field name="content_type" type="str" required="true" enum="markdown|json|html|accessibility_tree" />
    <field name="backend" type="str" required="true" enum="camofox|brightdata|oxylabs" />
    <field name="url" type="str" required="true" />
    <field name="screenshot_path" type="str" required="false" description="Path to screenshot if captured" />
    <field name="session_id" type="str" required="false" description="Camofox session ID for reuse" />
    <field name="latency_ms" type="int" required="false" />
    <field name="failover_used" type="bool" required="false" default="false" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <validation>
    - content must be non-empty.
  </validation>
  <produced_by>CamofoxAdapter, BrightDataBrowserAdapter, OxylabsBrowserAdapter</produced_by>
  <consumed_by>PLAN, CODE, QA</consumed_by>
</schema>
```

### 3.5 Pipeline Core Schemas

#### BuildPlanSchema v1.0

```xml
<schema name="BuildPlanSchema" version="1.0" owner="PLAN">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="summary" type="str" required="true" description="One-paragraph plan summary" />
    <field name="repo" type="str" required="true" />
    <field name="branch_name" type="str" required="true" pattern="auto-build/[a-z0-9/-]+" />
    <field name="commit_message" type="str" required="true" description="Conventional commits format" />
    <field name="pr_title" type="str" required="true" max_length="72" />
    <field name="pr_body" type="str" required="true" description="Markdown PR body with Summary/Changes/Testing sections" />
    <field name="tasks" type="list[BuildTask]" required="true" min_items="1" />
    <field name="research_context" type="str" required="false" description="EvidenceSchema combined_context that informed the plan" />
    <field name="estimated_files" type="int" required="true" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <sub_schema name="BuildTask">
    <field name="id" type="str" required="true" description="8-char UUID prefix" />
    <field name="description" type="str" required="true" />
    <field name="file_path" type="str" required="true" />
    <field name="action" type="str" required="true" enum="create|modify|delete" />
    <field name="content_preview" type="str" required="false" description="First 200 chars of intended content" />
    <field name="status" type="str" required="true" default="pending" enum="pending|in_progress|completed|failed" />
    <field name="error" type="str" required="false" />
  </sub_schema>
  <validation>
    - branch_name must start with "auto-build/".
    - Each task's file_path must be a valid relative path within the repo.
    - pr_title must not exceed 72 characters.
  </validation>
  <produced_by>PLAN</produced_by>
  <consumed_by>CODE</consumed_by>
</schema>
```

#### ChangeSetSchema v1.0

```xml
<schema name="ChangeSetSchema" version="1.0" owner="CODE">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="branch_name" type="str" required="true" />
    <field name="files" type="dict[str, str]" required="true" description="Mapping of file_path → file_content" />
    <field name="commit_sha" type="str" required="true" description="SHA of the commit created via Git Data API" />
    <field name="commit_message" type="str" required="true" />
    <field name="syntax_validated" type="bool" required="true" description="Whether syntax checks passed for all files" />
    <field name="validation_errors" type="list[str]" required="false" description="Any syntax validation errors found" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <validation>
    - files must be non-empty.
    - commit_sha must be a 40-character hex string.
    - syntax_validated must be true (otherwise the pipeline halts at QA).
  </validation>
  <produced_by>CODE</produced_by>
  <consumed_by>QA</consumed_by>
</schema>
```

#### QAReportSchema v1.0

```xml
<schema name="QAReportSchema" version="1.0" owner="QA">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="passed" type="bool" required="true" />
    <field name="syntax_check" type="CheckResult" required="true" />
    <field name="security_scan" type="SecurityResult" required="true" />
    <field name="lint_check" type="CheckResult" required="false" description="Optional — skipped if ruff not installed" />
    <field name="type_check" type="CheckResult" required="false" description="Optional — skipped if mypy not installed" />
    <field name="test_results" type="TestResult" required="false" description="Optional — skipped if no test files" />
    <field name="overall_message" type="str" required="true" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <sub_schema name="CheckResult">
    <field name="passed" type="bool" required="true" />
    <field name="errors" type="list[str]" required="true" default="[]" />
  </sub_schema>
  <sub_schema name="SecurityResult">
    <field name="passed" type="bool" required="true" />
    <field name="findings" type="list[str]" required="true" default="[]" description="Security findings (advisory unless CRITICAL)" />
    <field name="critical_count" type="int" required="true" default="0" />
  </sub_schema>
  <sub_schema name="TestResult">
    <field name="run" type="bool" required="true" description="Whether tests were executed" />
    <field name="passed" type="int" required="true" default="0" />
    <field name="failed" type="int" required="true" default="0" />
  </sub_schema>
  <validation>
    - passed is true only if syntax_check.passed is true and security_scan.critical_count is 0.
    - If passed is false, the pipeline halts — Policy Gate does not authorize shipping.
  </validation>
  <produced_by>QA</produced_by>
  <consumed_by>PolicyGate</consumed_by>
</schema>
```

#### ShipAuthorizationSchema v1.0

```xml
<schema name="ShipAuthorizationSchema" version="1.0" owner="PolicyGate">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="authorized" type="bool" required="true" description="Whether shipping is authorized" />
    <field name="qa_report" type="QAReportSchema" required="true" description="The QA report being evaluated" />
    <field name="policy_checks" type="list[PolicyCheck]" required="true" description="Results of each policy rule" />
    <field name="denial_reason" type="str" required="false" description="If not authorized, why" />
    <field name="merge_method" type="str" required="false" default="squash" enum="merge|squash|rebase" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <sub_schema name="PolicyCheck">
    <field name="rule" type="str" required="true" description="Policy rule name" />
    <field name="passed" type="bool" required="true" />
    <field name="message" type="str" required="false" />
  </sub_schema>
  <validation>
    - authorized is true only if all policy_checks have passed=true.
    - If authorized is false, denial_reason must be present.
    - Policy rules:
      1. qa_passed: QAReportSchema.passed must be true.
      2. no_critical_security: SecurityResult.critical_count must be 0.
      3. trusted_actor: The triggering actor must be in the trusted_actors list.
      4. branch_naming: BuildPlanSchema.branch_name must match "auto-build/" prefix.
      5. file_count_limit: ChangeSetSchema.files must not exceed 50 files.
  </validation>
  <produced_by>PolicyGate</produced_by>
  <consumed_by>GitHubAgent</consumed_by>
</schema>
```

#### ShipResultSchema v1.0

```xml
<schema name="ShipResultSchema" version="1.0" owner="GitHubAgent">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="pr_number" type="int" required="false" description="None if PR creation failed" />
    <field name="pr_url" type="str" required="false" />
    <field name="merged" type="bool" required="true" default="false" />
    <field name="merge_message" type="str" required="true" default="" />
    <field name="branch_name" type="str" required="true" />
    <field name="commit_sha" type="str" required="true" />
    <field name="branch_deleted" type="bool" required="false" default="false" description="Whether the feature branch was deleted after merge" />
    <field name="timestamp" type="datetime" required="true" />
  </fields>
  <validation>
    - If merged is true, pr_number and pr_url must be present.
    - If merged is false, merge_message must explain why.
  </validation>
  <produced_by>GitHubAgent</produced_by>
  <consumed_by>LangGraph (for PipelineResultSchema)</consumed_by>
</schema>
```

#### PipelineResultSchema v1.0

```xml
<schema name="PipelineResultSchema" version="1.0" owner="LangGraph">
  <fields>
    <field name="run_id" type="str" required="true" />
    <field name="status" type="PipelineStatus" required="true" enum="pending|planning|coding|qa|shipping|completed|failed|cancelled" />
    <field name="trigger" type="str" required="true" enum="manual|issue|pr|workflow_dispatch|webhook|cron" />
    <field name="repo_owner" type="str" required="true" />
    <field name="repo_name" type="str" required="true" />
    <field name="plan" type="BuildPlanSchema" required="false" />
    <field name="change_set" type="ChangeSetSchema" required="false" />
    <field name="qa_report" type="QAReportSchema" required="false" />
    <field name="ship_result" type="ShipResultSchema" required="false" />
    <field name="errors" type="list[str]" required="true" default="[]" />
    <field name="current_phase" type="str" required="true" />
    <field name="created_at" type="datetime" required="true" />
    <field name="updated_at" type="datetime" required="true" />
    <field name="duration_ms" type="int" required="false" description="Total pipeline wall-clock time" />
  </fields>
  <validation>
    - If status=completed, ship_result.merged must be true.
    - If status=failed, errors must be non-empty.
  </validation>
  <produced_by>LangGraph</produced_by>
  <consumed_by>Hermes</consumed_by>
</schema>
```

---

## 4. Vendor Adapter Mappings

### 4.1 Research Providers

```xml
<adapter_mappings category="research">
  <!-- BrightData MCP → EvidenceSchema -->
  <adapter vendor="BrightData" direction="bidirectional">
    <canonical_to_vendor>
      ResearchRequestSchema.topics[0] → MCP tool "search_engine" args: {query: topics[0]}
      ResearchRequestSchema.topics[0] → MCP tool "scrape_as_markdown" args: {url: topics[0]}
    </canonical_to_vendor>
    <vendor_to_canonical>
      MCP response text → EvidenceSource.content
      MCP response metadata → EvidenceSource.provider = "brightdata"
      MCP response content_type → EvidenceSource.content_type = "markdown"
    </vendor_to_canonical>
    <error_mapping>
      MCP "blocked" → FailoverEvent(reason="blocked", to_provider="oxylabs")
      MCP timeout → FailoverEvent(reason="timeout", to_provider="oxylabs")
    </error_mapping>
  </adapter>

  <!-- Oxylabs AI Studio → EvidenceSchema -->
  <adapter vendor="Oxylabs" direction="bidirectional">
    <canonical_to_vendor>
      ResearchRequestSchema.topics[0] → POST /ai-search {query: topics[0], limit: max_results, return_content: true}
      ResearchRequestSchema.topics[0] → POST /ai-scraper {url: topics[0], output_format: "markdown"}
    </canonical_to_vendor>
    <vendor_to_canonical>
      API response results[].content → EvidenceSource.content
      API response results[].url → EvidenceSource.url
      API response results[].title → EvidenceSource.title
      API response → EvidenceSource.provider = "oxylabs"
    </vendor_to_canonical>
    <error_mapping>
      HTTP 401 → FailoverEvent(reason="error", to_provider="venice")
      HTTP 429 → FailoverEvent(reason="rate_limit", to_provider="venice")
    </error_mapping>
  </adapter>

  <!-- Venice → EvidenceSchema -->
  <adapter vendor="Venice" direction="bidirectional">
    <canonical_to_vendor>
      ResearchRequestSchema.topics[0] → POST /api/v1/chat/completions {model: "llama-3.3-70b", messages: [{role: "user", content: topics[0]}], enable_web_scraping: true}
    </canonical_to_vendor>
    <vendor_to_canonical>
      API response choices[0].message.content → EvidenceSource.content
      API response → EvidenceSource.provider = "venice"
      API response → EvidenceSource.content_type = "text"
    </vendor_to_canonical>
    <error_mapping>
      HTTP 5xx → FailoverEvent(reason="error", to_provider=none)
      Venice is the last tier — no failover.
    </error_mapping>
  </adapter>
</adapter_mappings>
```

### 4.2 Model Providers

```xml
<adapter_mappings category="model">
  <!-- OpenRouter → ModelResultSchema -->
  <adapter vendor="OpenRouter" direction="bidirectional">
    <canonical_to_vendor>
      ModelTaskSchema.system_prompt → OpenAI messages[0] = {role: "system", content: system_prompt}
      ModelTaskSchema.user_prompt → OpenAI messages[1] = {role: "user", content: user_prompt}
      ModelTaskSchema.temperature → OpenAI temperature
      ModelTaskSchema.max_tokens → OpenAI max_tokens
      ModelTaskSchema.response_format → OpenAI response_format (if json_object)
    </canonical_to_vendor>
    <vendor_to_canonical>
      OpenAI response choices[0].message.content → ModelResultSchema.content
      OpenAI response model → ModelResultSchema.model
      OpenAI usage.prompt_tokens → ModelResultSchema.tokens_input
      OpenAI usage.completion_tokens → ModelResultSchema.tokens_output
    </vendor_to_canonical>
    <error_mapping>
      HTTP 429 → try next provider in fallback_providers
      HTTP 5xx → try next provider in fallback_providers
    </error_mapping>
  </adapter>

  <!-- OpenAI → ModelResultSchema (same shape as OpenRouter) -->
  <adapter vendor="OpenAI" direction="bidirectional" ref="OpenRouter" />

  <!-- Grok (xAI) → ModelResultSchema -->
  <adapter vendor="Grok" direction="bidirectional">
    <canonical_to_vendor>
      Same OpenAI-compatible shape, base_url = "https://api.x.ai/v1"
    </canonical_to_vendor>
    <vendor_to_canonical>
      Same as OpenRouter adapter
    </vendor_to_canonical>
  </adapter>

  <!-- Nous Portal → ModelResultSchema -->
  <adapter vendor="Nous" direction="bidirectional">
    <canonical_to_vendor>
      Same OpenAI-compatible shape, base_url from auth.json inference_base_url
      Auth via agent_key from auth.json (auto-refreshed)
    </canonical_to_vendor>
    <vendor_to_canonical>
      Same as OpenRouter adapter
    </vendor_to_canonical>
  </adapter>
</adapter_mappings>
```

### 4.3 Browser Providers

```xml
<adapter_mappings category="browser">
  <!-- Camofox → BrowserResultSchema -->
  <adapter vendor="Camofox" direction="bidirectional">
    <canonical_to_vendor>
      BrowserTaskSchema.url → POST /start {userId: run_id}
      BrowserTaskSchema.url → POST /tabs {userId, sessionKey, url}
      BrowserTaskSchema.task_prompt → POST /act {kind: "type", ref: ...} (if task_prompt)
    </canonical_to_vendor>
    <vendor_to_canonical>
      GET /snapshot response → BrowserResultSchema.content (accessibility tree)
      sessionKey → BrowserResultSchema.session_id
    </vendor_to_canonical>
    <error_mapping>
      CamofoxError → failover to BrightData browser
      Health check fail → failover to BrightData browser
    </error_mapping>
  </adapter>

  <!-- BrightData Browser → BrowserResultSchema -->
  <adapter vendor="BrightData" direction="bidirectional">
    <canonical_to_vendor>
      BrowserTaskSchema.url → MCP tool "scraping_browser_navigate" args: {url}
    </canonical_to_vendor>
    <vendor_to_canonical>
      MCP response → BrowserResultSchema.content = "markdown"
      MCP response → BrowserResultSchema.backend = "brightdata"
    </vendor_to_canonical>
    <error_mapping>
      MCP error → failover to Oxylabs browser agent
    </error_mapping>
  </adapter>

  <!-- Oxylabs Browser Agent → BrowserResultSchema -->
  <adapter vendor="Oxylabs" direction="bidirectional">
    <canonical_to_vendor>
      BrowserTaskSchema.url → POST /ai-browser-agent {url, task_prompt, output_format}
    </canonical_to_vendor>
    <vendor_to_canonical>
      API response content → BrowserResultSchema.content
      API response → BrowserResultSchema.backend = "oxylabs"
    </vendor_to_canonical>
  </adapter>
</adapter_mappings>
```

### 4.4 GitHub Agent

```xml
<adapter_mappings category="github">
  <adapter vendor="GitHub" direction="bidirectional">
    <canonical_to_vendor>
      ShipAuthorizationSchema → GitHubAgent.ship()
      BuildPlanSchema.branch_name → GitHub Git Data API: create branch ref
      ChangeSetSchema.files → GitHub Git Data API: create blobs + tree + commit
      BuildPlanSchema.pr_title → GitHub Pulls API: create PR
      ShipAuthorizationSchema.merge_method → GitHub Pulls API: merge
    </canonical_to_vendor>
    <vendor_to_canonical>
      GitHub PR response.number → ShipResultSchema.pr_number
      GitHub PR response.html_url → ShipResultSchema.pr_url
      GitHub merge response.sha → ShipResultSchema.commit_sha (post-merge)
      GitHub merge response.merged → ShipResultSchema.merged
    </vendor_to_canonical>
    <error_mapping>
      HTTP 422 (PR already exists) → reuse existing PR (deduplication)
      HTTP 401 on check-runs → ShipResultSchema.merged = false, merge_message = "cannot read CI checks"
      HTTP 409 (merge conflict) → ShipResultSchema.merged = false, merge_message = "merge conflict"
    </error_mapping>
  </adapter>
</adapter_mappings>
```

---

## 5. MCP Integration Pattern

MCP tools do not bypass the schema system. Their inputSchema and response
are adapted into canonical pipeline schemas:

```xml
<mcp_integration>
  <!-- BrightData MCP: tool inputSchema → ResearchRequestSchema → MCP invocation → EvidenceSchema -->
  <flow name="brightdata_research">
    <step name="adapt_in">
      ResearchRequestSchema.topics[0] → MCP tool "search_engine" inputSchema: {query: "..."}
    </step>
    <step name="invoke">
      MCPStdioClient.call_tool("search_engine", {query: topics[0]})
    </step>
    <step name="adapt_out">
      MCP response text → EvidenceSource.content
      MCP response → EvidenceSource.provider = "brightdata"
      EvidenceSource → EvidenceSchema.sources[]
    </step>
  </flow>

  <!-- Oxylabs MCP: same pattern -->
  <flow name="oxylabs_research">
    <step name="adapt_in">
      ResearchRequestSchema.topics[0] → MCP tool "ai_search" inputSchema: {query: "...", limit: 3}
    </step>
    <step name="invoke">
      MCPStdioClient.call_tool("ai_search", {query: topics[0], limit: 3})
    </step>
    <step name="adapt_out">
      MCP response text → EvidenceSource.content
      MCP response → EvidenceSource.provider = "oxylabs"
    </step>
  </flow>

  <!-- Hermes plugin tools follow the same pattern -->
  <flow name="hermes_tool">
    <step name="adapt_in">
      HermesRunRequestSchema → Hermes plugin tool inputSchema
    </step>
    <step name="invoke">
      Hermes orchestrator dispatches to the plugin
    </step>
    <step name="adapt_out">
      Plugin response → PipelineResultSchema
    </step>
  </flow>
</mcp_integration>
```

---

## 6. State Management and Persistence

### 6.1 State Store

```xml
<state_management>
  <store type="SQLite" path="~/.hermes/auto_build/pipeline_state.db">
    <table name="pipeline_runs">
      <column name="run_id" type="TEXT PRIMARY KEY" />
      <column name="status" type="TEXT NOT NULL" indexed="true" />
      <column name="trigger" type="TEXT NOT NULL" />
      <column name="issue_number" type="INTEGER" indexed="true" />
      <column name="repo_full_name" type="TEXT" />
      <column name="created_at" type="TEXT NOT NULL" />
      <column name="updated_at" type="TEXT NOT NULL" />
      <column name="state_json" type="TEXT NOT NULL" description="Full PipelineState as JSON" />
    </table>
  </store>

  <persistence_contract>
    - State is saved after every phase transition (before and after each agent runs).
    - The state_json column contains the full PipelineResultSchema serialized as JSON.
    - Resume: load by run_id, check status, reset to last completed phase.
    - Idempotency: idempotency_key (if present) is checked before creating a new run.
    - Audit: all runs are retained; deletion is a manual operation.
  </persistence_contract>
</state_management>
```

### 6.2 State Transition Diagram

```xml
<state_transitions>
  <transition from="pending" to="planning" trigger="LangGraphRouter dispatches to PLAN" />
  <transition from="planning" to="coding" trigger="BuildPlanSchema validates successfully" />
  <transition from="planning" to="failed" trigger="PLAN error or schema validation failure" />
  <transition from="coding" to="qa" trigger="ChangeSetSchema validates (syntax_validated=true)" />
  <transition from="coding" to="failed" trigger="CODE error or syntax validation failure" />
  <transition from="qa" to="shipping" trigger="QAReportSchema.passed=true AND PolicyGate authorizes" />
  <transition from="qa" to="failed" trigger="QAReportSchema.passed=false OR PolicyGate denies" />
  <transition from="shipping" to="completed" trigger="ShipResultSchema.merged=true" />
  <transition from="shipping" to="failed" trigger="ShipResultSchema.merged=false (PR created but not merged)" />
  <transition from="*" to="cancelled" trigger="User or system cancellation" />

  <resume_rules>
    <rule status="failed" with="commit_sha AND files_committed" action="reset to qa, clear errors" />
    <rule status="failed" with="plan AND NOT commit_sha" action="reset to coding, clear errors" />
    <rule status="failed" with="NOT plan" action="reset to pending, full restart" />
    <rule status="completed" action="no resume — terminal state" />
    <rule status="cancelled" action="no resume — terminal state" />
  </resume_rules>
</state_transitions>
```

---

## 7. Failure Handling and Retry Semantics

```xml
<failure_handling>
  <!-- Research Layer: 3-tier failover -->
  <layer name="research" max_retries="2">
    <failover_chain>
      <tier ordinal="1" provider="brightdata" on_failure="oxylabs" reasons="blocked|error|timeout" />
      <tier ordinal="2" provider="oxylabs" on_failure="venice" reasons="error|rate_limit|timeout" />
      <tier ordinal="3" provider="venice" on_failure="none" reasons="*" description="last resort — no further failover" />
    </failover_chain>
    <retry_policy>
      - Exponential backoff: base_delay * 2^(attempt-1), max 30s
      - Retries within a tier: 3 (network errors only)
      - Tier failover: immediate (no backoff between tiers)
    </retry_policy>
  </layer>

  <!-- Model Layer: provider failover -->
  <layer name="model" max_retries="3">
    <failover_chain>
      <provider name="preferred" on_failure="fallback[0]" />
      <provider name="fallback[0]" on_failure="fallback[1]" />
      <provider name="fallback[1]" on_failure="none" />
    </failover_chain>
    <retry_policy>
      - Exponential backoff: 2s, 4s, 8s
      - Retry on: NetworkError, TimeoutException, HTTP 5xx
      - No retry on: HTTP 400 (bad request — schema issue), HTTP 401 (auth)
    </retry_policy>
  </layer>

  <!-- Browser Layer: backend failover -->
  <layer name="browser" max_retries="2">
    <failover_chain>
      <backend name="camofox" on_failure="brightdata" reasons="error|health_fail" />
      <backend name="brightdata" on_failure="oxylabs" reasons="error|blocked" />
      <backend name="oxylabs" on_failure="none" />
    </failover_chain>
  </layer>

  <!-- Pipeline Core: phase-level retry -->
  <layer name="pipeline" max_retries="3">
    <retry_policy>
      - On PLAN failure: re-plan with failover context (up to 2 retries)
      - On CODE failure: retry code generation for failed tasks (up to 2 per task)
      - On QA failure: no auto-retry — halt and report
      - On SHIP failure: resume from shipping phase (deduplicate PR if exists)
    </retry_policy>
  </layer>

  <!-- GitHub API: rate limit and transient errors -->
  <layer name="github" max_retries="3">
    <retry_policy>
      - Proactive rate limit: wait if remaining < buffer (default 100)
      - Exponential backoff on 5xx: 1s, 2s, 4s
      - No retry on 401 (auth — halt), 404 (not found — halt), 422 (validation — adapt)
      - Poll CI checks: 30s interval, 600s max wait
    </retry_policy>
  </layer>
</failure_handling>
```

---

## 8. Schema Versioning and Migration

```xml
<schema_versioning>
  <versioning_rules>
    - All schemas carry a semver version field (e.g., "1.0", "1.1").
    - Breaking changes increment the major version (1.0 → 2.0).
    - Additive changes increment the minor version (1.0 → 1.1).
    - Each schema is registered in a schema registry with its version.
  </versioning_rules>

  <migration_strategy>
    - When a schema version changes, a migration function is registered:
      migrate_v1_to_v2(old: SchemaV1) → SchemaV2
    - Migrations are one-directional (old → new). No down-migration.
    - The state store records the schema version used for each run.
    - On resume, if the stored schema version differs from the current,
      the migration function is applied before loading.
    - If no migration function exists for the version gap, the run is
      marked as "stale" and a new run is started.
  </migration_strategy>

  <compatibility_matrix>
    <rule name="forward_compatible">
      A consumer that accepts Schema v1.0 also accepts v1.x (additive).
    </rule>
    <rule name="breaking_incompatible">
      A consumer that accepts Schema v1.0 rejects v2.0 (breaking).
    </rule>
    <rule name="vendor_adapter_versioning">
      Vendor adapters declare which canonical schema version they support.
      A version mismatch between adapter and schema is a startup error.
    </rule>
  </compatibility_matrix>
</schema_versioning>
```

---

## 9. Security and Event Auditing

```xml
<security_and_audit>
  <credential_management>
    - All credentials are stored in ~/.hermes/.env or auth.json.
    - Credentials are never logged, displayed, or included in schema payloads.
    - Adapters receive credentials via environment variables at construction time.
    - Credential rotation: adapters re-read env vars on each client construction.
  </credential_management>

  <event_audit>
    <audit_log type="structured" backend="structlog + SQLite">
      <event name="pipeline_start" fields="run_id,trigger,repo,trusted_actor,timestamp" />
      <event name="phase_transition" fields="run_id,from_phase,to_phase,timestamp" />
      <event name="schema_validation" fields="run_id,schema_name,version,passed,errors" />
      <event name="provider_failover" fields="run_id,from_provider,to_provider,reason,timestamp" />
      <event name="llm_call" fields="run_id,phase,provider,model,tokens_in,tokens_out,latency_ms" />
      <event name="github_api_call" fields="run_id,endpoint,method,status,rate_limit_remaining" />
      <event name="qa_result" fields="run_id,passed,syntax_errors,security_findings,test_results" />
      <event name="policy_decision" fields="run_id,authorized,checks,denial_reason" />
      <event name="ship_result" fields="run_id,pr_number,merged,merge_message" />
      <event name="pipeline_end" fields="run_id,status,duration_ms,timestamp" />
    </audit_log>
  </event_audit>

  <trusted_actor_policy>
    - Only GitHub usernames in the trusted_actors config list can trigger the pipeline.
    - The LangGraphRouter validates trusted_actor in HermesRunRequestSchema.
    - The PolicyGate re-validates trusted_actor in ShipAuthorizationSchema.
    - Untrusted triggers are rejected before any API calls are made.
  </trusted_actor_policy>

  <secret_scan>
    - The QA agent runs a regex-based security scan on all generated files.
    - Patterns: hardcoded API keys, private keys, passwords, tokens.
    - Test files (*.test.*, *test_*) are exempt (may contain fake secrets).
    - Critical findings (private keys, injection patterns) halt the pipeline.
    - Advisory findings (potential secrets) are logged but don't halt.
  </secret_scan>
</security_and_audit>
```

---

## 10. End-to-End Data Flow Sequence

```xml
<data_flow_sequence name="typical_run">
  <step ordinal="1" actor="Hermes" action="Receives trigger (issue labeled 'auto-build')">
    <output>HermesRunRequestSchema v1.0</output>
    <validation>Hermes validates trigger source and trusted actor</validation>
  </step>

  <step ordinal="2" actor="LangGraphRouter" action="Analyzes requirement, extracts research topics">
    <input>HermesRunRequestSchema v1.0</input>
    <output>RoutingPlanSchema v1.0</output>
    <validation>RoutingPlanSchema validates before dispatch</validation>
  </step>

  <step ordinal="3" actor="ResearchRouter" action="Dispatches to BrightData MCP with failover to Oxylabs/Venice">
    <input>ResearchRequestSchema v1.0</input>
    <output>EvidenceSchema v1.0</output>
    <validation>EvidenceSchema.combined_context must be non-empty</validation>
    <failover>If BrightData blocked → Oxylabs → Venice (each transition logged as FailoverEvent)</failover>
  </step>

  <step ordinal="4" actor="ModelRouter" action="Dispatches to LLM provider for plan generation">
    <input>ModelTaskSchema v1.0 (phase=plan)</input>
    <output>ModelResultSchema v1.0</output>
    <validation>If response_format=json_object, parsed must be valid JSON</validation>
  </step>

  <step ordinal="5" actor="PLAN" action="Produces structured build plan from LLM output + research context">
    <input>EvidenceSchema v1.0 + ModelResultSchema v1.0 (phase=plan)</input>
    <output>BuildPlanSchema v1.0</output>
    <validation>branch_name matches "auto-build/" prefix, tasks non-empty, pr_title ≤ 72 chars</validation>
  </step>

  <step ordinal="6" actor="CODE" action="Generates file content for each task, validates syntax, commits via Git Data API">
    <input>BuildPlanSchema v1.0 + ModelResultSchema v1.0 (phase=code)</input>
    <output>ChangeSetSchema v1.0</output>
    <validation>syntax_validated must be true</validation>
  </step>

  <step ordinal="7" actor="QA" action="Runs syntax, security, lint, type, and test checks">
    <input>ChangeSetSchema v1.0</input>
    <output>QAReportSchema v1.0</output>
    <validation>passed is true only if syntax passes and no critical security findings</validation>
  </step>

  <step ordinal="8" actor="PolicyGate" action="Evaluates policy rules against QA report">
    <input>QAReportSchema v1.0</input>
    <output>ShipAuthorizationSchema v1.0</output>
    <validation>authorized is true only if all policy checks pass</validation>
  </step>

  <step ordinal="9" actor="GitHubAgent" action="Creates branch, commits files, opens PR, waits for CI, merges on green">
    <input>ShipAuthorizationSchema v1.0</input>
    <output>ShipResultSchema v1.0</output>
    <validation>If merged=true, pr_number and pr_url must be present</validation>
  </step>

  <step ordinal="10" actor="LangGraph" action="Assembles final pipeline result">
    <input>ShipResultSchema v1.0</input>
    <output>PipelineResultSchema v1.0</output>
    <validation>If status=completed, ship_result.merged must be true</validation>
  </step>

  <step ordinal="11" actor="Hermes" action="Reports result to triggering surface (issue comment, CLI, etc.)">
    <input>PipelineResultSchema v1.0</input>
    <output>Human-readable summary posted as GitHub issue comment or CLI output</output>
  </step>
</data_flow_sequence>
```

---

## 11. Acceptance Criteria

```xml
<acceptance_criteria>
  <criterion id="AC-1" name="schema_validation_at_every_boundary">
    Every component transition validates the producer's output schema
    before the consumer is allowed to execute. A validation failure halts
    the pipeline and records the error. No component receives unvalidated
    input.
  </criterion>

  <criterion id="AC-2" name="vendor_adapter_bidirectionality">
    Every vendor (BrightData, Oxylabs, Venice, OpenRouter, OpenAI, Grok,
    Camofox, GitHub) has a bidirectional adapter that maps vendor-specific
    schemas to canonical pipeline schemas and back. Vendor schemas never
    leak into pipeline components.
  </criterion>

  <criterion id="AC-3" name="resume_after_crash">
    A pipeline run interrupted at any phase can be resumed from the last
    completed phase by loading its state from the SQLite store. Resume
    logic resets the status and clears errors for the failed phase.
  </criterion>

  <criterion id="AC-4" name="failover_with_audit_trail">
    Every provider failover is recorded as a FailoverEvent in the
    EvidenceSchema. The audit log captures the from_provider, to_provider,
    and reason for every failover.
  </criterion>

  <criterion id="AC-5" name="policy_gate_enforcement">
    No code is shipped without ShipAuthorizationSchema.authorized=true.
    The Policy Gate independently validates: QA passed, no critical
    security findings, trusted actor, branch naming, file count limit.
  </criterion>

  <criterion id="AC-6" name="idempotency">
    Webhook triggers with an idempotency_key are deduplicated. A duplicate
    run is rejected before any API calls are made.
  </criterion>

  <criterion id="AC-7" name="credential_safety">
    No credential value appears in any schema payload, log entry, audit
    event, or error message. Credentials are passed via environment
    variables and never serialized.
  </criterion>

  <criterion id="AC-8" name="schema_versioning">
    All schemas carry semver versions. Schema changes are backward-compatible
    within a major version. Breaking changes require a migration function.
  </criterion>

  <criterion id="AC-9" name="mcp_integration">
    MCP tools (BrightData, Oxylabs) are integrated through the schema system,
    not bypassing it. MCP inputSchema → ResearchRequestSchema → MCP
    invocation → EvidenceSchema.
  </criterion>

  <criterion id="AC-10" name="end_to_end_evidence">
    The pipeline has produced at least one completed run that created a PR
    on a real GitHub repository. The run is auditable from the SQLite state
    store with full phase transitions and timing.
  </criterion>
</acceptance_criteria>
```

---

## 12. Component Ownership Matrix

```xml
<ownership>
  <component name="Hermes" owner="Hermes Agent framework" responsibility="Trigger ingestion, result delivery, credential management" />
  <component name="LangGraphRouter" owner="Pipeline" responsibility="Requirement analysis, routing decisions, phase coordination" />
  <component name="ResearchRouter" owner="Pipeline" responsibility="Multi-provider research dispatch with failover" />
  <component name="ModelRouter" owner="Pipeline" responsibility="LLM provider dispatch with failover" />
  <component name="BrowserRouter" owner="Pipeline" responsibility="Browser backend dispatch with failover" />
  <component name="PLAN" owner="Pipeline" responsibility="Build plan generation from research + LLM output" />
  <component name="CODE" owner="Pipeline" responsibility="Code generation, syntax validation, Git Data API commits" />
  <component name="QA" owner="Pipeline" responsibility="Syntax, security, lint, type, and test checks" />
  <component name="PolicyGate" owner="Pipeline" responsibility="Policy enforcement before shipping" />
  <component name="GitHubAgent" owner="Pipeline" responsibility="Branch, commit, PR, CI monitoring, merge" />
  <component name="StateStore" owner="Pipeline" responsibility="SQLite persistence, resume, audit trail" />
  <component name="BrightDataAdapter" owner="Pipeline" responsibility="BrightData MCP ↔ EvidenceSchema" />
  <component name="OxylabsAdapter" owner="Pipeline" responsibility="Oxylabs API/MCP ↔ EvidenceSchema" />
  <component name="VeniceAdapter" owner="Pipeline" responsibility="Venice API ↔ EvidenceSchema" />
  <component name="OpenRouterAdapter" owner="Pipeline" responsibility="OpenRouter API ↔ ModelResultSchema" />
  <component name="OpenAIAdapter" owner="Pipeline" responsibility="OpenAI API ↔ ModelResultSchema" />
  <component name="GrokAdapter" owner="Pipeline" responsibility="xAI Grok API ↔ ModelResultSchema" />
  <component name="CamofoxAdapter" owner="Pipeline" responsibility="Camofox API ↔ BrowserResultSchema" />
</ownership>
```

---

## 13. Existing Implementation Mapping

This specification describes the target architecture. The current codebase
already implements the following components:

| Spec Component | Existing File | Status |
|---|---|---|
| HermesRunRequestSchema | `auto_build/state.py: PipelineState` | Implemented (needs schema versioning) |
| LangGraphRouter | `langgraph_orchestrator/router.py` | Implemented (scraping routing only) |
| ResearchRouter | `auto_build/planner.py: _research_requirement` | Partial (topic extraction + WebFetcher) |
| EvidenceSchema | `auto_build/web_fetcher.py: research()` | Partial (returns string, needs structured schema) |
| ModelRouter | `auto_build/llm.py: LLMClient` | Implemented (OpenAI-compatible, no failover) |
| ModelResultSchema | `auto_build/llm.py: chat()` return | Partial (returns string, needs structured schema) |
| PLAN | `auto_build/planner.py: Planner` | Implemented |
| BuildPlanSchema | `auto_build/state.py: BuildPlan` | Implemented |
| CODE | `auto_build/coder.py: Coder` | Implemented |
| ChangeSetSchema | `auto_build/state.py: files_committed + commit_sha` | Partial (dict, needs schema) |
| QA | `auto_build/qa.py: QAChecker` | Implemented |
| QAReportSchema | `auto_build/state.py: QAResult` | Implemented |
| PolicyGate | Not yet implemented | Needs implementation |
| ShipAuthorizationSchema | Not yet implemented | Needs implementation |
| GitHubAgent | `auto_build/shipper.py: Shipper` | Implemented |
| ShipResultSchema | `auto_build/state.py: ShipResult` | Implemented |
| PipelineResultSchema | `auto_build/state.py: PipelineState` | Implemented (needs schema versioning) |
| StateStore | `auto_build/state.py: StateStore` | Implemented (SQLite) |
| BrightDataAdapter | `langgraph_orchestrator/tools/brightdata.py` | Implemented (MCP, no schema adapter) |
| OxylabsAdapter | `auto_build/oxylabs_client.py` | Implemented (HTTP, no schema adapter) |
| VeniceAdapter | `langgraph_orchestrator/tools/venice.py` | Implemented (HTTP, no schema adapter) |
| CamofoxAdapter | `langgraph_orchestrator/tools/camofox.py` | Implemented (HTTP, no schema adapter) |
| Schema Versioning | Not yet implemented | Needs implementation |
| MCP Integration | `langgraph_orchestrator/mcp_client.py` | Implemented (no schema boundary) |

### Gaps to Close

1. **PolicyGate** — New component between QA and Shipper. Needs
   `ShipAuthorizationSchema` with policy rules (qa_passed, no_critical_security,
   trusted_actor, branch_naming, file_count_limit).

2. **Structured EvidenceSchema** — Currently `research()` returns a string.
   Needs to return a structured `EvidenceSchema` with sources, providers,
   and failover events.

3. **Schema Versioning** — All Pydantic models need a `schema_version` field
   and a migration registry.

4. **Vendor Adapter Formalization** — BrightData, Oxylabs, Venice, and
   Camofox wrappers need formal adapter classes that map vendor responses
   to canonical schemas (EvidenceSchema, ModelResultSchema, BrowserResultSchema).

5. **ModelRouter Failover** — LLMClient needs fallback provider support
   (preferred → fallback[0] → fallback[1]).

6. **MCP Schema Boundary** — MCP tool calls need to pass through
   ResearchRequestSchema → MCP invocation → EvidenceSchema, not call
   MCP tools directly from the planner.

---

*End of Specification*
