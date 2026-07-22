# LangGraph Orchestrator

A LangGraph-based orchestration layer with a 3-tier web scraping routing system (BrightData → Oxylabs → Venice) and a local Camofox anti-detection browser backend.

## Architecture

```
User Input Plane (CLI / Web / Telegram / Cron)
        │
        ▼
LangGraph Orchestrator (Plan → Get → Run → Summary)
   state graph, checkpointing, retries, branching
        │
        ├── LangChain Tool Wrappers
        │   ├── BrightData MCP (74 tools, PRO_MODE)
        │   ├── Oxylabs AI Studio MCP (6 tools)
        │   ├── Venice Chat + web scraping
        │   └── Camofox Browser (Docker, :9377)
        │
        └── Camofox Browser (Docker)
            Camoufox (Firefox fork), C++ fingerprint spoofing
            Sticky sessions, :9377
```

## Quick Start

### 1. Start Camofox browser (Docker)

```bash
docker compose up -d
# or
docker run -d --name camofox-browser -p 9377:9377 -p 6080:6080 \
  -e CAMOFOX_HOST=0.0.0.0 -e CAMOFOX_API_KEY=change-me \
  -v ~/.camofox-docker:/home/node/.camofox \
  ghcr.io/redf0x1/camofox-browser:latest
```

Verify: `curl http://localhost:9377/health` → `{"ok":true,"engine":"camoufox"}`

### 2. Install dependencies

```bash
cd langgraph-orchestrator
uv sync
```

### 3. Run a query (CLI)

```bash
uv run python scripts/orchestrator.py "What is Hermes Agent?"
uv run python scripts/orchestrator.py --health
uv run python scripts/orchestrator.py --stream "search query"
```

### 4. Run the server (Web/Telegram/Cron input plane)

```bash
uv run python -m uvicorn langgraph_orchestrator.server.app:app --host 0.0.0.0 --port 8100
```

Endpoints:
- `POST /query` — run a query, returns final state
- `POST /query/stream` — SSE stream of graph events
- `GET /health` — dependency health check
- `GET /graph` — graph visualization (Mermaid)

## Provider Routing

| Tier | Provider | When | Failover Trigger |
|------|----------|------|------------------|
| 1 | BrightData (MCP) | Default first choice | block/CAPTCHA/5xx/empty |
| 2 | Oxylabs (AI Studio) | BrightData failover | credits exhausted / render+proxy fail |
| 3 | Venice (web scraping) | <3 pages, needs LLM answer | — |
| Browser | Camofox | needs_rendering=True | BrightData scraping_browser fallback |

All requests are logged with provider attribution:
```
scrape-served url=... provider=brightdata tool=scrape_as_markdown status=success chars=2277
```

## Graph Structure

```
START → plan → get → run → summary → END
                       │
                       └─ (failure + retries<2) → retry_bump → plan
                       └─ (exhausted) → summary
```

- **Plan**: LLM analyzes query, decides needs_rendering/geo/steps/provider
- **Get**: Prepares tool call, reroutes to render path if needed
- **Run**: Executes via routing layer, 3-tier failover, records tool calls
- **retry_bump**: Increments retry counter, clears errors, forces provider escalation
- **Summary**: LLM synthesizes grounded answer from scraped content

State is checkpointed via SQLite (`~/.hermes/cache/orchestrator-checkpoints.db`) — enables resume, time travel, and thread isolation.

## Configuration

Credentials are loaded from `~/.hermes/.env`:
- `BLACKBOX_API_KEY` — LLM (api.blackbox.ai/v1, z-ai/glm-5.2)
- `BRIGHTDATA_API_TOKEN` — tier-1 scraping
- `OXYLABS_AI_STUDIO_API_KEY` — tier-2 failover
- `VENICE_API_KEY` — tier-3 + web scraping
- `CAMOFOX_URL` — browser backend (http://localhost:9377)

## Key Design Decisions

1. **MCP stdio over HTTP**: BrightData/Oxylabs use stdio MCP servers (npx/uvx subprocesses) — the hosted HTTP MCP endpoints don't accept bearer tokens directly. A shared `MCPStdioClient` manages long-lived subprocesses with JSON-RPC.

2. **Retry via dedicated node**: The `retry_bump` node increments the counter and clears errors, avoiding the infinite-loop bug where `plan` reset `retries=0` on every run.

3. **Provider escalation on retry**: When `retries > 0`, the plan node forces the next provider tier (brightdata → oxylabs → venice) instead of re-trying the same one.

4. **Graceful degradation**: Every node catches exceptions and records errors in state — the graph always completes and the summary node reports partial results.

5. **Camofox session/tab flow**: v2.4.6 requires `POST /start` (session) → `POST /tabs` (open tab, returns targetId) → `GET /snapshot?targetId=` (accessibility tree).
