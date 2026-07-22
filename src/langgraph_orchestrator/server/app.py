"""FastAPI input-plane server — accepts queries from CLI, Web UI, Telegram, Cron.

Endpoints:
  POST /query        — run a query (returns final state)
  POST /query/stream — SSE stream of graph events
  GET  /health       — server + dependency health
  GET  /graph        — graph structure (mermaid)
"""
from __future__ import annotations

import json
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from ..config import settings
from ..graph import get_graph, run_query
from ..tools.camofox import health as camofox_health

log = structlog.get_logger("server")

app = FastAPI(title="LangGraph Orchestrator", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    source: str = "web"
    thread_id: str | None = None


@app.get("/health")
async def health():
    """Check server + all dependencies."""
    deps = {
        "camofox": camofox_health(),
        "brightdata_token": bool(settings.brightdata_api_token),
        "oxylabs_key": bool(settings.oxylabs_ai_studio_api_key),
        "venice_key": bool(settings.venice_api_key),
        "blackbox_key": bool(settings.blackbox_api_key),
    }
    all_ok = all(deps.values())
    return {"ok": all_ok, "dependencies": deps, "model": settings.default_model}


@app.post("/query")
async def query(req: QueryRequest):
    """Run a query synchronously and return the final state."""
    log.info("query.received", query=req.query, source=req.source)
    try:
        result = run_query(req.query, source=req.source, thread_id=req.thread_id)
        # Trim large fields for the response
        resp = {
            "summary": result.get("summary", ""),
            "providers": [c.get("provider") for c in result.get("tool_calls", [])],
            "tool_calls": [
                {"provider": c.get("provider"), "tool": c.get("tool"), "status": c.get("status"), "chars": c.get("chars")}
                for c in result.get("tool_calls", [])
            ],
            "result_chars": result.get("result_chars", 0),
            "errors": result.get("errors", []),
            "thread_id": result.get("thread_id"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
        return resp
    except Exception as e:
        log.error("query.failed", error=str(e)[:200])
        return JSONResponse({"error": str(e)[:300]}, status_code=500)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Stream graph events via SSE."""
    from langgraph.graph import StateGraph
    graph = get_graph()

    async def event_gen():
        import uuid
        tid = req.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        initial = {
            "query": req.query, "source": req.source, "thread_id": tid,
            "started_at": _now(), "tool_calls": [], "logs": [], "errors": [],
            "retries": 0, "current_step_idx": 0,
        }
        try:
            for event in graph.stream(initial, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    yield f"data: {json.dumps({'node': node_name, 'state_keys': list(node_state.keys())})}\n\n"
            yield f"data: {json.dumps({'done': True, 'thread_id': tid})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/graph", response_class=HTMLResponse)
async def graph_viz():
    """Render the graph structure as Mermaid."""
    html = """<html><body>
<h2>LangGraph Orchestrator</h2>
<pre class="mermaid">
graph TD
    START --> plan[Plan Node<br/>LLM analyzes query]
    plan --> get[Get Node<br/>Select tool + route]
    get --> run[Run Node<br/>Execute via routing]
    run -->|success| summary[Summary Node<br/>LLM synthesizes answer]
    run -->|failure + retries&lt;2| plan
    summary --> END
</pre>
<script type="module">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs'; mermaid.initialize({startOnLoad:true});</script>
</body></html>"""
    return html


@app.get("/")
async def root():
    return {"service": "langgraph-orchestrator", "endpoints": ["/query", "/query/stream", "/health", "/graph"]}


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def serve(host: str | None = None, port: int | None = None):
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(
        app,
        host=host or settings.server_host,
        port=port or settings.server_port,
    )
