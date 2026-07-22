"""Shared MCP stdio client — spawns a subprocess (npx/uvx) and talks JSON-RPC.

This is the reliable path for BrightData (@brightdata/mcp) and Oxylabs (oxylabs-mcp).
The hosted HTTP MCP endpoints don't accept bearer tokens directly; the stdio
servers we verified in Hermes are the working integration.
"""
from __future__ import annotations

import subprocess
import json
import threading
import queue
import structlog
from typing import Any

log = structlog.get_logger("mcp_client")


class MCPStdioClient:
    """Manages a long-lived MCP stdio subprocess with JSON-RPC over stdin/stdout."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None, name: str = "mcp"):
        self.command = command
        self.env = env
        self.name = name
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._responses: dict[int, queue.Queue] = {}
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start the subprocess and do the MCP initialize handshake."""
        if self._proc and self._proc.poll() is None:
            return
        log.info("mcp.start", name=self.name, command=self.command[0])
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self.env,
            bufsize=1,
            text=True,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize handshake
        init_result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "langgraph-orchestrator", "version": "0.1.0"},
        })
        # Send initialized notification (no id, no response expected)
        self._notify("notifications/initialized", {})
        self._started = True
        log.info("mcp.ready", name=self.name, server=init_result.get("serverInfo", {}).get("name"))

    def _read_loop(self) -> None:
        """Background thread reading stdout lines and dispatching to response queues."""
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Responses have an "id" matching a request
            msg_id = msg.get("id")
            if msg_id is not None and msg_id in self._responses:
                self._responses[msg_id].put(msg)

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _send(self, payload: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        msg_id = self._next_id()
        q: queue.Queue = queue.Queue()
        self._responses[msg_id] = q
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        # Wait for response (timeout 120s for slow tools)
        try:
            resp = q.get(timeout=120)
        except queue.Empty:
            del self._responses[msg_id]
            raise TimeoutError(f"MCP {self.name} request {method} timed out")
        del self._responses[msg_id]
        if "error" in resp:
            raise RuntimeError(f"MCP {self.name} error: {resp['error']}")
        return resp.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the text content of the first content block."""
        if not self._started:
            self.start()
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            content = result.get("content", [{}])
            raise RuntimeError(f"Tool {name} error: {content[0].get('text', '')[:200]}")
        content = result.get("content", [{}])
        return content[0].get("text", "") if content else ""

    def list_tools(self) -> list[dict]:
        if not self._started:
            self.start()
        result = self._request("tools/list", {})
        return result.get("tools", [])

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            self._started = False


# --- Module-level singleton clients (lazily started, reused across calls) ---
import os
import shutil

_brightdata_client: MCPStdioClient | None = None
_oxylabs_client: MCPStdioClient | None = None


def get_brightdata_client() -> MCPStdioClient:
    global _brightdata_client
    if _brightdata_client is None:
        from .config import settings
        npx = shutil.which("npx") or "npx"
        env = dict(os.environ)
        env["API_TOKEN"] = settings.brightdata_api_token
        env["PRO_MODE"] = "true"
        _brightdata_client = MCPStdioClient(
            command=[npx, "-y", "@brightdata/mcp"], env=env, name="brightdata"
        )
    return _brightdata_client


def get_oxylabs_client() -> MCPStdioClient:
    global _oxylabs_client
    if _oxylabs_client is None:
        from .config import settings
        uvx = shutil.which("uvx") or "uvx"
        env = dict(os.environ)
        env["OXYLABS_AI_STUDIO_API_KEY"] = settings.oxylabs_ai_studio_api_key
        _oxylabs_client = MCPStdioClient(
            command=[uvx, "oxylabs-mcp"], env=env, name="oxylabs"
        )
    return _oxylabs_client
