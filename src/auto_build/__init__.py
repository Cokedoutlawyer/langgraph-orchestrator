"""auto_build — Multi-agent auto-build pipeline (PLAN → CODE → QA → SHIP).

Coordinates with hermes_github to provide a fully automated build workflow:
1. PLAN: Analyze an issue/requirement, produce a structured build plan
2. CODE: Execute the plan — write code, create commits on a feature branch
3. QA: Run tests, lint, type-check, security scan
4. SHIP: Create PR, monitor CI, auto-merge on green
"""
from __future__ import annotations

from .state import (
    PipelineState,
    PipelineStatus,
    BuildTask,
    BuildPlan,
    QAResult,
    ShipResult,
    StateStore,
)
from .orchestrator import Orchestrator
from .planner import Planner
from .coder import Coder
from .qa import QAChecker
from .shipper import Shipper
from .notify import Notifier
from .llm import LLMClient
from .web_fetcher import WebFetcher

__version__ = "1.0.0"

__all__ = [
    "PipelineState",
    "PipelineStatus",
    "BuildTask",
    "BuildPlan",
    "QAResult",
    "ShipResult",
    "StateStore",
    "Orchestrator",
    "Planner",
    "Coder",
    "QAChecker",
    "Shipper",
    "Notifier",
    "LLMClient",
    "WebFetcher",
]
