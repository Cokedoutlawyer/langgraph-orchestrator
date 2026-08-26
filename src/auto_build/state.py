"""Pipeline state models and SQLite persistence.

The state store tracks every pipeline run, its current phase, inputs, and
outputs. This enables:
- Resuming interrupted pipeline runs
- Auditing past builds
- Correlating builds with GitHub issues/PRs
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Status of a pipeline run."""
    PENDING = "pending"
    PLANNING = "planning"
    CODING = "coding"
    QA = "qa"
    SHIPPING = "shipping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildTask(BaseModel):
    """A single task within a build plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    file_path: str
    action: str = "create"  # create | modify | delete
    content_preview: str = ""  # first 200 chars of content
    status: str = "pending"  # pending | in_progress | completed | failed
    error: str | None = None


class BuildPlan(BaseModel):
    """Structured output from the Planner stage."""
    summary: str
    repo: str
    branch_name: str
    commit_message: str
    pr_title: str
    pr_body: str
    tasks: list[BuildTask] = Field(default_factory=list)
    estimated_files: int = 0


class QAResult(BaseModel):
    """Result of the QA stage."""
    passed: bool
    tests_run: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    lint_passed: bool = False
    lint_errors: list[str] = Field(default_factory=list)
    typecheck_passed: bool = False
    typecheck_errors: list[str] = Field(default_factory=list)
    security_passed: bool = False
    security_findings: list[str] = Field(default_factory=list)
    overall_message: str = ""


class ShipResult(BaseModel):
    """Result of the ship stage."""
    pr_number: int | None = None
    pr_url: str | None = None
    merged: bool = False
    merge_message: str = ""
    branch_name: str = ""
    commit_sha: str | None = None


class PipelineState(BaseModel):
    """Full state of a pipeline run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Input
    trigger: str = "manual"  # manual | issue | pr | workflow_dispatch
    issue_number: int | None = None
    issue_title: str = ""
    issue_body: str = ""
    repo_owner: str = ""
    repo_name: str = ""
    requirements: str = ""  # free-text requirements (from issue or manual)

    # Plan
    plan: BuildPlan | None = None

    # Code
    files_committed: dict[str, str] = Field(default_factory=dict)  # path → content
    commit_sha: str | None = None

    # QA
    qa_result: QAResult | None = None

    # Ship
    ship_result: ShipResult | None = None

    # Error tracking
    errors: list[str] = Field(default_factory=list)
    current_phase: str = ""

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class StateStore:
    """SQLite-backed pipeline state persistence.

    Stores pipeline runs as JSON blobs keyed by run_id.
    Enables resume-after-crash and audit trails.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".hermes" / "auto_build" / "pipeline_state.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    issue_number INTEGER,
                    repo_full_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON pipeline_runs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_issue ON pipeline_runs(issue_number)
            """)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def save(self, state: PipelineState) -> None:
        """Save or update a pipeline state."""
        state.touch()
        repo_full = f"{state.repo_owner}/{state.repo_name}" if state.repo_owner else ""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs
                    (run_id, status, trigger, issue_number, repo_full_name,
                     created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
                """,
                (
                    state.run_id,
                    state.status.value,
                    state.trigger,
                    state.issue_number,
                    repo_full,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    state.model_dump_json(),
                ),
            )
            conn.commit()

    def load(self, run_id: str) -> PipelineState | None:
        """Load a pipeline state by run_id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM pipeline_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return PipelineState.model_validate_json(row[0])

    def list_runs(
        self,
        status: PipelineStatus | None = None,
        limit: int = 50,
    ) -> list[PipelineState]:
        """List pipeline runs, optionally filtered by status."""
        query = "SELECT state_json FROM pipeline_runs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [PipelineState.model_validate_json(r[0]) for r in rows]

    def find_by_issue(self, issue_number: int) -> list[PipelineState]:
        """Find pipeline runs associated with an issue."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT state_json FROM pipeline_runs WHERE issue_number = ? ORDER BY created_at DESC",
                (issue_number,),
            ).fetchall()
        return [PipelineState.model_validate_json(r[0]) for r in rows]

    def delete(self, run_id: str) -> None:
        """Delete a pipeline run."""
        with self._conn() as conn:
            conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (run_id,))
            conn.commit()
