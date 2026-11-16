"""SQLite persistence layer for AgentOps execution runs and metrics."""

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional
import aiosqlite

from agentops.config import Settings, get_settings
from agentops.schemas import WorkflowExecutionState

logger = logging.getLogger("agentops.storage")


class ExecutionStore:
    """Async SQLite persistence store for multi-agent workflows."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")

    async def init_db(self) -> None:
        """Initialize database tables and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    request_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    repo_url TEXT,
                    status TEXT NOT NULL,
                    current_iteration INTEGER NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    total_input_tokens INTEGER NOT NULL,
                    total_output_tokens INTEGER NOT NULL,
                    total_cost_usd REAL,
                    total_latency_ms REAL NOT NULL,
                    final_response TEXT,
                    is_mock INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_created_at
                ON executions(created_at DESC)
            """)
            await db.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")

    async def save_execution(self, state: WorkflowExecutionState) -> None:
        """Insert or update workflow execution state."""
        state_json = state.model_dump_json()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO executions (
                    request_id, trace_id, task, repo_url, status,
                    current_iteration, max_iterations, total_tokens,
                    total_input_tokens, total_output_tokens, total_cost_usd,
                    total_latency_ms, final_response, is_mock, state_json,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.request_id,
                state.trace_id,
                state.task,
                state.repo_url,
                state.status,
                state.current_iteration,
                state.max_iterations,
                state.total_tokens,
                state.total_input_tokens,
                state.total_output_tokens,
                state.total_cost_usd,
                state.total_latency_ms,
                state.final_response,
                1 if state.is_mock else 0,
                state_json,
                state.created_at,
                state.completed_at,
            ))
            await db.commit()

    async def get_execution(self, request_id: str) -> Optional[WorkflowExecutionState]:
        """Retrieve full workflow state by request_id."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT state_json FROM executions WHERE request_id = ?",
                (request_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return WorkflowExecutionState.model_validate_json(row["state_json"])

    async def list_executions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List summary of executions ordered by creation date descending."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT request_id, trace_id, task, repo_url, status,
                       current_iteration, max_iterations, total_tokens,
                       total_cost_usd, total_latency_ms, is_mock, created_at, completed_at
                FROM executions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated execution and latency metrics."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN status != 'approved' THEN 1 ELSE 0 END) as failed_requests,
                    AVG(current_iteration) as avg_iterations,
                    AVG(total_tokens) as avg_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(total_latency_ms) as avg_latency_ms,
                    AVG(total_cost_usd) as avg_cost_usd,
                    SUM(total_cost_usd) as total_cost_usd
                FROM executions
            """)
            summary_row = await cursor.fetchone()
            summary = dict(summary_row) if summary_row else {}

            # Fetch latencies to calculate P50 and P95
            lat_cursor = await db.execute("SELECT total_latency_ms FROM executions ORDER BY total_latency_ms ASC")
            lat_rows = await lat_cursor.fetchall()
            latencies = [r["total_latency_ms"] for r in lat_rows] if lat_rows else []

            p50 = 0.0
            p95 = 0.0
            if latencies:
                n = len(latencies)
                p50 = latencies[int(n * 0.50)]
                p95 = latencies[min(int(n * 0.95), n - 1)]

            summary["p50_latency_ms"] = round(p50, 2)
            summary["p95_latency_ms"] = round(p95, 2)
            summary["avg_latency_ms"] = round(summary.get("avg_latency_ms") or 0.0, 2)
            summary["avg_tokens"] = round(summary.get("avg_tokens") or 0.0, 1)
            summary["avg_iterations"] = round(summary.get("avg_iterations") or 1.0, 2)
            return summary
