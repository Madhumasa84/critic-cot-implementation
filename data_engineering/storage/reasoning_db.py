"""
SQLite storage for Critic-CoT reasoning traces.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "reasoning_traces.db"


class ReasoningTraceDB:
    """
    Store complete reasoning traces, step-level details, critiques, and daily metrics.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    question TEXT,
                    expected_answer TEXT,
                    strategy TEXT,
                    final_answer TEXT,
                    is_correct INTEGER,
                    model TEXT,
                    total_iterations INTEGER,
                    total_steps INTEGER,
                    api_latency_ms REAL,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    trace_json TEXT
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    round_number INTEGER,
                    step_number INTEGER,
                    step_text TEXT,
                    is_correct INTEGER,
                    error_description TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS critiques (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    round_number INTEGER,
                    critique_text TEXT,
                    error_step INTEGER,
                    has_error INTEGER,
                    tool_verified INTEGER,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_runs INTEGER,
                    avg_accuracy REAL,
                    avg_latency REAL,
                    total_cost REAL,
                    strategies_used TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_traces_strategy
                    ON traces(strategy);
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp
                    ON traces(timestamp);
                CREATE INDEX IF NOT EXISTS idx_steps_trace_round
                    ON steps(trace_id, round_number);
                CREATE INDEX IF NOT EXISTS idx_critiques_trace_round
                    ON critiques(trace_id, round_number);
                """
            )

    def save_trace(self, trace_data: Dict[str, Any]) -> str:
        trace_id = trace_data.get("trace_id") or f"trace_{datetime.now().timestamp()}"
        timestamp = trace_data.get("timestamp") or datetime.now().isoformat()
        question_payload = trace_data.get("question", {})
        strategy_payload = trace_data.get("strategy", {})
        metadata = trace_data.get("metadata", {})
        iterations = trace_data.get("iterations", [])

        with self._connect() as conn:
            conn.execute("DELETE FROM steps WHERE trace_id = ?", (trace_id,))
            conn.execute("DELETE FROM critiques WHERE trace_id = ?", (trace_id,))

            conn.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id,
                    timestamp,
                    question,
                    expected_answer,
                    strategy,
                    final_answer,
                    is_correct,
                    model,
                    total_iterations,
                    total_steps,
                    api_latency_ms,
                    total_tokens,
                    cost_usd,
                    trace_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    timestamp,
                    question_payload.get("text", ""),
                    question_payload.get("expected_answer", ""),
                    strategy_payload.get("name", ""),
                    trace_data.get("final_answer", ""),
                    1 if trace_data.get("is_correct") else 0,
                    metadata.get("model", ""),
                    len(iterations),
                    metadata.get("total_steps", 0),
                    metadata.get("api_latency_ms", 0.0),
                    metadata.get("total_tokens", 0),
                    metadata.get("cost_usd", 0.0),
                    json.dumps(trace_data, ensure_ascii=True),
                ),
            )

            for iteration in iterations:
                self._insert_iteration(conn, trace_id, iteration)

        return trace_id

    def _insert_iteration(
        self,
        conn: sqlite3.Connection,
        trace_id: str,
        iteration: Dict[str, Any],
    ) -> None:
        round_number = int(iteration.get("round", 0))
        solution = iteration.get("solution", {})
        critique = iteration.get("critique")

        steps = solution.get("steps") or self._extract_steps(solution.get("full_text", ""))
        error_step = critique.get("error_step") if isinstance(critique, dict) else None
        critique_text = critique.get("full_text", "") if isinstance(critique, dict) else ""

        for step_number, step_text in enumerate(steps, start=1):
            is_correct, error_description = self._step_status(
                step_number=step_number,
                error_step=error_step,
                critique_text=critique_text,
            )
            conn.execute(
                """
                INSERT INTO steps (
                    trace_id,
                    round_number,
                    step_number,
                    step_text,
                    is_correct,
                    error_description
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    round_number,
                    step_number,
                    step_text,
                    is_correct,
                    error_description,
                ),
            )

        if isinstance(critique, dict):
            conn.execute(
                """
                INSERT INTO critiques (
                    trace_id,
                    round_number,
                    critique_text,
                    error_step,
                    has_error,
                    tool_verified
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    round_number,
                    critique.get("full_text", ""),
                    critique.get("error_step"),
                    1 if critique.get("has_error") else 0,
                    1 if critique.get("tool_verified") else 0,
                ),
            )

    def _step_status(
        self,
        step_number: int,
        error_step: Optional[int],
        critique_text: str,
    ) -> tuple[Optional[int], Optional[str]]:
        if error_step is None:
            return None, None
        if step_number < error_step:
            return 1, None
        if step_number == error_step:
            return 0, critique_text[:500] if critique_text else "Critic marked this step as incorrect."
        return None, None

    def _extract_steps(self, solution_text: str) -> List[str]:
        if not solution_text:
            return []

        lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
        step_lines = [line for line in lines if line.lower().startswith("step ")]
        if step_lines:
            return step_lines
        return lines[:1] if lines else []

    def _fetch_rows(self, query: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def _as_table(self, rows: List[Dict[str, Any]]) -> Any:
        if pd is not None:
            return pd.DataFrame(rows)
        return rows

    def get_traces(
        self,
        strategy: Optional[str] = None,
        limit: int = 100,
        correct_only: Optional[bool] = None,
    ) -> Any:
        query = "SELECT * FROM traces"
        clauses: List[str] = []
        params: List[Any] = []

        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if correct_only is not None:
            clauses.append("is_correct = ?")
            params.append(1 if correct_only else 0)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return self._as_table(self._fetch_rows(query, params))

    def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        traces = self._fetch_rows("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        steps = self._fetch_rows(
            """
            SELECT round_number, step_number, step_text, is_correct, error_description
            FROM steps
            WHERE trace_id = ?
            ORDER BY round_number, step_number
            """,
            (trace_id,),
        )
        critiques = self._fetch_rows(
            """
            SELECT round_number, critique_text, error_step, has_error, tool_verified
            FROM critiques
            WHERE trace_id = ?
            ORDER BY round_number
            """,
            (trace_id,),
        )
        return {
            "trace": traces[0] if traces else None,
            "steps": steps,
            "critiques": critiques,
        }

    def get_accuracy_by_strategy(self) -> Any:
        rows = self._fetch_rows(
            """
            SELECT
                strategy,
                COUNT(*) AS total_samples,
                SUM(is_correct) AS correct_count,
                ROUND(AVG(is_correct) * 100, 2) AS accuracy_pct,
                ROUND(AVG(api_latency_ms), 2) AS avg_latency_ms,
                ROUND(AVG(total_tokens), 2) AS avg_tokens,
                ROUND(SUM(cost_usd), 6) AS total_cost
            FROM traces
            GROUP BY strategy
            ORDER BY accuracy_pct DESC, strategy ASC
            """
        )
        return self._as_table(rows)

    def get_error_analysis(self) -> Any:
        rows = self._fetch_rows(
            """
            SELECT
                t.strategy,
                c.error_step,
                COUNT(*) AS error_count,
                ROUND(AVG(t.api_latency_ms), 2) AS avg_latency_ms
            FROM critiques c
            JOIN traces t ON t.trace_id = c.trace_id
            WHERE c.has_error = 1
            GROUP BY t.strategy, c.error_step
            ORDER BY error_count DESC, t.strategy ASC
            """
        )
        return self._as_table(rows)

    def get_daily_metrics(self) -> Any:
        rows = self._fetch_rows(
            "SELECT * FROM daily_metrics ORDER BY date DESC"
        )
        return self._as_table(rows)

    def get_recent_failures(self, limit: int = 20) -> Any:
        rows = self._fetch_rows(
            """
            SELECT trace_id, timestamp, question, expected_answer, final_answer, strategy
            FROM traces
            WHERE is_correct = 0
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return self._as_table(rows)

    def search_questions(self, keyword: str, limit: int = 20) -> Any:
        rows = self._fetch_rows(
            """
            SELECT trace_id, timestamp, question, strategy, final_answer, is_correct
            FROM traces
            WHERE question LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{keyword}%", limit),
        )
        return self._as_table(rows)

    def update_daily_metrics(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM daily_metrics")
            conn.execute(
                """
                INSERT INTO daily_metrics (
                    date,
                    total_runs,
                    avg_accuracy,
                    avg_latency,
                    total_cost,
                    strategies_used
                )
                SELECT
                    DATE(timestamp) AS date,
                    COUNT(*) AS total_runs,
                    ROUND(AVG(is_correct) * 100, 2) AS avg_accuracy,
                    ROUND(AVG(api_latency_ms), 2) AS avg_latency,
                    ROUND(SUM(cost_usd), 6) AS total_cost,
                    GROUP_CONCAT(DISTINCT strategy) AS strategies_used
                FROM traces
                GROUP BY DATE(timestamp)
                """
            )

    def export_to_csv(
        self,
        filename: Optional[str] = None,
        table: str = "traces",
    ) -> str:
        allowed_tables = {"traces", "steps", "critiques", "daily_metrics"}
        if table not in allowed_tables:
            raise ValueError(f"Unsupported table: {table}")

        export_path = Path(filename) if filename else self.db_path.parent / f"{table}.csv"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._fetch_rows(f"SELECT * FROM {table}")
        fieldnames = list(rows[0].keys()) if rows else []

        with export_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)

        return str(export_path)

    def export_all_to_csv(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        target_dir = Path(output_dir) if output_dir else self.db_path.parent / "exports"
        target_dir.mkdir(parents=True, exist_ok=True)

        exported = {}
        for table in ("traces", "steps", "critiques", "daily_metrics"):
            exported[table] = self.export_to_csv(str(target_dir / f"{table}.csv"), table=table)
        return exported

    def close(self) -> None:
        # Connections are short-lived and managed per operation.
        return None


if __name__ == "__main__":
    db = ReasoningTraceDB()
    sample_trace = {
        "trace_id": "demo_trace",
        "timestamp": datetime.now().isoformat(),
        "question": {
            "text": "What is 2 + 2?",
            "expected_answer": "4",
        },
        "strategy": {"name": "baseline"},
        "final_answer": "4",
        "is_correct": True,
        "metadata": {
            "model": "demo-model",
            "api_latency_ms": 120.5,
            "total_tokens": 42,
            "cost_usd": 0.000042,
            "total_steps": 2,
        },
        "iterations": [
            {
                "round": 0,
                "solution": {
                    "full_text": "Step 1: 2 + 2 = 4\nStep 2: Therefore the answer is 4",
                    "steps": [
                        "Step 1: 2 + 2 = 4",
                        "Step 2: Therefore the answer is 4",
                    ],
                }
            }
        ],
    }
    db.save_trace(sample_trace)
    db.update_daily_metrics()
    print(db.get_accuracy_by_strategy())
    print(db.export_all_to_csv())
