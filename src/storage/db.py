from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..models.evaluation import EvaluationResult


class ExperimentDB:
    """SQLite-backed store for experiments and evaluation results."""

    def __init__(self, db_path: str = "experiments/results.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now')),
                status      TEXT DEFAULT 'running',
                metadata    TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                test_case_id  TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                prompt_name   TEXT NOT NULL,
                raw_response  TEXT,
                latency_ms    REAL,
                input_tokens  INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                overall_score REAL DEFAULT 0.0,
                dimensions    TEXT DEFAULT '[]',
                metadata      TEXT DEFAULT '{}',
                created_at    TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
        """)
        self.conn.commit()

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO experiments (id, name, description, metadata) VALUES (?, ?, ?, ?)",
            (experiment_id, name, description, json.dumps(metadata or {})),
        )
        self.conn.commit()

    def save_result(self, result: EvaluationResult) -> None:
        self.conn.execute(
            """INSERT INTO results
               (experiment_id, test_case_id, prompt_version, prompt_name,
                raw_response, latency_ms, input_tokens, output_tokens,
                overall_score, dimensions, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.experiment_id,
                result.test_case_id,
                result.prompt_version,
                result.prompt_name,
                result.raw_response,
                result.latency_ms,
                result.input_tokens,
                result.output_tokens,
                result.overall_score,
                json.dumps([d.model_dump() for d in result.dimensions]),
                json.dumps(result.metadata),
            ),
        )
        self.conn.commit()

    def complete_experiment(self, experiment_id: str) -> None:
        self.conn.execute(
            "UPDATE experiments SET status = 'completed' WHERE id = ?",
            (experiment_id,),
        )
        self.conn.commit()

    def list_experiments(self) -> List[Tuple]:
        """Returns list of (id, name, description, created_at, status, metadata_dict)."""
        cursor = self.conn.execute(
            "SELECT id, name, description, created_at, status, metadata "
            "FROM experiments ORDER BY created_at DESC"
        )
        return [
            (r[0], r[1], r[2], r[3], r[4], json.loads(r[5] or "{}"))
            for r in cursor.fetchall()
        ]

    def get_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM results WHERE experiment_id = ? ORDER BY prompt_name, test_case_id",
            (experiment_id,),
        )
        columns = [col[0] for col in cursor.description]
        rows = []
        for row in cursor.fetchall():
            d = dict(zip(columns, row))
            d["dimensions"] = json.loads(d.get("dimensions") or "[]")
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            rows.append(d)
        return rows
