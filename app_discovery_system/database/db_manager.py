"""Menedżer bazy danych SQLite (WAL mode) z obsługą kampanii, PID procesów, logów, pomysłów i MVP."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app_discovery_system.config import DEFAULT_DB_PATH
from app_discovery_system.models.schemas import AppIdea, MVPSpecification

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DatabaseManager:
    """Bezpieczny wątkowo i procesowo menedżer SQLite (WAL mode)."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        with self._lock:
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            with self._connect() as conn:
                conn.executescript(schema_sql)
                conn.commit()

    def create_campaign(self, name: str, pid: Optional[int] = None) -> int:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO campaigns (name, status, pid, vllm_calls_count, created_at, updated_at)
                VALUES (?, 'RUNNING', ?, 0, ?, ?)
                """,
                (name, pid, now, now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_campaign_status(
        self,
        campaign_id: int,
        status: str,
        pid: Optional[int] = None,
        vllm_calls_count: Optional[int] = None,
        executive_summary: Optional[str] = None,
        final_recommendation: Optional[str] = None,
        final_report_md: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        fields = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]
        if pid is not None:
            fields.append("pid = ?")
            params.append(pid)
        if vllm_calls_count is not None:
            fields.append("vllm_calls_count = ?")
            params.append(vllm_calls_count)
        if executive_summary is not None:
            fields.append("executive_summary = ?")
            params.append(executive_summary)
        if final_recommendation is not None:
            fields.append("final_recommendation = ?")
            params.append(final_recommendation)
        if final_report_md is not None:
            fields.append("final_report_md = ?")
            params.append(final_report_md)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        params.append(campaign_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()

    def is_stop_requested(self, campaign_id: int) -> bool:
        """Sprawdza w SQLite, czy użytkownik kliknął przycisk Stop."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT status FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
            if not row:
                return True
            return row["status"] in ("STOPPED", "STOP_REQUESTED")

    def increment_vllm_calls(self, campaign_id: int, delta: int = 1) -> int:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE campaigns SET vllm_calls_count = vllm_calls_count + ?, updated_at = ? WHERE id = ?",
                (delta, now, campaign_id),
            )
            conn.commit()
            row = conn.execute("SELECT vllm_calls_count FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
            return int(row["vllm_calls_count"]) if row else 0

    def log_agent_event(self, campaign_id: int, agent_name: str, status: str, message: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_logs (campaign_id, agent_name, status, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (campaign_id, agent_name, status, message, now),
            )
            conn.commit()

    def save_ideas(self, campaign_id: int, ideas: List[AppIdea]) -> List[int]:
        now = _utc_now()
        inserted_ids: List[int] = []
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM ideas WHERE campaign_id = ?", (campaign_id,))
            sorted_ideas = sorted(ideas, key=lambda x: x.score_total, reverse=True)
            for idx, idea in enumerate(sorted_ideas, start=1):
                idea.rank = idx
                cursor = conn.execute(
                    """
                    INSERT INTO ideas (
                        campaign_id, rank, title, category, problem_statement,
                        target_audience, value_proposition, score_total,
                        score_breakdown_json, is_top3, claims_json,
                        competitors_analysis, pricing_analysis, build_difficulty,
                        sales_channels, recommendation, approved_for_mvp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        campaign_id,
                        idea.rank,
                        idea.title,
                        idea.category.value,
                        idea.problem_statement,
                        idea.target_audience,
                        idea.value_proposition,
                        idea.score_total,
                        idea.score_breakdown.model_dump_json(),
                        1 if idea.is_top3 else 0,
                        json.dumps([c.model_dump() for c in idea.claims], ensure_ascii=False),
                        idea.competitors_analysis,
                        idea.pricing_analysis,
                        idea.build_difficulty,
                        idea.sales_channels,
                        idea.recommendation,
                        now,
                    ),
                )
                inserted_ids.append(int(cursor.lastrowid))
            conn.commit()
        return inserted_ids

    def approve_idea_for_mvp(self, idea_id: int) -> Optional[Dict[str, Any]]:
        """Oznacza wybrany pomysł jako zatwierdzony przez użytkownika do budowy MVP."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
            if not row:
                return None
            conn.execute("UPDATE ideas SET approved_for_mvp = 1 WHERE id = ?", (idea_id,))
            conn.commit()
            updated = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
            return dict(updated) if updated else None



    def save_mvp_artifact(
        self,
        campaign_id: int,
        idea_id: int,
        spec: MVPSpecification,
        mvp_spec_md: str,
        sandbox_dir: str,
    ) -> int:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mvp_artifacts (
                    campaign_id, idea_id, idea_title, mvp_spec_md,
                    architecture_overview, tech_stack_json, core_features_json,
                    database_schema_sql, api_endpoints_json, generated_files_json,
                    sandbox_dir, qa_test_plan, static_security_audit,
                    safe_run_instructions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    idea_id,
                    spec.idea_title,
                    mvp_spec_md,
                    spec.architecture_overview,
                    json.dumps(spec.tech_stack, ensure_ascii=False),
                    json.dumps(spec.core_features, ensure_ascii=False),
                    spec.database_schema_sql,
                    json.dumps(spec.api_endpoints, ensure_ascii=False),
                    json.dumps(spec.generated_files, ensure_ascii=False),
                    sandbox_dir,
                    spec.qa_test_plan,
                    spec.static_security_audit,
                    spec.safe_run_instructions,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_campaign(self, campaign_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
            return dict(row) if row else None

    def get_latest_campaign(self) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM campaigns ORDER BY id DESC LIMIT 1").fetchone()

    def get_ideas_for_campaign(self, campaign_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ideas WHERE campaign_id = ? ORDER BY score_total DESC, rank ASC",
                (campaign_id,),
            ).fetchall()
            return [self._decode_idea(dict(r)) for r in rows]

    def get_idea_by_id(self, idea_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
            return self._decode_idea(dict(row)) if row else None

    @staticmethod
    def _decode_idea(d: Dict[str, Any]) -> Dict[str, Any]:
        d["score_breakdown"] = json.loads(d.get("score_breakdown_json") or "{}")
        d["claims"] = json.loads(d.get("claims_json") or "[]")
        d["is_top3"] = bool(d.get("is_top3"))
        d["approved_for_mvp"] = bool(d.get("approved_for_mvp"))
        return d

    def get_agent_logs(self, campaign_id: int, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_logs WHERE campaign_id = ? ORDER BY id ASC LIMIT ?",
                (campaign_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_mvp_artifacts(
        self, campaign_id: Optional[int] = None, idea_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if idea_id is not None:
                rows = conn.execute(
                    "SELECT * FROM mvp_artifacts WHERE idea_id = ? ORDER BY id DESC", (idea_id,)
                ).fetchall()
            elif campaign_id is not None:
                rows = conn.execute(
                    "SELECT * FROM mvp_artifacts WHERE campaign_id = ? ORDER BY id DESC", (campaign_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM mvp_artifacts ORDER BY id DESC").fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["tech_stack"] = json.loads(d.get("tech_stack_json") or "[]")
                d["core_features"] = json.loads(d.get("core_features_json") or "[]")
                d["api_endpoints"] = json.loads(d.get("api_endpoints_json") or "[]")
                d["generated_files"] = json.loads(d.get("generated_files_json") or "{}")
                result.append(d)
            return result

            return dict(row) if row else None

    def list_campaigns(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM campaigns ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
