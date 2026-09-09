from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_STATUS = {"pending", "reviewing", "resolved"}
ALLOWED_DECISIONS = {"", "confirmed_ai", "confirmed_real", "insufficient_evidence"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def database_path() -> Path:
    return Path(os.environ.get("AIGC_REVIEW_DB", "/tmp/aigc_review_queue.sqlite3"))


def persistence_mode() -> str:
    return "configured_persistent_path" if "AIGC_REVIEW_DB" in os.environ else "temporary_local_queue"


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                binary_label TEXT NOT NULL,
                generated_probability REAL NOT NULL,
                platform_label TEXT NOT NULL,
                platform_candidate TEXT NOT NULL,
                platform_confidence REAL NOT NULL,
                platform_margin REAL NOT NULL,
                model_agreement INTEGER NOT NULL,
                drift_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                user_note TEXT NOT NULL,
                reviewer_decision TEXT NOT NULL,
                reviewer_note TEXT NOT NULL
            )
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    value["model_agreement"] = bool(value["model_agreement"])
    try:
        value["reason_codes"] = json.loads(value["reason_codes"])
    except (TypeError, json.JSONDecodeError):
        value["reason_codes"] = []
    return value


def create_case(payload: dict[str, Any]) -> dict[str, Any]:
    init_db()
    now = utc_now()
    case_id = f"RVW-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    values = {
        "case_id": case_id,
        "created_at": now,
        "updated_at": now,
        "status": "pending",
        "file_name": str(payload.get("file_name", "unknown"))[:255],
        "file_sha256": str(payload.get("file_sha256", ""))[:64],
        "binary_label": str(payload.get("binary_label", "uncertain"))[:32],
        "generated_probability": float(payload.get("generated_probability", 0.0)),
        "platform_label": str(payload.get("platform_label", "unknown_platform"))[:64],
        "platform_candidate": str(payload.get("platform_candidate", ""))[:64],
        "platform_confidence": float(payload.get("platform_confidence", 0.0)),
        "platform_margin": float(payload.get("platform_margin", 0.0)),
        "model_agreement": int(bool(payload.get("model_agreement", False))),
        "drift_score": float(payload.get("drift_score", 0.0)),
        "risk_level": str(payload.get("risk_level", "review"))[:32],
        "reason_codes": json.dumps(payload.get("reason_codes") or [], ensure_ascii=False),
        "user_note": str(payload.get("user_note", ""))[:2000],
        "reviewer_decision": "",
        "reviewer_note": "",
    }
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    with connect() as connection:
        connection.execute(
            f"INSERT INTO review_cases ({', '.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
    return row_to_dict(row)


def list_cases(*, limit: int = 50, status: str = "") -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 200))
    with connect() as connection:
        if status in ALLOWED_STATUS:
            rows = connection.execute(
                "SELECT * FROM review_cases WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM review_cases ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [row_to_dict(row) for row in rows]


def update_case(case_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    init_db()
    status = str(payload.get("status", ""))
    decision = str(payload.get("reviewer_decision", ""))
    if status not in ALLOWED_STATUS:
        raise ValueError("invalid status")
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("invalid reviewer decision")
    reviewer_note = str(payload.get("reviewer_note", ""))[:2000]
    with connect() as connection:
        connection.execute(
            """
            UPDATE review_cases
            SET status = ?, reviewer_decision = ?, reviewer_note = ?, updated_at = ?
            WHERE case_id = ?
            """,
            (status, decision, reviewer_note, utc_now(), case_id),
        )
        row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
    return row_to_dict(row) if row else None


def export_csv() -> str:
    rows = list_cases(limit=200)
    output = io.StringIO()
    if not rows:
        return "case_id,status\n"
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        row = dict(row)
        row["reason_codes"] = json.dumps(row["reason_codes"], ensure_ascii=False)
        writer.writerow(row)
    return output.getvalue()
