from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SalesMemory:
    user_phone: str
    device_id: str
    conversation_id: str
    intent: dict[str, Any]
    rows: list[dict[str, Any]]
    answer_summary: str


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_sales_memory(self, memory: SalesMemory) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sales_memories (
                    user_phone,
                    device_id,
                    conversation_id,
                    intent_json,
                    rows_json,
                    answer_summary,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_phone, device_id, conversation_id)
                DO UPDATE SET
                    intent_json = excluded.intent_json,
                    rows_json = excluded.rows_json,
                    answer_summary = excluded.answer_summary,
                    updated_at = excluded.updated_at
                """,
                (
                    memory.user_phone,
                    memory.device_id,
                    memory.conversation_id,
                    json.dumps(memory.intent, ensure_ascii=False),
                    json.dumps(memory.rows, ensure_ascii=False),
                    memory.answer_summary,
                    now,
                    now,
                ),
            )

    def get_sales_memory(
        self,
        user_phone: str,
        device_id: str,
        conversation_id: str,
    ) -> SalesMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    user_phone,
                    device_id,
                    conversation_id,
                    intent_json,
                    rows_json,
                    answer_summary
                FROM sales_memories
                WHERE user_phone = ? AND device_id = ? AND conversation_id = ?
                """,
                (user_phone, device_id, conversation_id),
            ).fetchone()
        if row is None:
            return None
        return SalesMemory(
            user_phone=str(row["user_phone"]),
            device_id=str(row["device_id"]),
            conversation_id=str(row["conversation_id"]),
            intent=_load_json_object(row["intent_json"]),
            rows=_load_json_list(row["rows_json"]),
            answer_summary=str(row["answer_summary"] or ""),
        )

    def set_preference(self, user_phone: str, device_id: str, key: str, value: str) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (
                    user_phone,
                    device_id,
                    key,
                    value,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_phone, device_id, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_phone, device_id, key, value, now, now),
            )

    def get_preference(self, user_phone: str, device_id: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value
                FROM user_preferences
                WHERE user_phone = ? AND device_id = ? AND key = ?
                """,
                (user_phone, device_id, key),
            ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sales_memories (
                    user_phone TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (user_phone, device_id, conversation_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_phone TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (user_phone, device_id, key)
                )
                """
            )


def _load_json_object(raw: str) -> dict[str, Any]:
    loaded = json.loads(raw)
    return loaded if isinstance(loaded, dict) else {}


def _load_json_list(raw: str) -> list[dict[str, Any]]:
    loaded = json.loads(raw)
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]
