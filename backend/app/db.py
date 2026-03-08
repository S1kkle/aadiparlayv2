from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class SqliteTTLCache:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                      key TEXT PRIMARY KEY,
                      value_json TEXT NOT NULL,
                      expires_at_epoch INTEGER NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at_epoch)")
                conn.commit()

    def get_json(self, key: str) -> Any | None:
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value_json, expires_at_epoch FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                if int(row["expires_at_epoch"]) <= now:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
                return json.loads(row["value_json"])

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires = int(time.time()) + int(ttl_seconds)
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO cache(key, value_json, expires_at_epoch)
                    VALUES(?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                      value_json = excluded.value_json,
                      expires_at_epoch = excluded.expires_at_epoch
                    """,
                    (key, payload, expires),
                )
                conn.commit()

    def clear(self) -> int:
        """
        Clears all cached entries. Returns number of rows deleted.
        """
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM cache")
                conn.commit()
                return int(cur.rowcount or 0)

    def clear_prefix(self, prefix: str) -> int:
        """Delete all cache entries whose key starts with the given prefix."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM cache WHERE key LIKE ?",
                    (prefix + "%",),
                )
                conn.commit()
                return int(cur.rowcount or 0)

    def _ensure_history_table(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_history (
                      id TEXT PRIMARY KEY,
                      timestamp TEXT NOT NULL,
                      sport TEXT NOT NULL,
                      props_json TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def save_history(self, entry_id: str, timestamp: str, sport: str, props_json: str) -> None:
        self._ensure_history_table()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prediction_history(id, timestamp, sport, props_json) VALUES(?, ?, ?, ?)",
                    (entry_id, timestamp, sport, props_json),
                )
                conn.commit()

    def get_history(self, limit: int = 30) -> list[dict[str, Any]]:
        self._ensure_history_table()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, sport, props_json FROM prediction_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [
                    {"id": r["id"], "timestamp": r["timestamp"], "sport": r["sport"], "props": json.loads(r["props_json"])}
                    for r in rows
                ]

