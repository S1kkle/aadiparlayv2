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

