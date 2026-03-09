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

    # ── Learning log ───────────────────────────────────────────────────

    def _ensure_learning_table(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_log (
                      id TEXT PRIMARY KEY,
                      history_id TEXT NOT NULL,
                      timestamp TEXT NOT NULL,
                      player_name TEXT NOT NULL,
                      sport TEXT NOT NULL,
                      stat TEXT NOT NULL,
                      line REAL NOT NULL,
                      side TEXT NOT NULL,
                      model_prob REAL,
                      implied_prob REAL,
                      edge REAL,
                      ai_bias INTEGER,
                      ai_confidence REAL,
                      actual_value REAL,
                      hit INTEGER,
                      miss_reason TEXT,
                      miss_category TEXT,
                      resolved INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_reports (
                      id TEXT PRIMARY KEY,
                      week_start TEXT NOT NULL,
                      week_end TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      total_picks INTEGER,
                      hits INTEGER,
                      misses INTEGER,
                      hit_rate REAL,
                      miss_breakdown_json TEXT,
                      suggestions_json TEXT
                    )
                    """
                )
                conn.commit()

    def save_learning_entry(self, entry: dict[str, Any]) -> None:
        self._ensure_learning_table()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO learning_log
                    (id, history_id, timestamp, player_name, sport, stat, line, side,
                     model_prob, implied_prob, edge, ai_bias, ai_confidence,
                     actual_value, hit, miss_reason, miss_category, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["id"], entry["history_id"], entry["timestamp"],
                        entry["player_name"], entry["sport"], entry["stat"],
                        entry["line"], entry["side"],
                        entry.get("model_prob"), entry.get("implied_prob"),
                        entry.get("edge"), entry.get("ai_bias"),
                        entry.get("ai_confidence"), entry.get("actual_value"),
                        entry.get("hit"), entry.get("miss_reason"),
                        entry.get("miss_category"), entry.get("resolved", 0),
                    ),
                )
                conn.commit()

    def get_learning_entries(self, *, resolved_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_learning_table()
        where = "WHERE resolved = 1" if resolved_only else ""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM learning_log {where} ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]

    def get_unresolved_learning_entries(self) -> list[dict[str, Any]]:
        self._ensure_learning_table()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM learning_log WHERE resolved = 0 ORDER BY timestamp ASC",
                ).fetchall()
                return [dict(r) for r in rows]

    def save_learning_report(self, report: dict[str, Any]) -> None:
        self._ensure_learning_table()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO learning_reports
                    (id, week_start, week_end, created_at, total_picks, hits, misses,
                     hit_rate, miss_breakdown_json, suggestions_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report["id"], report["week_start"], report["week_end"],
                        report["created_at"], report["total_picks"],
                        report["hits"], report["misses"], report["hit_rate"],
                        report.get("miss_breakdown_json", "{}"),
                        report.get("suggestions_json", "[]"),
                    ),
                )
                conn.commit()

    def get_learning_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        self._ensure_learning_table()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM learning_reports ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    d["miss_breakdown"] = json.loads(d.pop("miss_breakdown_json", "{}"))
                    d["suggestions"] = json.loads(d.pop("suggestions_json", "[]"))
                    out.append(d)
                return out

