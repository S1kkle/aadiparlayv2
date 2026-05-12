from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class SqliteTTLCache:
    """SQLite-backed cache + history + learning + calibration store.

    Performance notes:
    - Uses WAL journal mode (concurrent reads, single writer) configured per
      connection on first use.
    - Maintains a small bounded connection pool to avoid the connect-per-call
      overhead under concurrent load.
    - Synchronous=NORMAL is the recommended setting for WAL-mode app DBs
      (durable on power loss, fast on commit).
    """

    _POOL_SIZE = 8

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pool: "queue.LifoQueue[sqlite3.Connection]" = queue.LifoQueue(maxsize=self._POOL_SIZE)
        self._pool_count = 0
        self._pool_count_lock = threading.Lock()
        self._init_db()

    def _new_connection(self) -> sqlite3.Connection:
        # check_same_thread=False → safe because we wrap pool checkout/return
        # with synchronization. Single writer by SQLite design.
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL allows concurrent reads while one writer is active. NORMAL sync
        # is durable across power loss with WAL but ~2x faster than FULL.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=134217728")  # 128 MB
            conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.OperationalError:
            pass
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Borrow a pooled connection for the duration of the with-block."""
        conn: sqlite3.Connection | None = None
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            with self._pool_count_lock:
                if self._pool_count < self._POOL_SIZE:
                    conn = self._new_connection()
                    self._pool_count += 1
            if conn is None:
                conn = self._pool.get(timeout=30.0)
        try:
            yield conn
        finally:
            try:
                self._pool.put_nowait(conn)
            except queue.Full:
                conn.close()
                with self._pool_count_lock:
                    self._pool_count = max(0, self._pool_count - 1)

    # Backwards-compatible single-call connection (still pooled).
    def _connect(self) -> sqlite3.Connection:
        # Used only internally for pre-existing call patterns; new code
        # should prefer the _connection() context manager.
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._pool_count_lock:
                if self._pool_count < self._POOL_SIZE:
                    self._pool_count += 1
                    return self._new_connection()
        return self._pool.get(timeout=30.0)

    def _release(self, conn: sqlite3.Connection) -> None:
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._pool_count_lock:
                self._pool_count = max(0, self._pool_count - 1)

    def _init_db(self) -> None:
        with self._connection() as conn:
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
        with self._connection() as conn:
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
        with self._connection() as conn:
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
        """Clears all cached entries. Returns number of rows deleted."""
        with self._connection() as conn:
            cur = conn.execute("DELETE FROM cache")
            conn.commit()
            return int(cur.rowcount or 0)

    def clear_prefix(self, prefix: str) -> int:
        """Delete all cache entries whose key starts with the given prefix."""
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE key LIKE ?",
                (prefix + "%",),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def _ensure_history_table(self) -> None:
        with self._connection() as conn:
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
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prediction_history(id, timestamp, sport, props_json) VALUES(?, ?, ?, ?)",
                (entry_id, timestamp, sport, props_json),
            )
            conn.commit()

    def get_history(self, limit: int = 30) -> list[dict[str, Any]]:
        self._ensure_history_table()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, sport, props_json FROM prediction_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"id": r["id"], "timestamp": r["timestamp"], "sport": r["sport"], "props": json.loads(r["props_json"])}
                for r in rows
            ]

    # ── Learning log ───────────────────────────────────────────────────

    # Schema migrations are additive: we never drop columns, only ALTER ADD.
    # Add a new tuple to _LEARNING_MIGRATIONS to introduce a new column.
    _LEARNING_MIGRATIONS: list[tuple[str, str]] = [
        ("series_json", "ALTER TABLE learning_log ADD COLUMN series_json TEXT"),
        ("stat_field", "ALTER TABLE learning_log ADD COLUMN stat_field TEXT"),
        ("position", "ALTER TABLE learning_log ADD COLUMN position TEXT"),
        ("decimal_price", "ALTER TABLE learning_log ADD COLUMN decimal_price REAL"),
        ("payout_multiplier", "ALTER TABLE learning_log ADD COLUMN payout_multiplier REAL"),
        # CLV tracking — captures the line at-pick vs the closing line, the
        # single best leading-indicator of long-term ROI for sharp bettors.
        ("line_at_pick", "ALTER TABLE learning_log ADD COLUMN line_at_pick REAL"),
        ("close_line", "ALTER TABLE learning_log ADD COLUMN close_line REAL"),
        ("close_implied_prob", "ALTER TABLE learning_log ADD COLUMN close_implied_prob REAL"),
        # CLV in cents per leg (negative = picked late vs market, positive = beat the close)
        ("clv_cents", "ALTER TABLE learning_log ADD COLUMN clv_cents REAL"),
        # Bankroll accounting — what was actually risked / won
        ("stake_amount", "ALTER TABLE learning_log ADD COLUMN stake_amount REAL"),
        ("payout_amount", "ALTER TABLE learning_log ADD COLUMN payout_amount REAL"),
        ("profit", "ALTER TABLE learning_log ADD COLUMN profit REAL"),
        # Provenance — which prompt revision and calibration snapshot the
        # pick was made under, so we can attribute wins/losses to versions.
        ("prompt_version", "ALTER TABLE learning_log ADD COLUMN prompt_version TEXT"),
        ("model_params_id", "ALTER TABLE learning_log ADD COLUMN model_params_id TEXT"),
        # Underdog entry context (Standard/Insurance/Flex; per-leg boost)
        ("entry_type", "ALTER TABLE learning_log ADD COLUMN entry_type TEXT"),
    ]

    def _ensure_learning_table(self) -> None:
        with self._connection() as conn:
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
            for _col, ddl in self._LEARNING_MIGRATIONS:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column exists
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_resolved ON learning_log(resolved, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_history_id ON learning_log(history_id)"
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
        cols = [
            "id", "history_id", "timestamp", "player_name", "sport", "stat",
            "line", "side", "model_prob", "implied_prob", "edge", "ai_bias",
            "ai_confidence", "actual_value", "hit", "miss_reason",
            "miss_category", "resolved", "series_json", "stat_field",
            "position", "decimal_price", "payout_multiplier",
            "line_at_pick", "close_line", "close_implied_prob", "clv_cents",
            "stake_amount", "payout_amount", "profit",
            "prompt_version", "model_params_id", "entry_type",
        ]
        placeholders = ",".join(["?"] * len(cols))
        values = tuple(
            entry.get(c) if c not in ("id", "history_id", "timestamp",
                                      "player_name", "sport", "stat",
                                      "line", "side") else entry[c]
            for c in cols
        )
        # `resolved` defaults to 0 when not provided.
        if entry.get("resolved") is None:
            values = tuple(0 if c == "resolved" else v for c, v in zip(cols, values))
        sql = (
            f"INSERT OR REPLACE INTO learning_log ({','.join(cols)}) "
            f"VALUES ({placeholders})"
        )
        with self._connection() as conn:
            conn.execute(sql, values)
            conn.commit()

    def update_learning_clv(
        self,
        entry_id: str,
        *,
        close_line: float | None,
        close_implied_prob: float | None,
        clv_cents: float | None,
    ) -> None:
        """Update CLV columns after the closing line is captured."""
        self._ensure_learning_table()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE learning_log
                SET close_line = ?, close_implied_prob = ?, clv_cents = ?
                WHERE id = ?
                """,
                (close_line, close_implied_prob, clv_cents, entry_id),
            )
            conn.commit()

    def update_learning_settlement(
        self,
        entry_id: str,
        *,
        actual_value: float | None,
        hit: int | None,
        stake_amount: float | None = None,
        payout_amount: float | None = None,
        profit: float | None = None,
    ) -> None:
        """Update settlement fields once the game finishes and stake is known."""
        self._ensure_learning_table()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE learning_log
                SET actual_value = COALESCE(?, actual_value),
                    hit = COALESCE(?, hit),
                    resolved = CASE WHEN ? IS NOT NULL THEN 1 ELSE resolved END,
                    stake_amount = COALESCE(?, stake_amount),
                    payout_amount = COALESCE(?, payout_amount),
                    profit = COALESCE(?, profit)
                WHERE id = ?
                """,
                (
                    actual_value, hit, hit,
                    stake_amount, payout_amount, profit, entry_id,
                ),
            )
            conn.commit()

    def get_learning_entries(self, *, resolved_only: bool = False, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_learning_table()
        where = "WHERE resolved = 1" if resolved_only else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM learning_log {where} ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_learning_pipeline_snapshot(self) -> dict[str, int]:
        """Row counts for Learning Log empty-state diagnostics (read-only)."""
        self._ensure_learning_table()
        self._ensure_calibration_table()
        with self._connection() as conn:
            learning_total = int(
                conn.execute("SELECT COUNT(*) FROM learning_log").fetchone()[0]
            )
            learning_resolved = int(
                conn.execute(
                    "SELECT COUNT(*) FROM learning_log WHERE resolved = 1"
                ).fetchone()[0]
            )
            learning_resolved_scored = int(
                conn.execute(
                    "SELECT COUNT(*) FROM learning_log WHERE resolved = 1 AND hit IN (0, 1)"
                ).fetchone()[0]
            )
            calibration_runs = int(
                conn.execute("SELECT COUNT(*) FROM calibration_runs").fetchone()[0]
            )
            learning_reports = int(
                conn.execute("SELECT COUNT(*) FROM learning_reports").fetchone()[0]
            )
        return {
            "learning_total": learning_total,
            "learning_resolved": learning_resolved,
            "learning_resolved_scored": learning_resolved_scored,
            "calibration_runs": calibration_runs,
            "learning_reports": learning_reports,
        }

    def get_unresolved_learning_entries(self) -> list[dict[str, Any]]:
        self._ensure_learning_table()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM learning_log WHERE resolved = 0 ORDER BY timestamp ASC",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_learning_entries_missing_clv(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """Picks that have no captured close_line yet — used by the closing-line
        background job to fill in CLV after game start."""
        self._ensure_learning_table()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM learning_log
                WHERE close_line IS NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_learning_report(self, report: dict[str, Any]) -> None:
        self._ensure_learning_table()
        with self._connection() as conn:
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
        with self._connection() as conn:
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

    # ── Calibration runs ──────────────────────────────────────────────

    _CALIBRATION_MIGRATIONS: list[tuple[str, str]] = [
        ("brier", "ALTER TABLE calibration_runs ADD COLUMN brier REAL"),
        ("log_loss", "ALTER TABLE calibration_runs ADD COLUMN log_loss REAL"),
        ("ece", "ALTER TABLE calibration_runs ADD COLUMN ece REAL"),
        # Isotonic regression bin spec, JSON-encoded
        ("isotonic_json", "ALTER TABLE calibration_runs ADD COLUMN isotonic_json TEXT"),
    ]

    def _ensure_calibration_table(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_runs (
                  id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  accuracy_real REAL,
                  accuracy_synthetic REAL,
                  total_real INTEGER,
                  total_synthetic INTEGER,
                  params_json TEXT NOT NULL,
                  applied INTEGER DEFAULT 0,
                  source TEXT NOT NULL
                )
                """
            )
            for _col, ddl in self._CALIBRATION_MIGRATIONS:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def save_calibration_run(self, run: dict[str, Any]) -> None:
        self._ensure_calibration_table()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibration_runs
                (id, created_at, accuracy_real, accuracy_synthetic,
                 total_real, total_synthetic, params_json, applied, source,
                 brier, log_loss, ece, isotonic_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["id"], run["created_at"],
                    run.get("accuracy_real"), run.get("accuracy_synthetic"),
                    run.get("total_real"), run.get("total_synthetic"),
                    run["params_json"], run.get("applied", 0),
                    run.get("source", "scheduled"),
                    run.get("brier"), run.get("log_loss"), run.get("ece"),
                    run.get("isotonic_json"),
                ),
            )
            conn.commit()

    def get_latest_calibration(self, *, applied_only: bool = True) -> dict[str, Any] | None:
        self._ensure_calibration_table()
        where = "WHERE applied = 1" if applied_only else ""
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT * FROM calibration_runs {where} ORDER BY created_at DESC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["params"] = json.loads(d.pop("params_json"))
            if d.get("isotonic_json"):
                try:
                    d["isotonic"] = json.loads(d["isotonic_json"])
                except (TypeError, ValueError):
                    d["isotonic"] = None
            return d

    def get_calibration_history(self, limit: int = 20) -> list[dict[str, Any]]:
        self._ensure_calibration_table()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM calibration_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["params"] = json.loads(d.pop("params_json"))
                if d.get("isotonic_json"):
                    try:
                        d["isotonic"] = json.loads(d["isotonic_json"])
                    except (TypeError, ValueError):
                        d["isotonic"] = None
                out.append(d)
            return out

