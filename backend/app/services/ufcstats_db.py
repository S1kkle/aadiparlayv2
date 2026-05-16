"""UFCStats SQLite schema and read accessors.

The schema is intentionally minimal — we persist only the fields the
ranker / hazard model / style clustering actually consume. Three tables:

  ufcstats_events     — one row per UFC event (date, name)
  ufcstats_fights     — one row per fight (event_id, fighters, method,
                        round/time finished, weight class)
  ufcstats_fight_rds  — one row per (fight_id, fighter_id, round_num)
                        with the per-round stats that ESPN doesn't expose

The DB file is the same SQLite cache the rest of the app uses
(SqliteTTLCache). We reuse it rather than creating a separate file
because the deployment story (Render free tier) prefers a single
persisted file and the cache schema is already migration-aware.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.db import SqliteTTLCache

log = logging.getLogger(__name__)


# Schema-migration DDLs — applied idempotently on first call to
# `_ensure_tables`.
_DDLS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS ufcstats_events (
      id              TEXT PRIMARY KEY,
      name            TEXT NOT NULL,
      event_date      TEXT,
      location        TEXT,
      scraped_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ufcstats_fights (
      id              TEXT PRIMARY KEY,
      event_id        TEXT NOT NULL,
      event_date      TEXT,
      weight_class    TEXT,
      method          TEXT,
      round_finished  INTEGER,
      time_finished   TEXT,
      scheduled_rounds INTEGER,
      fighter_a       TEXT NOT NULL,
      fighter_b       TEXT NOT NULL,
      winner          TEXT,
      scraped_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ufcstats_fight_rds (
      fight_id           TEXT NOT NULL,
      fighter            TEXT NOT NULL,
      round_num          INTEGER NOT NULL,
      sig_strikes_landed INTEGER,
      sig_strikes_attempted INTEGER,
      total_strikes_landed  INTEGER,
      takedowns_landed   INTEGER,
      takedowns_attempted INTEGER,
      sub_attempts       INTEGER,
      knockdowns         INTEGER,
      control_time_sec   INTEGER,
      head_strikes       INTEGER,
      body_strikes       INTEGER,
      leg_strikes        INTEGER,
      PRIMARY KEY (fight_id, fighter, round_num)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ufcfights_fighter_a ON ufcstats_fights(fighter_a)",
    "CREATE INDEX IF NOT EXISTS idx_ufcfights_fighter_b ON ufcstats_fights(fighter_b)",
    "CREATE INDEX IF NOT EXISTS idx_ufcfights_event_date ON ufcstats_fights(event_date)",
    "CREATE INDEX IF NOT EXISTS idx_ufcrds_fighter ON ufcstats_fight_rds(fighter)",
]


def _ensure_tables(cache: SqliteTTLCache) -> None:
    """Idempotent schema bootstrap. Calling on every read is cheap because
    `CREATE IF NOT EXISTS` is metadata-only after first execution."""
    # SqliteTTLCache exposes `_connection` as a context manager that hands
    # out a pooled connection — re-use it rather than opening a parallel
    # connection to the same file (avoids busy-locks).
    with cache._connection() as conn:
        for ddl in _DDLS:
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def upsert_event(
    cache: SqliteTTLCache,
    *,
    event_id: str,
    name: str,
    event_date: str | None,
    location: str | None,
) -> None:
    _ensure_tables(cache)
    with cache._connection() as conn:
        conn.execute(
            """
            INSERT INTO ufcstats_events(id, name, event_date, location, scraped_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name = excluded.name,
              event_date = excluded.event_date,
              location = excluded.location,
              scraped_at = excluded.scraped_at
            """,
            (event_id, name, event_date, location, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def upsert_fight(
    cache: SqliteTTLCache,
    *,
    fight_id: str,
    event_id: str,
    event_date: str | None,
    weight_class: str | None,
    method: str | None,
    round_finished: int | None,
    time_finished: str | None,
    scheduled_rounds: int | None,
    fighter_a: str,
    fighter_b: str,
    winner: str | None,
) -> None:
    _ensure_tables(cache)
    with cache._connection() as conn:
        conn.execute(
            """
            INSERT INTO ufcstats_fights(
              id, event_id, event_date, weight_class, method,
              round_finished, time_finished, scheduled_rounds,
              fighter_a, fighter_b, winner, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              method = excluded.method,
              round_finished = excluded.round_finished,
              time_finished = excluded.time_finished,
              winner = excluded.winner,
              scraped_at = excluded.scraped_at
            """,
            (
                fight_id, event_id, event_date, weight_class, method,
                round_finished, time_finished, scheduled_rounds,
                fighter_a, fighter_b, winner,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def upsert_fight_round(
    cache: SqliteTTLCache,
    *,
    fight_id: str,
    fighter: str,
    round_num: int,
    sig_strikes_landed: int | None = None,
    sig_strikes_attempted: int | None = None,
    total_strikes_landed: int | None = None,
    takedowns_landed: int | None = None,
    takedowns_attempted: int | None = None,
    sub_attempts: int | None = None,
    knockdowns: int | None = None,
    control_time_sec: int | None = None,
    head_strikes: int | None = None,
    body_strikes: int | None = None,
    leg_strikes: int | None = None,
) -> None:
    _ensure_tables(cache)
    with cache._connection() as conn:
        conn.execute(
            """
            INSERT INTO ufcstats_fight_rds(
              fight_id, fighter, round_num,
              sig_strikes_landed, sig_strikes_attempted, total_strikes_landed,
              takedowns_landed, takedowns_attempted,
              sub_attempts, knockdowns, control_time_sec,
              head_strikes, body_strikes, leg_strikes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fight_id, fighter, round_num) DO UPDATE SET
              sig_strikes_landed = excluded.sig_strikes_landed,
              sig_strikes_attempted = excluded.sig_strikes_attempted,
              total_strikes_landed = excluded.total_strikes_landed,
              takedowns_landed = excluded.takedowns_landed,
              takedowns_attempted = excluded.takedowns_attempted,
              sub_attempts = excluded.sub_attempts,
              knockdowns = excluded.knockdowns,
              control_time_sec = excluded.control_time_sec,
              head_strikes = excluded.head_strikes,
              body_strikes = excluded.body_strikes,
              leg_strikes = excluded.leg_strikes
            """,
            (
                fight_id, fighter, round_num,
                sig_strikes_landed, sig_strikes_attempted, total_strikes_landed,
                takedowns_landed, takedowns_attempted,
                sub_attempts, knockdowns, control_time_sec,
                head_strikes, body_strikes, leg_strikes,
            ),
        )
        conn.commit()


# ── Read accessors used by ranker / hazard model / style clustering ──


def _norm(name: str) -> str:
    return name.strip().lower()


def get_fighter_recent_fights(
    cache: SqliteTTLCache,
    *,
    fighter: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return up to `limit` most-recent fights for a fighter as dicts."""
    _ensure_tables(cache)
    nm = _norm(fighter)
    with cache._connection() as conn:
        rows = conn.execute(
            """
            SELECT id, event_date, weight_class, method, round_finished,
                   time_finished, scheduled_rounds, fighter_a, fighter_b, winner
            FROM ufcstats_fights
            WHERE LOWER(fighter_a) = ? OR LOWER(fighter_b) = ?
            ORDER BY event_date DESC
            LIMIT ?
            """,
            (nm, nm, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_fight_rounds(
    cache: SqliteTTLCache,
    *,
    fight_id: str,
    fighter: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_tables(cache)
    with cache._connection() as conn:
        if fighter:
            rows = conn.execute(
                """
                SELECT * FROM ufcstats_fight_rds
                WHERE fight_id = ? AND LOWER(fighter) = ?
                ORDER BY round_num ASC
                """,
                (fight_id, _norm(fighter)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ufcstats_fight_rds WHERE fight_id = ? ORDER BY round_num ASC",
                (fight_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_fighter_per_minute_aggregates(
    cache: SqliteTTLCache,
    *,
    fighter: str,
    last_n_fights: int = 8,
) -> dict[str, Any]:
    """Compute per-minute aggregates from the most recent N fights:

      sig_strikes_per_min        — sig strikes landed / total fight minutes
      sig_strikes_absorbed_pm    — opponent sig strikes landed / fight min
      takedowns_per_15min        — takedowns landed × (15 / fight min)
      control_time_pct           — control_time_sec / fight_seconds
      sub_attempts_per_15min     — same scaling as TDs
      head_strike_pct            — head strikes / total sig strikes

    Returns {} if no fights are in the DB for this fighter.
    """
    fights = get_fighter_recent_fights(cache, fighter=fighter, limit=last_n_fights)
    if not fights:
        return {}
    nm = _norm(fighter)

    total_seconds = 0
    sig_land = 0
    sig_absorbed = 0
    td_land = 0
    sub_att = 0
    control_sec = 0
    head_sig = 0
    body_sig = 0
    leg_sig = 0

    for fight in fights:
        rds = get_fight_rounds(cache, fight_id=fight["id"])
        if not rds:
            continue
        # Sum fighter's rounds + opponent's rounds.
        f_rds = [r for r in rds if _norm(r["fighter"]) == nm]
        o_rds = [r for r in rds if _norm(r["fighter"]) != nm]
        if not f_rds:
            continue
        # Compute total seconds: full rounds of 300s except for the final
        # round which used time_finished. If not present, assume full.
        n_rounds = max(r["round_num"] for r in f_rds)
        sched = fight.get("scheduled_rounds") or 3
        for rd in range(1, n_rounds + 1):
            if rd < n_rounds:
                total_seconds += 300
            else:
                tf = fight.get("time_finished")
                # Parse "M:SS" — if it doesn't parse or fight went the distance,
                # use full 5 minutes.
                if isinstance(tf, str) and ":" in tf and rd == fight.get("round_finished"):
                    try:
                        m, s = tf.split(":")
                        total_seconds += int(m) * 60 + int(s)
                    except (ValueError, TypeError):
                        total_seconds += 300
                else:
                    total_seconds += 300

        for r in f_rds:
            sig_land += r.get("sig_strikes_landed") or 0
            td_land += r.get("takedowns_landed") or 0
            sub_att += r.get("sub_attempts") or 0
            control_sec += r.get("control_time_sec") or 0
            head_sig += r.get("head_strikes") or 0
            body_sig += r.get("body_strikes") or 0
            leg_sig += r.get("leg_strikes") or 0
        for r in o_rds:
            sig_absorbed += r.get("sig_strikes_landed") or 0

    if total_seconds <= 0:
        return {}
    minutes = total_seconds / 60.0
    return {
        "n_fights": len(fights),
        "total_minutes": round(minutes, 1),
        "sig_strikes_per_min": round(sig_land / max(1.0, minutes), 3),
        "sig_strikes_absorbed_pm": round(sig_absorbed / max(1.0, minutes), 3),
        "takedowns_per_15min": round((td_land * 15.0) / max(1.0, minutes), 3),
        "sub_attempts_per_15min": round((sub_att * 15.0) / max(1.0, minutes), 3),
        "control_time_pct": round(control_sec / max(1, total_seconds), 3),
        "head_strike_pct": round(head_sig / max(1, sig_land), 3) if sig_land else 0.0,
        "body_strike_pct": round(body_sig / max(1, sig_land), 3) if sig_land else 0.0,
        "leg_strike_pct": round(leg_sig / max(1, sig_land), 3) if sig_land else 0.0,
    }


def get_finish_history(
    cache: SqliteTTLCache, *, fighter: str, last_n: int = 12,
) -> list[dict[str, Any]]:
    """Recent fights as (round_finished, method) tuples — feeds the hazard model."""
    fights = get_fighter_recent_fights(cache, fighter=fighter, limit=last_n)
    out: list[dict[str, Any]] = []
    nm = _norm(fighter)
    for f in fights:
        method = (f.get("method") or "").upper()
        winner = _norm(f.get("winner") or "")
        is_winner = winner == nm
        out.append({
            "round_finished": f.get("round_finished"),
            "method": method,
            "won": is_winner,
            "scheduled_rounds": f.get("scheduled_rounds") or 3,
            "weight_class": f.get("weight_class"),
        })
    return out


def get_scraper_status(cache: SqliteTTLCache) -> dict[str, Any]:
    _ensure_tables(cache)
    with cache._connection() as conn:
        n_events = conn.execute("SELECT COUNT(*) AS n FROM ufcstats_events").fetchone()["n"]
        n_fights = conn.execute("SELECT COUNT(*) AS n FROM ufcstats_fights").fetchone()["n"]
        n_rounds = conn.execute("SELECT COUNT(*) AS n FROM ufcstats_fight_rds").fetchone()["n"]
        latest = conn.execute(
            "SELECT MAX(event_date) AS d FROM ufcstats_events"
        ).fetchone()["d"]
        last_scrape = conn.execute(
            "SELECT MAX(scraped_at) AS t FROM ufcstats_events"
        ).fetchone()["t"]
    return {
        "n_events": int(n_events or 0),
        "n_fights": int(n_fights or 0),
        "n_rounds": int(n_rounds or 0),
        "latest_event_date": latest,
        "last_scrape": last_scrape,
    }
