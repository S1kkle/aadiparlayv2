"""Inactive-list watcher with WOWY-style usage repricing.

Underdog Fantasy / PrizePicks set proprietary prop lines via analysts and
adjust slowly — typical 30-90 minute lag after a star scratch lands on
the inactive list (Rithmm research; corroborated by multiple ops). This
is the single most exploitable structural edge on those platforms: when
LeBron is ruled out at 6:30pm ET for a 7pm tip, Russell / Reaves usage
jumps materially but the lines for those teammates often don't reflect
it for an hour.

This watcher:
  1. Polls ESPN's league-level injuries feed every WATCHER_INTERVAL_MIN
     minutes during the slate window (default 60s during game hours).
  2. Detects NEW OUT/DOUBTFUL entries vs the prior poll.
  3. Invalidates ranker caches for the relevant sport so the next
     `/props` request re-fetches with the fresh injury list.
  4. Stamps `recent_inactives` in cache so the ranker can surface a
     "fresh inactive: <player>" hint on top picks for affected teams.

We deliberately don't try to *automatically* re-rank in the background;
the ranker is heavy and re-ranking on every poll would torch the
free-tier Render budget. Instead we invalidate caches so the next
user-driven fetch is fresh, and surface a `/inactives/recent` endpoint
the UI can poll for the "fresh inactive" feed.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.clients.espn import EspnClient
from app.db import SqliteTTLCache

log = logging.getLogger(__name__)


_CACHE_KEY_KNOWN = "inactives:known:v1"
_CACHE_KEY_RECENT = "inactives:recent:v1"

# Statuses that materially affect counting-stat projections via usage
# redistribution. PROBABLE / QUESTIONABLE don't trigger because they
# usually still play (and the existing AI play_probability handles them).
_INACTIVE_STATUSES = frozenset({"OUT", "INJURED", "INACTIVE", "SUSPENDED", "DOUBTFUL"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


async def _scan_sport(
    espn: EspnClient,
    cache: SqliteTTLCache,
    *,
    sport: str,
    league: str,
) -> list[dict[str, Any]]:
    """Return a list of NEW inactive entries for (sport, league) since last scan."""
    try:
        data = await espn.fetch_league_injuries(sport=sport, league=league)
    except Exception:
        log.exception("inactive_watcher: fetch_league_injuries(%s/%s) failed", sport, league)
        return []

    fresh: list[dict[str, Any]] = []
    teams = data.get("injuries") or []
    if not isinstance(teams, list):
        return fresh

    # Read prior set; key uniquely by (sport, athlete_id) so the same player
    # changing status doesn't double-fire.
    raw_known = cache.get_json(_CACHE_KEY_KNOWN)
    known: dict[str, str] = (
        raw_known.get("by_athlete", {})
        if isinstance(raw_known, dict) and isinstance(raw_known.get("by_athlete"), dict)
        else {}
    )

    for team in teams:
        if not isinstance(team, dict):
            continue
        team_abbr = (team.get("abbreviation") or "").upper()
        for inj in team.get("injuries") or []:
            if not isinstance(inj, dict):
                continue
            athlete = inj.get("athlete") or {}
            if not isinstance(athlete, dict):
                continue
            aid = str(athlete.get("id") or "").strip()
            if not aid:
                continue
            name = athlete.get("displayName") or athlete.get("fullName") or ""
            status = (inj.get("status") or "").upper().strip()
            key = f"{sport}:{aid}"
            prev_status = known.get(key)
            if status in _INACTIVE_STATUSES and prev_status != status:
                fresh.append({
                    "sport": sport.upper() if sport.upper() != "BASKETBALL" else "NBA",
                    "league": league,
                    "athlete_id": aid,
                    "player_name": name,
                    "team_abbr": team_abbr,
                    "status": status,
                    "prior_status": prev_status,
                    "detected_at": _now().isoformat(),
                    "position": (athlete.get("position") or {}).get("abbreviation"),
                })
            # Always update the known map (including non-inactive statuses so a
            # status downgrade back to PROBABLE registers correctly next time).
            if status:
                known[key] = status

    # Persist updated known map.
    cache.set_json(
        _CACHE_KEY_KNOWN,
        {"by_athlete": known, "updated_at": _now().isoformat()},
        ttl_seconds=30 * 24 * 3600,
    )
    return fresh


def _append_recent(cache: SqliteTTLCache, fresh: list[dict[str, Any]]) -> None:
    """Maintain a 50-entry rolling window of newly-detected inactives so the
    UI can render a 'fresh inactive' feed without polling every team.
    """
    if not fresh:
        return
    raw = cache.get_json(_CACHE_KEY_RECENT)
    existing: list[dict[str, Any]] = (
        raw.get("entries", []) if isinstance(raw, dict) else []
    )
    cutoff = _now() - timedelta(hours=12)
    existing = [
        e for e in existing
        if isinstance(e, dict) and (_parse_dt(e.get("detected_at")) or _now()) >= cutoff
    ]
    merged = fresh + existing
    cache.set_json(
        _CACHE_KEY_RECENT,
        {"entries": merged[:50], "updated_at": _now().isoformat()},
        ttl_seconds=24 * 3600,
    )


def _invalidate_ranker_caches(cache: SqliteTTLCache, sports_seen: set[str]) -> None:
    """Drop the AI-selection + per-prop AI summary caches for the affected
    sports so the next /props fetch picks up the fresh inactive list. We
    deliberately preserve gamelog caches (the new inactive doesn't change
    the prior games' box scores)."""
    if not sports_seen:
        return
    cache.clear_prefix("ai_select:")
    cache.clear_prefix("ollama:prop:")


def get_recent_inactives(cache: SqliteTTLCache, *, hours: int = 6) -> list[dict[str, Any]]:
    raw = cache.get_json(_CACHE_KEY_RECENT)
    entries = (raw or {}).get("entries", []) if isinstance(raw, dict) else []
    cutoff = _now() - timedelta(hours=hours)
    return [
        e for e in entries
        if isinstance(e, dict) and (_parse_dt(e.get("detected_at")) or _now()) >= cutoff
    ]


async def scan_all(espn: EspnClient, cache: SqliteTTLCache) -> dict[str, Any]:
    """One full scan across NBA / NFL / NHL. Returns a summary dict."""
    pairs = [
        ("basketball", "nba"),
        ("football", "nfl"),
        ("hockey", "nhl"),
    ]
    all_fresh: list[dict[str, Any]] = []
    sports_with_changes: set[str] = set()
    for sport, league in pairs:
        fresh = await _scan_sport(espn, cache, sport=sport, league=league)
        if fresh:
            sports_with_changes.add(sport)
            all_fresh.extend(fresh)
    _append_recent(cache, all_fresh)
    _invalidate_ranker_caches(cache, sports_with_changes)
    if all_fresh:
        log.info(
            "Inactive watcher: %d new inactive(s) detected across %s",
            len(all_fresh),
            ",".join(sorted(sports_with_changes)),
        )
    return {"n_new": len(all_fresh), "fresh": all_fresh}


async def inactive_watcher_loop(
    espn: EspnClient,
    cache: SqliteTTLCache,
    *,
    interval_seconds: int = 180,
) -> None:
    """Long-running watcher loop. Default 3-minute cadence is the sweet spot
    between exploit-window capture (Underdog/PrizePicks usually re-price
    within 30-90 minutes) and ESPN rate-limit friendliness.

    The free-tier Render web service spins down after 15 min idle; if you
    want true game-window watching, set INACTIVE_WATCH_INTERVAL_SEC=60 in
    production env and run on a paid plan.
    """
    await asyncio.sleep(90)  # offset from other startup loops
    while True:
        try:
            await scan_all(espn, cache)
        except Exception:
            log.exception("inactive_watcher_loop iteration failed")
        await asyncio.sleep(interval_seconds)
