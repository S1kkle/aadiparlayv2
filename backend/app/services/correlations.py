"""Empirical player-stat pairwise correlations.

The default same-player parlay penalty is a hardcoded 0.85 (≈ +0.20
correlation assumption between any two stats of the same player).
Reality varies hugely: a primary creator's points and assists run
+0.4 together while a high-usage gunner's points and assists can be
near-zero or negative. Hardcoding 0.85 over-penalizes some pairs and
under-penalizes others.

This module computes the per-(player, stat_a, stat_b) Pearson
correlation from resolved learning_log entries (their persisted
`series_json` per pick provides the underlying values). The output
is a multiplier `corr_factor ∈ [0.65, 1.0]` per pair, used by
`Ranker._correlation_factor` to replace the static 0.85 when an
empirical estimate exists.

Refresh cadence: weekly (or daily during the season); cached as
`correlations:player_stat:v1`.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import SqliteTTLCache

log = logging.getLogger(__name__)


_CACHE_KEY = "correlations:player_stat:v1"
_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = min(len(x), len(y))
    if n < 5:
        return None
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    sxx = sum((v - mx) ** 2 for v in x[:n])
    syy = sum((v - my) ** 2 for v in y[:n])
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _key(player: str, stat_a: str, stat_b: str) -> str:
    a, b = sorted([stat_a, stat_b])
    return f"{player.strip().lower()}|{a}|{b}"


def compute_player_stat_correlations(
    entries: list[dict[str, Any]],
    *,
    min_pairs: int = 5,
    window_days: int = 120,
) -> dict[str, dict[str, Any]]:
    """Compute per-(player, stat_a, stat_b) correlation from resolved entries.

    Returns mapping: key -> {"r": pearson_r, "n": pair_count}.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    # Group entries by (player, stat) → list of (timestamp, actual_value)
    by_player_stat: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for e in entries:
        ts_raw = e.get("timestamp")
        actual = e.get("actual_value")
        player = (e.get("player_name") or "").strip()
        stat = e.get("stat_field") or e.get("stat") or ""
        if not (ts_raw and isinstance(actual, (int, float)) and player and stat):
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if ts < cutoff:
            continue
        by_player_stat[(player.lower(), stat)].append((str(ts_raw), float(actual)))

    # For each player, find stat-pairs with overlapping timestamps.
    by_player: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    for (player, stat), series in by_player_stat.items():
        by_player[player][stat] = sorted(series, key=lambda t: t[0])

    out: dict[str, dict[str, Any]] = {}
    for player, stats_map in by_player.items():
        stat_names = list(stats_map.keys())
        for i in range(len(stat_names)):
            for j in range(i + 1, len(stat_names)):
                sa, sb = stat_names[i], stat_names[j]
                # Pair by timestamp — only games where both stats were resolved
                map_b = {ts: v for ts, v in stats_map[sb]}
                xs: list[float] = []
                ys: list[float] = []
                for ts, v in stats_map[sa]:
                    if ts in map_b:
                        xs.append(v)
                        ys.append(map_b[ts])
                if len(xs) < min_pairs:
                    continue
                r = _pearson(xs, ys)
                if r is None:
                    continue
                out[_key(player, sa, sb)] = {"r": round(r, 4), "n": len(xs)}
    return out


def correlation_penalty(r: float | None, *, default_r: float = 0.20) -> float:
    """Convert a Pearson correlation into a multiplicative parlay penalty.

    Joint hit-rate for two binary correlated outcomes (Underdog parlay
    pricing assumes independence): factor ≈ 1 - r * (1 - p_marginal_avg).
    We approximate p_marginal_avg ≈ 0.55 (typical model_prob for picked
    legs), so factor ≈ 1 - 0.45·r. Bounded to [0.65, 1.0] so even an
    r=1.0 doesn't completely zero out a pick (which would over-punish
    a real correlation that's not actually 100%).
    """
    rho = r if r is not None else default_r
    factor = 1.0 - 0.45 * max(0.0, rho)
    return max(0.65, min(1.0, factor))


def refresh_correlations(cache: SqliteTTLCache) -> dict[str, Any]:
    try:
        entries = cache.get_learning_entries(resolved_only=True, limit=100000)
    except Exception as exc:
        log.exception("correlations: failed to load learning entries")
        return {"status": "error", "error": str(exc)}

    corr = compute_player_stat_correlations(entries)
    cache.set_json(
        _CACHE_KEY,
        {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(corr),
            "data": corr,
        },
        ttl_seconds=_TTL_SECONDS,
    )
    if corr:
        log.info("Player-stat correlations refreshed: %d player-pair entries", len(corr))
    return {"status": "ok", "n_pairs": len(corr)}


def get_player_stat_correlation(
    cache: SqliteTTLCache, player: str, stat_a: str, stat_b: str
) -> float | None:
    """Look up the empirical Pearson correlation for a player's two stats."""
    if not player or not stat_a or not stat_b:
        return None
    try:
        payload = cache.get_json(_CACHE_KEY)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    entry = data.get(_key(player, stat_a, stat_b))
    if not isinstance(entry, dict):
        return None
    r = entry.get("r")
    return float(r) if isinstance(r, (int, float)) else None
