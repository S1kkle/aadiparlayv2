"""Online Bayesian per-player priors with hierarchical pooling.

Combines two ideas:

1. **Per-player conjugate Bayesian updates** — after each resolved game,
   incorporate the new observation into a per-(player, stat) posterior
   in closed form. No SGD, no retraining; literally one line of math
   per observation. Posterior mean and effective sample size become
   the player's "prior" the next time the model fits him.

2. **Hierarchical (player → position → league) partial pooling** —
   when a player's effective sample size is small, the posterior blends
   toward the position archetype, which itself blends toward the league
   prior. This is the EPAA / DARKO-style three-level pooling that
   research-grade NBA models use.

Storage: cache key `online_priors:v1` holds a JSON dict keyed by
`(sport, stat_field, player_lowercase)` with `{mu, n_eff, last_updated}`.

The classical conjugate update for an unknown-mean / known-variance
gaussian (which is what we have once σ is fit from the recent series):

    n_post = n_prior + 1
    mu_post = (n_prior * mu_prior + obs) / n_post

`n_eff` is bounded above so a 200-game player doesn't have a posterior
so concentrated that one outlier game can't move it. Cap default is 40.

For the hierarchical pool the position-archetype layer is derived from
the same online posteriors (averaged across players of that position).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.db import SqliteTTLCache
from app.services.stat_model import league_prior_mean

log = logging.getLogger(__name__)


_CACHE_KEY = "online_priors:v1"
_TTL = 30 * 24 * 3600  # 30 days
_N_EFF_CAP = 40
_POSITION_POOL_K = 6.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(sport: str, stat_field: str, player: str) -> str:
    return f"{sport.upper()}|{stat_field}|{player.strip().lower()}"


def _load(cache: SqliteTTLCache) -> dict[str, dict[str, Any]]:
    raw = cache.get_json(_CACHE_KEY)
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        return raw["data"]
    return {}


def _save(cache: SqliteTTLCache, data: dict[str, dict[str, Any]]) -> None:
    cache.set_json(
        _CACHE_KEY,
        {"data": data, "updated_at": _now_iso()},
        ttl_seconds=_TTL,
    )


def update_player_prior(
    cache: SqliteTTLCache,
    *,
    sport: str,
    stat_field: str,
    player: str,
    position: str | None,
    observation: float,
    weight_class: str | None = None,
    n_eff_cap: int = _N_EFF_CAP,
) -> dict[str, float]:
    """Apply one conjugate update to the player's posterior. O(1) per call.

    For MMA, the optional `weight_class` is persisted so the hierarchical
    prior lookup can use it as a pooling layer.
    """
    if not sport or not stat_field or not player:
        return {}
    data = _load(cache)
    k = _key(sport, stat_field, player)
    prev = data.get(k)
    if prev:
        mu_prior = float(prev.get("mu", observation))
        n_prior = float(prev.get("n_eff", 1))
        # Preserve weight_class from prior entry when caller didn't supply it.
        if not weight_class and prev.get("weight_class"):
            weight_class = prev.get("weight_class")
    else:
        # Seed the prior with the league baseline so the first observation
        # doesn't fully define the player.
        mu_prior = league_prior_mean(
            sport=sport, stat_field=stat_field, position=position,
            fallback=observation,
        ) or observation
        n_prior = 1.0

    n_post = min(float(n_eff_cap), n_prior + 1.0)
    mu_post = (n_prior * mu_prior + float(observation)) / (n_prior + 1.0)

    data[k] = {
        "mu": round(mu_post, 4),
        "n_eff": round(n_post, 2),
        "position": position or "",
        "weight_class": weight_class or "",
        "last_updated": _now_iso(),
    }
    _save(cache, data)
    return data[k]


def get_hierarchical_prior(
    cache: SqliteTTLCache,
    *,
    sport: str,
    stat_field: str,
    player: str,
    position: str | None,
    weight_class: str | None = None,
) -> float | None:
    """Multi-level partial pool: player → (weight_class for MMA) → position → league.

    For MMA, `weight_class` (heavyweight / lightweight / women's flyweight…)
    adds an additional pooling layer between fighter posterior and the
    global league prior. MMA per-fighter sample sizes are 8-20 fights vs
    NBA's 500-1000+ games, so partial pooling is more load-bearing here
    than in team sports — see Diving Into Data's Bayesian MMA ranking
    work for the motivation.

    Each level contributes weight proportional to its effective sample
    size; small-sample players are pulled toward the position posterior,
    which itself is pulled toward the league baseline by _POSITION_POOL_K.
    """
    data = _load(cache)
    league = league_prior_mean(sport=sport, stat_field=stat_field, position=position)

    # Level 1: player posterior
    player_entry = data.get(_key(sport, stat_field, player))
    if player_entry:
        mu_p = float(player_entry.get("mu"))
        n_p = float(player_entry.get("n_eff", 0))
    else:
        mu_p = None
        n_p = 0.0

    # Level 2 (MMA only): weight-class pool. Heavyweight fighters average
    # ~25% fewer significant strikes per fight than featherweights because
    # KO rate is materially higher; failing to pool by class biases small-
    # sample heavyweights toward the cross-class league mean.
    sport_upper = sport.upper()
    mu_wc_pooled: float | None = None
    if sport_upper == "MMA" and weight_class:
        wc_upper = weight_class.strip().lower()
        prefix = f"{sport_upper}|{stat_field}|"
        wc_values: list[tuple[float, float]] = []
        for entry_key, entry in data.items():
            if not entry_key.startswith(prefix):
                continue
            if (entry.get("weight_class") or "").lower() == wc_upper:
                wc_values.append((float(entry.get("mu", 0)), float(entry.get("n_eff", 0))))
        if wc_values:
            total_w = sum(n for _, n in wc_values)
            mu_wc = sum(mu * n for mu, n in wc_values) / max(1.0, total_w)
            if league is not None:
                mu_wc_pooled = (
                    total_w * mu_wc + _POSITION_POOL_K * league
                ) / (total_w + _POSITION_POOL_K)
            else:
                mu_wc_pooled = mu_wc

    # Level 3 / fallback level 2 for non-MMA: position pool. Falls back to
    # league when no other players have data for this position+stat yet.
    pos_upper = (position or "").strip().upper()
    if pos_upper:
        prefix = f"{sport_upper}|{stat_field}|"
        pos_values: list[tuple[float, float]] = []
        for entry_key, entry in data.items():
            if not entry_key.startswith(prefix):
                continue
            if (entry.get("position") or "").upper() == pos_upper:
                pos_values.append((float(entry.get("mu", 0)), float(entry.get("n_eff", 0))))
        if pos_values:
            total_w = sum(n for _, n in pos_values)
            mu_pos = sum(mu * n for mu, n in pos_values) / max(1.0, total_w)
            if league is not None:
                mu_pos_pooled = (
                    total_w * mu_pos + _POSITION_POOL_K * league
                ) / (total_w + _POSITION_POOL_K)
            else:
                mu_pos_pooled = mu_pos
        else:
            mu_pos_pooled = league
    else:
        mu_pos_pooled = league

    # MMA weight-class pool takes precedence over position pool when present.
    pool_target = mu_wc_pooled if mu_wc_pooled is not None else mu_pos_pooled

    if mu_p is not None and n_p > 0:
        if pool_target is not None:
            return (n_p * mu_p + _POSITION_POOL_K * pool_target) / (n_p + _POSITION_POOL_K)
        return mu_p
    return pool_target


def replay_from_learning_log(
    cache: SqliteTTLCache, *, limit: int = 50000
) -> dict[str, Any]:
    """One-shot replay of historical resolved learning_log entries to seed
    the online priors. Idempotent: priors track their last_updated stamp,
    and we cap n_eff to prevent runaway concentration.

    Returns a structured dict for the operator API.
    """
    try:
        entries = cache.get_learning_entries(resolved_only=True, limit=limit)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    # Replay in chronological order so older updates compound correctly.
    entries.sort(key=lambda e: e.get("timestamp") or "")

    updated = 0
    for e in entries:
        actual = e.get("actual_value")
        if not isinstance(actual, (int, float)):
            continue
        sport = (e.get("sport") or "").upper()
        stat_field = e.get("stat_field") or e.get("stat") or ""
        player = e.get("player_name") or ""
        position = e.get("position")
        if not (sport and stat_field and player):
            continue
        update_player_prior(
            cache,
            sport=sport, stat_field=stat_field, player=player, position=position,
            observation=float(actual),
        )
        updated += 1

    data = _load(cache)
    return {"status": "ok", "updated": updated, "n_players_stats": len(data)}


def get_all_priors(cache: SqliteTTLCache) -> dict[str, dict[str, Any]]:
    return _load(cache)
