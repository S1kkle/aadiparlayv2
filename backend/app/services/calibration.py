"""
Calibration service: backtests the stat model against real resolved outcomes
and synthetic player archetypes, then grid-searches for optimal parameters.

Designed to run on a schedule (every ~2 days) or manually via API.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from app.db import SqliteTTLCache
from app.services.stat_model import (
    fit_isotonic,
    get_calibrated_params,
    get_default_params,
    load_calibrated_params,
    load_isotonic_calibrator,
    prob_over_for_field,
)

log = logging.getLogger(__name__)

MIN_REAL_ENTRIES = 100  # was 20; 20 binary samples lets noise pass 65% ~25% of the time
REAL_ACCURACY_THRESHOLD = 0.58  # 55% break-even → require modest cushion above noise
SYNTHETIC_ACCURACY_THRESHOLD = 0.60

LOW_COUNT_STATS = frozenset({
    "blocks", "steals", "knockDowns", "takedownsLanded",
    "submissions", "advances", "interceptions",
    "rushingTouchdowns", "receivingTouchdowns", "passingTouchdowns",
    "goals",
})

NBA_STATS = [
    "points", "totalRebounds", "assists",
    "threePointFieldGoalsMade", "steals", "blocks",
]

# MMA stats we know how to backtest synthetically. Limited to count-style
# fight totals — round-of-finish / fight-time props need the hazard model
# in Batch 2 to evaluate properly and don't fit the same Gaussian-fit-then-
# distribution-dispatch backtest harness.
MMA_STATS = [
    "sigStrikesLanded",
    "totalStrikesLanded",
    "takedownsLanded",
    "knockDowns",
    "submissions",
]

# MMA fighter archetypes — (mean per-fight, std per-fight) for each stat.
# Calibrated against public UFCStats career-average distributions across
# weight classes. The intent is to validate the MMA branch independently
# of NBA archetypes during calibration runs when real MMA learning_log
# data is sparse. Per-fight values; treated by `generate_synthetic_data`
# the same as NBA archetypes (Gaussian draw with archetype mean ± std).
MMA_ARCHETYPES = [
    # Striker archetypes (heavy SLpM, low TDA)
    {"name": "Volume_Striker_LW",
     "sigStrikesLanded": (95, 30), "totalStrikesLanded": (130, 35),
     "takedownsLanded": (0.2, 0.4), "knockDowns": (0.4, 0.6),
     "submissions": (0.1, 0.3)},
    {"name": "Volume_Striker_WW",
     "sigStrikesLanded": (85, 28), "totalStrikesLanded": (115, 30),
     "takedownsLanded": (0.3, 0.5), "knockDowns": (0.5, 0.7),
     "submissions": (0.1, 0.3)},
    {"name": "Power_Striker_HW",
     "sigStrikesLanded": (40, 20), "totalStrikesLanded": (55, 25),
     "takedownsLanded": (0.1, 0.3), "knockDowns": (0.9, 0.9),
     "submissions": (0.0, 0.2)},
    {"name": "Counter_Striker_MW",
     "sigStrikesLanded": (55, 18), "totalStrikesLanded": (70, 20),
     "takedownsLanded": (0.3, 0.5), "knockDowns": (0.6, 0.7),
     "submissions": (0.1, 0.3)},
    # Wrestler archetypes (low SLpM, high TDA, moderate sub)
    {"name": "Wrestle_Boxer_LW",
     "sigStrikesLanded": (60, 22), "totalStrikesLanded": (85, 28),
     "takedownsLanded": (2.5, 1.5), "knockDowns": (0.2, 0.4),
     "submissions": (0.3, 0.5)},
    {"name": "Dominant_Wrestler_WW",
     "sigStrikesLanded": (45, 18), "totalStrikesLanded": (70, 25),
     "takedownsLanded": (3.5, 1.8), "knockDowns": (0.1, 0.3),
     "submissions": (0.4, 0.6)},
    {"name": "Greco_Wrestler_HW",
     "sigStrikesLanded": (30, 15), "totalStrikesLanded": (50, 20),
     "takedownsLanded": (2.0, 1.2), "knockDowns": (0.3, 0.5),
     "submissions": (0.2, 0.4)},
    # Grappler archetypes (low SLpM, mid TDA, high sub attempts)
    {"name": "BJJ_Specialist_LW",
     "sigStrikesLanded": (40, 18), "totalStrikesLanded": (60, 22),
     "takedownsLanded": (1.8, 1.0), "knockDowns": (0.1, 0.3),
     "submissions": (1.2, 0.9)},
    {"name": "BJJ_Specialist_WW",
     "sigStrikesLanded": (50, 20), "totalStrikesLanded": (75, 25),
     "takedownsLanded": (2.0, 1.2), "knockDowns": (0.2, 0.4),
     "submissions": (1.0, 0.8)},
    # Hybrid / well-rounded
    {"name": "Hybrid_LW",
     "sigStrikesLanded": (70, 25), "totalStrikesLanded": (95, 28),
     "takedownsLanded": (1.5, 1.0), "knockDowns": (0.4, 0.6),
     "submissions": (0.4, 0.6)},
    {"name": "Hybrid_FW",
     "sigStrikesLanded": (75, 27), "totalStrikesLanded": (100, 30),
     "takedownsLanded": (1.3, 1.0), "knockDowns": (0.3, 0.5),
     "submissions": (0.5, 0.6)},
    # Women's divisions
    {"name": "WStrawweight_Volume",
     "sigStrikesLanded": (80, 25), "totalStrikesLanded": (110, 30),
     "takedownsLanded": (1.0, 0.8), "knockDowns": (0.2, 0.4),
     "submissions": (0.3, 0.5)},
    {"name": "WFlyweight_Wrestler",
     "sigStrikesLanded": (55, 20), "totalStrikesLanded": (80, 25),
     "takedownsLanded": (2.0, 1.2), "knockDowns": (0.1, 0.3),
     "submissions": (0.5, 0.6)},
    # Decision-fighter archetype (low finish rate)
    {"name": "Decision_Fighter",
     "sigStrikesLanded": (85, 22), "totalStrikesLanded": (115, 25),
     "takedownsLanded": (1.5, 1.0), "knockDowns": (0.1, 0.3),
     "submissions": (0.2, 0.4)},
    # Early-career / inconsistent
    {"name": "Young_Prospect",
     "sigStrikesLanded": (60, 30), "totalStrikesLanded": (85, 35),
     "takedownsLanded": (1.0, 1.2), "knockDowns": (0.3, 0.5),
     "submissions": (0.3, 0.5)},
]

PLAYER_ARCHETYPES = [
    {"name": "Elite_Scorer", "points": (28, 5), "totalRebounds": (5, 2), "assists": (8, 3), "threePointFieldGoalsMade": (3, 1.5), "steals": (1.5, 0.8), "blocks": (0.5, 0.5)},
    {"name": "Pass_First_PG", "points": (22, 5), "totalRebounds": (4, 1.5), "assists": (10, 3), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (1.2, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Combo_Guard", "points": (25, 6), "totalRebounds": (4, 1.5), "assists": (6, 2.5), "threePointFieldGoalsMade": (3, 1.5), "steals": (1, 0.7), "blocks": (0.5, 0.5)},
    {"name": "Young_PG", "points": (20, 6), "totalRebounds": (4, 2), "assists": (7, 3), "threePointFieldGoalsMade": (1.5, 1.2), "steals": (1.3, 0.8), "blocks": (0.3, 0.4)},
    {"name": "Two_Way_Wing", "points": (26, 5), "totalRebounds": (7, 2.5), "assists": (5, 2.5), "threePointFieldGoalsMade": (2, 1.2), "steals": (1.2, 0.7), "blocks": (1, 0.8)},
    {"name": "Stretch_Four", "points": (22, 5), "totalRebounds": (8, 3), "assists": (3, 2), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (0.8, 0.6), "blocks": (1, 0.8)},
    {"name": "Point_Forward", "points": (20, 5), "totalRebounds": (9, 3), "assists": (7, 3), "threePointFieldGoalsMade": (1, 0.9), "steals": (1, 0.7), "blocks": (0.8, 0.6)},
    {"name": "Unicorn_Big", "points": (25, 4), "totalRebounds": (12, 3), "assists": (9, 3), "threePointFieldGoalsMade": (1, 0.8), "steals": (1.3, 0.8), "blocks": (0.7, 0.6)},
    {"name": "Rim_Protector", "points": (16, 4), "totalRebounds": (10, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.3, 0.5), "steals": (0.5, 0.5), "blocks": (2.5, 1.3)},
    {"name": "Modern_Center", "points": (20, 5), "totalRebounds": (11, 3), "assists": (3, 2), "threePointFieldGoalsMade": (1, 0.8), "steals": (0.8, 0.6), "blocks": (2, 1)},
    {"name": "Starter_SG", "points": (18, 5), "totalRebounds": (4, 1.5), "assists": (4, 2), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (1, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Starter_SF", "points": (16, 5), "totalRebounds": (6, 2.5), "assists": (3, 2), "threePointFieldGoalsMade": (1.5, 1), "steals": (0.8, 0.6), "blocks": (0.8, 0.6)},
    {"name": "Starter_PF", "points": (15, 4), "totalRebounds": (8, 2.5), "assists": (2, 1.5), "threePointFieldGoalsMade": (1, 0.8), "steals": (0.6, 0.5), "blocks": (1.5, 1)},
    {"name": "Starter_C", "points": (14, 4), "totalRebounds": (9, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.5, 0.6), "steals": (0.5, 0.5), "blocks": (1.8, 1)},
    {"name": "Sixth_Man", "points": (16, 5), "totalRebounds": (3, 1.5), "assists": (4, 2), "threePointFieldGoalsMade": (2, 1.2), "steals": (1, 0.7), "blocks": (0.3, 0.4)},
    {"name": "3D_Specialist", "points": (12, 4), "totalRebounds": (4, 2), "assists": (2, 1.5), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (0.8, 0.6), "blocks": (0.3, 0.4)},
    {"name": "Energy_Big", "points": (10, 3), "totalRebounds": (7, 2.5), "assists": (1, 1), "threePointFieldGoalsMade": (0.3, 0.5), "steals": (0.5, 0.5), "blocks": (1.2, 0.8)},
    {"name": "Streaky_Shooter", "points": (22, 9), "totalRebounds": (4, 2), "assists": (3, 2.5), "threePointFieldGoalsMade": (3, 2), "steals": (0.8, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Mr_Consistent", "points": (18, 2.5), "totalRebounds": (5, 1.2), "assists": (5, 1.5), "threePointFieldGoalsMade": (2, 0.7), "steals": (1, 0.4), "blocks": (0.5, 0.3)},
    {"name": "High_Usage_SG", "points": (24, 7), "totalRebounds": (3, 1.5), "assists": (5, 3), "threePointFieldGoalsMade": (3, 2), "steals": (1.5, 1), "blocks": (0.3, 0.4)},
    {"name": "Physical_PF", "points": (17, 4), "totalRebounds": (10, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.5, 0.6), "steals": (0.7, 0.6), "blocks": (1, 0.8)},
    {"name": "Veteran_Guard", "points": (15, 4), "totalRebounds": (3, 1.5), "assists": (6, 2), "threePointFieldGoalsMade": (2, 1), "steals": (1, 0.6), "blocks": (0.3, 0.3)},
    {"name": "Raw_Rookie", "points": (14, 6), "totalRebounds": (5, 2.5), "assists": (3, 2.5), "threePointFieldGoalsMade": (1, 1.2), "steals": (0.7, 0.6), "blocks": (0.8, 0.7)},
    {"name": "Defensive_Wing", "points": (10, 3.5), "totalRebounds": (5, 2), "assists": (2, 1.5), "threePointFieldGoalsMade": (1.5, 1), "steals": (2, 0.9), "blocks": (1, 0.7)},
    {"name": "Floor_General", "points": (14, 4), "totalRebounds": (3, 1.5), "assists": (9, 3), "threePointFieldGoalsMade": (1.5, 1), "steals": (1.2, 0.7), "blocks": (0.2, 0.3)},
]


# ── Math primitives (mirrors stat_model.py to avoid circular imports) ─

def _fit_normal_weighted(series: list[float], *, decay: float) -> tuple[float, float] | None:
    if not series:
        return None
    n = len(series)
    if n < 2:
        return (series[0], max(1e-6, 0.0))
    weights = [decay ** i for i in range(n)]
    w_sum = sum(weights)
    mu = sum(w * x for w, x in zip(weights, series)) / w_sum
    var = sum(w * (x - mu) ** 2 for w, x in zip(weights, series)) / w_sum
    sigma = math.sqrt(var) if var > 0 else 0.0
    return (mu, max(1e-6, sigma))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_over(line: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if mu > line else (0.0 if mu < line else 0.5)
    z = (line - mu) / sigma
    return 1.0 - _normal_cdf(z)


def _poisson_prob_over(line: float, lam: float) -> float:
    if lam <= 0:
        return 0.0 if line >= 0 else 1.0
    k = int(math.floor(line))
    if k < 0:
        return 1.0
    cdf = 0.0
    for i in range(k + 1):
        log_pmf = -lam + i * math.log(lam) - math.lgamma(i + 1)
        cdf += math.exp(log_pmf)
    return max(0.0, min(1.0, 1.0 - cdf))


# ── Config dataclass ─────────────────────────────────────────────────

@dataclass
class ModelConfig:
    decay: float = 0.88
    shrinkage_k: float = 3.0
    vol_assists: float = 0.87
    vol_steals: float = 1.16
    vol_blocks: float = 1.16
    vol_3pm: float = 0.87
    prob_cap: float = 0.66
    prox_penalty_min: float = 0.30
    prox_gap_threshold: float = 0.5
    pick_threshold: float = 0.64

    def vol_for(self, field: str) -> float:
        m = {
            "assists": self.vol_assists,
            "steals": self.vol_steals,
            "blocks": self.vol_blocks,
            "threePointFieldGoalsMade": self.vol_3pm,
        }
        return m.get(field, 1.0)

    def to_params_dict(self) -> dict[str, float]:
        return {
            "decay": self.decay,
            "shrinkage_k": self.shrinkage_k,
            "vol_assists": self.vol_assists,
            "vol_steals": self.vol_steals,
            "vol_blocks": self.vol_blocks,
            "vol_3pm": self.vol_3pm,
            "prob_cap": self.prob_cap,
            "prox_penalty_min": self.prox_penalty_min,
            "pick_threshold": self.pick_threshold,
        }

    @classmethod
    def from_params_dict(cls, d: dict[str, float]) -> ModelConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ── Synthetic data generation ────────────────────────────────────────

def generate_synthetic_data(
    n_games: int = 25,
    n_copies: int = 2,
    seed: int = 42,
    *,
    include_mma: bool = True,
    mma_n_fights: int = 12,
) -> list[dict]:
    """Generate synthetic player-data archetypes for calibration backtests.

    NBA archetypes always emitted. MMA archetypes appended when
    `include_mma=True` (the default) — fewer "games" (fights) per
    archetype because MMA careers are short. Identifiable by the
    archetype name prefix "MMA_".
    """
    rng = random.Random(seed)
    player_data = []
    for copy_idx in range(n_copies):
        for arch in PLAYER_ARCHETYPES:
            stats: dict[str, list[float]] = {}
            for field in NBA_STATS:
                if field not in arch:
                    continue
                mean, std = arch[field]
                mean_adj = mean * rng.uniform(0.9, 1.1)
                std_adj = std * rng.uniform(0.85, 1.15)
                games = []
                for _ in range(n_games):
                    if field in LOW_COUNT_STATS or mean < 3:
                        val = max(0, round(rng.gauss(mean_adj, std_adj)))
                    else:
                        val = max(0, round(rng.gauss(mean_adj, std_adj), 1))
                    games.append(val)
                if len(games) >= 5:
                    stats[field] = games
            if stats:
                player_data.append({"name": f"{arch['name']}_v{copy_idx}", "stats": stats})

    if include_mma:
        for copy_idx in range(n_copies):
            for arch in MMA_ARCHETYPES:
                stats: dict[str, list[float]] = {}
                for field in MMA_STATS:
                    if field not in arch:
                        continue
                    mean, std = arch[field]
                    mean_adj = mean * rng.uniform(0.85, 1.15)
                    std_adj = std * rng.uniform(0.80, 1.20)
                    fights = []
                    for _ in range(mma_n_fights):
                        # All MMA counting stats are integer-valued.
                        val = max(0, round(rng.gauss(mean_adj, std_adj)))
                        fights.append(float(val))
                    if len(fights) >= 5:
                        stats[field] = fights
                if stats:
                    player_data.append({
                        "name": f"MMA_{arch['name']}_v{copy_idx}",
                        "stats": stats,
                    })
    return player_data


# ── Real data from learning_log ──────────────────────────────────────

def build_real_dataset(cache: SqliteTTLCache) -> list[dict]:
    """Convert resolved learning_log entries into the backtest format.

    Groups entries by (player_name, stat) and treats them as a time series.
    Each entry already has actual_value, line, side, and hit.
    """
    entries = cache.get_learning_entries(resolved_only=True, limit=5000)
    if not entries:
        return []

    STAT_TO_FIELD: dict[str, str] = {
        "points": "points", "pts": "points",
        "rebounds": "totalRebounds", "total rebounds": "totalRebounds", "rebs": "totalRebounds",
        "assists": "assists", "asts": "assists",
        "steals": "steals", "blocks": "blocks", "turnovers": "turnovers",
        "3-pointers made": "threePointFieldGoalsMade",
        "3-pt made": "threePointFieldGoalsMade",
        "three pointers made": "threePointFieldGoalsMade",
    }

    grouped: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        if e.get("actual_value") is None:
            continue
        stat_raw = (e.get("stat") or "").lower().strip()
        field = STAT_TO_FIELD.get(stat_raw, stat_raw)
        key = (e["player_name"], field)
        grouped.setdefault(key, []).append(e)

    player_data = []
    for (player, field), ents in grouped.items():
        ents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        series = [float(e["actual_value"]) for e in ents]
        if len(series) >= 5:
            player_data.append({
                "name": player,
                "stats": {field: series},
                "entries": ents,
            })

    return player_data


# ── Backtest engine ──────────────────────────────────────────────────

def _simulate_prediction(
    history: list[float], game_idx: int, line: float,
    side: str, field: str, cfg: ModelConfig,
    *,
    prior_mean: float | None = None,
) -> float | None:
    prior = history[game_idx + 1:]
    if len(prior) < 3:
        return None

    result = _fit_normal_weighted(prior, decay=cfg.decay)
    if result is None:
        return None
    mu, sigma = result
    sigma *= cfg.vol_for(field)

    # Shrink toward an OUTSIDE-THE-MARKET baseline (caller-supplied league/position
    # prior). Fall back to the player's own historical mean when no prior is given,
    # which makes the shrinkage a no-op rather than a market-collapsing bug.
    n = len(prior)
    baseline = prior_mean if prior_mean is not None else mu
    mu = (n * mu + cfg.shrinkage_k * baseline) / (n + cfg.shrinkage_k)

    # Use distribution-family dispatch (NegBin / Poisson / Gamma / Normal).
    p_over = prob_over_for_field(line=line, mean=mu, variance=sigma * sigma, field_name=field)

    model_prob = p_over if side == "over" else (1.0 - p_over)
    if model_prob > cfg.prob_cap:
        model_prob = cfg.prob_cap

    if n >= 3 and sigma > 0:
        sorted_prior = sorted(prior)
        median = sorted_prior[len(sorted_prior) // 2]
        gap = abs(line - median) / sigma
        if gap < cfg.prox_gap_threshold:
            if gap <= 0.1:
                penalty = cfg.prox_penalty_min
            else:
                penalty = cfg.prox_penalty_min + (gap - 0.1) / (cfg.prox_gap_threshold - 0.1) * (1.0 - cfg.prox_penalty_min)
            excess = model_prob - 0.5
            model_prob = 0.5 + excess * penalty

    return model_prob


def _backtest_player_stat(
    series: list[float], field: str, cfg: ModelConfig, n_test: int = 10,
    *,
    prior_mean: float | None = None,
) -> list[dict]:
    results = []
    for gi in range(min(n_test, len(series) - 4)):
        actual = series[gi]
        prior = series[gi + 1:]
        if len(prior) < 3:
            break
        prior_avg = sum(prior[:10]) / min(len(prior), 10)
        line = round(prior_avg * 2) / 2
        if line == actual:
            continue
        # Use the player's full prior-window mean as the shrinkage prior — a
        # rolling proxy for league baseline that decouples shrinkage from the
        # generated betting line.
        if prior_mean is None:
            prior_mean = sum(prior) / len(prior)
        for side in ["over", "under"]:
            mp = _simulate_prediction(
                series, gi, line, side, field, cfg, prior_mean=prior_mean,
            )
            if mp is None or mp < cfg.pick_threshold:
                continue
            hit = (actual > line) if side == "over" else (actual < line)
            results.append({"field": field, "model_prob": mp, "hit": hit, "side": side})
    return results


_STAT_FIELD_ALIASES: dict[str, str] = {
    "rebounds": "totalRebounds",
    "total rebounds": "totalRebounds",
    "rebs": "totalRebounds",
    "pts": "points",
    "asts": "assists",
    "3-pointers made": "threePointFieldGoalsMade",
    "3-pt made": "threePointFieldGoalsMade",
    "three pointers made": "threePointFieldGoalsMade",
    "passing yards": "passingYards",
    "rushing yards": "rushingYards",
    "receiving yards": "receivingYards",
    "passing touchdowns": "passingTouchdowns",
    "shots on goal": "shots",
}


def _normalize_field(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    return _STAT_FIELD_ALIASES.get(s, s)


def _backtest_real_entries(entries: list[dict], cfg: ModelConfig) -> list[dict]:
    """Backtest against resolved learning_log entries.

    NEW BEHAVIOR: when `series_json` is present we TRULY re-simulate the
    prediction with the candidate cfg (i.e., changing decay / shrinkage /
    vol multipliers actually changes the predicted probability).

    LEGACY BEHAVIOR: for entries without persisted series we fall back to
    the historical model_prob and just re-threshold (this is what the old
    code always did, and is a flawed-but-stable degradation).
    """
    results = []
    for e in entries:
        actual = e.get("actual_value")
        line_val = e.get("line")
        side = e.get("side", "over")
        if actual is None or line_val is None:
            continue
        line = float(line_val)
        stat_raw = e.get("stat_field") or e.get("stat") or ""
        field = _normalize_field(stat_raw)

        # ── Path A: persisted series → real re-simulation ──
        series_json = e.get("series_json")
        if isinstance(series_json, str) and series_json:
            try:
                series = [float(x) for x in json.loads(series_json) if isinstance(x, (int, float))]
            except Exception:
                series = []
            if len(series) >= 4:
                # We don't have the actual game in the series; use the player's own
                # series mean as the shrinkage prior (decoupled from the line).
                prior_mean = sum(series) / len(series) if series else None
                # Treat history index 0 as "next game"; everything is prior already.
                # Build a fake "history" so _simulate_prediction can use prior = history[1:]
                synthetic_history = [float(actual)] + series
                mp = _simulate_prediction(
                    synthetic_history, 0, line, side, field, cfg,
                    prior_mean=prior_mean,
                )
                if mp is None or mp < cfg.pick_threshold:
                    continue
                hit = (float(actual) > line) if side == "over" else (float(actual) < line)
                results.append({"field": field, "model_prob": mp, "hit": hit, "side": side, "source": "series"})
                continue

        # ── Path B: legacy fallback (no persisted series) ──
        mp = 0.55
        model_prob_orig = e.get("model_prob")
        if model_prob_orig is not None:
            mp = float(model_prob_orig)
        if mp > cfg.prob_cap:
            mp = cfg.prob_cap
        if mp < cfg.pick_threshold:
            continue
        hit = (float(actual) > line) if side == "over" else (float(actual) < line)
        results.append({"field": field, "model_prob": mp, "hit": hit, "side": side, "source": "legacy"})

    return results


def _brier_score(results: list[dict]) -> float:
    """Mean squared error between predicted prob and observed binary outcome.
    Lower is better; range [0, 1]. The fundamental metric for probabilistic
    betting models — strictly proper scoring rule that simultaneously rewards
    sharpness and calibration."""
    if not results:
        return 1.0
    return sum((r["model_prob"] - (1.0 if r["hit"] else 0.0)) ** 2 for r in results) / len(results)


def _log_loss(results: list[dict], *, eps: float = 1e-6) -> float:
    """Cross-entropy / log loss; complements Brier and penalizes confident wrongs harder."""
    if not results:
        return 10.0
    total = 0.0
    for r in results:
        p = max(eps, min(1.0 - eps, r["model_prob"]))
        y = 1.0 if r["hit"] else 0.0
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(results)


def _expected_calibration_error(results: list[dict], *, n_bins: int = 10) -> float:
    """ECE: weighted average of |confidence - empirical_freq| across confidence bins.
    Lower is better; perfectly calibrated → 0."""
    if not results:
        return 1.0
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for r in results:
        # bin index by predicted probability
        idx = min(n_bins - 1, max(0, int(r["model_prob"] * n_bins)))
        bins[idx].append(r)
    total_n = len(results)
    ece = 0.0
    for bin_results in bins:
        if not bin_results:
            continue
        bin_n = len(bin_results)
        avg_conf = sum(r["model_prob"] for r in bin_results) / bin_n
        avg_acc = sum(1 for r in bin_results if r["hit"]) / bin_n
        ece += (bin_n / total_n) * abs(avg_conf - avg_acc)
    return ece


def _reliability_diagram(results: list[dict], *, n_bins: int = 10) -> list[dict]:
    """Bin-by-bin (predicted_prob, empirical_hit_rate, count) for downstream display."""
    if not results:
        return []
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for r in results:
        idx = min(n_bins - 1, max(0, int(r["model_prob"] * n_bins)))
        bins[idx].append(r)
    out: list[dict] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        avg_conf = sum(r["model_prob"] for r in bucket) / len(bucket)
        avg_acc = sum(1 for r in bucket if r["hit"]) / len(bucket)
        out.append({
            "bin": i,
            "lo": round(i / n_bins, 3),
            "hi": round((i + 1) / n_bins, 3),
            "n": len(bucket),
            "avg_predicted_prob": round(avg_conf, 4),
            "empirical_hit_rate": round(avg_acc, 4),
        })
    return out


def _summarize_results(results: list[dict]) -> dict:
    if not results:
        return {
            "accuracy": 0.0, "total": 0, "hits": 0,
            "brier": 1.0, "log_loss": 10.0, "ece": 1.0,
            "reliability": [],
        }
    hits = sum(1 for r in results if r["hit"])
    total = len(results)
    return {
        "accuracy": hits / total,
        "total": total,
        "hits": hits,
        "brier": round(_brier_score(results), 5),
        "log_loss": round(_log_loss(results), 5),
        "ece": round(_expected_calibration_error(results), 5),
        "reliability": _reliability_diagram(results),
    }


def run_backtest(player_data: list[dict], cfg: ModelConfig, n_test: int = 8) -> dict:
    all_r: list[dict] = []
    for pd_entry in player_data:
        for sf, series in pd_entry["stats"].items():
            if len(series) < 6:
                continue
            all_r.extend(_backtest_player_stat(series, sf, cfg, n_test))
    return _summarize_results(all_r)


def run_backtest_real(entries: list[dict], cfg: ModelConfig) -> dict:
    all_r = _backtest_real_entries(entries, cfg)
    return _summarize_results(all_r)


# ── Grid search ──────────────────────────────────────────────────────

def _near(val: float, d: float, lo: float = 0.01, hi: float = 2.0) -> list[float]:
    return [max(lo, val - d), val, min(hi, val + d)]


def _score_metric(m: dict, *, min_total: int = 30) -> float:
    """Composite "lower is better" score. Drives grid search on Brier (calibration
    matters more than accuracy for betting per published NBA evidence) while still
    requiring decent breadth. Configs below the support threshold are penalized."""
    if m.get("total", 0) < min_total:
        return 1e9
    brier = m.get("brier", 1.0)
    # Encourage configs that pick more bets (resolution) without sacrificing calibration.
    accuracy_bonus = max(0.0, m.get("accuracy", 0.0) - 0.5) * 0.05
    return brier - accuracy_bonus


_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "decay": (0.70, 0.97),
    "shrinkage_k": (0.5, 14.0),
    "vol_assists": (0.7, 1.6),
    "vol_steals": (0.7, 1.6),
    "vol_blocks": (0.7, 1.6),
    "vol_3pm": (0.7, 1.6),
    "prob_cap": (0.60, 0.92),
    "prox_penalty_min": (0.30, 1.0),
    "pick_threshold": (0.50, 0.70),
}


def _sample_random_cfg(rng: random.Random) -> ModelConfig:
    return ModelConfig(
        decay=rng.uniform(*_PARAM_BOUNDS["decay"]),
        shrinkage_k=rng.uniform(*_PARAM_BOUNDS["shrinkage_k"]),
        vol_assists=rng.uniform(*_PARAM_BOUNDS["vol_assists"]),
        vol_steals=rng.uniform(*_PARAM_BOUNDS["vol_steals"]),
        vol_blocks=rng.uniform(*_PARAM_BOUNDS["vol_blocks"]),
        vol_3pm=rng.uniform(*_PARAM_BOUNDS["vol_3pm"]),
        prob_cap=rng.uniform(*_PARAM_BOUNDS["prob_cap"]),
        prox_penalty_min=rng.uniform(*_PARAM_BOUNDS["prox_penalty_min"]),
        pick_threshold=rng.uniform(*_PARAM_BOUNDS["pick_threshold"]),
    )


def _perturb_cfg(cfg: ModelConfig, scale: float, rng: random.Random) -> ModelConfig:
    """Sample neighbour with `scale` (e.g. 0.10) of each bound's range."""
    def jitter(val: float, key: str) -> float:
        lo, hi = _PARAM_BOUNDS[key]
        span = hi - lo
        return max(lo, min(hi, val + rng.uniform(-scale * span, scale * span)))

    return ModelConfig(
        decay=jitter(cfg.decay, "decay"),
        shrinkage_k=jitter(cfg.shrinkage_k, "shrinkage_k"),
        vol_assists=jitter(cfg.vol_assists, "vol_assists"),
        vol_steals=jitter(cfg.vol_steals, "vol_steals"),
        vol_blocks=jitter(cfg.vol_blocks, "vol_blocks"),
        vol_3pm=jitter(cfg.vol_3pm, "vol_3pm"),
        prob_cap=jitter(cfg.prob_cap, "prob_cap"),
        prox_penalty_min=jitter(cfg.prox_penalty_min, "prox_penalty_min"),
        pick_threshold=jitter(cfg.pick_threshold, "pick_threshold"),
    )


def grid_search(
    player_data: list[dict],
    n_test: int = 8,
    real_entries: list[dict] | None = None,
    *,
    n_random: int = 200,
    n_refine: int = 80,
    seed: int | None = None,
) -> tuple[ModelConfig, dict, dict | None]:
    """Random search + local refinement. Returns (best_cfg, primary_metrics,
    real_metrics).

    OBJECTIVE PRIORITY:
    - When n_real >= MIN_REAL_ENTRIES we score on real-data Brier (the only
      thing that ultimately matters for ROI). Synthetic data is used only as
      a sanity-check.
    - Otherwise we score on synthetic Brier (legacy behaviour) so the model
      is still calibrated even with sparse real data.

    The random+refine loop evaluates ~280 configs total (vs 6,000+ for the
    old grid search) and finds equally-good optima per Bergstra & Bengio
    "Random Search for Hyper-Parameter Optimization" (JMLR 2012).
    """
    rng = random.Random(seed)

    # Phase 1: Latin-hypercube-ish random search across full bounded space.
    best_score = 1e9
    best_cfg = ModelConfig()
    best_m: dict = {}

    use_real_objective = bool(real_entries) and len(real_entries) >= MIN_REAL_ENTRIES

    def evaluate(cfg: ModelConfig) -> tuple[dict, float]:
        synth_m = run_backtest(player_data, cfg, n_test)
        if use_real_objective:
            real_m = run_backtest_real(real_entries or [], cfg)
            score = _score_metric(real_m, min_total=50)
            return real_m, score
        return synth_m, _score_metric(synth_m)

    log.info(
        "Calibration: phase 1 random search (%d configs, objective=%s)",
        n_random, "real" if use_real_objective else "synthetic",
    )
    t0 = time.time()
    for _ in range(n_random):
        cfg = _sample_random_cfg(rng)
        m, score = evaluate(cfg)
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_m = m

    log.info(
        "  Phase 1 done in %.0fs -> brier=%.4f acc=%.2f%%",
        time.time() - t0, best_m.get("brier", 1.0), best_m.get("accuracy", 0.0) * 100,
    )

    # Phase 2: Thompson-sampling local refinement. We maintain a small set
    # of "elite" configs and on each iteration either (a) perturb an elite
    # at a random scale (exploitation) or (b) draw a fresh random config
    # (exploration). Probability of exploration anneals down from 0.30 to
    # 0.05 across the budget. This is closer to a contextual-bandit-style
    # search than the prior fixed-3-scale annealing — it converges faster
    # on the small per-cycle budgets we run in production (~80 evals).
    # References: Agrawal & Goyal "Thompson Sampling for Contextual
    # Bandits" (arxiv 1209.3352); Russo et al. "A Tutorial on Thompson
    # Sampling" (stanford TS_Tutorial.pdf).
    t1 = time.time()
    elites: list[tuple[float, ModelConfig]] = [(best_score, best_cfg)]
    for step in range(n_refine):
        # Annealing schedule for exploration probability.
        progress = step / max(1, n_refine - 1)
        explore_prob = 0.30 - 0.25 * progress  # 0.30 → 0.05
        if rng.random() < explore_prob or not elites:
            cfg = _sample_random_cfg(rng)
        else:
            # Sample an elite proportionally to its inverse-score (Boltzmann-
            # style softmax over scores). Lower score = better = higher pick.
            weights = [1.0 / max(0.01, s - elites[0][0] + 0.05) for s, _ in elites]
            total_w = sum(weights)
            pick = rng.random() * total_w
            cum = 0.0
            chosen = elites[0][1]
            for w, (_, c) in zip(weights, elites):
                cum += w
                if pick <= cum:
                    chosen = c
                    break
            # Local perturbation scale shrinks with progress.
            scale = 0.10 * (1.0 - 0.7 * progress)  # 0.10 → 0.03
            cfg = _perturb_cfg(chosen, scale, rng)

        m, score = evaluate(cfg)
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_m = m
        # Maintain top-5 elites for future exploitation draws.
        elites.append((score, cfg))
        elites.sort(key=lambda x: x[0])
        elites = elites[:5]

    log.info(
        "  Phase 2 Thompson refine done in %.0fs -> brier=%.4f acc=%.2f%% (elites=%d)",
        time.time() - t1, best_m.get("brier", 1.0), best_m.get("accuracy", 0.0) * 100,
        len(elites),
    )

    # Always produce both real and synthetic metrics for the chosen config so
    # the caller has clean side-by-side reporting regardless of which signal
    # drove the search.
    synthetic_m = run_backtest(player_data, best_cfg, n_test)
    real_m = None
    if real_entries:
        real_m = run_backtest_real(real_entries, best_cfg)
        log.info(
            "Real-data validation: brier=%.4f acc=%.2f%% (%d entries)",
            real_m.get("brier", 1.0), real_m.get("accuracy", 0.0) * 100, real_m.get("total", 0),
        )

    return best_cfg, synthetic_m, real_m


# ── Main calibration service ─────────────────────────────────────────

class CalibrationService:
    def __init__(self, *, cache: SqliteTTLCache) -> None:
        self._cache = cache

    def load_active_params_from_db(self) -> dict[str, float] | None:
        """Load the most recently applied calibration from the DB and activate it."""
        latest = self._cache.get_latest_calibration(applied_only=True)
        if latest is None:
            return None
        params = latest["params"]
        load_calibrated_params(params)
        # Stamp the active calibration ID so newly-saved learning entries
        # carry it as model_params_id (provenance).
        try:
            from app.services.ranker import Ranker as _Ranker
            _Ranker.set_active_calibration_id(latest.get("id"))
        except Exception:
            pass
        # Reload isotonic calibrator if one was saved with this run.
        iso = latest.get("isotonic")
        if iso:
            try:
                load_isotonic_calibrator([(float(x), float(y)) for x, y in iso])
            except (TypeError, ValueError):
                load_isotonic_calibrator(None)
        else:
            load_isotonic_calibrator(None)
        log.info("Loaded calibrated params from DB run %s (accuracy: real=%.2f%%, synth=%.2f%%, isotonic=%s)",
                 latest["id"],
                 (latest.get("accuracy_real") or 0) * 100,
                 (latest.get("accuracy_synthetic") or 0) * 100,
                 "yes" if iso else "no")
        return params

    def run_calibration(self, *, source: str = "scheduled") -> dict[str, Any]:
        """Execute a full calibration run.

        1. Build real dataset from learning_log (if enough entries)
        2. Generate synthetic holdout
        3. Grid search optimizing on synthetic (primary), validate on real
        4. Accept new params only if both thresholds met
        5. Save run to DB, optionally apply
        """
        t_start = time.time()

        real_data = build_real_dataset(self._cache)
        real_entries_flat: list[dict] = []
        for pd_entry in real_data:
            real_entries_flat.extend(pd_entry.get("entries", []))
        has_real = len(real_entries_flat) >= MIN_REAL_ENTRIES

        log.info("Calibration starting: %d real entries, has_real=%s, source=%s",
                 len(real_entries_flat), has_real, source)

        synthetic_data = generate_synthetic_data(n_games=25, n_copies=2)

        best_cfg, synth_metrics, real_metrics = grid_search(
            synthetic_data,
            n_test=8,
            real_entries=real_entries_flat if has_real else None,
        )

        synth_acc = synth_metrics.get("accuracy", 0.0)
        real_acc = real_metrics["accuracy"] if real_metrics else None

        should_apply = synth_acc >= SYNTHETIC_ACCURACY_THRESHOLD
        if has_real and real_acc is not None:
            should_apply = should_apply and (real_acc >= REAL_ACCURACY_THRESHOLD)

        params_dict = best_cfg.to_params_dict()
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Post-hoc isotonic calibration is now a SEPARATE stage:
        #
        #   Stage A (above) — grid_search optimizes raw probability parameters
        #     (decay, shrinkage_k, vol_*, prob_cap, ...) against Brier.
        #   Stage B (here)  — re-simulate every resolved real entry under the
        #     winning Stage A config to get FRESH probs, then fit isotonic on
        #     (fresh_prob, outcome) pairs.
        #
        # The previous implementation fit isotonic on `e.get("model_prob")` —
        # those values were the predictions saved at pick-time under the
        # PREVIOUS calibration. After Stage A picks new params, the old probs
        # are stale, so the isotonic mapping was correcting a model that no
        # longer existed. Per Niculescu-Mizil & Caruana (ICML 2005), isotonic
        # must be fit on the same model whose output it will be applied to.
        isotonic_breakpoints = None
        if has_real:
            fresh_results = _backtest_real_entries(real_entries_flat, best_cfg)
            probs: list[float] = []
            outcomes: list[int] = []
            for r in fresh_results:
                try:
                    probs.append(float(r["model_prob"]))
                    outcomes.append(1 if r["hit"] else 0)
                except (TypeError, ValueError, KeyError):
                    continue
            if len(probs) >= 50:
                isotonic_breakpoints = fit_isotonic(probs, outcomes)
                if isotonic_breakpoints:
                    log.info(
                        "Isotonic calibrator fit on %d FRESH (re-simulated) real entries "
                        "under Stage-A best_cfg (%d breakpoints)",
                        len(probs), len(isotonic_breakpoints),
                    )
            else:
                log.info(
                    "Isotonic skipped: only %d re-simulated probs (need >= 50). "
                    "Falling back to no post-hoc calibration this run.",
                    len(probs),
                )

        run_record = {
            "id": run_id,
            "created_at": now,
            "accuracy_real": real_acc,
            "accuracy_synthetic": synth_acc,
            "total_real": real_metrics["total"] if real_metrics else 0,
            "total_synthetic": synth_metrics.get("total", 0),
            "params_json": json.dumps(params_dict),
            "applied": 1 if should_apply else 0,
            "source": source,
            "brier": real_metrics.get("brier") if real_metrics else synth_metrics.get("brier"),
            "log_loss": real_metrics.get("log_loss") if real_metrics else synth_metrics.get("log_loss"),
            "ece": real_metrics.get("ece") if real_metrics else synth_metrics.get("ece"),
            "isotonic_json": json.dumps(isotonic_breakpoints) if isotonic_breakpoints else None,
        }
        self._cache.save_calibration_run(run_record)

        if should_apply:
            load_calibrated_params(params_dict)
            load_isotonic_calibrator(isotonic_breakpoints)
            try:
                from app.services.ranker import Ranker as _Ranker
                _Ranker.set_active_calibration_id(run_id)
            except Exception:
                pass
            log.info("Calibration applied: synth=%.2f%%, real=%s, isotonic=%s, params=%s",
                     synth_acc * 100,
                     f"{real_acc * 100:.2f}%" if real_acc is not None else "N/A",
                     "yes" if isotonic_breakpoints else "no",
                     params_dict)
        else:
            log.info("Calibration NOT applied (thresholds not met): synth=%.2f%%, real=%s",
                     synth_acc * 100,
                     f"{real_acc * 100:.2f}%" if real_acc is not None else "N/A")

        elapsed = time.time() - t_start
        return {
            "run_id": run_id,
            "elapsed_seconds": round(elapsed, 1),
            "applied": should_apply,
            "accuracy_synthetic": round(synth_acc, 4),
            "accuracy_real": round(real_acc, 4) if real_acc is not None else None,
            "total_synthetic": synth_metrics.get("total", 0),
            "total_real": real_metrics["total"] if real_metrics else 0,
            "real_entries_available": len(real_entries_flat),
            "params": params_dict,
            "source": source,
            "brier_synthetic": synth_metrics.get("brier"),
            "brier_real": real_metrics.get("brier") if real_metrics else None,
            "ece_synthetic": synth_metrics.get("ece"),
            "ece_real": real_metrics.get("ece") if real_metrics else None,
            "log_loss_synthetic": synth_metrics.get("log_loss"),
            "log_loss_real": real_metrics.get("log_loss") if real_metrics else None,
            "reliability_synthetic": synth_metrics.get("reliability", []),
            "reliability_real": real_metrics.get("reliability", []) if real_metrics else [],
        }

    def get_current_params(self) -> dict[str, Any]:
        active = get_calibrated_params()
        defaults = get_default_params()
        latest = self._cache.get_latest_calibration(applied_only=True)
        return {
            "active_params": active,
            "default_params": defaults,
            "last_calibration": {
                "id": latest["id"],
                "created_at": latest["created_at"],
                "accuracy_real": latest.get("accuracy_real"),
                "accuracy_synthetic": latest.get("accuracy_synthetic"),
                "source": latest.get("source"),
            } if latest else None,
        }

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._cache.get_calibration_history(limit=limit)
