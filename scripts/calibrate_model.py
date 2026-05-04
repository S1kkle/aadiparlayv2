"""
Statistical Model Calibration Script

Uses realistic synthetic player stat distributions (based on real NBA archetypes)
to backtest the prop prediction model. For each simulated player's game history,
predicts outcomes using only prior games, then measures accuracy.

Grid-searches over key model parameters to maximize hit rate to 65%+.
Runs entirely locally — no API calls needed.

Usage:
    python scripts/calibrate_model.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ── Stat model functions (mirrors stat_model.py) ────────────────────────

def fit_normal_weighted(series: list[float], *, decay: float = 0.85) -> tuple[float, float] | None:
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


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over_fn(line: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if mu > line else (0.0 if mu < line else 0.5)
    z = (line - mu) / sigma
    return 1.0 - normal_cdf(z)


def poisson_prob_over_fn(line: float, lam: float) -> float:
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


LOW_COUNT_STATS = frozenset({
    "blocks", "steals", "knockDowns", "takedownsLanded",
    "submissions", "advances", "interceptions",
    "rushingTouchdowns", "receivingTouchdowns", "passingTouchdowns",
    "goals",
})


# ── Synthetic data generation ────────────────────────────────────────────

PLAYER_ARCHETYPES = [
    # (name, {stat: (mean, std), ...})
    # Star guards
    {"name": "Elite_Scorer", "points": (28, 5), "totalRebounds": (5, 2), "assists": (8, 3), "threePointFieldGoalsMade": (3, 1.5), "steals": (1.5, 0.8), "blocks": (0.5, 0.5)},
    {"name": "Pass_First_PG", "points": (22, 5), "totalRebounds": (4, 1.5), "assists": (10, 3), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (1.2, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Combo_Guard", "points": (25, 6), "totalRebounds": (4, 1.5), "assists": (6, 2.5), "threePointFieldGoalsMade": (3, 1.5), "steals": (1, 0.7), "blocks": (0.5, 0.5)},
    {"name": "Young_PG", "points": (20, 6), "totalRebounds": (4, 2), "assists": (7, 3), "threePointFieldGoalsMade": (1.5, 1.2), "steals": (1.3, 0.8), "blocks": (0.3, 0.4)},
    # Star forwards
    {"name": "Two_Way_Wing", "points": (26, 5), "totalRebounds": (7, 2.5), "assists": (5, 2.5), "threePointFieldGoalsMade": (2, 1.2), "steals": (1.2, 0.7), "blocks": (1, 0.8)},
    {"name": "Stretch_Four", "points": (22, 5), "totalRebounds": (8, 3), "assists": (3, 2), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (0.8, 0.6), "blocks": (1, 0.8)},
    {"name": "Point_Forward", "points": (20, 5), "totalRebounds": (9, 3), "assists": (7, 3), "threePointFieldGoalsMade": (1, 0.9), "steals": (1, 0.7), "blocks": (0.8, 0.6)},
    # Star centers
    {"name": "Unicorn_Big", "points": (25, 4), "totalRebounds": (12, 3), "assists": (9, 3), "threePointFieldGoalsMade": (1, 0.8), "steals": (1.3, 0.8), "blocks": (0.7, 0.6)},
    {"name": "Rim_Protector", "points": (16, 4), "totalRebounds": (10, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.3, 0.5), "steals": (0.5, 0.5), "blocks": (2.5, 1.3)},
    {"name": "Modern_Center", "points": (20, 5), "totalRebounds": (11, 3), "assists": (3, 2), "threePointFieldGoalsMade": (1, 0.8), "steals": (0.8, 0.6), "blocks": (2, 1)},
    # Quality starters
    {"name": "Starter_SG", "points": (18, 5), "totalRebounds": (4, 1.5), "assists": (4, 2), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (1, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Starter_SF", "points": (16, 5), "totalRebounds": (6, 2.5), "assists": (3, 2), "threePointFieldGoalsMade": (1.5, 1), "steals": (0.8, 0.6), "blocks": (0.8, 0.6)},
    {"name": "Starter_PF", "points": (15, 4), "totalRebounds": (8, 2.5), "assists": (2, 1.5), "threePointFieldGoalsMade": (1, 0.8), "steals": (0.6, 0.5), "blocks": (1.5, 1)},
    {"name": "Starter_C", "points": (14, 4), "totalRebounds": (9, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.5, 0.6), "steals": (0.5, 0.5), "blocks": (1.8, 1)},
    # Bench/role
    {"name": "Sixth_Man", "points": (16, 5), "totalRebounds": (3, 1.5), "assists": (4, 2), "threePointFieldGoalsMade": (2, 1.2), "steals": (1, 0.7), "blocks": (0.3, 0.4)},
    {"name": "3D_Specialist", "points": (12, 4), "totalRebounds": (4, 2), "assists": (2, 1.5), "threePointFieldGoalsMade": (2.5, 1.3), "steals": (0.8, 0.6), "blocks": (0.3, 0.4)},
    {"name": "Energy_Big", "points": (10, 3), "totalRebounds": (7, 2.5), "assists": (1, 1), "threePointFieldGoalsMade": (0.3, 0.5), "steals": (0.5, 0.5), "blocks": (1.2, 0.8)},
    # Edge cases
    {"name": "Streaky_Shooter", "points": (22, 9), "totalRebounds": (4, 2), "assists": (3, 2.5), "threePointFieldGoalsMade": (3, 2), "steals": (0.8, 0.7), "blocks": (0.3, 0.4)},
    {"name": "Mr_Consistent", "points": (18, 2.5), "totalRebounds": (5, 1.2), "assists": (5, 1.5), "threePointFieldGoalsMade": (2, 0.7), "steals": (1, 0.4), "blocks": (0.5, 0.3)},
    {"name": "High_Usage_SG", "points": (24, 7), "totalRebounds": (3, 1.5), "assists": (5, 3), "threePointFieldGoalsMade": (3, 2), "steals": (1.5, 1), "blocks": (0.3, 0.4)},
    {"name": "Physical_PF", "points": (17, 4), "totalRebounds": (10, 3), "assists": (2, 1.5), "threePointFieldGoalsMade": (0.5, 0.6), "steals": (0.7, 0.6), "blocks": (1, 0.8)},
    {"name": "Veteran_Guard", "points": (15, 4), "totalRebounds": (3, 1.5), "assists": (6, 2), "threePointFieldGoalsMade": (2, 1), "steals": (1, 0.6), "blocks": (0.3, 0.3)},
    {"name": "Raw_Rookie", "points": (14, 6), "totalRebounds": (5, 2.5), "assists": (3, 2.5), "threePointFieldGoalsMade": (1, 1.2), "steals": (0.7, 0.6), "blocks": (0.8, 0.7)},
    {"name": "Defensive_Wing", "points": (10, 3.5), "totalRebounds": (5, 2), "assists": (2, 1.5), "threePointFieldGoalsMade": (1.5, 1), "steals": (2, 0.9), "blocks": (1, 0.7)},
    {"name": "Floor_General", "points": (14, 4), "totalRebounds": (3, 1.5), "assists": (9, 3), "threePointFieldGoalsMade": (1.5, 1), "steals": (1.2, 0.7), "blocks": (0.2, 0.3)},
]

NBA_STATS = ["points", "totalRebounds", "assists", "threePointFieldGoalsMade", "steals", "blocks"]


def log(msg: str = ""):
    print(msg, flush=True)


def generate_player_data(n_games: int = 25, n_copies: int = 2) -> list[dict]:
    """Generate multiple copies of each archetype with random variation for robustness."""
    rng = random.Random(42)
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
                player_data.append({
                    "name": f"{arch['name']}_v{copy_idx}",
                    "stats": stats,
                })

    return player_data


# ── Backtesting engine ───────────────────────────────────────────────────

@dataclass
class ModelConfig:
    decay: float = 0.85
    shrinkage_k: float = 8.0
    vol_assists: float = 1.20
    vol_steals: float = 1.25
    vol_blocks: float = 1.25
    vol_3pm: float = 1.20
    prob_cap: float = 0.82
    prox_penalty_min: float = 0.65
    prox_gap_threshold: float = 0.5
    pick_threshold: float = 0.55

    def vol_for(self, field: str) -> float:
        m = {
            "assists": self.vol_assists,
            "steals": self.vol_steals,
            "blocks": self.vol_blocks,
            "threePointFieldGoalsMade": self.vol_3pm,
        }
        return m.get(field, 1.0)


def simulate_prediction(
    history: list[float], game_idx: int, line: float,
    side: str, field: str, cfg: ModelConfig,
) -> float | None:
    prior = history[game_idx + 1:]
    if len(prior) < 3:
        return None

    result = fit_normal_weighted(prior, decay=cfg.decay)
    if result is None:
        return None
    mu, sigma = result

    sigma *= cfg.vol_for(field)

    n = len(prior)
    mu = (n * mu + cfg.shrinkage_k * line) / (n + cfg.shrinkage_k)

    if field in LOW_COUNT_STATS:
        p_over = poisson_prob_over_fn(line=line, lam=mu)
    else:
        p_over = prob_over_fn(line=line, mu=mu, sigma=sigma)

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


def backtest_player_stat(series: list[float], field: str, cfg: ModelConfig, n_test: int = 10) -> list[dict]:
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

        for side in ["over", "under"]:
            mp = simulate_prediction(series, gi, line, side, field, cfg)
            if mp is None or mp < cfg.pick_threshold:
                continue
            hit = (actual > line) if side == "over" else (actual < line)
            results.append({"field": field, "model_prob": mp, "hit": hit, "side": side})

    return results


def run_backtest_all(player_data: list[dict], cfg: ModelConfig, n_test: int = 8) -> dict:
    all_r = []
    for pd_entry in player_data:
        for sf, series in pd_entry["stats"].items():
            if len(series) < 6:
                continue
            all_r.extend(backtest_player_stat(series, sf, cfg, n_test))

    if not all_r:
        return {"accuracy": 0, "total": 0, "hits": 0}

    hits = sum(1 for r in all_r if r["hit"])
    total = len(all_r)

    hc = [r for r in all_r if r["model_prob"] >= 0.60]
    hc_hits = sum(1 for r in hc if r["hit"])

    by_stat: dict[str, dict] = {}
    for r in all_r:
        f = r["field"]
        if f not in by_stat:
            by_stat[f] = {"hits": 0, "total": 0}
        by_stat[f]["total"] += 1
        if r["hit"]:
            by_stat[f]["hits"] += 1
    for f in by_stat:
        by_stat[f]["accuracy"] = by_stat[f]["hits"] / by_stat[f]["total"]

    return {
        "accuracy": hits / total,
        "total": total,
        "hits": hits,
        "high_conf_accuracy": hc_hits / len(hc) if hc else 0,
        "high_conf_total": len(hc),
        "by_stat": by_stat,
    }


# ── Grid search ──────────────────────────────────────────────────────────

def _near(val, d, lo=0.01, hi=2.0):
    return [max(lo, val - d), val, min(hi, val + d)]


def grid_search(player_data: list[dict], n_test: int = 8) -> tuple[ModelConfig, dict]:
    best_acc = 0.0
    best_cfg = ModelConfig()
    best_m: dict = {}

    log("\n=== Phase 1: Coarse grid ===")
    t0 = time.time()
    n = 0

    for decay in [0.78, 0.85, 0.92]:
        for sk in [2.0, 5.0, 8.0, 12.0]:
            for va in [0.9, 1.1, 1.30]:
                for vs in [0.9, 1.15, 1.40]:
                    for cap in [0.75, 0.82, 0.88]:
                        for pm in [0.55, 0.75, 1.0]:
                            for pt in [0.52, 0.56, 0.60]:
                                cfg = ModelConfig(
                                    decay=decay, shrinkage_k=sk,
                                    vol_assists=va, vol_steals=vs,
                                    vol_blocks=vs, vol_3pm=va,
                                    prob_cap=cap, prox_penalty_min=pm,
                                    pick_threshold=pt,
                                )
                                m = run_backtest_all(player_data, cfg, n_test)
                                n += 1
                                if m["total"] >= 30 and m["accuracy"] > best_acc:
                                    best_acc = m["accuracy"]
                                    best_cfg = cfg
                                    best_m = m

    log(f"  {n} configs in {time.time()-t0:.0f}s")
    log(f"  Best: {best_acc:.2%} ({best_m.get('total', 0)} preds)")

    log("\n=== Phase 2: Fine-tune ===")
    t1 = time.time()
    n2 = 0
    for decay in _near(best_cfg.decay, 0.03, 0.6, 0.99):
        for sk in _near(best_cfg.shrinkage_k, 1.5, 0.5, 15):
            for va in _near(best_cfg.vol_assists, 0.08, 0.7, 1.6):
                for vs in _near(best_cfg.vol_steals, 0.08, 0.7, 1.6):
                    for cap in _near(best_cfg.prob_cap, 0.03, 0.6, 0.95):
                        for pm in _near(best_cfg.prox_penalty_min, 0.08, 0.3, 1.0):
                            for pt in _near(best_cfg.pick_threshold, 0.02, 0.5, 0.7):
                                cfg = ModelConfig(
                                    decay=decay, shrinkage_k=sk,
                                    vol_assists=va, vol_steals=vs,
                                    vol_blocks=vs, vol_3pm=va,
                                    prob_cap=cap, prox_penalty_min=pm,
                                    pick_threshold=pt,
                                )
                                m = run_backtest_all(player_data, cfg, n_test)
                                n2 += 1
                                if m["total"] >= 30 and m["accuracy"] > best_acc:
                                    best_acc = m["accuracy"]
                                    best_cfg = cfg
                                    best_m = m
    log(f"  {n2} configs in {time.time()-t1:.0f}s -> {best_acc:.2%}")

    log("\n=== Phase 3: Ultra-fine ===")
    t2 = time.time()
    n3 = 0
    for decay in _near(best_cfg.decay, 0.01, 0.6, 0.99):
        for sk in _near(best_cfg.shrinkage_k, 0.5, 0.5, 15):
            for va in _near(best_cfg.vol_assists, 0.03, 0.7, 1.6):
                for vs in _near(best_cfg.vol_steals, 0.03, 0.7, 1.6):
                    for cap in _near(best_cfg.prob_cap, 0.01, 0.6, 0.95):
                        for pm in _near(best_cfg.prox_penalty_min, 0.03, 0.3, 1.0):
                            for pt in _near(best_cfg.pick_threshold, 0.01, 0.5, 0.7):
                                cfg = ModelConfig(
                                    decay=decay, shrinkage_k=sk,
                                    vol_assists=va, vol_steals=vs,
                                    vol_blocks=vs, vol_3pm=va,
                                    prob_cap=cap, prox_penalty_min=pm,
                                    pick_threshold=pt,
                                )
                                m = run_backtest_all(player_data, cfg, n_test)
                                n3 += 1
                                if m["total"] >= 30 and m["accuracy"] > best_acc:
                                    best_acc = m["accuracy"]
                                    best_cfg = cfg
                                    best_m = m
    log(f"  {n3} configs in {time.time()-t2:.0f}s -> {best_acc:.2%}")

    return best_cfg, best_m


def main():
    t_start = time.time()
    log("=" * 60)
    log("PROP MODEL CALIBRATION")
    log("Target: 65%+ prediction accuracy")
    log("Using synthetic NBA player distributions")
    log("=" * 60)

    log("\nGenerating player data...")
    player_data = generate_player_data(n_games=25, n_copies=2)
    log(f"  {len(player_data)} profiles, ~{sum(sum(len(v) for v in p['stats'].values()) for p in player_data)} game values")

    log("\n=== Current defaults ===")
    dcfg = ModelConfig()
    dm = run_backtest_all(player_data, dcfg)
    log(f"  Accuracy: {dm['accuracy']:.2%} ({dm['total']} predictions)")
    if dm.get("by_stat"):
        for stat, sm in sorted(dm["by_stat"].items(), key=lambda x: -x[1]["accuracy"]):
            log(f"    {stat}: {sm['accuracy']:.2%} ({sm['total']})")

    best_cfg, best_m = grid_search(player_data)

    elapsed = time.time() - t_start
    log("\n" + "=" * 60)
    log(f"CALIBRATION COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 60)
    log(f"\nDefault accuracy:   {dm['accuracy']:.2%}")
    log(f"Optimized accuracy: {best_m['accuracy']:.2%} ({best_m['total']} predictions)")
    if best_m.get("high_conf_total", 0) > 0:
        log(f"High-confidence:    {best_m['high_conf_accuracy']:.2%} ({best_m['high_conf_total']})")
    log(f"\nOptimal parameters:")
    log(f"  decay            = {best_cfg.decay}")
    log(f"  shrinkage_k      = {best_cfg.shrinkage_k}")
    log(f"  vol_assists      = {best_cfg.vol_assists}")
    log(f"  vol_steals       = {best_cfg.vol_steals}")
    log(f"  vol_blocks       = {best_cfg.vol_blocks}")
    log(f"  vol_3pm          = {best_cfg.vol_3pm}")
    log(f"  prob_cap         = {best_cfg.prob_cap}")
    log(f"  prox_penalty_min = {best_cfg.prox_penalty_min}")
    log(f"  pick_threshold   = {best_cfg.pick_threshold}")

    if best_m.get("by_stat"):
        log(f"\nAccuracy by stat:")
        for stat, sm in sorted(best_m["by_stat"].items(), key=lambda x: -x[1]["accuracy"]):
            log(f"  {stat}: {sm['accuracy']:.2%} ({sm['total']})")

    met = best_m["accuracy"] >= 0.65
    log(f"\n*** TARGET 65% {'ACHIEVED' if met else 'NOT MET'} ***")

    results = {
        "accuracy": best_m["accuracy"],
        "default_accuracy": dm["accuracy"],
        "total_predictions": best_m["total"],
        "params": {
            "decay": best_cfg.decay,
            "shrinkage_k": best_cfg.shrinkage_k,
            "vol_assists": best_cfg.vol_assists,
            "vol_steals": best_cfg.vol_steals,
            "vol_blocks": best_cfg.vol_blocks,
            "vol_3pm": best_cfg.vol_3pm,
            "prob_cap": best_cfg.prob_cap,
            "prox_penalty_min": best_cfg.prox_penalty_min,
            "pick_threshold": best_cfg.pick_threshold,
        },
        "by_stat": {k: {"accuracy": v["accuracy"], "total": v["total"]}
                    for k, v in best_m.get("by_stat", {}).items()},
    }

    out_path = Path(__file__).parent / "calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
