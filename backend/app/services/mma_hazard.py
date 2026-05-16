"""Discrete-time hazard model for MMA round-of-finish / fight-time props.

Underdog offers props like:
  - Fight time over/under 2.5 rounds
  - Round of finish per fighter (1, 2, 3+)
  - Round of KO per fighter (1, 2, 3+)
  - Round of submission per fighter (1, 2, 3+)
  - Finishes (does the fight end inside the distance)

These are NOT counting-stat props and don't fit the Poisson / NB / Gamma
dispatch. They are **survival** problems: P(fight ends by time t).
A Cox PH model would be ideal but requires more data than we'll have
out of the gate. We use a discrete-time hazard model instead — one
hazard parameter per round, plus weight-class baselines.

The model:

  h_t = P(fight ends in round t | survived to round t)

Decomposed by method:
  h_t = h_ko_t + h_sub_t + h_dec_t   (decision hazard is only at round=scheduled)

  S_t = product_{i=1..t} (1 - h_i)        # survival to end of round t
  P(round of finish = t) = h_t * S_{t-1}

For per-fighter modeling we estimate h_ko_t and h_sub_t from the
fighter's own finish history (last 12 fights). Heavy shrinkage toward
weight-class baselines for the (very common) case where the fighter
has < 3 finishes — otherwise the estimate is dominated by noise.

Weight-class baselines come from public UFC outcome data (Fight Matrix
per-weight-class finish rates; Fightomic finish-rate-by-division
breakdowns):

  Heavyweight       66% finish rate
  Light Heavyweight 55%
  Middleweight      48%
  Welterweight      43%
  Lightweight       42%
  Featherweight     38%
  Bantamweight      34%
  Flyweight         32%
  W. Strawweight    33%
  W. Flyweight      35%
  W. Bantamweight   33%

Within finishes, KO/Sub split varies by class (HW is KO-heavy, FW is
sub-heavier). Per-round distribution: round 1 ≈ 50%, round 2 ≈ 28%,
round 3 ≈ 16%, rounds 4-5 ≈ 6% combined for 5-round fights.
"""
from __future__ import annotations

import logging
from typing import Any

from app.db import SqliteTTLCache
from app.services.ufcstats_db import get_finish_history

log = logging.getLogger(__name__)


# Baseline per-class finish rate (KO+Sub combined). Tuned from Fight
# Matrix outcome data 2018-2024.
_FINISH_RATE_BASELINE: dict[str, float] = {
    "heavyweight": 0.66,
    "light heavyweight": 0.55,
    "middleweight": 0.48,
    "welterweight": 0.43,
    "lightweight": 0.42,
    "featherweight": 0.38,
    "bantamweight": 0.34,
    "flyweight": 0.32,
    "women's strawweight": 0.33,
    "women's flyweight": 0.35,
    "women's bantamweight": 0.33,
    "catchweight": 0.45,
}

# KO share of finishes by class (rest is sub). Heavyweight is overwhelmingly
# KO; lightweight is mixed; women's classes tilt sub-heavier.
_KO_SHARE_BASELINE: dict[str, float] = {
    "heavyweight": 0.85,
    "light heavyweight": 0.78,
    "middleweight": 0.72,
    "welterweight": 0.68,
    "lightweight": 0.62,
    "featherweight": 0.58,
    "bantamweight": 0.55,
    "flyweight": 0.55,
    "women's strawweight": 0.48,
    "women's flyweight": 0.50,
    "women's bantamweight": 0.50,
    "catchweight": 0.65,
}

# Per-round share of finishes (rounds 1..5). Public UFC data: ~50/28/16/4/2.
_FINISH_ROUND_DIST_3RD: list[float] = [0.50, 0.30, 0.20]
_FINISH_ROUND_DIST_5RD: list[float] = [0.42, 0.25, 0.18, 0.10, 0.05]


def _normalize_wc(weight_class: str | None) -> str:
    return (weight_class or "").strip().lower()


def _baseline_finish_rate(wc: str | None) -> float:
    return _FINISH_RATE_BASELINE.get(_normalize_wc(wc), 0.45)


def _baseline_ko_share(wc: str | None) -> float:
    return _KO_SHARE_BASELINE.get(_normalize_wc(wc), 0.65)


def _round_dist(scheduled_rounds: int) -> list[float]:
    return _FINISH_ROUND_DIST_5RD if scheduled_rounds >= 5 else _FINISH_ROUND_DIST_3RD


def _classify_method(method: str | None) -> str | None:
    """UFCStats method strings vary. Returns 'ko', 'sub', 'dec', or None."""
    if not method:
        return None
    m = method.lower()
    if "ko" in m or "tko" in m:
        return "ko"
    if "sub" in m:
        return "sub"
    if "dec" in m:
        return "dec"
    if "draw" in m or "nc" in m or "no contest" in m:
        return None
    return None


def estimate_fighter_finish_propensity(
    cache: SqliteTTLCache,
    *,
    fighter: str,
    weight_class: str | None,
    pool_k: float = 4.0,
) -> dict[str, float]:
    """Bayesian-shrunk per-fighter finish propensity.

    Returns:
      {
        finish_rate: P(this fighter finishes a fight),
        ko_share:    P(finish is KO | finish),
        sub_share:   1 - ko_share,
        n_career:    sample size used,
      }

    Uses the fighter's UFCStats history (wins-only) with `pool_k`
    pseudo-observations pulled from the weight-class baseline.
    """
    history = get_finish_history(cache, fighter=fighter, last_n=15)
    wins_by_finish = [h for h in history if h.get("won")]
    n_wins = len(wins_by_finish)
    ko_wins = sum(1 for h in wins_by_finish if _classify_method(h.get("method")) == "ko")
    sub_wins = sum(1 for h in wins_by_finish if _classify_method(h.get("method")) == "sub")
    finish_wins = ko_wins + sub_wins

    base_rate = _baseline_finish_rate(weight_class)
    base_ko_share = _baseline_ko_share(weight_class)

    # Partial pool toward baseline.
    finish_rate = (finish_wins + pool_k * base_rate) / max(1.0, n_wins + pool_k)
    if finish_wins > 0:
        empirical_ko_share = ko_wins / finish_wins
    else:
        empirical_ko_share = base_ko_share
    ko_share = (
        empirical_ko_share * finish_wins + base_ko_share * pool_k
    ) / max(1.0, finish_wins + pool_k)

    return {
        "finish_rate": round(finish_rate, 4),
        "ko_share": round(ko_share, 4),
        "sub_share": round(1.0 - ko_share, 4),
        "n_career": n_wins,
    }


def round_hazards(
    *,
    finisher_a_rate: float,
    finisher_b_rate: float,
    scheduled_rounds: int = 3,
) -> list[float]:
    """Compute per-round hazard h_t = P(fight ends in round t | survived).

    Combines both fighters' finish propensity. We treat the fight-level
    finish rate as 1 - (1 - finisher_a_rate) * (1 - finisher_b_rate),
    i.e. "finish if either fighter finishes" — an independence
    approximation that's reasonable for KO (each fighter has an
    independent shot) and acceptable for sub (slight over-counting when
    both are submission threats).

    Then distribute the total finish probability across rounds via
    `_round_dist`, converting share-of-finishes to hazard via:
      h_t = round_share_t * total_finish_p / S_{t-1}
    """
    finish_p = 1.0 - (1.0 - finisher_a_rate) * (1.0 - finisher_b_rate)
    finish_p = max(0.05, min(0.95, finish_p))
    shares = _round_dist(scheduled_rounds)
    hazards: list[float] = []
    survival = 1.0
    for share in shares:
        h_t = (share * finish_p) / max(1e-6, survival)
        h_t = max(0.0, min(0.99, h_t))
        hazards.append(h_t)
        survival *= 1.0 - h_t
    # Append a "decision hazard" pseudo-round for survival accounting.
    return hazards


def prob_round_of_finish(
    cache: SqliteTTLCache,
    *,
    fighter: str,
    opponent: str,
    weight_class: str | None,
    target_round: int,
    method_filter: str | None = None,
    scheduled_rounds: int = 3,
) -> float | None:
    """P(this fighter finishes the fight in `target_round`).

    `method_filter`:
      - None: any finish method
      - "ko": only KO/TKO finishes
      - "sub": only submission finishes

    Returns None when target_round is out of bounds.
    """
    if target_round < 1 or target_round > scheduled_rounds:
        return None

    f_self = estimate_fighter_finish_propensity(
        cache, fighter=fighter, weight_class=weight_class,
    )
    f_opp = estimate_fighter_finish_propensity(
        cache, fighter=opponent, weight_class=weight_class,
    )

    # Hazards on the joint finish event.
    hazards = round_hazards(
        finisher_a_rate=f_self["finish_rate"],
        finisher_b_rate=f_opp["finish_rate"],
        scheduled_rounds=scheduled_rounds,
    )
    survival = 1.0
    for i in range(target_round - 1):
        survival *= 1.0 - hazards[i]
    finish_in_round = hazards[target_round - 1] * survival

    # Now split: P(finish is by THIS fighter) ≈ f_self / (f_self + f_opp).
    self_share_of_finish = f_self["finish_rate"] / max(
        1e-6, f_self["finish_rate"] + f_opp["finish_rate"]
    )
    self_finish_in_round = finish_in_round * self_share_of_finish

    if method_filter == "ko":
        return round(self_finish_in_round * f_self["ko_share"], 4)
    if method_filter == "sub":
        return round(self_finish_in_round * f_self["sub_share"], 4)
    return round(self_finish_in_round, 4)


def prob_fight_total_rounds(
    cache: SqliteTTLCache,
    *,
    fighter_a: str,
    fighter_b: str,
    weight_class: str | None,
    line_rounds: float,
    scheduled_rounds: int = 3,
) -> float:
    """P(fight goes OVER `line_rounds`).

    Standard Underdog line is 2.5 (3-round fights) or 4.5 (5-round main
    events). Maps to survival probability past the appropriate round.
    """
    f_a = estimate_fighter_finish_propensity(
        cache, fighter=fighter_a, weight_class=weight_class,
    )
    f_b = estimate_fighter_finish_propensity(
        cache, fighter=fighter_b, weight_class=weight_class,
    )
    hazards = round_hazards(
        finisher_a_rate=f_a["finish_rate"],
        finisher_b_rate=f_b["finish_rate"],
        scheduled_rounds=scheduled_rounds,
    )
    # Need survival past round k where k = ceil(line_rounds).
    # For line=2.5 in a 3-round fight: survives past round 2 = S_2.
    threshold_round = int(line_rounds + 0.5)  # 2.5 → 3, 4.5 → 5
    threshold_round = min(threshold_round, scheduled_rounds + 1)
    survival = 1.0
    for i in range(threshold_round - 1):
        survival *= 1.0 - hazards[i]
    # The line is in ROUNDS, and "over 2.5" means the fight enters round 3.
    # That equals survival past end of round 2 = product of (1-h_1)(1-h_2).
    return round(survival, 4)


def prob_finishes_inside(
    cache: SqliteTTLCache,
    *,
    fighter_a: str,
    fighter_b: str,
    weight_class: str | None,
    scheduled_rounds: int = 3,
) -> float:
    """P(fight ends by stoppage inside the distance). Useful for the
    'Finishes' Underdog prop type."""
    f_a = estimate_fighter_finish_propensity(
        cache, fighter=fighter_a, weight_class=weight_class,
    )
    f_b = estimate_fighter_finish_propensity(
        cache, fighter=fighter_b, weight_class=weight_class,
    )
    finish_p = 1.0 - (1.0 - f_a["finish_rate"]) * (1.0 - f_b["finish_rate"])
    return round(max(0.05, min(0.95, finish_p)), 4)
