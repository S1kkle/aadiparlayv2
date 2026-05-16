"""MMA-specific feature engineering.

MMA is structurally different from team sports and requires its own
adjustments — these are not appropriate for NBA/NFL/NHL pipelines:

1. **Opponent-style exchange factor** — In MMA, a striker's opponents
   typically also strike a lot (fights stay standing); a wrestler's
   opponents typically strike less (fights end up on the ground). We
   proxy opponent matchup by the opponent's own career strike volume
   relative to a weight-class baseline. A high-volume striker bumps μ
   up modestly; a control wrestler pulls μ down.

2. **Layoff / inactivity decay** — Layoffs > 12 months degrade striking
   output reliably (see Bloody Elbow / Sportskeeda ring-rust analyses).
   The effect is asymmetric: rust manifests more in volume (μ) than
   accuracy, so we pull μ toward the league prior, scaled by months
   since last fight. < 90 days = no adjustment, > 18 months = full
   regression to prior.

3. **Era decay on career stats** — A career stat aggregated since 2014
   includes pre-modern-MMA performance and biases projections for
   active fighters. We weight career history by per-fight exponential
   decay so the 8 most-recent fights dominate.

4. **Weight-class baselines** — Heavyweight (66% finish rate) and
   women's strawweight (~33% finish rate) are genuinely different
   distributions; using a global MMA prior under-corrects per Fight
   Matrix outcome data. Weight-class layer sits between fighter
   posterior and global MMA prior.

References:
  - Fight Matrix UFC outcomes by weight class (finish rates)
  - Bloody Elbow / Sportskeeda ring-rust longitudinal studies
  - Shurzy: SLpM/SApM differentials predict prop outcomes
  - Cage Calculus / Fightomic: opponent-quality adjustment
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


# League-average career SLpM (significant strikes landed per minute) for
# UFC fighters. Used as the denominator for opponent-strength adjustment.
# Sourced from UFCStats aggregate ~ 4.0-4.5 SLpM is league average.
_LEAGUE_SLPM = 4.2
_LEAGUE_TDA = 1.0  # takedowns landed per fight (UFC career average)
_LEAGUE_SUB_AVG = 0.4  # submission attempts per fight

# Weight-class baselines for partial pooling. Used by online_priors to
# add a per-weight-class layer between fighter and global. Heavyweight
# fighters throw fewer strikes per fight on average (fights end fast);
# featherweights throw the most.
WEIGHT_CLASS_NORMALIZATION: dict[str, float] = {
    "heavyweight": 0.82,            # KO-heavy, short fights
    "light heavyweight": 0.90,
    "middleweight": 0.96,
    "welterweight": 1.00,           # reference class
    "lightweight": 1.04,
    "featherweight": 1.08,          # highest volume
    "bantamweight": 1.06,
    "flyweight": 1.05,
    "women's bantamweight": 0.98,
    "women's flyweight": 1.00,
    "women's strawweight": 1.02,
    "catchweight": 1.00,
}


def _parse_fight_date(date_str: Any) -> datetime | None:
    if not isinstance(date_str, str):
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def days_since_last_fight(
    fights: list[dict[str, Any]], *, now: datetime | None = None
) -> int | None:
    """Days since the most-recent fight in the list.

    `fights` is assumed to be ordered newest-first (matching ESPN's
    eventlog return order). Returns None when no parseable date exists.
    """
    if not fights:
        return None
    now = now or datetime.now(timezone.utc)
    for f in fights:
        dt = _parse_fight_date(f.get("date"))
        if dt is not None:
            delta = now - dt
            return max(0, int(delta.days))
    return None


def layoff_adjustment(
    days: int | None,
    *,
    rust_start_days: int = 90,
    full_rust_days: int = 540,
) -> float:
    """Returns a multiplier on μ for layoff / ring rust.

    Piecewise:
      - days < 90        : 1.00 (no rust)
      - 90 ≤ days < 540  : linearly interpolated down to 0.92
      - days ≥ 540       : 0.92 (capped; max regression to prior)

    The 8% cap is empirical — ring-rust research suggests volume drops
    of 5-12% on returns from 12+ month layoffs, with high variance. We
    don't apply more because the variance is large enough that a
    bigger penalty over-corrects on active 35-year-old veterans.
    """
    if days is None or days < rust_start_days:
        return 1.0
    if days >= full_rust_days:
        return 0.92
    span = full_rust_days - rust_start_days
    progress = (days - rust_start_days) / span
    return 1.0 - 0.08 * progress


def opponent_exchange_factor(
    opponent_career: dict[str, Any] | None,
    *,
    stat_field: str,
    league_slpm: float = _LEAGUE_SLPM,
) -> float:
    """How much to bump fighter μ based on opponent's striking volume.

    Logic:
      - Striking stats (sigStrikesLanded, totalStrikesLanded): proxy
        opponent style via career SLpM. A high-volume opponent typically
        engages in more striking exchanges → fighter μ bumps up. A
        low-volume control fighter → fighter μ bumps down.
      - Takedown stats: proxy via opponent's takedownAvg. A high-TD
        opponent suggests the fight goes to ground → opponent's *own*
        takedowns up, but the fighter's takedowns are LESS likely (they
        get defended) → slight DOWN adjustment for `takedownsLanded`.
      - Submission stats: proxy via opponent's submissionAvg
        (vulnerability — high-sub-volume opponents have lower TD
        defense and tend to scramble, raising sub opportunity).

    Returns a factor in [0.85, 1.15] — we deliberately cap because
    opponent's own averages are an imperfect proxy for opponent defense,
    and over-confident adjustments amplify error.
    """
    if not opponent_career:
        return 1.0

    if stat_field in ("sigStrikesLanded", "totalStrikesLanded"):
        opp_slpm = opponent_career.get("strikeLPM")
        if not isinstance(opp_slpm, (int, float)) or opp_slpm <= 0:
            return 1.0
        ratio = float(opp_slpm) / max(0.1, league_slpm)
        factor = 1.0 + 0.15 * (ratio - 1.0)
        return max(0.85, min(1.15, factor))

    if stat_field == "takedownsLanded":
        opp_tda = opponent_career.get("takedownAvg")
        if not isinstance(opp_tda, (int, float)):
            return 1.0
        ratio = float(opp_tda) / max(0.1, _LEAGUE_TDA)
        # Wrestler-vs-wrestler usually means LESS clean takedowns (they
        # cancel out). Invert the ratio: more opp TDs → fewer fighter TDs.
        factor = 1.0 - 0.10 * (ratio - 1.0)
        return max(0.85, min(1.15, factor))

    if stat_field in ("submissions", "submissionsLanded"):
        opp_sub = opponent_career.get("submissionAvg")
        if not isinstance(opp_sub, (int, float)):
            return 1.0
        ratio = float(opp_sub) / max(0.05, _LEAGUE_SUB_AVG)
        factor = 1.0 + 0.10 * (ratio - 1.0)
        return max(0.85, min(1.15, factor))

    return 1.0


def era_weighted_career_mean(
    fight_series: list[float],
    *,
    decay: float = 0.85,
) -> float | None:
    """Exponentially-decayed career mean for prior use.

    The static `_PRIORS["MMA"]` table is a single number per stat. When
    we have a fighter's full career history, we can compute a per-fighter
    "era-adjusted" baseline that weights recent fights more heavily.
    Falls back to None when the series is empty.

    Decay default 0.85 → ~10-fight half-life, matching the ~9-12 month
    activity horizon that public MMA-stats blogs use for skill drift.
    """
    if not fight_series:
        return None
    n = len(fight_series)
    weights = [decay ** i for i in range(n)]
    w_sum = sum(weights)
    return sum(w * x for w, x in zip(weights, fight_series)) / w_sum


def weight_class_factor(weight_class: str | None) -> float:
    """Normalizing factor for weight-class-specific volume baselines."""
    if not weight_class:
        return 1.0
    return WEIGHT_CLASS_NORMALIZATION.get(weight_class.strip().lower(), 1.0)
