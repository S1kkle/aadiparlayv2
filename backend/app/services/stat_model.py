from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalParams:
    mu: float
    sigma: float


@dataclass(frozen=True)
class StatProfile:
    """Rich statistical profile computed from a game series."""
    mean: float
    median: float
    sigma: float
    floor: float  # 10th percentile value
    ceiling: float  # 90th percentile value
    consistency: float  # 0..1, how often within 1 sigma of mean
    current_streak: int  # positive = consecutive overs, negative = unders
    cv: float  # coefficient of variation (sigma/mean), lower = more consistent


LOW_COUNT_STATS = frozenset({
    "blocks", "steals", "knockDowns", "takedownsLanded",
    "submissions", "advances", "interceptions",
    "rushingTouchdowns", "receivingTouchdowns", "passingTouchdowns",
    "goals",
})


def fit_normal(series: list[float], *, sigma_floor: float = 1e-6) -> NormalParams | None:
    if not series:
        return None
    mu = sum(series) / len(series)
    if len(series) < 2:
        return NormalParams(mu=mu, sigma=max(sigma_floor, 0.0))
    var = sum((x - mu) ** 2 for x in series) / (len(series) - 1)
    sigma = math.sqrt(var) if var > 0 else 0.0
    return NormalParams(mu=mu, sigma=max(sigma_floor, sigma))


def fit_normal_weighted(
    series: list[float],
    *,
    decay: float = 0.85,
    sigma_floor: float = 1e-6,
) -> NormalParams | None:
    """Fit normal with exponential decay weights.  Index 0 = most recent = highest weight."""
    if not series:
        return None
    n = len(series)
    if n < 2:
        return NormalParams(mu=series[0], sigma=max(sigma_floor, 0.0))
    weights = [decay ** i for i in range(n)]
    w_sum = sum(weights)
    mu = sum(w * x for w, x in zip(weights, series)) / w_sum
    var = sum(w * (x - mu) ** 2 for w, x in zip(weights, series)) / w_sum
    sigma = math.sqrt(var) if var > 0 else 0.0
    return NormalParams(mu=mu, sigma=max(sigma_floor, sigma))


def compute_stat_profile(series: list[float], *, line: float, side: str) -> StatProfile | None:
    """Compute a full statistical profile from game values (index 0 = most recent)."""
    if not series:
        return None
    n = len(series)
    mean = sum(series) / n
    sorted_vals = sorted(series)
    median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    sigma = 0.0
    if n >= 2:
        var = sum((x - mean) ** 2 for x in series) / (n - 1)
        sigma = math.sqrt(var) if var > 0 else 0.0

    floor_idx = max(0, int(n * 0.1))
    ceil_idx = min(n - 1, int(n * 0.9))
    floor_val = sorted_vals[floor_idx]
    ceil_val = sorted_vals[ceil_idx]

    within_1sigma = sum(1 for v in series if abs(v - mean) <= max(sigma, 0.01))
    consistency = within_1sigma / n

    streak = 0
    for v in series:
        hit = (v > line) if side == "over" else (v < line)
        if hit:
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break

    cv = (sigma / abs(mean)) if abs(mean) > 0.01 else 0.0

    return StatProfile(
        mean=round(mean, 2),
        median=round(median, 2),
        sigma=round(sigma, 2),
        floor=round(floor_val, 2),
        ceiling=round(ceil_val, 2),
        consistency=round(consistency, 3),
        current_streak=streak,
        cv=round(cv, 3),
    )


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over(*, line: float, params: NormalParams) -> float:
    if params.sigma <= 0:
        if params.mu > line:
            return 1.0
        if params.mu < line:
            return 0.0
        return 0.5
    z = (line - params.mu) / params.sigma
    return 1.0 - normal_cdf(z)


def poisson_prob_over(*, line: float, lam: float) -> float:
    """P(X > line) where X ~ Poisson(lam).  Better for discrete low-count stats."""
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


def is_low_count_stat(field_name: str | None) -> bool:
    return field_name in LOW_COUNT_STATS if field_name else False


def bayesian_shrink(
    sample_mean: float,
    n: int,
    prior_mean: float,
    *,
    shrinkage_k: float = 3.0,
) -> float:
    """Shrink sample_mean toward prior_mean when n is small."""
    return (n * sample_mean + shrinkage_k * prior_mean) / (n + shrinkage_k)


def line_percentile(*, line: float, params: NormalParams) -> float:
    """What percentile of the distribution the line sits at.  0.5 = mean."""
    if params.sigma <= 0:
        return 0.5
    z = (line - params.mu) / params.sigma
    return normal_cdf(z)


# ── Sharp betting additions ───────────────────────────────────────────


def devig_multiplicative(implied_over: float, implied_under: float) -> tuple[float, float]:
    """Remove vig using the multiplicative (proportional) method.

    Sportsbooks inflate both sides so they sum to >1.  This divides each
    side by the total overround to recover true probabilities.
    """
    total = implied_over + implied_under
    if total <= 0:
        return (0.5, 0.5)
    return (implied_over / total, implied_under / total)


def devig_power(implied_over: float, implied_under: float, *, tol: float = 1e-8, max_iter: int = 50) -> tuple[float, float]:
    """Remove vig using the power method (better for favorite-longshot bias).

    Finds exponent k such that implied_over^k + implied_under^k = 1.
    """
    if implied_over <= 0 or implied_under <= 0:
        return devig_multiplicative(implied_over, implied_under)

    lo, hi = 0.5, 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = implied_over ** mid + implied_under ** mid
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    return (implied_over ** mid, implied_under ** mid)


def kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    *,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion — optimal bet sizing as fraction of bankroll.

    Full Kelly = (p * b - q) / b, where p = win prob, q = 1-p, b = net decimal odds.
    We return fraction * full_kelly (default 25% = quarter Kelly, standard for sharps).
    Returns 0 if edge is non-positive.
    """
    if model_prob <= 0 or model_prob >= 1 or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (model_prob * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return round(min(full_kelly * fraction, 0.10), 4)  # cap at 10% of bankroll


def per_minute_rate(total: float, minutes: float) -> float | None:
    """Compute per-minute production rate. Returns None if insufficient minutes."""
    if minutes < 5:
        return None
    return total / minutes


def projected_value(per_min_rate: float, projected_minutes: float) -> float:
    """Project stat value from per-minute rate and expected minutes."""
    return per_min_rate * projected_minutes


def blowout_minutes_discount(spread: float, avg_minutes: float) -> float:
    """Estimate minutes reduction for blowouts.

    Sharp principle: spreads >10 pts reduce starter minutes by ~10-15%.
    Returns projected minutes after discount.
    """
    abs_spread = abs(spread)
    if abs_spread <= 7:
        return avg_minutes
    if abs_spread <= 10:
        return avg_minutes * 0.95
    if abs_spread <= 14:
        return avg_minutes * 0.90
    return avg_minutes * 0.85


def edge_confidence(
    edge: float,
    n_games: int,
    cv: float,
) -> float:
    """Variance-weighted edge confidence.

    Sharps don't just look at raw edge — they weight it by:
    1. Sample size (more games = more reliable)
    2. Player consistency (low CV = more reliable edge)

    Returns a 0..1 confidence score.
    """
    sample_factor = min(1.0, n_games / 15.0)

    cv_factor = 1.0
    if cv > 0:
        cv_factor = max(0.3, 1.0 - (cv - 0.3) * 0.7)
        cv_factor = min(1.0, cv_factor)

    raw = abs(edge) * sample_factor * cv_factor
    return round(min(1.0, raw * 10.0), 3)


def continuous_shrinkage(sample_mean: float, n: int, prior_mean: float) -> float:
    """Continuous Bayesian shrinkage that scales smoothly with sample size.

    Unlike the current binary shrinkage (only for n<5), this always applies
    some shrinkage, but it diminishes as sample size grows.
    Equivalent to a prior with strength ~8 games.
    """
    k = 8.0
    weight = n / (n + k)
    return weight * sample_mean + (1 - weight) * prior_mean
