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
