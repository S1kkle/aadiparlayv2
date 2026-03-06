from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalParams:
    mu: float
    sigma: float


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
