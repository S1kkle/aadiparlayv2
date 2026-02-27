from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalParams:
    mu: float
    sigma: float


def fit_normal(series: list[float], *, sigma_floor: float = 1e-6) -> NormalParams | None:
    if not series:
        return None
    mu = sum(series) / len(series)
    if len(series) < 2:
        return NormalParams(mu=mu, sigma=max(sigma_floor, 0.0))
    var = sum((x - mu) ** 2 for x in series) / (len(series) - 1)
    sigma = math.sqrt(var) if var > 0 else 0.0
    return NormalParams(mu=mu, sigma=max(sigma_floor, sigma))


def normal_cdf(x: float) -> float:
    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over(*, line: float, params: NormalParams) -> float:
    # P(X > line) with continuity-ish adjustment omitted (stats are not discrete here).
    if params.sigma <= 0:
        if params.mu > line:
            return 1.0
        if params.mu < line:
            return 0.0
        return 0.5
    z = (line - params.mu) / params.sigma
    return 1.0 - normal_cdf(z)

