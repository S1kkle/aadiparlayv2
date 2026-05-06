from __future__ import annotations

import math
import threading
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


def negbin_prob_over(*, line: float, mean: float, variance: float) -> float:
    """P(X > line) where X ~ NegativeBinomial fit by method-of-moments to (mean, variance).

    Negative Binomial handles over-dispersed count data (variance > mean), which is
    the common case for points/rebounds/assists where Poisson under-fits the tails
    (cf. Binomial Basketball / Excelladz prop modeling articles).

    Parameterization: r (number of successes), p (success prob), with
        mean = r * (1-p) / p
        variance = r * (1-p) / p^2
    Solving:
        p = mean / variance
        r = mean^2 / (variance - mean)
    Falls back to Poisson when variance <= mean (under-dispersion).
    """
    if mean <= 0:
        return 0.0 if line >= 0 else 1.0
    if variance <= mean:  # under-dispersed → Poisson is optimal
        return poisson_prob_over(line=line, lam=mean)
    p = mean / variance
    if p <= 0 or p >= 1:
        return poisson_prob_over(line=line, lam=mean)
    r = (mean * mean) / (variance - mean)
    if r <= 0:
        return poisson_prob_over(line=line, lam=mean)
    k = int(math.floor(line))
    if k < 0:
        return 1.0
    # P(X <= k) = sum_{i=0..k} C(i+r-1, i) p^r (1-p)^i, computed via log-pmf.
    cdf = 0.0
    log_p = math.log(p)
    log_one_minus_p = math.log(1.0 - p)
    log_r = math.lgamma(r)
    for i in range(k + 1):
        log_pmf = (
            math.lgamma(i + r) - math.lgamma(i + 1) - log_r
            + r * log_p + i * log_one_minus_p
        )
        cdf += math.exp(log_pmf)
    return max(0.0, min(1.0, 1.0 - cdf))


def gamma_prob_over(*, line: float, mean: float, variance: float) -> float:
    """P(X > line) where X ~ Gamma fit by method-of-moments to (mean, variance).

    Gamma is the right family for non-negative, right-skewed continuous stats
    like passing/rushing/receiving yards. Falls back to Normal when the moments
    are degenerate.
    """
    if mean <= 0 or variance <= 0:
        # Mean 0 with positive line: definitely under; mean > 0 but no variance:
        # Normal degenerate case is fine here.
        if mean > line:
            return 1.0
        if mean < line:
            return 0.0
        return 0.5
    if line <= 0:
        return 1.0
    # k (shape) = mean^2 / variance, theta (scale) = variance / mean.
    k = (mean * mean) / variance
    theta = variance / mean
    if k <= 0 or theta <= 0:
        # Degenerate — fall back to normal-ish approximation.
        sigma = math.sqrt(variance)
        z = (line - mean) / max(sigma, 1e-6)
        return 1.0 - normal_cdf(z)
    # P(X > line) = upper regularized incomplete gamma Q(k, line/theta).
    return _regularized_gamma_q(k, line / theta)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Q(a, x) = 1 - P(a, x) — upper regularized incomplete gamma function.

    Uses series for x < a+1 and continued fractions otherwise (NR §6.2 style).
    """
    if x <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    log_prefix = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        # Series representation of P(a,x).
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(200):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        p = math.exp(log_prefix) * total
        return max(0.0, min(1.0, 1.0 - p))
    # Continued fraction for Q(a,x).
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = math.exp(log_prefix) * h
    return max(0.0, min(1.0, q))


# Stat → distribution family used to compute P(X > line).
# - "negbin": discrete count, often over-dispersed (NBA/NHL counting stats)
# - "poisson": discrete count, generally Poisson-like (TDs, knock-downs, etc.)
# - "gamma": continuous non-negative, right-skewed (yards)
# - "normal": fallback Gaussian
_DISTRIBUTION_FAMILY: dict[str, str] = {
    # NBA
    "points": "negbin",
    "totalRebounds": "negbin",
    "assists": "negbin",
    "steals": "poisson",
    "blocks": "poisson",
    "turnovers": "negbin",
    "threePointFieldGoalsMade": "negbin",
    "freeThrowsMade": "negbin",
    # NFL (yards = gamma; counts = negbin/poisson)
    "passingYards": "gamma",
    "rushingYards": "gamma",
    "receivingYards": "gamma",
    "receptions": "negbin",
    "passingTouchdowns": "poisson",
    "rushingTouchdowns": "poisson",
    "receivingTouchdowns": "poisson",
    "interceptions": "poisson",
    "completions": "negbin",
    "passingAttempts": "negbin",
    "rushingAttempts": "negbin",
    # NHL
    "goals": "poisson",
    "shots": "negbin",
    "shotsTotal": "negbin",
    "blockedShots": "poisson",
    "saves": "negbin",
    "goalsAgainst": "poisson",
    # MMA
    "sigStrikesLanded": "negbin",
    "totalStrikesLanded": "negbin",
    "takedownsLanded": "poisson",
    "knockDowns": "poisson",
    "submissions": "poisson",
    "advances": "poisson",
}


def stat_distribution_family(field_name: str | None) -> str:
    if not field_name:
        return "normal"
    return _DISTRIBUTION_FAMILY.get(field_name, "normal")


def prob_over_for_field(
    *, line: float, mean: float, variance: float, field_name: str | None,
) -> float:
    """Dispatch P(X > line) using the right distribution for the stat type.

    Falls through to Normal for unknown stats and degenerate variance.
    """
    family = stat_distribution_family(field_name)
    if variance <= 0 or mean <= 0:
        # Degenerate — Normal handles trivial cases gracefully.
        sigma = math.sqrt(max(variance, 0.0))
        return prob_over(line=line, params=NormalParams(mu=mean, sigma=max(sigma, 1e-6)))
    if family == "negbin":
        return negbin_prob_over(line=line, mean=mean, variance=variance)
    if family == "poisson":
        return poisson_prob_over(line=line, lam=mean)
    if family == "gamma":
        return gamma_prob_over(line=line, mean=mean, variance=variance)
    sigma = math.sqrt(variance)
    return prob_over(line=line, params=NormalParams(mu=mean, sigma=sigma))


def is_low_count_stat(field_name: str | None) -> bool:
    return field_name in LOW_COUNT_STATS if field_name else False


def bayesian_shrink(
    sample_mean: float,
    n: int,
    prior_mean: float,
    *,
    shrinkage_k: float = 3.0,
) -> float:
    """Shrink sample_mean toward prior_mean when n is small.

    IMPORTANT: prior_mean MUST be an outside-the-market baseline (league/position
    average), NEVER the betting line — shrinking toward the line is a self-
    fulfilling prophecy that destroys real edge.
    """
    return (n * sample_mean + shrinkage_k * prior_mean) / (n + shrinkage_k)


# League/position priors per stat. Values are rough averages of starter-level
# production from publicly available season stats; intended as a regularizer
# for small-sample players, not a precise projection.
# Format: PRIORS[sport][stat_field][position_abbr or "*"] -> mean
_PRIORS: dict[str, dict[str, dict[str, float]]] = {
    "NBA": {
        "points":                {"G": 12.0, "PG": 14.0, "SG": 12.0, "SF": 11.0, "PF": 11.0, "C": 11.0, "F": 11.0, "*": 11.5},
        "totalRebounds":         {"G": 3.5,  "PG": 3.5,  "SG": 3.5,  "SF": 4.5,  "PF": 6.5,  "C": 7.5,  "F": 6.0,  "*": 5.0},
        "assists":               {"G": 4.0,  "PG": 5.5,  "SG": 3.0,  "SF": 2.5,  "PF": 2.0,  "C": 2.0,  "F": 2.5,  "*": 3.0},
        "steals":                {"*": 0.8},
        "blocks":                {"G": 0.3,  "PG": 0.3,  "SG": 0.3,  "SF": 0.5,  "PF": 0.7,  "C": 1.0,  "F": 0.6,  "*": 0.6},
        "turnovers":             {"G": 1.6,  "PG": 2.2,  "SG": 1.4,  "SF": 1.4,  "PF": 1.3,  "C": 1.5,  "F": 1.4,  "*": 1.5},
        "threePointFieldGoalsMade": {"*": 1.5},
        "freeThrowsMade":        {"*": 1.8},
        "minutes":               {"*": 22.0},
    },
    "NFL": {
        "passingYards":          {"QB": 220.0, "*": 50.0},
        "passingTouchdowns":     {"QB": 1.4,   "*": 0.2},
        "interceptions":         {"QB": 0.7,   "*": 0.1},
        "rushingYards":          {"RB": 55.0, "QB": 18.0, "WR": 4.0,  "*": 12.0},
        "rushingTouchdowns":     {"RB": 0.4,  "QB": 0.2,  "*": 0.1},
        "receivingYards":        {"WR": 50.0, "TE": 30.0, "RB": 18.0, "*": 25.0},
        "receivingTouchdowns":   {"WR": 0.4,  "TE": 0.3,  "RB": 0.2,  "*": 0.2},
        "receptions":            {"WR": 4.0,  "TE": 2.5,  "RB": 2.0,  "*": 2.5},
    },
    "NHL": {
        "goals":                 {"*": 0.4},
        "assists":                {"*": 0.6},
        "points":                {"*": 1.0},
        "shots":                 {"D": 1.5, "F": 2.5, "C": 2.5, "LW": 2.5, "RW": 2.5, "*": 2.0},
        "shotsTotal":            {"D": 1.5, "F": 2.5, "*": 2.0},
        "blockedShots":          {"D": 1.4, "F": 0.6, "*": 1.0},
        "saves":                 {"G": 24.0, "*": 24.0},
        "goalsAgainst":          {"G": 2.6,  "*": 2.6},
    },
    "MMA": {
        "sigStrikesLanded":      {"*": 60.0},
        "totalStrikesLanded":    {"*": 80.0},
        "takedownsLanded":       {"*": 1.2},
        "knockDowns":            {"*": 0.3},
        "submissions":           {"*": 0.4},
        "advances":              {"*": 0.6},
    },
}


def league_prior_mean(
    *, sport: str | None, stat_field: str | None, position: str | None = None,
    fallback: float | None = None,
) -> float | None:
    """Return a league/position-baseline prior mean for the given stat.

    Falls back through: position-specific -> "*" wildcard -> caller fallback -> None.
    """
    if not sport or not stat_field:
        return fallback
    sport_key = sport.upper()
    sport_priors = _PRIORS.get(sport_key)
    if not sport_priors:
        return fallback
    stat_priors = sport_priors.get(stat_field)
    if not stat_priors:
        return fallback
    if position:
        pos = position.strip().upper()
        # Try exact match, then prefix match (e.g. "PG" matches "PG/SG")
        if pos in stat_priors:
            return stat_priors[pos]
        for key in stat_priors:
            if key != "*" and (pos.startswith(key) or key.startswith(pos)):
                return stat_priors[key]
    if "*" in stat_priors:
        return stat_priors["*"]
    return fallback


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

    IMPORTANT: prior_mean MUST be an outside-the-market baseline (league/position
    average). Shrinking toward the betting line forces the model to agree with
    the book and erases real edge.
    """
    k = _active_params.get("shrinkage_k", _DEFAULT_PARAMS["shrinkage_k"])
    weight = n / (n + k)
    return weight * sample_mean + (1 - weight) * prior_mean


# ── Dynamic calibration params ────────────────────────────────────────

_params_lock = threading.Lock()

_DEFAULT_PARAMS: dict[str, float] = {
    "decay": 0.88,
    "shrinkage_k": 3.0,
    "vol_assists": 0.87,
    "vol_steals": 1.16,
    "vol_blocks": 1.16,
    "vol_3pm": 0.87,
    "prob_cap": 0.66,
    "prox_penalty_min": 0.30,
    "pick_threshold": 0.64,
}

_active_params: dict[str, float] = dict(_DEFAULT_PARAMS)


def load_calibrated_params(params: dict[str, float]) -> None:
    """Hot-reload calibrated parameters at runtime (thread-safe)."""
    with _params_lock:
        for k in _DEFAULT_PARAMS:
            if k in params:
                _active_params[k] = float(params[k])


def get_calibrated_params() -> dict[str, float]:
    """Return a snapshot of the currently active calibrated parameters."""
    with _params_lock:
        return dict(_active_params)


def get_default_params() -> dict[str, float]:
    return dict(_DEFAULT_PARAMS)


# ── Learning-driven refinements ───────────────────────────────────────

_STAT_VOLATILITY_BASE: dict[str, float] = {
    "turnovers": 1.10,
    "double_doubles": 1.25,
    "triple_doubles": 1.35,
    "goals": 1.10,
    "knockDowns": 1.20,
    "submissions": 1.20,
}


def stat_volatility_multiplier(field_name: str | None) -> float:
    """Inflate sigma for inherently noisy stat types.

    Assists depend on teammate shot-making, steals/blocks on opponent behavior.
    These stats have higher game-to-game variance than points or rebounds.
    Calibratable fields (assists, steals, blocks, 3pm) read from _active_params.
    """
    if field_name is None:
        return 1.0
    calibrated_map = {
        "assists": _active_params.get("vol_assists", 0.87),
        "steals": _active_params.get("vol_steals", 1.16),
        "blocks": _active_params.get("vol_blocks", 1.16),
        "three_points_made": _active_params.get("vol_3pm", 0.87),
        "threePointFieldGoalsMade": _active_params.get("vol_3pm", 0.87),
    }
    if field_name in calibrated_map:
        return calibrated_map[field_name]
    return _STAT_VOLATILITY_BASE.get(field_name, 1.0)


def line_proximity_penalty(*, line: float, median: float, sigma: float) -> float:
    """Penalize picks where the line is very close to the player's median.

    When |line - median| < 0.5*sigma, the pick is essentially a coin flip.
    Returns a 0..1 multiplier on the excess probability (amount above 0.5).
    """
    if sigma <= 0:
        return 1.0
    gap = abs(line - median) / sigma
    if gap >= 0.5:
        return 1.0
    floor = _active_params.get("prox_penalty_min", _DEFAULT_PARAMS["prox_penalty_min"])
    if gap <= 0.1:
        return floor
    return floor + (gap - 0.1) / (0.5 - 0.1) * (1.0 - floor)


def edge_skepticism(edge: float) -> float:
    """Dampen extreme edges — edges >15% against efficient markets are usually model error.

    Sportsbooks are good at setting lines. A 27% edge on a role player's points
    almost certainly means the model is wrong, not that the book is.
    """
    abs_e = abs(edge)
    if abs_e <= 0.10:
        return edge
    if abs_e <= 0.20:
        damped = 0.10 + (abs_e - 0.10) * 0.5
    else:
        damped = 0.15 + (abs_e - 0.20) * 0.25
    return math.copysign(min(damped, 0.20), edge)


_MODEL_PROB_CAP_BASE: dict[str, float] = {
    "assists": 0.66,
    "steals": 0.64,
    "blocks": 0.64,
    "turnovers": 0.66,
    "knockDowns": 0.64,
    "submissions": 0.64,
    "double_doubles": 0.60,
    "triple_doubles": 0.56,
    "three_points_made": 0.66,
    "threePointFieldGoalsMade": 0.66,
}


# ── Logistic regression for confidence tiering ────────────────────────


_TIER_FEATURE_NAMES = (
    "intercept",
    "model_prob",
    "edge",
    "kelly_fraction",
    "edge_confidence",
    "stat_consistency",
    "hit_rate_last10",
    "abs_streak_aligned",
    "n_games_norm",
    "ai_confidence",
)


_tier_lock = threading.Lock()
_tier_weights: list[float] | None = None
_tier_metrics: dict[str, float] | None = None


def get_tier_model() -> dict | None:
    with _tier_lock:
        if _tier_weights is None:
            return None
        return {
            "weights": list(_tier_weights),
            "feature_names": list(_TIER_FEATURE_NAMES),
            "metrics": dict(_tier_metrics or {}),
        }


def set_tier_model(weights: list[float], *, metrics: dict | None = None) -> None:
    with _tier_lock:
        global _tier_weights, _tier_metrics
        if len(weights) != len(_TIER_FEATURE_NAMES):
            raise ValueError(
                f"Tier model expects {len(_TIER_FEATURE_NAMES)} weights, got {len(weights)}"
            )
        _tier_weights = list(weights)
        _tier_metrics = dict(metrics or {})


def _tier_features_from_dict(data: dict) -> list[float]:
    """Build the 10-feature row for tier prediction from a dict-shaped prop/entry.

    Robust to None / missing fields. The order MUST match _TIER_FEATURE_NAMES.
    """
    def _f(k: str, default: float = 0.0) -> float:
        v = data.get(k)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    streak_aligned = 0.0
    streak = data.get("current_streak")
    side = data.get("side", "over")
    if isinstance(streak, (int, float)):
        if side == "over" and streak > 0:
            streak_aligned = float(min(abs(streak), 8))
        elif side == "under" and streak < 0:
            streak_aligned = float(min(abs(streak), 8))

    n_games = data.get("recent_games")
    n = len(n_games) if isinstance(n_games, list) else 0
    n_norm = min(n / 15.0, 1.0)

    return [
        1.0,  # intercept
        _f("model_prob", 0.5),
        _f("edge", 0.0),
        _f("kelly_fraction", 0.0),
        _f("edge_confidence", 0.0),
        _f("stat_consistency", 0.5),
        _f("hit_rate_last10", 0.5),
        streak_aligned,
        n_norm,
        _f("ai_confidence", 0.5),
    ]


def predict_hit_prob(prop_dict: dict) -> float | None:
    """Predict P(hit) for a prop using the trained logistic-regression model.
    Returns None when no model is trained yet (caller should fall back)."""
    model = get_tier_model()
    if model is None:
        return None
    weights = model["weights"]
    x = _tier_features_from_dict(prop_dict)
    z = sum(w * v for w, v in zip(weights, x))
    # Sigmoid with overflow protection.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_tier_logistic(
    rows: list[dict],
    *,
    lr: float = 0.1,
    n_iter: int = 400,
    l2: float = 0.01,
) -> dict | None:
    """Fit a small logistic regression on resolved learning_log rows.

    `rows` shape (per row): {features keyed by _TIER_FEATURE_NAMES (excluding
    intercept), "hit": 0/1}. Returns metrics dict or None if not enough data.

    Uses plain-Python batch gradient descent; no scipy/sklearn dependency.
    """
    if len(rows) < 30:
        return None

    n_features = len(_TIER_FEATURE_NAMES)
    X = [row.get("features") for row in rows if isinstance(row.get("features"), list) and len(row["features"]) == n_features]
    y = [int(row["hit"]) for row in rows if isinstance(row.get("features"), list) and len(row["features"]) == n_features]
    if len(X) < 30:
        return None
    n = len(X)
    weights = [0.0] * n_features

    for _ in range(n_iter):
        grads = [0.0] * n_features
        for xi, yi in zip(X, y):
            z = sum(w * v for w, v in zip(weights, xi))
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
            err = p - yi
            for k in range(n_features):
                grads[k] += err * xi[k]
        # Update with L2 regularization (skip intercept).
        for k in range(n_features):
            reg = 0.0 if k == 0 else l2 * weights[k]
            weights[k] -= lr * (grads[k] / n + reg)

    # Diagnostics.
    correct = 0
    brier = 0.0
    log_loss = 0.0
    for xi, yi in zip(X, y):
        z = sum(w * v for w, v in zip(weights, xi))
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
        if (p >= 0.5) == bool(yi):
            correct += 1
        brier += (p - yi) ** 2
        p_clip = max(1e-6, min(1 - 1e-6, p))
        log_loss += -(yi * math.log(p_clip) + (1 - yi) * math.log(1 - p_clip))

    metrics = {
        "n_train": n,
        "accuracy": correct / n,
        "brier": brier / n,
        "log_loss": log_loss / n,
    }
    set_tier_model(weights, metrics=metrics)
    return {"weights": weights, "metrics": metrics, "feature_names": list(_TIER_FEATURE_NAMES)}


def model_prob_cap(field_name: str | None) -> float:
    """Maximum model probability by stat type.

    Volatile stats (assists, steals, blocks) should never show 80%+ confidence
    because their game-to-game variance is too high.
    The default cap is read from _active_params["prob_cap"].
    """
    default_cap = _active_params.get("prob_cap", _DEFAULT_PARAMS["prob_cap"])
    if field_name is None:
        return default_cap
    per_stat = _MODEL_PROB_CAP_BASE.get(field_name)
    if per_stat is not None:
        return min(per_stat, default_cap)
    return default_cap
