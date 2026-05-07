"""Underdog Pick'em payout tables — single source of truth.

Sources:
- Underdog help center "Pick'em Power Plays" + "Pick'em Insurance" + "Pick'em Flex"
- GamedayMath payout-math article (cross-checks the per-leg break-even)
- Occupy Fantasy Pick'em strategy article

The frontend mirrors these via [web/src/lib/underdog.ts]. If you change a
number here, change it there too (or generate the TS file from this module).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EntryType = Literal["standard", "flex", "insurance"]


# Standard / Power Play (all-or-nothing). Keys = number of legs.
STANDARD_PAYOUTS: dict[int, float] = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
    6: 37.5,
    7: 75.0,
    8: 150.0,
}


# Insurance: pays a smaller multiplier when one leg loses, otherwise the full
# Insurance multiplier. Format: { legs: {wins_required: payout_multiplier} }
# A 5-pick Insurance entry pays 10x for 5/5 and 2.5x for 4/5; 0/0 otherwise.
INSURANCE_PAYOUTS: dict[int, dict[int, float]] = {
    3: {3: 3.0, 2: 1.0},
    4: {4: 6.0, 3: 1.5},
    5: {5: 10.0, 4: 2.5},
}


# Flex: allows one or two losses at reduced payouts. Format same as Insurance.
# 3/4/5-pick Flex allow 1 loss; 6/7/8-pick Flex allow 2 losses.
FLEX_PAYOUTS: dict[int, dict[int, float]] = {
    3: {3: 2.25, 2: 1.0},
    4: {4: 5.0, 3: 1.5},
    5: {5: 10.0, 4: 2.5},
    6: {6: 25.0, 5: 2.0, 4: 0.4},
    7: {7: 50.0, 6: 5.0, 5: 0.5},
    8: {8: 100.0, 7: 10.0, 6: 1.0},
}


def standard_payout(legs: int) -> float:
    """Return the Standard / Power Play multiplier for `legs` correct picks.

    For unknown sizes, falls back to a power-of-2 extrapolation purely as a
    diagnostic; callers should not rely on it for sizing real bets.
    """
    if legs <= 1:
        return 0.0
    if legs in STANDARD_PAYOUTS:
        return STANDARD_PAYOUTS[legs]
    return float(2 ** (legs - 1)) * 1.5


def insurance_payout_table(legs: int) -> dict[int, float] | None:
    return INSURANCE_PAYOUTS.get(legs)


def flex_payout_table(legs: int) -> dict[int, float] | None:
    return FLEX_PAYOUTS.get(legs)


@dataclass(frozen=True)
class EntryEv:
    entry_type: EntryType
    legs: int
    expected_value_per_dollar: float
    expected_payout_multiplier: float
    win_probability_full: float
    breakdown: dict[int, float]  # wins_required -> probability of that scenario


def _binom_pmf_at_least(p: float, n: int, k: int) -> float:
    """P(X >= k) for X ~ Binomial(n, p) under independence assumption."""
    from math import comb
    if k > n:
        return 0.0
    if k <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def _binom_pmf_eq(p: float, n: int, k: int) -> float:
    from math import comb
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def expected_value(
    entry_type: EntryType,
    legs: int,
    leg_probs: list[float],
) -> EntryEv | None:
    """Return EV per $1 for a given Underdog entry type assuming per-leg
    independence. `leg_probs` length must equal `legs`.

    For Standard, EV = (joint_prob * payout) - 1.
    For Insurance/Flex, weighted EV across the wins-required tiers.

    Falls back to None when the leg count + entry type combination isn't
    supported by Underdog's documented payouts.
    """
    if legs != len(leg_probs):
        return None
    if entry_type == "standard":
        payout = standard_payout(legs)
        if payout <= 0:
            return None
        joint = 1.0
        for p in leg_probs:
            joint *= max(0.0, min(1.0, p))
        ev = (joint * payout) - 1.0
        return EntryEv(
            entry_type="standard",
            legs=legs,
            expected_value_per_dollar=ev,
            expected_payout_multiplier=joint * payout,
            win_probability_full=joint,
            breakdown={legs: joint},
        )

    table = (
        insurance_payout_table(legs) if entry_type == "insurance"
        else flex_payout_table(legs)
    )
    if not table:
        return None

    # Use mean leg probability as the binomial p. This loses some info versus
    # iterating the 2^legs joint distribution but Underdog only resolves on
    # number-of-wins, so the binomial collapse is exact when probs are equal
    # and a close approximation otherwise.
    p_mean = sum(leg_probs) / max(1, len(leg_probs))
    joint_full = 1.0
    for p in leg_probs:
        joint_full *= max(0.0, min(1.0, p))

    breakdown: dict[int, float] = {}
    expected_payout = 0.0
    for wins_required, mult in table.items():
        prob_at_exact = _binom_pmf_eq(p_mean, legs, wins_required)
        breakdown[wins_required] = prob_at_exact
        expected_payout += prob_at_exact * mult

    ev = expected_payout - 1.0
    return EntryEv(
        entry_type=entry_type,
        legs=legs,
        expected_value_per_dollar=ev,
        expected_payout_multiplier=expected_payout,
        win_probability_full=joint_full,
        breakdown=breakdown,
    )


def best_entry_type(
    legs: int,
    leg_probs: list[float],
) -> EntryEv | None:
    """Return the entry type with the highest EV per dollar for these legs.

    Returns None if no documented payout structure applies.
    """
    candidates: list[EntryEv] = []
    for et in ("standard", "insurance", "flex"):
        ev = expected_value(et, legs, leg_probs)  # type: ignore[arg-type]
        if ev is not None:
            candidates.append(ev)
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.expected_value_per_dollar)
