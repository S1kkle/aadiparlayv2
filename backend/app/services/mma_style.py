"""Simple style-cluster assignment from UFCStats per-minute aggregates.

Full k-means clustering requires fitting on the whole fighter
population. For an MVP, we use a rule-based classifier that maps the
per-minute aggregates from `ufcstats_db.get_fighter_per_minute_aggregates`
into a small set of clusters:

  - volume_striker     — high SLpM, low control time, mostly head/body strikes
  - power_striker      — moderate SLpM, high KD rate (proxy: high body strikes pct)
  - wrestle_boxer      — moderate SLpM + high takedowns + moderate control time
  - dominant_wrestler  — low SLpM + very high control time + high TDs
  - bjj_specialist     — low SLpM + high sub_attempts + high control time
  - hybrid             — middling everything

The vector representation is also returned so a downstream cosine
similarity can compute *style-matchup interaction* — the single highest-
ROI feature engineering move per the research synthesis.

Future work: replace with proper k-means once we have ≥100 UFCStats
fighters in the DB and a fitting cron. The interface returns the same
shape so the swap is drop-in.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from app.db import SqliteTTLCache
from app.services.ufcstats_db import get_fighter_per_minute_aggregates

log = logging.getLogger(__name__)


_STYLES = [
    "volume_striker",
    "power_striker",
    "wrestle_boxer",
    "dominant_wrestler",
    "bjj_specialist",
    "hybrid",
    "unknown",
]


def style_vector(aggregates: dict[str, Any]) -> dict[str, float]:
    """Numeric feature vector used for clustering / cosine similarity.

    All features normalized to [0, 1] using soft scaling against typical
    UFC ranges. Missing fields default to mid-range so the vector
    distance is conservative.
    """
    slpm = float(aggregates.get("sig_strikes_per_min", 4.0))
    abs_pm = float(aggregates.get("sig_strikes_absorbed_pm", 4.0))
    td_15 = float(aggregates.get("takedowns_per_15min", 1.0))
    sub_15 = float(aggregates.get("sub_attempts_per_15min", 0.5))
    ctrl_pct = float(aggregates.get("control_time_pct", 0.15))
    head_pct = float(aggregates.get("head_strike_pct", 0.55))
    leg_pct = float(aggregates.get("leg_strike_pct", 0.20))
    body_pct = float(aggregates.get("body_strike_pct", 0.20))

    def _norm(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    return {
        "slpm_norm": _norm(slpm, 1.0, 8.0),
        "absorbed_norm": _norm(abs_pm, 1.0, 8.0),
        "td_norm": _norm(td_15, 0.0, 5.0),
        "sub_norm": _norm(sub_15, 0.0, 3.0),
        "ctrl_norm": _norm(ctrl_pct, 0.0, 0.50),
        "head_strike_pct": head_pct,
        "leg_strike_pct": leg_pct,
        "body_strike_pct": body_pct,
    }


def classify_style(aggregates: dict[str, Any]) -> str:
    """Rule-based style assignment from per-minute aggregates.

    Returns one of `_STYLES`. Falls back to "unknown" when aggregates
    is empty (UFCStats hasn't been scraped for this fighter yet).
    """
    if not aggregates:
        return "unknown"
    v = style_vector(aggregates)
    slpm = v["slpm_norm"]
    td = v["td_norm"]
    sub = v["sub_norm"]
    ctrl = v["ctrl_norm"]

    if td > 0.55 and ctrl > 0.45 and slpm < 0.45:
        return "dominant_wrestler"
    if sub > 0.60 and ctrl > 0.40:
        return "bjj_specialist"
    if td > 0.45 and slpm > 0.45:
        return "wrestle_boxer"
    if slpm > 0.65 and ctrl < 0.30:
        return "volume_striker"
    if slpm > 0.45 and v["body_strike_pct"] > 0.25:
        return "power_striker"
    return "hybrid"


def matchup_factor(style_a: str, style_b: str) -> float:
    """Multiplicative striker-vs-grappler matchup factor for striking volume.

    `style_a` is the fighter whose μ we're adjusting; `style_b` is the
    opponent. Returns a value in [0.78, 1.18] following the rock-paper-
    scissors logic in MMA:
      - volume striker vs dominant wrestler → striking μ DOWN (fight ends
        up on the ground)
      - volume striker vs another striker → striking μ UP (more exchanges)
      - dominant wrestler vs striker → wrestler's striking μ flat-to-down
        (they take the fight to the ground, throwing little)
    """
    bonus_table: dict[tuple[str, str], float] = {
        ("volume_striker", "volume_striker"):     1.10,
        ("volume_striker", "power_striker"):      1.05,
        ("volume_striker", "wrestle_boxer"):      0.95,
        ("volume_striker", "dominant_wrestler"):  0.80,
        ("volume_striker", "bjj_specialist"):     0.85,
        ("volume_striker", "hybrid"):             1.00,
        ("power_striker", "volume_striker"):      1.02,
        ("power_striker", "dominant_wrestler"):   0.82,
        ("power_striker", "bjj_specialist"):      0.88,
        ("wrestle_boxer", "volume_striker"):      0.95,
        ("wrestle_boxer", "dominant_wrestler"):   0.92,
        ("wrestle_boxer", "bjj_specialist"):      0.98,
        ("dominant_wrestler", "volume_striker"):  0.85,
        ("dominant_wrestler", "power_striker"):   0.85,
        ("dominant_wrestler", "wrestle_boxer"):   0.95,
        ("bjj_specialist", "volume_striker"):     0.90,
        ("bjj_specialist", "dominant_wrestler"):  1.00,
    }
    return bonus_table.get((style_a, style_b), 1.00)


def get_fighter_style(
    cache: SqliteTTLCache, *, fighter: str, last_n_fights: int = 8,
) -> tuple[str, dict[str, Any]]:
    """Returns (style_label, aggregates_dict). aggregates is empty when
    no UFCStats data exists for this fighter (caller should fall back)."""
    agg = get_fighter_per_minute_aggregates(
        cache, fighter=fighter, last_n_fights=last_n_fights,
    )
    style = classify_style(agg)
    return style, agg
