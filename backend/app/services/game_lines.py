"""Synthesize game-line "props" (moneyline, spread, game total, team total)
from ESPN's bundled sportsbook odds.

Why: Underdog Pick'em itself doesn't expose moneyline / spread / point-total
markets through the same `over_under_lines` payload we already parse, but the
user wants those markets available so they can be researched and combined
into mixed parlays. ESPN bundles a `pickcenter` block on every event summary
that gives consensus spread + total + per-team moneyline from a real
sportsbook (typically ESPN BET / Caesars). That's our source.

These synthetic Props use a stable `underdog_option_id` of the form
``espn:game:<event_id>:<market>:<side>`` so the rest of the pipeline (parlay
slip, learning_log, history) can key off them like any other prop. They're
clearly tagged with `market_type` so the frontend's "Game Lines" filter
picks them up and the parlay slip's smart payout detection switches to the
true decimal-odds product when any of these synthetic legs is added.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from math import erf, sqrt
from typing import Any

from app.clients.espn import EspnClient
from app.models.core import Prop, SportId

log = logging.getLogger(__name__)


# Sport-calibrated sigmas for Normal-CDF probability modeling. These are
# the same values used inside Ranker._apply_game_line_model — kept in sync
# so synthetic props get reasonable probabilities baked in at generation
# time (the ranker's model pass will recompute them, but having a sensible
# default avoids zeros / NaNs when display happens before the model pass).
SIGMA_TOTAL = {"NBA": 10.5, "NFL": 10.0, "NHL": 1.5}
SIGMA_TEAM = {"NBA": 8.0, "NFL": 7.5, "NHL": 1.0}
SIGMA_MARGIN = {"NBA": 12.0, "NFL": 13.0, "NHL": 2.0}

# Standard sportsbook -110 vig: decimal 1.91. Used for spread + total legs
# when ESPN doesn't surface a per-side price (it usually doesn't — totals
# and spreads are conventionally priced at -110 across all books).
DEFAULT_DECIMAL_PRICE = 1.91


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.5
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def _american_to_decimal(american: int) -> float:
    if american == 0:
        return 1.0
    if american < 0:
        return 1.0 + (100.0 / -american)
    return 1.0 + (american / 100.0)


def _round_half_int(x: float) -> float:
    """Round to the nearest 0.5 — sportsbooks always offer half-point lines
    on totals and spreads to avoid pushes."""
    return round(x * 2) / 2.0


def _scheduled_at_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _make_id(event_id: str, market: str, side: str) -> str:
    return f"espn:game:{event_id}:{market}:{side}"


def _build_props_for_event(
    *,
    sport: SportId,
    event_id: str,
    odds: dict[str, Any],
) -> list[Prop]:
    """Produce up to 6 synthetic props from a single event-odds payload.

    - Moneyline: one leg per team (uses ESPN's American moneyline as the price)
    - Spread:    one leg per team at the published number, priced -110
    - Total:     over + under at a half-point line, priced -110
    - Team total: skipped here — Underdog rarely offers it and ESPN doesn't
      publish it directly. We could synthesize it from total/2 ± spread/2
      but the implied edge would always be ~0 against our own model.

    Returns [] when the event doesn't have enough data to be useful.
    """
    out: list[Prop] = []
    spread_home = odds.get("spread_home")
    total = odds.get("total")
    ml_home = odds.get("moneyline_home")
    ml_away = odds.get("moneyline_away")
    home_abbr = odds.get("home_abbr")
    away_abbr = odds.get("away_abbr")
    game_title = odds.get("game_title") or (
        f"{away_abbr} @ {home_abbr}" if home_abbr and away_abbr else None
    )
    scheduled_at = _scheduled_at_dt(odds.get("scheduled_at"))

    # ── Moneyline ────────────────────────────────────────────────────
    if isinstance(ml_home, int) and home_abbr:
        decimal = _american_to_decimal(ml_home)
        implied = 1.0 / decimal if decimal > 0 else 0.5
        out.append(
            Prop(
                sport=sport,
                player_name=f"{home_abbr} ML",
                underdog_option_id=_make_id(event_id, "ml", "home"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=home_abbr,
                opponent_abbr=away_abbr,
                is_home=True,
                market_type="moneyline",
                subject_kind="team",
                subject_name=home_abbr,
                stat="Moneyline",
                display_stat="Moneyline",
                line=0.0,
                side="over",  # encodes "win" — see normalizer / ranker
                american_price=ml_home,
                decimal_price=decimal,
                implied_prob=round(implied, 4),
                breakeven_prob=round(implied, 4),
                model_prob=round(implied, 4),
                notes=[f"ESPN ML provider: {odds.get('provider') or 'consensus'}"],
            )
        )
    if isinstance(ml_away, int) and away_abbr:
        decimal = _american_to_decimal(ml_away)
        implied = 1.0 / decimal if decimal > 0 else 0.5
        out.append(
            Prop(
                sport=sport,
                player_name=f"{away_abbr} ML",
                underdog_option_id=_make_id(event_id, "ml", "away"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=away_abbr,
                opponent_abbr=home_abbr,
                is_home=False,
                market_type="moneyline",
                subject_kind="team",
                subject_name=away_abbr,
                stat="Moneyline",
                display_stat="Moneyline",
                line=0.0,
                side="over",
                american_price=ml_away,
                decimal_price=decimal,
                implied_prob=round(implied, 4),
                breakeven_prob=round(implied, 4),
                model_prob=round(implied, 4),
                notes=[f"ESPN ML provider: {odds.get('provider') or 'consensus'}"],
            )
        )

    # ── Spread ───────────────────────────────────────────────────────
    if isinstance(spread_home, (int, float)) and home_abbr and away_abbr:
        spread_home_val = _round_half_int(float(spread_home))
        spread_away_val = -spread_home_val
        # Spread legs priced -110 by convention. Probability ~50/50 (the line
        # is calibrated to the public's split). The ranker will refine this
        # via the proper Normal-CDF model on its game-line pass.
        sigma = SIGMA_MARGIN.get(sport, 12.0)
        # P(home_margin > spread_home) under N(mean=-spread_home, sigma)
        mu_margin_home = -spread_home_val  # expected home margin
        p_cover_home = 1.0 - _normal_cdf(spread_home_val, mu_margin_home, sigma)
        p_cover_away = 1.0 - p_cover_home
        out.append(
            Prop(
                sport=sport,
                player_name=f"{home_abbr} {spread_home_val:+.1f}",
                underdog_option_id=_make_id(event_id, "spread", "home"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=home_abbr,
                opponent_abbr=away_abbr,
                is_home=True,
                market_type="spread",
                subject_kind="team",
                subject_name=home_abbr,
                stat="Spread",
                display_stat=f"Spread {spread_home_val:+.1f}",
                line=spread_home_val,
                side="over",
                american_price=-110,
                decimal_price=DEFAULT_DECIMAL_PRICE,
                implied_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                breakeven_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                model_prob=round(p_cover_home, 4),
                notes=[f"ESPN spread (home perspective): {spread_home_val:+.1f}"],
            )
        )
        out.append(
            Prop(
                sport=sport,
                player_name=f"{away_abbr} {spread_away_val:+.1f}",
                underdog_option_id=_make_id(event_id, "spread", "away"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=away_abbr,
                opponent_abbr=home_abbr,
                is_home=False,
                market_type="spread",
                subject_kind="team",
                subject_name=away_abbr,
                stat="Spread",
                display_stat=f"Spread {spread_away_val:+.1f}",
                line=spread_away_val,
                side="over",
                american_price=-110,
                decimal_price=DEFAULT_DECIMAL_PRICE,
                implied_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                breakeven_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                model_prob=round(p_cover_away, 4),
                notes=[f"ESPN spread (away perspective): {spread_away_val:+.1f}"],
            )
        )

    # ── Game total ──────────────────────────────────────────────────
    if isinstance(total, (int, float)) and home_abbr and away_abbr:
        total_line = _round_half_int(float(total))
        sigma_t = SIGMA_TOTAL.get(sport, 10.0)
        # P(actual > line) when actual is N(total, sigma_t). Since the
        # offered line equals the consensus total, this is exactly 0.5
        # at line = total. Half-point shifts move it slightly.
        p_over = 1.0 - _normal_cdf(total_line, float(total), sigma_t)
        out.append(
            Prop(
                sport=sport,
                player_name=f"{game_title or 'Game'} Total",
                underdog_option_id=_make_id(event_id, "total", "over"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=home_abbr,
                opponent_abbr=away_abbr,
                is_home=True,
                market_type="game_total",
                subject_kind="game",
                subject_name=game_title,
                stat="Total Points",
                display_stat=f"Total Points",
                line=total_line,
                side="over",
                american_price=-110,
                decimal_price=DEFAULT_DECIMAL_PRICE,
                implied_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                breakeven_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                model_prob=round(p_over, 4),
                vegas_total=float(total),
                vegas_spread=float(spread_home) if isinstance(spread_home, (int, float)) else None,
                notes=[f"ESPN total: {total_line}"],
            )
        )
        out.append(
            Prop(
                sport=sport,
                player_name=f"{game_title or 'Game'} Total",
                underdog_option_id=_make_id(event_id, "total", "under"),
                game_title=game_title,
                scheduled_at=scheduled_at,
                team_abbr=home_abbr,
                opponent_abbr=away_abbr,
                is_home=True,
                market_type="game_total",
                subject_kind="game",
                subject_name=game_title,
                stat="Total Points",
                display_stat=f"Total Points",
                line=total_line,
                side="under",
                american_price=-110,
                decimal_price=DEFAULT_DECIMAL_PRICE,
                implied_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                breakeven_prob=round(1.0 / DEFAULT_DECIMAL_PRICE, 4),
                model_prob=round(1.0 - p_over, 4),
                vegas_total=float(total),
                vegas_spread=float(spread_home) if isinstance(spread_home, (int, float)) else None,
                notes=[f"ESPN total: {total_line}"],
            )
        )

    return out


async def fetch_game_line_props(
    *,
    espn: EspnClient,
    sport: SportId,
    team_abbrs: list[str],
    cache: Any | None = None,
    cache_ttl_seconds: int = 600,
) -> list[Prop]:
    """For each team in `team_abbrs`, find the next upcoming event and pull
    its full game-line market from ESPN. Dedupes by event_id so each game
    only generates one set of props.

    `cache` is an optional `SqliteTTLCache` — when provided, the per-event
    market payload is cached so repeated calls are cheap.
    """
    if sport not in ("NBA", "NFL", "NHL"):
        return []

    sl = espn.sport_league_for_scope(sport)
    if sl is None:
        return []
    sport_slug, league_slug = sl

    # Resolve unique team -> team_id, then team_id -> next event id.
    upper_abbrs = sorted({a.strip().upper() for a in team_abbrs if isinstance(a, str) and a.strip()})
    if not upper_abbrs:
        return []

    sem = asyncio.Semaphore(8)

    async def event_id_for(abbr: str) -> str | None:
        async with sem:
            tid = await espn._team_id_by_abbr(sport=sport_slug, league=league_slug, abbr=abbr)
            if tid is None:
                return None
            return await espn.fetch_upcoming_event_id(
                sport=sport_slug, league=league_slug, team_id=tid
            )

    event_ids_raw = await asyncio.gather(*(event_id_for(a) for a in upper_abbrs))
    unique_event_ids = sorted({eid for eid in event_ids_raw if isinstance(eid, str) and eid})
    if not unique_event_ids:
        return []

    async def odds_for(event_id: str) -> tuple[str, dict[str, Any]]:
        cache_key = f"espn:game_line_market:{sport_slug}:{league_slug}:{event_id}"
        if cache is not None:
            cached = cache.get_json(cache_key)
            if isinstance(cached, dict):
                return (event_id, cached)
        async with sem:
            payload = await espn.fetch_event_market_lines(
                sport=sport_slug, league=league_slug, event_id=event_id
            )
        if cache is not None:
            try:
                cache.set_json(cache_key, payload, ttl_seconds=cache_ttl_seconds)
            except Exception:
                pass
        return (event_id, payload)

    odds_results = await asyncio.gather(*(odds_for(eid) for eid in unique_event_ids))

    out: list[Prop] = []
    for event_id, odds in odds_results:
        try:
            out.extend(_build_props_for_event(sport=sport, event_id=event_id, odds=odds))
        except Exception as exc:
            log.info("game-line synthesis failed for %s: %s", event_id, exc)
            continue
    return out
