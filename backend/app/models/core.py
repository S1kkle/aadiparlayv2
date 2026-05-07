from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


SportId = Literal["NBA", "NFL", "NHL", "SOCCER", "MMA", "UNKNOWN"]
Side = Literal["over", "under"]


class GameStat(BaseModel):
    game_date: datetime | None = None
    opponent_abbr: str | None = None
    value: float


class UnderdogOption(BaseModel):
    id: str
    choice: str | None = None  # higher/lower/...
    american_price: str | None = None
    decimal_price: str | None = None
    payout_multiplier: str | None = None
    selection_subheader: str | None = None
    status: str | None = None


class UnderdogOverUnderLine(BaseModel):
    id: str
    over_under_id: str
    stable_id: str | None = None
    stat_value: str | None = None
    status: str | None = None
    updated_at: datetime | None = None
    options: list[UnderdogOption] = Field(default_factory=list)
    over_under: dict[str, Any] = Field(default_factory=dict)


class Prop(BaseModel):
    # identity
    sport: SportId
    league: str | None = None  # ESPN league slug: nba/nfl/nhl/...
    player_name: str
    player_position: str | None = None  # ESPN position abbreviation when available (e.g., G/F/C, QB, RW)
    espn_athlete_id: int | None = None
    underdog_player_id: str | None = None
    underdog_option_id: str
    game_title: str | None = None
    scheduled_at: datetime | None = None
    team_abbr: str | None = None
    opponent_abbr: str | None = None

    # market
    stat: str
    display_stat: str | None = None
    line: float
    side: Side
    stat_field: str | None = None  # ESPN gamelog field used (if any)
    recent_games: list[GameStat] = Field(default_factory=list)
    vs_opponent_games: list[GameStat] = Field(default_factory=list)

    # context
    is_home: bool | None = None
    is_b2b: bool | None = None
    rest_days: int | None = None
    avg_minutes: float | None = None
    projected_minutes: float | None = None  # avg_minutes after blowout/rest adjustment
    vegas_total: float | None = None  # game over/under total when known
    vegas_spread: float | None = None  # team-perspective point spread (negative = favorite)
    blowout_risk: bool | None = None
    injury_status: str | None = None  # ESPN canonical: OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, DAY-TO-DAY, etc.
    injury_haircut_applied: float | None = None  # multiplier applied to model_prob (0.0 = picked filtered out)

    # trend / profile
    trend_short_avg: float | None = None  # last 3 game average
    trend_direction: str | None = None  # "up" | "down" | "flat"
    hit_rate_last10: float | None = None  # fraction 0..1
    hit_rate_str: str | None = None  # e.g. "7/10"
    stat_median: float | None = None
    stat_floor: float | None = None  # 10th percentile
    stat_ceiling: float | None = None  # 90th percentile
    stat_consistency: float | None = None  # 0..1
    current_streak: int | None = None  # +N = N overs in a row, -N = unders
    line_percentile: float | None = None  # where line sits in distribution

    # odds/prices
    american_price: int | None = None
    decimal_price: float | None = None
    payout_multiplier: float | None = None  # actual per-leg multiplier from Underdog (truth, when present)
    selection_subheader: str | None = None  # e.g. "Boost", "Rescue" — flags non-standard payouts
    is_boosted: bool | None = None
    breakeven_prob: float | None = None  # 1/decimal_price — break-even win rate per leg

    # model outputs
    model_prob: float | None = None
    implied_prob: float | None = None
    no_vig_prob: float | None = None  # true implied prob after vig removal
    edge: float | None = None
    ev: float | None = None
    volatility: float | None = None
    kelly_fraction: float | None = None  # fractional Kelly bet size (0..0.10)
    edge_confidence: float | None = None  # variance-weighted edge confidence (0..1)
    per_minute_rate: float | None = None  # stat production per minute played

    # ai outputs
    ai_bias: int | None = None  # -1,0,1
    ai_confidence: float | None = None
    ai_summary: str | None = None
    ai_tailwinds: list[str] = Field(default_factory=list)
    ai_risk_factors: list[str] = Field(default_factory=list)
    ai_prob_adjustment: float | None = None  # e.g. +0.05 or -0.03

    # provenance — which prompt revision and calibration snapshot was used.
    # Persisted to learning_log so we can attribute pick performance to
    # specific prompt / model revisions.
    prompt_version: str | None = None
    model_params_id: str | None = None

    # derived
    confidence_tier: str | None = None  # "high" | "medium" | "low"
    model_ai_agree: bool | None = None

    # final
    score: float | None = None
    notes: list[str] = Field(default_factory=list)


class RankedPropsResponse(BaseModel):
    scope: str
    updated_at: datetime
    props: list[Prop]


class HealthResponse(BaseModel):
    status: Literal["ok"]

