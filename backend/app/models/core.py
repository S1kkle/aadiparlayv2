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

    # odds/prices
    american_price: int | None = None
    decimal_price: float | None = None

    # model outputs
    model_prob: float | None = None
    implied_prob: float | None = None
    edge: float | None = None
    ev: float | None = None
    volatility: float | None = None

    # ai outputs
    ai_bias: int | None = None  # -1,0,1
    ai_confidence: float | None = None
    ai_summary: str | None = None
    ai_tailwinds: list[str] = Field(default_factory=list)
    ai_risk_factors: list[str] = Field(default_factory=list)

    # final
    score: float | None = None
    notes: list[str] = Field(default_factory=list)


class RankedPropsResponse(BaseModel):
    scope: str
    updated_at: datetime
    props: list[Prop]


class HealthResponse(BaseModel):
    status: Literal["ok"]

