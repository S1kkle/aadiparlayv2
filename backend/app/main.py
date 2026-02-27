from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.clients.espn import EspnClient, EspnConfig
from app.clients.ollama import OllamaClient, OllamaConfig
from app.clients.underdog import UnderdogClient, UnderdogConfig
from app.db import SqliteTTLCache
from app.models.core import HealthResponse, RankedPropsResponse, SportId
from app.services.ranker import Ranker, RankerConfig


load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int) -> int:
    val = _env(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = _env(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


app = FastAPI(title="Underdog Prop Ranker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cache = SqliteTTLCache(db_path=os.path.join(os.path.dirname(__file__), "..", ".cache", "cache.sqlite3"))

espn_client = EspnClient(
    EspnConfig(
        site_api_base=_env("ESPN_SITE_API_BASE", "https://site.api.espn.com") or "https://site.api.espn.com",
        site_web_api_base=_env("ESPN_SITE_WEB_API_BASE", "https://site.web.api.espn.com") or "https://site.web.api.espn.com",
        core_api_base=_env("ESPN_CORE_API_BASE", "https://sports.core.api.espn.com") or "https://sports.core.api.espn.com",
    ),
    cache=cache,
)

ollama_client = OllamaClient(
    OllamaConfig(
        base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
        model=_env("OLLAMA_MODEL", "llama3.1") or "llama3.1",
    )
)

ud_client = UnderdogClient(
    UnderdogConfig(
        over_under_lines_url=_env("UD_OVER_UNDER_LINES_URL", "https://api.underdogfantasy.com/beta/v5/over_under_lines")
        or "https://api.underdogfantasy.com/beta/v5/over_under_lines",
        auth_token=_env("UD_AUTH_TOKEN"),
        user_location_token=_env("UD_USER_LOCATION_TOKEN"),
        product=_env("UD_PRODUCT"),
        product_experience_id=_env("UD_PRODUCT_EXPERIENCE_ID"),
        state_config_id=_env("UD_STATE_CONFIG_ID"),
    )
)

ranker = Ranker(
    RankerConfig(
        last_n=_env_int("RANK_LAST_N", 10),
        w_edge=_env_float("RANK_W_EDGE", 0.55),
        w_ev=_env_float("RANK_W_EV", 0.25),
        w_vol=_env_float("RANK_W_VOL", 0.20),
        w_ai=_env_float("RANK_W_AI", 0.35),
    ),
    ud_client=ud_client,
    cache=cache,
    espn=espn_client,
    ollama=ollama_client,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/props", response_model=RankedPropsResponse)
async def get_ranked_props(
    sport: SportId = Query("UNKNOWN"),
    scope: Literal["all", "featured"] = Query("all"),
    refresh: bool = Query(False),
    max_props: int = Query(10, ge=1, le=500),
    ai_limit: int = Query(10, ge=0, le=200),
) -> RankedPropsResponse:
    props = await ranker.rank_props(
        scope=scope, sport=sport, refresh=refresh, max_props=max_props, ai_limit=ai_limit
    )
    return RankedPropsResponse(scope=f"{scope}:{sport}", updated_at=datetime.now(timezone.utc), props=props)

