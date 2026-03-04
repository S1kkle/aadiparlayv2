from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from app.clients.espn import EspnClient, EspnConfig
from app.clients.groq import GroqClient, GroqConfig
from app.clients.ollama import OllamaClient, OllamaConfig
from app.clients.underdog import UnderdogClient, UnderdogConfig
from app.db import SqliteTTLCache
from app.models.core import HealthResponse, RankedPropsResponse, SportId
from app.props_jobs import PropsJobRequest, PropsJobStore, sse_format
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

# CORS: allow any origin (frontend doesn't send cookies). Fixes Vercel -> Render cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

# LLM: Groq (cloud, always-on) if API key set; else Ollama (local)
_groq_key = _env("GROQ_API_KEY")
if _groq_key and _groq_key.strip():
    llm_client: OllamaClient | GroqClient = GroqClient(
        GroqConfig(
            api_key=_groq_key,
            model=_env("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant",
        )
    )
else:
    llm_client = OllamaClient(
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
    ollama=llm_client,
)

jobs = PropsJobStore()

async def _warm_nba_league_matchup_cache() -> None:
    # Precompute league snapshots so the AI can mention "vs league average" deltas quickly.
    stat_keys = ["points", "assists", "totalRebounds", "blocks", "steals", "turnovers"]
    for k in stat_keys:
        try:
            await espn_client.compute_league_allowed_rank_snapshot(
                sport="basketball", league="nba", stat_key=k, last_n_games=3
            )
        except Exception:
            continue


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_warm_nba_league_matchup_cache())


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
    require_ai: bool = Query(False),
    require_ai_count: int = Query(10, ge=1, le=50),
) -> RankedPropsResponse:
    props = await ranker.rank_props(
        scope=scope,
        sport=sport,
        refresh=refresh,
        max_props=max_props,
        ai_limit=ai_limit,
        require_ai=require_ai,
        require_ai_count=require_ai_count,
    )
    return RankedPropsResponse(scope=f"{scope}:{sport}", updated_at=datetime.now(timezone.utc), props=props)


@app.post("/cache/clear")
async def clear_cache() -> dict[str, int | str]:
    deleted = cache.clear()
    asyncio.create_task(_warm_nba_league_matchup_cache())
    return {"status": "ok", "deleted": deleted}


@app.post("/props/job")
async def start_props_job(req: dict) -> dict[str, str]:
    """
    Starts a background job (used by All-sports mode) and streams progress via SSE.
    """
    # Manual parsing to keep deps minimal
    r = PropsJobRequest(
        sport=req.get("sport") or "UNKNOWN",
        scope=req.get("scope") or "all",
        refresh=bool(req.get("refresh") or False),
        max_props=int(req.get("max_props") or 10),
        ai_limit=int(req.get("ai_limit") or 10),
        require_ai_count=int(req.get("require_ai_count") or 10),
    )
    job = await jobs.create()

    async def _runner() -> None:
        try:
            await job.emit(
                {
                    "type": "progress",
                    "stage": "starting",
                    "ai_succeeded": 0,
                    "ai_attempted": 0,
                    "ai_target": r.require_ai_count,
                    "analyzed": 0,
                }
            )

            attempted = 0
            succeeded = 0

            async def on_ai_progress(ev: dict) -> None:
                nonlocal attempted, succeeded
                if ev.get("type") == "ai_prop_done":
                    attempted += 1
                    if ev.get("ok"):
                        succeeded += 1
                if ev.get("type") in ("ai_prop_done", "ai_batch"):
                    raw_analyzed = ev.get("analyzed")
                    if isinstance(raw_analyzed, (int, float)):
                        analyzed = int(raw_analyzed)
                    elif isinstance(raw_analyzed, str) and raw_analyzed.strip().isdigit():
                        analyzed = int(raw_analyzed.strip())
                    else:
                        analyzed = 0
                    await job.emit(
                        {
                            "type": "progress",
                            "stage": "ai",
                            "ai_succeeded": succeeded,
                            "ai_attempted": attempted,
                            "ai_target": r.require_ai_count,
                            "analyzed": analyzed,
                        }
                    )

            props = await ranker.rank_props(
                scope=r.scope,
                sport=r.sport,
                refresh=r.refresh,
                max_props=r.max_props,
                ai_limit=r.ai_limit,
                require_ai=True,
                require_ai_count=r.require_ai_count,
                on_ai_progress=on_ai_progress,
            )

            resp = RankedPropsResponse(
                scope=f"{r.scope}:{r.sport}", updated_at=datetime.now(timezone.utc), props=props
            )
            await jobs.set_done(job, resp)
        except Exception as e:
            await jobs.set_error(job, str(e))

    asyncio.create_task(_runner())
    return {"job_id": job.id}


@app.get("/props/job/{job_id}/events")
async def props_job_events(job_id: str) -> StreamingResponse:
    job = await jobs.get(job_id)
    if job is None:
        return StreamingResponse(
            content=iter([sse_format({"type": "error", "error": "job not found"})]),
            media_type="text/event-stream",
        )

    async def gen():
        # send initial status snapshot
        yield sse_format({"type": "status", "job_id": job.id, "status": job.status})
        while True:
            ev = await job.events.get()
            yield sse_format(ev)
            if ev.get("type") in ("done", "error"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/props/job/{job_id}/result")
async def props_job_result(job_id: str):
    job = await jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    if job.status == "running" or job.result is None:
        return JSONResponse(status_code=202, content={"status": job.status})
    if job.status == "error":
        return JSONResponse(status_code=500, content={"status": "error", "error": job.error or "unknown"})
    return job.result

