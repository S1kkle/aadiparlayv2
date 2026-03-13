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
from app.models.core import HealthResponse, Prop, RankedPropsResponse, SportId
from app.props_jobs import PropsJobRequest, PropsJobStore, sse_format
from app.services.learning import LearningService
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

learning_svc = LearningService(cache=cache, espn=espn_client, llm=llm_client)

async def _warm_league_matchup_caches() -> None:
    nba_keys = ["points", "assists", "totalRebounds", "blocks", "steals", "turnovers"]
    for k in nba_keys:
        try:
            await espn_client.compute_league_allowed_rank_snapshot(
                sport="basketball", league="nba", stat_key=k, last_n_games=3
            )
        except Exception:
            continue
    nfl_keys = ["passingYards", "rushingYards", "receivingYards", "receptions", "passingTouchdowns"]
    for k in nfl_keys:
        try:
            await espn_client.compute_league_allowed_rank_snapshot(
                sport="football", league="nfl", stat_key=k, last_n_games=3
            )
        except Exception:
            continue
    nhl_keys = ["goals", "assists", "points", "shots"]
    for k in nhl_keys:
        try:
            await espn_client.compute_league_allowed_rank_snapshot(
                sport="hockey", league="nhl", stat_key=k, last_n_games=3
            )
        except Exception:
            continue


@app.on_event("startup")
async def _startup() -> None:
    # Purge stale gamelog caches from old key formats (v1, v2) on startup
    cache.clear_prefix("espn:gamelog:v2:")
    cache.clear_prefix("espn:gamelog:basketball:")
    cache.clear_prefix("espn:gamelog:football:")
    cache.clear_prefix("espn:gamelog:hockey:")
    cache.clear_prefix("ai_select:")
    asyncio.create_task(_warm_league_matchup_caches())


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
    asyncio.create_task(_warm_league_matchup_caches())
    return {"status": "ok", "deleted": deleted}


@app.post("/cache/clear-gamelogs")
async def clear_gamelogs() -> dict[str, int | str]:
    """Clear only gamelog + AI selection caches (keeps matchup snapshots)."""
    d1 = cache.clear_prefix("espn:gamelog:")
    d2 = cache.clear_prefix("ai_select:")
    d3 = cache.clear_prefix("ollama:prop:")
    return {"status": "ok", "deleted": d1 + d2 + d3}


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

            async def on_model_done(all_props: list[Prop]) -> None:
                """Emit all stat-model results so frontend can display them before AI."""
                serialized = [p.model_dump(mode="json") for p in all_props]
                await job.emit({
                    "type": "model_done",
                    "props": serialized,
                })

            async def on_ai_progress(ev: dict) -> None:
                nonlocal attempted, succeeded
                if ev.get("type") == "stage":
                    await job.emit({
                        "type": "progress",
                        "stage": ev.get("stage", ""),
                        "detail": ev.get("detail", ""),
                        "ai_succeeded": succeeded,
                        "ai_attempted": attempted,
                        "ai_target": r.require_ai_count,
                        "analyzed": 0,
                    })
                    return
                if ev.get("type") == "ai_prop_done":
                    attempted += 1
                    if ev.get("ok"):
                        succeeded += 1
                    prop_obj = ev.get("prop")
                    if isinstance(prop_obj, Prop) and isinstance(prop_obj.ai_summary, str) and prop_obj.ai_summary.strip():
                        await job.emit({
                            "type": "ai_update",
                            "option_id": prop_obj.underdog_option_id,
                            "ai": {
                                "ai_summary": prop_obj.ai_summary,
                                "ai_bias": prop_obj.ai_bias,
                                "ai_confidence": prop_obj.ai_confidence,
                                "ai_tailwinds": prop_obj.ai_tailwinds,
                                "ai_risk_factors": prop_obj.ai_risk_factors,
                                "ai_prob_adjustment": prop_obj.ai_prob_adjustment,
                            },
                        })
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
                on_model_done=on_model_done,
            )

            resp = RankedPropsResponse(
                scope=f"{r.scope}:{r.sport}", updated_at=datetime.now(timezone.utc), props=props
            )

            # Auto-save top picks to history so Learning Mode can resolve them later
            try:
                top_for_history = [p for p in props if isinstance(p.ai_summary, str) and p.ai_summary.strip()][:15]
                if not top_for_history:
                    top_for_history = props[:15]
                if top_for_history:
                    entry_id = str(_uuid.uuid4())
                    timestamp = datetime.now(timezone.utc).isoformat()
                    cache.save_history(
                        entry_id,
                        timestamp,
                        r.sport,
                        _json.dumps([p.model_dump(mode="json") for p in top_for_history], default=str),
                    )
            except Exception:
                pass

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


import json as _json
import uuid as _uuid


@app.get("/history")
async def get_history():
    entries = cache.get_history(limit=30)
    return {"count": len(entries), "entries": entries}


@app.post("/history")
async def save_history(req: dict):
    entry_id = req.get("id") or str(_uuid.uuid4())
    timestamp = req.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sport = req.get("sport", "UNKNOWN")
    props_list = req.get("props", [])
    cache.save_history(entry_id, timestamp, sport, _json.dumps(props_list, default=str))
    return {"status": "ok", "id": entry_id}


@app.post("/history/seed")
async def seed_history(req: dict):
    """Bulk-seed history entries from localStorage. Skips entries that already exist."""
    entries = req.get("entries", [])
    if not isinstance(entries, list):
        return {"status": "ok", "seeded": 0}
    existing = cache.get_history(limit=500)
    existing_ids = {e["id"] for e in existing}
    seeded = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        eid = entry.get("id", "")
        if eid in existing_ids:
            continue
        sport = entry.get("sport", "UNKNOWN")
        props_list = entry.get("props", [])
        timestamp = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
        cache.save_history(eid, timestamp, sport, _json.dumps(props_list, default=str))
        existing_ids.add(eid)
        seeded += 1
    return {"status": "ok", "seeded": seeded}


@app.post("/parlay/recommend")
async def recommend_parlay(req: dict):
    """AI-recommended parlay from current props."""
    sport: SportId = req.get("sport") or "UNKNOWN"
    legs = int(req.get("legs") or 2)
    if legs not in (2, 3, 4, 5, 6):
        return JSONResponse(status_code=400, content={"error": "legs must be 2-6"})

    props_data = req.get("props")
    if not isinstance(props_data, list) or not props_data:
        return JSONResponse(status_code=400, content={"error": "props list required"})

    props_objs: list[Prop] = []
    for raw in props_data:
        try:
            props_objs.append(Prop(**raw))
        except Exception:
            continue

    if len(props_objs) < legs:
        return JSONResponse(status_code=400, content={"error": f"Need at least {legs} props"})

    try:
        result = await ranker.recommend_parlay(props=props_objs, legs=legs)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Learning Mode endpoints ───────────────────────────────────────────

@app.post("/learning/resolve")
async def learning_resolve():
    """Check actual game results for past picks and mark them as hits/misses."""
    result = await learning_svc.resolve_outcomes()
    return result


@app.post("/learning/analyze-misses")
async def learning_analyze_misses():
    """Run AI analysis on missed picks to diagnose why they missed."""
    result = await learning_svc.analyze_misses()
    return result


@app.post("/learning/weekly-report")
async def learning_weekly_report():
    """Generate a weekly improvement report from miss patterns."""
    result = await learning_svc.generate_weekly_report()
    return result


@app.get("/learning/entries")
async def learning_entries(resolved: bool = Query(False), limit: int = Query(100)):
    """Get learning log entries."""
    entries = cache.get_learning_entries(resolved_only=resolved, limit=limit)
    return {"entries": entries}


@app.get("/learning/reports")
async def learning_reports(limit: int = Query(5)):
    """Get weekly learning reports."""
    reports = cache.get_learning_reports(limit=limit)
    return {"reports": reports}


@app.post("/learning/run-full")
async def learning_run_full():
    """Run the entire learning pipeline: resolve -> analyze misses -> weekly report."""
    resolve_result = await learning_svc.resolve_outcomes()
    analyze_result = await learning_svc.analyze_misses()
    try:
        report = await learning_svc.generate_weekly_report()
    except Exception as e:
        report = {"error": str(e)}
    return {
        "resolve": resolve_result,
        "analyze": analyze_result,
        "report": report,
    }


@app.get("/debug/player/{sport}/{player_name}")
async def debug_player(sport: str, player_name: str):
    """Trace player data pipeline for debugging stale/missing data."""
    trace: dict = {"player": player_name, "sport": sport, "steps": []}
    sl = espn_client.sport_league_for_scope(sport.upper())
    if sl is None:
        trace["error"] = f"Unknown sport: {sport}"
        return trace
    sport_slug, league_slug = sl

    aid = await espn_client.find_best_athlete_id(
        sport=sport_slug, league=league_slug, full_name=player_name
    )
    trace["steps"].append({"step": "find_athlete", "athlete_id": aid})
    if aid is None:
        trace["error"] = "Athlete not found via search"
        return trace

    try:
        gamelog = await espn_client.fetch_gamelog(
            sport=sport_slug, league=league_slug, athlete_id=aid
        )
        names = gamelog.get("names", [])
        season_types = gamelog.get("seasonTypes", [])
        event_count = 0
        first_events: list[dict] = []
        if season_types:
            st0 = season_types[0] if isinstance(season_types[0], dict) else {}
            for cat in (st0.get("categories") or []):
                evts = cat.get("events") or []
                event_count += len(evts)
                for ev in evts[:3]:
                    if isinstance(ev, dict):
                        first_events.append({
                            "stats_len": len(ev.get("stats", [])),
                            "stats_preview": (ev.get("stats") or [])[:6],
                            "eventId": ev.get("eventId"),
                        })

        events_meta = gamelog.get("events", {})
        event_dates: list[str] = []
        if isinstance(events_meta, dict):
            for eid, emeta in list(events_meta.items())[:5]:
                if isinstance(emeta, dict):
                    gd = emeta.get("gameDate")
                    opp = emeta.get("opponent", {}).get("abbreviation") if isinstance(emeta.get("opponent"), dict) else None
                    event_dates.append(f"{eid}: {gd} vs {opp}")

        trace["steps"].append({
            "step": "gamelog",
            "names_count": len(names),
            "names_sample": names[:10],
            "season_types_count": len(season_types),
            "total_events": event_count,
            "first_events": first_events,
            "recent_event_dates": event_dates,
        })
    except Exception as e:
        trace["steps"].append({"step": "gamelog", "error": str(e)})

    stat_fields = ["points", "assists", "totalRebounds", "blocks", "steals"]
    for field in stat_fields:
        try:
            series = espn_client.extract_stat_series(gamelog, field_name=field, last_n=5)
            if series:
                trace["steps"].append({"step": f"series_{field}", "values": series})
        except Exception:
            pass

    return trace


@app.get("/debug/mma/{fighter_name}")
async def debug_mma(fighter_name: str):
    """Trace MMA pipeline step-by-step for debugging."""
    trace: dict = {"fighter": fighter_name, "steps": []}

    # Step 1: find athlete ID
    aid = await espn_client.find_mma_athlete_id(full_name=fighter_name)
    trace["steps"].append({"step": "find_mma_athlete_id", "result": aid})
    if aid is None:
        trace["error"] = "Athlete not found"
        return trace

    # Step 2: fetch eventlog raw
    eventlog_url = f"{espn_client._cfg.core_api_base}/v2/sports/mma/athletes/{aid}/eventlog"
    try:
        eventlog = await espn_client._get_json(eventlog_url)
        items = (eventlog.get("events") or {}).get("items") or []
        trace["steps"].append({
            "step": "eventlog",
            "total_items": len(items),
            "played_items": sum(1 for i in items if isinstance(i, dict) and i.get("played")),
            "first_3_items": [
                {
                    "played": i.get("played"),
                    "has_competitor_ref": bool((i.get("competitor") or {}).get("$ref")),
                    "has_competition_ref": bool((i.get("competition") or {}).get("$ref")),
                    "competitor_ref_preview": str((i.get("competitor") or {}).get("$ref", ""))[:120],
                }
                for i in items[:3]
                if isinstance(i, dict)
            ],
        })
    except Exception as e:
        trace["steps"].append({"step": "eventlog", "error": str(e)})
        return trace

    # Step 3: try first played fight's stats
    played = [i for i in items if isinstance(i, dict) and i.get("played")]
    if not played:
        trace["steps"].append({"step": "stats", "error": "no played items"})
        return trace

    first = played[0]
    comp_ref = ((first.get("competitor") or {}).get("$ref") or "").replace("http://", "https://")
    stats_url = comp_ref + "/statistics"
    trace["steps"].append({"step": "stats_url", "url": stats_url[:200]})

    try:
        stats_data = await espn_client._get_json(stats_url)
        splits = stats_data.get("splits") or {}
        cats = splits.get("categories") or []
        all_stats: dict = {}
        for cat in cats:
            for s in (cat.get("stats") or []):
                nm = s.get("name")
                val = s.get("value")
                if nm:
                    all_stats[nm] = val
        trace["steps"].append({
            "step": "stats_parse",
            "raw_keys": list(stats_data.keys())[:10],
            "splits_keys": list(splits.keys())[:10],
            "num_categories": len(cats),
            "parsed_stats": all_stats,
        })
    except Exception as e:
        trace["steps"].append({"step": "stats_fetch", "error": str(e), "url": stats_url[:200]})

    # Step 4: full fight history
    try:
        fights = await espn_client.fetch_mma_fight_history(athlete_id=aid, last_n=10)
        trace["steps"].append({
            "step": "fight_history",
            "count": len(fights),
            "fights": [
                {"date": f.get("date"), "opp": f.get("opponent_name"), "stats_keys": list(f.get("stats", {}).keys())}
                for f in fights[:3]
            ],
        })
    except Exception as e:
        trace["steps"].append({"step": "fight_history", "error": str(e)})

    # Step 5: career stats
    try:
        career = await espn_client.fetch_mma_career_stats(athlete_id=aid)
        trace["steps"].append({"step": "career_stats", "stats": career})
    except Exception as e:
        trace["steps"].append({"step": "career_stats", "error": str(e)})

    return trace

