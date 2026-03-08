from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote_plus, urlsplit, urlunsplit

import httpx

from app.db import SqliteTTLCache


EspnLeague = Literal["nba", "nfl", "nhl", "ufc"]
EspnSport = Literal["basketball", "football", "hockey", "mma"]


def _canon_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


def _append_path(ref_url: str, suffix: str) -> str:
    """Append a path segment to an ESPN $ref URL, inserting before query params."""
    parts = urlsplit(ref_url)
    new_path = parts.path.rstrip("/") + suffix
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


@dataclass(frozen=True)
class EspnConfig:
    site_api_base: str  # e.g. https://site.api.espn.com
    site_web_api_base: str  # e.g. https://site.web.api.espn.com
    core_api_base: str  # e.g. https://sports.core.api.espn.com


class EspnClient:
    def __init__(self, cfg: EspnConfig, *, cache: SqliteTTLCache) -> None:
        self._cfg = cfg
        self._cache = cache

    async def _get_json(self, url: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()

    async def get_team_abbr_to_id(self, *, sport: EspnSport, league: EspnLeague) -> dict[str, int]:
        """
        Returns mapping of team abbreviation -> ESPN team id.
        """
        cache_key = f"espn:teams:{sport}:{league}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("abbr_to_id"), dict):
            out: dict[str, int] = {}
            for k, v in cached["abbr_to_id"].items():
                if isinstance(k, str) and isinstance(v, int):
                    out[k.upper()] = v
            if out:
                return out

        # Populate cache by calling the same logic used by _team_id_by_abbr.
        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/teams"
        data = await self._get_json(url)
        abbr_to_id: dict[str, int] = {}
        sports = data.get("sports") or []
        if isinstance(sports, list) and sports:
            leagues = (sports[0].get("leagues") or []) if isinstance(sports[0], dict) else []
            teams = (leagues[0].get("teams") or []) if isinstance(leagues, list) and leagues else []
            for t in teams:
                team = t.get("team") if isinstance(t, dict) else None
                if not isinstance(team, dict):
                    continue
                tabbr = team.get("abbreviation")
                tid = team.get("id")
                if isinstance(tabbr, str) and isinstance(tid, (int, str)):
                    try:
                        abbr_to_id[tabbr.upper()] = int(tid)
                    except ValueError:
                        continue

        self._cache.set_json(cache_key, {"abbr_to_id": abbr_to_id}, ttl_seconds=24 * 60 * 60)
        return abbr_to_id

    @staticmethod
    def sport_league_for_scope(scope: str) -> tuple[EspnSport, EspnLeague] | None:
        if scope == "NBA":
            return ("basketball", "nba")
        if scope == "NFL":
            return ("football", "nfl")
        if scope == "NHL":
            return ("hockey", "nhl")
        if scope == "MMA":
            return ("mma", "ufc")
        return None

    async def find_best_athlete_id(self, *, sport: EspnSport, league: EspnLeague, full_name: str) -> int | None:
        """
        Searches ESPN core API for athletes and picks the best name match.
        """
        canon_target = _canon_name(full_name)
        if not canon_target:
            return None

        cache_key = f"espn:athlete_search:{sport}:{league}:{canon_target}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and "athlete_id" in cached:
            return cached["athlete_id"]

        search_url = f"{self._cfg.core_api_base}/v2/sports/{sport}/leagues/{league}/athletes?search={quote_plus(full_name)}"
        data = await self._get_json(search_url)
        items = data.get("items") or []
        if not isinstance(items, list) or not items:
            self._cache.set_json(cache_key, {"athlete_id": None}, ttl_seconds=86400)
            return None

        # The search returns $ref URLs; fetch a few to get names.
        best: tuple[int, int] | None = None  # (score, athlete_id)
        for ref_obj in items[:10]:
            if not isinstance(ref_obj, dict):
                continue
            ref = ref_obj.get("$ref")
            if not isinstance(ref, str):
                continue
            # refs sometimes come back as http; normalize to https
            ref = ref.replace("http://", "https://")
            try:
                athlete = await self._get_json(ref)
            except Exception:
                continue
            aid = athlete.get("id")
            name = athlete.get("fullName") or athlete.get("displayName") or ""
            if not isinstance(aid, (int, str)) or not isinstance(name, str):
                continue
            try:
                aid_int = int(aid)
            except ValueError:
                continue

            canon = _canon_name(name)
            if canon == canon_target:
                best = (10_000, aid_int)
                break
            # simple overlap score
            score = 0
            tset = set(canon_target.split())
            aset = set(canon.split())
            score = int(100 * (len(tset & aset) / max(1, len(tset))))
            if best is None or score > best[0]:
                best = (score, aid_int)

        athlete_id = best[1] if best else None
        self._cache.set_json(cache_key, {"athlete_id": athlete_id}, ttl_seconds=86400)
        return athlete_id

    async def resolve_athlete_id_from_team_roster(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        team_abbr: str,
        full_name: str,
    ) -> int | None:
        """
        More reliable than global athlete search: map team abbreviation -> ESPN team id -> roster -> athlete id.
        """
        canon_target = _canon_name(full_name)
        if not canon_target:
            return None

        team_abbr = team_abbr.strip().upper()
        cache_key = f"espn:roster_lookup:v2:{sport}:{league}:{team_abbr}:{canon_target}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and "athlete_id" in cached:
            return cached["athlete_id"]

        team_id = await self._team_id_by_abbr(sport=sport, league=league, abbr=team_abbr)
        if team_id is None:
            self._cache.set_json(cache_key, {"athlete_id": None}, ttl_seconds=24 * 60 * 60)
            return None

        roster = await self._team_roster(sport=sport, league=league, team_id=team_id)
        best: tuple[int, int] | None = None
        for athlete in roster:
            if not isinstance(athlete, dict):
                continue
            aid = athlete.get("id")
            name = athlete.get("fullName") or athlete.get("displayName") or ""
            if not isinstance(aid, (int, str)) or not isinstance(name, str):
                continue
            try:
                aid_int = int(aid)
            except ValueError:
                continue
            canon = _canon_name(name)
            if canon == canon_target:
                best = (10_000, aid_int)
                break
            tset = set(canon_target.split())
            aset = set(canon.split())
            score = int(100 * (len(tset & aset) / max(1, len(tset))))
            if best is None or score > best[0]:
                best = (score, aid_int)

        athlete_id = best[1] if best and best[0] >= 60 else None
        self._cache.set_json(cache_key, {"athlete_id": athlete_id}, ttl_seconds=24 * 60 * 60)
        return athlete_id

    async def resolve_athlete_profile_from_team_roster(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        team_abbr: str,
        full_name: str,
    ) -> tuple[int | None, str | None]:
        """
        Like resolve_athlete_id_from_team_roster, but also returns the athlete position abbreviation when present.
        """
        canon_target = _canon_name(full_name)
        if not canon_target:
            return (None, None)

        team_abbr = team_abbr.strip().upper()
        cache_key = f"espn:roster_profile:v2:{sport}:{league}:{team_abbr}:{canon_target}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and ("athlete_id" in cached or "position" in cached):
            aid = cached.get("athlete_id")
            pos = cached.get("position")
            return (int(aid) if isinstance(aid, int) else None, pos if isinstance(pos, str) else None)

        team_id = await self._team_id_by_abbr(sport=sport, league=league, abbr=team_abbr)
        if team_id is None:
            self._cache.set_json(cache_key, {"athlete_id": None, "position": None}, ttl_seconds=24 * 60 * 60)
            return (None, None)

        roster = await self._team_roster(sport=sport, league=league, team_id=team_id)
        best: tuple[int, int, str | None] | None = None  # (score, athlete_id, pos_abbr)
        for athlete in roster:
            if not isinstance(athlete, dict):
                continue
            aid = athlete.get("id")
            name = athlete.get("fullName") or athlete.get("displayName") or ""
            if not isinstance(aid, (int, str)) or not isinstance(name, str):
                continue
            try:
                aid_int = int(aid)
            except ValueError:
                continue

            pos_abbr = None
            pos = athlete.get("position")
            if isinstance(pos, dict):
                pa = pos.get("abbreviation")
                if isinstance(pa, str) and pa.strip():
                    pos_abbr = pa.strip().upper()

            canon = _canon_name(name)
            if canon == canon_target:
                best = (10_000, aid_int, pos_abbr)
                break
            tset = set(canon_target.split())
            aset = set(canon.split())
            score = int(100 * (len(tset & aset) / max(1, len(tset))))
            if best is None or score > best[0]:
                best = (score, aid_int, pos_abbr)

        athlete_id = best[1] if best and best[0] >= 60 else None
        position = best[2] if athlete_id is not None else None
        self._cache.set_json(cache_key, {"athlete_id": athlete_id, "position": position}, ttl_seconds=24 * 60 * 60)
        return (athlete_id, position)

    # ── MMA-specific methods ──────────────────────────────────────────────

    async def find_mma_athlete_id(self, *, full_name: str) -> int | None:
        """
        Searches for an MMA fighter using ESPN's common search API (the
        core API /athletes?search= param does not filter for MMA).
        """
        canon_target = _canon_name(full_name)
        if not canon_target:
            return None

        cache_key = f"espn:mma_athlete_search:{canon_target}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and "athlete_id" in cached:
            return cached["athlete_id"]

        search_url = (
            f"{self._cfg.site_api_base}/apis/common/v3/search"
            f"?query={quote_plus(full_name)}&limit=10&type=player"
        )
        try:
            data = await self._get_json(search_url)
        except Exception:
            self._cache.set_json(cache_key, {"athlete_id": None}, ttl_seconds=3600)
            return None

        items = data.get("items") or []
        best_id: int | None = None
        best_score = -1
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("sport") != "mma":
                continue
            name = item.get("displayName") or ""
            canon = _canon_name(name)
            aid_raw = item.get("id")
            try:
                aid = int(aid_raw)
            except (TypeError, ValueError):
                continue

            if canon == canon_target:
                best_id = aid
                break
            tset = set(canon_target.split())
            aset = set(canon.split())
            score = int(100 * (len(tset & aset) / max(1, len(tset))))
            if score > best_score:
                best_score = score
                best_id = aid

        self._cache.set_json(cache_key, {"athlete_id": best_id}, ttl_seconds=86400)
        return best_id

    async def fetch_mma_fight_history(
        self,
        *,
        athlete_id: int,
        last_n: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Builds a fight-by-fight stat history from ESPN eventlog + per-fight
        competitor stats. Returns list (most-recent first) of:
          {date: str|None, opponent_name: str|None, winner: bool|None,
           stats: {stat_name: float}}
        """
        cache_key = f"espn:mma_fight_history:{athlete_id}:last{last_n}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, list):
            return cached

        eventlog_url = f"{self._cfg.core_api_base}/v2/sports/mma/athletes/{athlete_id}/eventlog"
        try:
            eventlog = await self._get_json(eventlog_url)
        except Exception:
            return []

        items = (eventlog.get("events") or {}).get("items") or []
        fight_refs: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("played"):
                continue
            comp_ref = (item.get("competitor") or {}).get("$ref") or ""
            competition_ref = (item.get("competition") or {}).get("$ref") or ""
            if comp_ref:
                fight_refs.append({
                    "comp_ref": comp_ref.replace("http://", "https://"),
                    "competition_ref": competition_ref.replace("http://", "https://"),
                })
            if len(fight_refs) >= last_n:
                break

        # Process fights sequentially to avoid overwhelming ESPN with parallel requests.
        results: list[dict[str, Any]] = []
        for info in fight_refs:
            stats_dict: dict[str, float] = {}
            date_str: str | None = None
            opponent_name: str | None = None
            winner: bool | None = None

            # Per-fight stats (required — skip fight if this fails)
            try:
                stats_url = _append_path(info["comp_ref"], "/statistics")
                stats_data = await self._get_json(stats_url)
                for cat in (stats_data.get("splits", {}).get("categories") or []):
                    for s in (cat.get("stats") or []):
                        nm = s.get("name")
                        val = s.get("value")
                        if nm and isinstance(val, (int, float)):
                            stats_dict[nm] = float(val)
            except Exception:
                continue

            if not stats_dict:
                continue

            # Winner status (optional)
            try:
                comp_data = await self._get_json(info["comp_ref"])
                winner = comp_data.get("winner")
            except Exception:
                pass

            # Competition detail for date + opponent (optional)
            if info.get("competition_ref"):
                try:
                    detail = await self._get_json(info["competition_ref"])
                    dr = detail.get("date")
                    if isinstance(dr, str):
                        date_str = dr
                    for c in (detail.get("competitors") or []):
                        if not isinstance(c, dict):
                            continue
                        cid = c.get("id")
                        try:
                            cid_int = int(cid) if isinstance(cid, (int, str)) else None
                        except ValueError:
                            cid_int = None
                        if cid_int is not None and cid_int != athlete_id:
                            ath_ref = (c.get("athlete") or {}).get("$ref")
                            if isinstance(ath_ref, str):
                                try:
                                    opp = await self._get_json(ath_ref.replace("http://", "https://"))
                                    opponent_name = opp.get("fullName") or opp.get("displayName")
                                except Exception:
                                    pass
                            break
                except Exception:
                    pass

            results.append({
                "date": date_str,
                "opponent_name": opponent_name,
                "winner": winner,
                "stats": stats_dict,
            })

        results.sort(key=lambda x: x.get("date") or "", reverse=True)
        if results:
            self._cache.set_json(cache_key, results, ttl_seconds=12 * 60 * 60)
        return results

    async def fetch_mma_career_stats(self, *, athlete_id: int) -> dict[str, float]:
        """
        Returns career aggregate stats dict  e.g.
        {strikeLPM: 2.45, strikeAccuracy: 71.55, takedownAccuracy: 56.16, ...}
        """
        cache_key = f"espn:mma_career_stats:{athlete_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached.get("stats"):
            return cached["stats"]

        url = f"{self._cfg.core_api_base}/v2/sports/mma/athletes/{athlete_id}/statistics"
        try:
            data = await self._get_json(url)
        except Exception:
            return {}

        out: dict[str, float] = {}
        for cat in (data.get("splits", {}).get("categories") or []):
            for s in (cat.get("stats") or []):
                nm = s.get("name")
                val = s.get("value")
                if nm and isinstance(val, (int, float)):
                    out[nm] = float(val)

        self._cache.set_json(cache_key, {"stats": out}, ttl_seconds=24 * 60 * 60)
        return out

    # ── Team-sport helpers ───────────────────────────────────────────────

    async def _team_id_by_abbr(self, *, sport: EspnSport, league: EspnLeague, abbr: str) -> int | None:
        cache_key = f"espn:teams:{sport}:{league}"
        cached = self._cache.get_json(cache_key)
        if not isinstance(cached, dict) or "abbr_to_id" not in cached:
            url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/teams"
            data = await self._get_json(url)
            abbr_to_id: dict[str, int] = {}
            sports = data.get("sports") or []
            if isinstance(sports, list) and sports:
                leagues = (sports[0].get("leagues") or []) if isinstance(sports[0], dict) else []
                teams = (leagues[0].get("teams") or []) if isinstance(leagues, list) and leagues else []
                for t in teams:
                    team = t.get("team") if isinstance(t, dict) else None
                    if not isinstance(team, dict):
                        continue
                    tabbr = team.get("abbreviation")
                    tid = team.get("id")
                    if isinstance(tabbr, str) and isinstance(tid, (int, str)):
                        try:
                            abbr_to_id[tabbr.upper()] = int(tid)
                        except ValueError:
                            continue
            self._cache.set_json(cache_key, {"abbr_to_id": abbr_to_id}, ttl_seconds=24 * 60 * 60)
            cached = {"abbr_to_id": abbr_to_id}
        abbr_to_id = cached.get("abbr_to_id") or {}
        if not isinstance(abbr_to_id, dict):
            return None
        tid = abbr_to_id.get(abbr.upper())
        return int(tid) if isinstance(tid, int) else None

    async def fetch_team_schedule(self, *, sport: EspnSport, league: EspnLeague, team_id: int) -> dict[str, Any]:
        cache_key = f"espn:schedule:{sport}:{league}:{team_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached:
            return cached
        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule"
        data = await self._get_json(url)
        self._cache.set_json(cache_key, data, ttl_seconds=60 * 60)
        return data

    async def fetch_event_summary(self, *, sport: EspnSport, league: EspnLeague, event_id: str) -> dict[str, Any]:
        cache_key = f"espn:summary:{sport}:{league}:{event_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached:
            return cached
        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/summary?event={event_id}"
        data = await self._get_json(url)
        self._cache.set_json(cache_key, data, ttl_seconds=12 * 60 * 60)
        return data

    async def get_recent_completed_event_ids(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        team_id: int,
        limit: int = 5,
    ) -> list[str]:
        sched = await self.fetch_team_schedule(sport=sport, league=league, team_id=team_id)
        events = sched.get("events") or []
        if not isinstance(events, list):
            return []
        out: list[str] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            comps = e.get("competitions") or []
            comp0 = comps[0] if isinstance(comps, list) and comps else None
            if not isinstance(comp0, dict):
                continue
            st = ((comp0.get("status") or {}).get("type") or {})
            if not isinstance(st, dict) or not st.get("completed"):
                continue
            eid = e.get("id")
            if isinstance(eid, str) and eid.strip():
                out.append(eid.strip())
            if len(out) >= max(1, int(limit)):
                return out
        return out

    @staticmethod
    def _extract_team_total_stat_from_summary(
        summary: dict[str, Any], *, team_id: int, stat_name: str
    ) -> float | None:
        """
        Reads a TEAM total stat from summary.boxscore.teams[*].statistics by `stat_name`.
        """
        box = summary.get("boxscore") or {}
        teams = box.get("teams") or []
        if not isinstance(teams, list):
            return None
        for t in teams:
            if not isinstance(t, dict):
                continue
            team = t.get("team") or {}
            if not isinstance(team, dict):
                continue
            tid = team.get("id")
            try:
                tid_int = int(tid) if isinstance(tid, (int, str)) else None
            except ValueError:
                tid_int = None
            if tid_int != team_id:
                continue
            stats = t.get("statistics") or []
            if not isinstance(stats, list):
                return None
            for s in stats:
                if not isinstance(s, dict):
                    continue
                if s.get("name") != stat_name:
                    continue
                val = s.get("value")
                if isinstance(val, (int, float)):
                    return float(val)
                dv = s.get("displayValue")
                if isinstance(dv, str):
                    try:
                        return float(dv)
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _extract_points_from_summary_header(summary: dict[str, Any], *, team_id: int) -> tuple[float | None, float | None]:
        """
        Returns (points_for, points_against) for team_id from summary.header.
        """
        header = summary.get("header") or {}
        comps = header.get("competitions") or []
        comp0 = comps[0] if isinstance(comps, list) and comps else None
        if not isinstance(comp0, dict):
            return (None, None)
        competitors = comp0.get("competitors") or []
        if not isinstance(competitors, list) or len(competitors) < 2:
            return (None, None)

        pts_for = None
        pts_against = None
        for c in competitors:
            if not isinstance(c, dict):
                continue
            team = c.get("team") or {}
            if not isinstance(team, dict):
                continue
            tid = team.get("id")
            try:
                tid_int = int(tid) if isinstance(tid, (int, str)) else None
            except ValueError:
                tid_int = None
            score_raw = c.get("score")
            try:
                score = float(score_raw) if isinstance(score_raw, (int, float, str)) else None
            except ValueError:
                score = None
            if tid_int == team_id:
                pts_for = score
            else:
                pts_against = score
        return (pts_for, pts_against)

    async def compute_team_allowed_average(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        defense_team_abbr: str,
        stat_key: str,
        last_n_games: int = 5,
    ) -> float | None:
        """
        Computes "allowed per game" for the given defense team over recent completed games.
        stat_key options (NBA): points | assists | totalRebounds | steals | blocks | turnovers
        """
        defense_team_abbr = defense_team_abbr.strip().upper()
        if not defense_team_abbr:
            return None
        team_id = await self._team_id_by_abbr(sport=sport, league=league, abbr=defense_team_abbr)
        if team_id is None:
            return None

        cache_key = f"espn:allowed_avg:{sport}:{league}:{defense_team_abbr}:{stat_key}:last{int(last_n_games)}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("avg"), (int, float)):
            return float(cached["avg"])

        eids = await self.get_recent_completed_event_ids(
            sport=sport, league=league, team_id=team_id, limit=max(1, int(last_n_games))
        )
        vals: list[float] = []
        for eid in eids:
            try:
                summ = await self.fetch_event_summary(sport=sport, league=league, event_id=eid)
            except Exception:
                continue

            if stat_key == "points":
                _, pts_against = self._extract_points_from_summary_header(summ, team_id=team_id)
                if isinstance(pts_against, (int, float)):
                    vals.append(float(pts_against))
                continue

            # For non-points stats, use boxscore team totals: allowed = opponent team's stat total.
            box = summ.get("boxscore") or {}
            teams = box.get("teams") or []
            if not isinstance(teams, list) or len(teams) < 2:
                continue
            opp_team_id = None
            for t in teams:
                if not isinstance(t, dict):
                    continue
                team = t.get("team") or {}
                if not isinstance(team, dict):
                    continue
                tid = team.get("id")
                try:
                    tid_int = int(tid) if isinstance(tid, (int, str)) else None
                except ValueError:
                    tid_int = None
                if tid_int is not None and tid_int != team_id:
                    opp_team_id = tid_int
                    break
            if opp_team_id is None:
                continue

            v = self._extract_team_total_stat_from_summary(summ, team_id=opp_team_id, stat_name=stat_key)
            if isinstance(v, (int, float)):
                vals.append(float(v))

        if not vals:
            self._cache.set_json(cache_key, {"avg": None}, ttl_seconds=60 * 60)
            return None

        avg = sum(vals) / len(vals)
        self._cache.set_json(cache_key, {"avg": avg}, ttl_seconds=6 * 60 * 60)
        return avg

    async def compute_team_allowed_by_position_average(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        defense_team_abbr: str,
        offense_position_abbr: str,
        stat_key: str,
        last_n_games: int = 5,
    ) -> float | None:
        """
        Estimates "allowed per game to a position" using ESPN boxscore player lines.

        NBA stat_key options here are PLAYER keys: points | rebounds | assists | blocks | steals | turnovers
        """
        defense_team_abbr = defense_team_abbr.strip().upper()
        offense_position_abbr = offense_position_abbr.strip().upper()
        if not defense_team_abbr or not offense_position_abbr:
            return None

        team_id = await self._team_id_by_abbr(sport=sport, league=league, abbr=defense_team_abbr)
        if team_id is None:
            return None

        cache_key = (
            f"espn:allowed_pos_avg:{sport}:{league}:{defense_team_abbr}:{offense_position_abbr}:{stat_key}:last{int(last_n_games)}"
        )
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("avg"), (int, float)):
            return float(cached["avg"])

        eids = await self.get_recent_completed_event_ids(
            sport=sport, league=league, team_id=team_id, limit=max(1, int(last_n_games))
        )
        vals: list[float] = []
        for eid in eids:
            try:
                summ = await self.fetch_event_summary(sport=sport, league=league, event_id=eid)
            except Exception:
                continue
            box = summ.get("boxscore") or {}
            players = box.get("players") or []
            if not isinstance(players, list) or len(players) < 2:
                continue

            offense_team_block = None
            for blk in players:
                if not isinstance(blk, dict):
                    continue
                team = blk.get("team") or {}
                if not isinstance(team, dict):
                    continue
                tid = team.get("id")
                try:
                    tid_int = int(tid) if isinstance(tid, (int, str)) else None
                except ValueError:
                    tid_int = None
                if tid_int is not None and tid_int != team_id:
                    offense_team_block = blk
                    break
            if not isinstance(offense_team_block, dict):
                continue

            stat_groups = offense_team_block.get("statistics") or []
            if not isinstance(stat_groups, list) or not stat_groups:
                continue
            g0 = stat_groups[0]
            if not isinstance(g0, dict):
                continue
            keys = g0.get("keys") or []
            if not isinstance(keys, list) or stat_key not in keys:
                continue
            idx = keys.index(stat_key)
            athletes = g0.get("athletes") or []
            if not isinstance(athletes, list):
                continue

            total = 0.0
            had_any = False
            for a in athletes:
                if not isinstance(a, dict):
                    continue
                ath = a.get("athlete") or {}
                if not isinstance(ath, dict):
                    continue
                pos = ath.get("position") or {}
                if not isinstance(pos, dict):
                    continue
                pa = pos.get("abbreviation")
                if not isinstance(pa, str) or pa.strip().upper() != offense_position_abbr:
                    continue
                stats = a.get("stats")
                if not isinstance(stats, list) or idx >= len(stats):
                    continue
                raw = stats[idx]
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                total += v
                had_any = True
            if had_any:
                vals.append(total)

        if not vals:
            self._cache.set_json(cache_key, {"avg": None}, ttl_seconds=60 * 60)
            return None

        avg = sum(vals) / len(vals)
        self._cache.set_json(cache_key, {"avg": avg}, ttl_seconds=6 * 60 * 60)
        return avg

    async def compute_league_allowed_rank_snapshot(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        stat_key: str,
        last_n_games: int = 3,
    ) -> dict[str, Any]:
        """
        Computes a league snapshot for allowed-per-game (higher = "softer") and caches it.
        Returns: {teams: {ABBR: {avg_allowed, rank, percentile}}, league_avg, n}
        """
        cache_key = f"espn:league_allowed_snapshot:{sport}:{league}:{stat_key}:last{int(last_n_games)}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("teams"), dict):
            return cached

        abbr_to_id = await self.get_team_abbr_to_id(sport=sport, league=league)
        items: list[tuple[str, float]] = []
        for abbr in sorted(abbr_to_id.keys()):
            try:
                avg = await self.compute_team_allowed_average(
                    sport=sport, league=league, defense_team_abbr=abbr, stat_key=stat_key, last_n_games=last_n_games
                )
            except Exception:
                avg = None
            if isinstance(avg, (int, float)):
                items.append((abbr, float(avg)))

        if not items:
            out = {"teams": {}, "league_avg": None, "n": 0}
            self._cache.set_json(cache_key, out, ttl_seconds=60 * 30)
            return out

        # Sort by allowed desc: rank 1 = most allowed ("softest")
        items.sort(key=lambda x: x[1], reverse=True)
        n = len(items)
        league_avg = sum(v for _, v in items) / n
        teams: dict[str, Any] = {}
        for i, (abbr, v) in enumerate(items, start=1):
            percentile = (n - i) / max(1, (n - 1))  # 1.0 = stingiest, 0.0 = softest
            teams[abbr] = {"avg_allowed": v, "rank": i, "percentile": percentile}

        out = {"teams": teams, "league_avg": league_avg, "n": n, "stat_key": stat_key, "last_n_games": last_n_games}
        self._cache.set_json(cache_key, out, ttl_seconds=6 * 60 * 60)
        return out

    def get_cached_league_allowed_rank_snapshot(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        stat_key: str,
        last_n_games: int = 3,
    ) -> dict[str, Any] | None:
        """
        Returns cached league snapshot if present; does NOT compute or fetch network data.
        """
        cache_key = f"espn:league_allowed_snapshot:{sport}:{league}:{stat_key}:last{int(last_n_games)}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("teams"), dict):
            return cached
        return None

    async def _team_roster(self, *, sport: EspnSport, league: EspnLeague, team_id: int) -> list[dict[str, Any]]:
        cache_key = f"espn:roster:{sport}:{league}:{team_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("athletes"), list):
            cached_list = [a for a in cached["athletes"] if isinstance(a, dict)]
            # Back-compat: older cache entries (esp. NHL) may be position groups with `items`.
            needs_flatten = any(isinstance(a.get("items"), list) for a in cached_list)
            if needs_flatten:
                flat: list[dict[str, Any]] = []
                for a in cached_list:
                    items = a.get("items")
                    if isinstance(items, list) and items:
                        for it in items:
                            if isinstance(it, dict):
                                flat.append(it)
                    else:
                        flat.append(a)
                self._cache.set_json(cache_key, {"athletes": flat}, ttl_seconds=6 * 60 * 60)
                return flat
            return cached_list

        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/roster"
        data = await self._get_json(url)
        athletes = data.get("athletes") or []

        athletes_list: list[dict[str, Any]] = []
        if isinstance(athletes, list):
            # NBA often returns a flat list of athlete dicts.
            # NHL often returns position groups like: [{position:..., items:[athlete,...]}, ...]
            for a in athletes:
                if not isinstance(a, dict):
                    continue
                items = a.get("items")
                if isinstance(items, list) and items:
                    for it in items:
                        if isinstance(it, dict):
                            athletes_list.append(it)
                else:
                    athletes_list.append(a)

        self._cache.set_json(cache_key, {"athletes": athletes_list}, ttl_seconds=6 * 60 * 60)
        return athletes_list

    def _current_espn_season(self, sport: EspnSport) -> int:
        """ESPN season year for the current campaign (NBA 2025-26 season = 2026)."""
        now = datetime.now()
        if sport == "basketball":
            return now.year + 1 if now.month >= 10 else now.year
        if sport == "hockey":
            return now.year + 1 if now.month >= 9 else now.year
        if sport == "football":
            return now.year + 1 if now.month >= 8 else now.year
        return now.year

    async def fetch_gamelog(self, *, sport: EspnSport, league: EspnLeague, athlete_id: int) -> dict[str, Any]:
        season = self._current_espn_season(sport)
        cache_key = f"espn:gamelog:v3:{sport}:{league}:{athlete_id}:{season}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached:
            return cached

        url = (
            f"{self._cfg.site_web_api_base}/apis/common/v3/sports/{sport}/{league}"
            f"/athletes/{athlete_id}/gamelog?season={season}"
        )
        data: dict[str, Any] = {}
        try:
            data = await self._get_json(url)
        except Exception:
            pass

        # Retry without season param if empty
        if not data or not data.get("seasonTypes"):
            try:
                url_no_season = (
                    f"{self._cfg.site_web_api_base}/apis/common/v3/sports/{sport}/{league}"
                    f"/athletes/{athlete_id}/gamelog"
                )
                data = await self._get_json(url_no_season)
            except Exception:
                pass

        if data:
            self._cache.set_json(cache_key, data, ttl_seconds=2 * 60 * 60)
        return data

    async def fetch_league_injuries(self, *, sport: EspnSport, league: EspnLeague) -> dict[str, Any]:
        """
        League-level injuries feed. Shape example:
        { timestamp, status, season, injuries: [ { id, displayName, injuries: [...] } ] }
        """
        cache_key = f"espn:injuries:{sport}:{league}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached:
            return cached

        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/injuries"
        data = await self._get_json(url)
        self._cache.set_json(cache_key, data, ttl_seconds=10 * 60)
        return data

    async def get_team_injuries_summary(
        self,
        *,
        sport: EspnSport,
        league: EspnLeague,
        team_abbr: str,
        limit: int = 6,
    ) -> list[str]:
        """
        Returns short injury strings for the given team abbreviation.
        """
        team_abbr = team_abbr.strip().upper()
        if not team_abbr:
            return []

        data = await self.fetch_league_injuries(sport=sport, league=league)
        team_entries = data.get("injuries") or []
        if not isinstance(team_entries, list):
            return []

        out: list[str] = []
        for t in team_entries:
            if not isinstance(t, dict):
                continue
            injuries = t.get("injuries") or []
            if not isinstance(injuries, list):
                continue
            for inj in injuries:
                if not isinstance(inj, dict):
                    continue
                athlete = inj.get("athlete") or {}
                if not isinstance(athlete, dict):
                    continue
                team = athlete.get("team") or {}
                if not isinstance(team, dict):
                    continue
                abbr = team.get("abbreviation")
                if not isinstance(abbr, str) or abbr.strip().upper() != team_abbr:
                    continue

                name = athlete.get("displayName")
                status = inj.get("status")
                details = inj.get("details") or {}
                injury_type = None
                return_date = None
                if isinstance(details, dict):
                    injury_type = details.get("type") or details.get("detail") or details.get("location")
                    return_date = details.get("returnDate")

                parts = []
                if isinstance(name, str) and name.strip():
                    parts.append(name.strip())
                if isinstance(status, str) and status.strip():
                    parts.append(status.strip())
                if isinstance(injury_type, str) and injury_type.strip():
                    parts.append(f"({injury_type.strip()})")
                if isinstance(return_date, str) and return_date.strip():
                    parts.append(f"return:{return_date.strip()}")
                s = " ".join(parts).strip()
                if s:
                    out.append(s[:200])
                if len(out) >= limit:
                    return out
        return out

    @staticmethod
    def extract_stat_series(gamelog: dict[str, Any], *, field_name: str, last_n: int) -> list[float]:
        """
        Returns the most recent N values for a given ESPN gamelog field name.

        For NBA, `field_name` examples include:
        - points
        - totalRebounds
        - assists
        """
        names = gamelog.get("names") or []
        if not isinstance(names, list) or field_name not in names:
            return []
        idx = names.index(field_name)

        season_types = gamelog.get("seasonTypes") or []
        if not isinstance(season_types, list) or not season_types:
            return []

        # Prefer the first seasonType (usually current regular season).
        st0 = season_types[0]
        if not isinstance(st0, dict):
            return []
        categories = st0.get("categories") or []
        if not isinstance(categories, list):
            return []

        # Gather events in the order provided (typically most recent first).
        values: list[float] = []
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            events = cat.get("events") or []
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                stats = ev.get("stats")
                if not isinstance(stats, list) or idx >= len(stats):
                    continue
                raw = stats[idx]
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
                if len(values) >= last_n:
                    return values
        return values

    @staticmethod
    def extract_game_stat_lines(
        gamelog: dict[str, Any], *, field_name: str, last_n: int
    ) -> list[dict[str, Any]]:
        """
        Returns last N game stat lines with opponent + date context.

        Output dict keys: game_date (datetime|None), opponent_abbr (str|None), value (float)
        """
        names = gamelog.get("names") or []
        if not isinstance(names, list) or field_name not in names:
            return []
        idx = names.index(field_name)

        season_types = gamelog.get("seasonTypes") or []
        if not isinstance(season_types, list) or not season_types:
            return []

        st0 = season_types[0]
        if not isinstance(st0, dict):
            return []
        categories = st0.get("categories") or []
        if not isinstance(categories, list):
            return []

        events_meta = gamelog.get("events") or {}
        if not isinstance(events_meta, dict):
            events_meta = {}

        out: list[dict[str, Any]] = []
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            events = cat.get("events") or []
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                event_id = ev.get("eventId")
                stats = ev.get("stats")
                if not isinstance(event_id, str) or not isinstance(stats, list) or idx >= len(stats):
                    continue
                raw = stats[idx]
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue

                meta = events_meta.get(event_id) if isinstance(event_id, str) else None
                game_date = None
                opp_abbr = None
                if isinstance(meta, dict):
                    dt_raw = meta.get("gameDate")
                    if isinstance(dt_raw, str):
                        try:
                            game_date = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                        except ValueError:
                            game_date = None
                    opp = meta.get("opponent")
                    if isinstance(opp, dict):
                        oa = opp.get("abbreviation")
                        if isinstance(oa, str):
                            opp_abbr = oa

                out.append({"game_date": game_date, "opponent_abbr": opp_abbr, "value": value})
                if len(out) >= last_n:
                    return out
        return out

    @staticmethod
    def extract_game_stat_lines_vs_opponent(
        gamelog: dict[str, Any], *, field_name: str, opponent_abbr: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Finds all games (current season type block) vs a specific opponent, up to `limit`.
        """
        opponent_abbr = opponent_abbr.strip().upper()

        names = gamelog.get("names") or []
        if not isinstance(names, list) or field_name not in names:
            return []
        idx = names.index(field_name)

        season_types = gamelog.get("seasonTypes") or []
        if not isinstance(season_types, list) or not season_types:
            return []

        st0 = season_types[0]
        if not isinstance(st0, dict):
            return []
        categories = st0.get("categories") or []
        if not isinstance(categories, list):
            return []

        events_meta = gamelog.get("events") or {}
        if not isinstance(events_meta, dict):
            events_meta = {}

        matches: list[dict[str, Any]] = []
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            events = cat.get("events") or []
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                event_id = ev.get("eventId")
                stats = ev.get("stats")
                if not isinstance(event_id, str) or not isinstance(stats, list) or idx >= len(stats):
                    continue
                meta = events_meta.get(event_id)
                if not isinstance(meta, dict):
                    continue
                opp = meta.get("opponent")
                if not isinstance(opp, dict):
                    continue
                oa = opp.get("abbreviation")
                if not isinstance(oa, str) or oa.strip().upper() != opponent_abbr:
                    continue

                raw = stats[idx]
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue

                game_date = None
                dt_raw = meta.get("gameDate")
                if isinstance(dt_raw, str):
                    try:
                        game_date = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                    except ValueError:
                        game_date = None

                matches.append({"game_date": game_date, "opponent_abbr": oa.strip().upper(), "value": value})
        matches.sort(key=lambda x: (x.get("game_date") is None, x.get("game_date")), reverse=True)
        return matches[: max(0, int(limit))]

