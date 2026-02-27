from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote_plus

import httpx

from app.db import SqliteTTLCache


EspnLeague = Literal["nba", "nfl", "nhl"]
EspnSport = Literal["basketball", "football", "hockey"]


def _canon_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


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

    @staticmethod
    def sport_league_for_scope(scope: str) -> tuple[EspnSport, EspnLeague] | None:
        # scope is our SportId-ish value
        if scope == "NBA":
            return ("basketball", "nba")
        if scope == "NFL":
            return ("football", "nfl")
        if scope == "NHL":
            return ("hockey", "nhl")
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
        cache_key = f"espn:roster_lookup:{sport}:{league}:{team_abbr}:{canon_target}"
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

    async def _team_roster(self, *, sport: EspnSport, league: EspnLeague, team_id: int) -> list[dict[str, Any]]:
        cache_key = f"espn:roster:{sport}:{league}:{team_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("athletes"), list):
            return cached["athletes"]

        url = f"{self._cfg.site_api_base}/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/roster"
        data = await self._get_json(url)
        athletes = data.get("athletes") or []
        athletes_list: list[dict[str, Any]] = [a for a in athletes if isinstance(a, dict)]
        self._cache.set_json(cache_key, {"athletes": athletes_list}, ttl_seconds=6 * 60 * 60)
        return athletes_list

    async def fetch_gamelog(self, *, sport: EspnSport, league: EspnLeague, athlete_id: int) -> dict[str, Any]:
        cache_key = f"espn:gamelog:{sport}:{league}:{athlete_id}"
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached:
            return cached

        url = f"{self._cfg.site_web_api_base}/apis/common/v3/sports/{sport}/{league}/athletes/{athlete_id}/gamelog"
        data = await self._get_json(url)
        self._cache.set_json(cache_key, data, ttl_seconds=6 * 60 * 60)
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

