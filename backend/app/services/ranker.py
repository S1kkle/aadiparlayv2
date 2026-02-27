import asyncio
import math
from dataclasses import dataclass
from typing import Any, Literal

from app.clients.espn import EspnClient
from app.clients.ollama import OllamaClient
from app.clients.underdog import UnderdogClient
from app.db import SqliteTTLCache
from app.models.core import GameStat, Prop, SportId
from app.services.underdog_normalizer import (
    american_to_implied_prob as ud_american_to_implied_prob,
    normalize_underdog_over_under_lines,
)
from app.services.stat_model import fit_normal, prob_over


@dataclass(frozen=True)
class RankerConfig:
    last_n: int
    w_edge: float
    w_ev: float
    w_vol: float
    w_ai: float


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, (len(values) - 1))
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def american_to_decimal(american: int) -> float:
    if american < 0:
        return 1.0 + (100.0 / (-american))
    return 1.0 + (american / 100.0)


def _stat_field_candidates(*, sport: SportId, stat: str) -> list[str]:
    s = stat.strip().lower()

    if sport == "NBA":
        mapping = {
            "points": ["points"],
            "rebounds": ["totalRebounds"],
            "assists": ["assists"],
            "blocks": ["blocks"],
            "steals": ["steals"],
            "turnovers": ["turnovers"],
        }
        return mapping.get(s, [])

    if sport == "NFL":
        mapping = {
            "passing_yards": ["passingYards"],
            "passing_touchdowns": ["passingTouchdowns"],
            "passing_tds": ["passingTouchdowns"],
            "interceptions": ["interceptions"],
            "rushing_yards": ["rushingYards"],
            "rushing_touchdowns": ["rushingTouchdowns"],
            "rushing_tds": ["rushingTouchdowns"],
            "receptions": ["receptions"],
            "receiving_yards": ["receivingYards"],
            "receiving_touchdowns": ["receivingTouchdowns"],
            "receiving_tds": ["receivingTouchdowns"],
        }
        return mapping.get(s, [])

    if sport == "NHL":
        mapping = {
            "goals": ["goals"],
            "assists": ["assists"],
            "points": ["points"],
            "shots": ["shotsTotal"],
            "shots_on_goal": ["shotsTotal"],
        }
        return mapping.get(s, [])

    return []


class Ranker:
    def __init__(
        self,
        cfg: RankerConfig,
        *,
        ud_client: UnderdogClient,
        cache: SqliteTTLCache,
        espn: EspnClient | None = None,
        ollama: OllamaClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._ud = ud_client
        self._cache = cache
        self._espn = espn
        self._ollama = ollama

    async def rank_props(
        self,
        *,
        scope: Literal["all", "featured"],
        sport: SportId,
        refresh: bool,
        max_props: int = 120,
        ai_limit: int = 25,
    ) -> list[Prop]:
        # cache Underdog pull briefly (their API is rate-limited / geo-gated)
        cache_key = f"underdog:over_under_lines:{scope}"
        payload: dict[str, Any] | None = None
        if not refresh:
            payload = self._cache.get_json(cache_key)
        if payload is None:
            payload = await self._ud.fetch_all_over_under_lines()
            self._cache.set_json(cache_key, payload, ttl_seconds=10)

        props = normalize_underdog_over_under_lines(payload)
        if sport != "UNKNOWN":
            props = [p for p in props if p.sport == sport]
        # NOTE: Do NOT truncate before scoring.
        # Underdog often returns props grouped by a game; truncating here biases the result.

        # 1) implied probability from Underdog
        for p in props:
            if p.american_price is None:
                p.american_price = -110
            if p.decimal_price is None:
                p.decimal_price = american_to_decimal(p.american_price)
            if p.implied_prob is None:
                p.implied_prob = ud_american_to_implied_prob(p.american_price)

        # 2) ESPN model probability (when possible)
        if self._espn is not None:
            await self._apply_espn_model(props)

        # fallback for any props we couldn't model
        for p in props:
            if p.model_prob is None:
                p.model_prob = p.implied_prob
                p.notes.append("Model fallback: no ESPN stat series available.")

        edge_vals: list[float] = []
        ev_vals: list[float] = []
        vol_vals: list[float] = []
        for p in props:
            p.edge = (p.model_prob or 0.0) - (p.implied_prob or 0.0)
            p.ev = (p.model_prob or 0.0) * ((p.decimal_price or 1.0) - 1.0) - (1.0 - (p.model_prob or 0.0))
            edge_vals.append(p.edge or 0.0)
            ev_vals.append(p.ev or 0.0)
            vol_vals.append(float(p.volatility or 0.0))

        edge_z = _zscore(edge_vals)
        ev_z = _zscore(ev_vals)
        vol_z = _zscore(vol_vals)

        # preliminary score (no AI yet) so we can AI-rank a subset
        z_by_option: dict[str, tuple[float, float, float]] = {}
        for i, p in enumerate(props):
            z_by_option[p.underdog_option_id] = (edge_z[i], ev_z[i], vol_z[i])
            p.score = self._cfg.w_edge * edge_z[i] + self._cfg.w_ev * ev_z[i] - self._cfg.w_vol * vol_z[i]

        # Select the top-N purely statistically first (unbiased by feed order),
        # then run AI on exactly those returned picks so the UI always has AI for top 10.
        props.sort(key=lambda p: (p.score is None, -(p.score or 0.0)))
        selected = props[:max_props] if max_props > 0 else props

        # 3) Ollama qualitative analysis (best-effort) for selected only
        if self._ollama is not None and ai_limit > 0:
            await self._apply_ollama(selected)

        # Final score (stats + AI) for selected only
        for p in selected:
            ez, evz, vz = z_by_option.get(p.underdog_option_id, (0.0, 0.0, 0.0))
            ai_signal = float((p.ai_bias or 0) * (p.ai_confidence or 0.0))
            p.score = self._cfg.w_edge * ez + self._cfg.w_ev * evz - self._cfg.w_vol * vz + self._cfg.w_ai * ai_signal

        selected.sort(key=lambda p: (p.score is None, -(p.score or 0.0)))
        return selected

    async def _apply_espn_model(self, props: list[Prop]) -> None:
        assert self._espn is not None

        # Resolve athlete IDs per (sport, player_name) and fetch gamelog once
        unique_players: list[tuple[SportId, str]] = []
        seen: set[tuple[SportId, str]] = set()
        for p in props:
            key = (p.sport, p.player_name)
            if key in seen:
                continue
            if p.sport not in ("NBA", "NFL", "NHL"):
                continue
            seen.add(key)
            unique_players.append(key)

        gamelog_by_player: dict[tuple[SportId, str], dict[str, Any] | None] = {}

        sem = asyncio.Semaphore(10)

        async def fetch_one(sp: SportId, name: str) -> None:
            sl = self._espn.sport_league_for_scope(sp)
            if sl is None:
                gamelog_by_player[(sp, name)] = None
                return
            sport_slug, league_slug = sl
            # prefer team-roster mapping when we have an abbreviation; fallback to global search
            team_abbr = None
            for p in props:
                if p.sport == sp and p.player_name == name and p.team_abbr:
                    team_abbr = p.team_abbr
                    break
            async with sem:
                if team_abbr:
                    athlete_id = await self._espn.resolve_athlete_id_from_team_roster(
                        sport=sport_slug, league=league_slug, team_abbr=team_abbr, full_name=name
                    )
                else:
                    athlete_id = await self._espn.find_best_athlete_id(
                        sport=sport_slug, league=league_slug, full_name=name
                    )
            if athlete_id is None:
                gamelog_by_player[(sp, name)] = None
                return
            try:
                async with sem:
                    gamelog_by_player[(sp, name)] = await self._espn.fetch_gamelog(
                        sport=sport_slug, league=league_slug, athlete_id=athlete_id
                    )
            except Exception:
                gamelog_by_player[(sp, name)] = None

        await asyncio.gather(*(fetch_one(sp, name) for sp, name in unique_players))

        # Apply model per prop
        for p in props:
            if p.sport not in ("NBA", "NFL", "NHL"):
                continue
            gamelog = gamelog_by_player.get((p.sport, p.player_name))
            if not gamelog:
                p.notes.append("No ESPN gamelog found for athlete.")
                continue

            candidates = _stat_field_candidates(sport=p.sport, stat=p.stat)
            if not candidates:
                p.notes.append(f"Stat '{p.stat}' not mapped to ESPN gamelog fields yet.")
                continue

            series: list[float] = []
            field_used: str | None = None
            for field in candidates:
                series = self._espn.extract_stat_series(gamelog, field_name=field, last_n=self._cfg.last_n)
                if series:
                    field_used = field
                    break
            if not series:
                p.notes.append("ESPN gamelog has no values for this stat field.")
                continue

            params = fit_normal(series)
            if params is None:
                p.notes.append("Unable to fit distribution (insufficient ESPN data).")
                continue
            p.volatility = float(params.sigma)
            p.stat_field = field_used

            if field_used:
                lines = self._espn.extract_game_stat_lines(gamelog, field_name=field_used, last_n=self._cfg.last_n)
                p.recent_games = [
                    GameStat(game_date=x.get("game_date"), opponent_abbr=x.get("opponent_abbr"), value=float(x["value"]))
                    for x in lines
                    if "value" in x
                ]
                if p.opponent_abbr:
                    vs_lines = self._espn.extract_game_stat_lines_vs_opponent(
                        gamelog, field_name=field_used, opponent_abbr=p.opponent_abbr, limit=10
                    )
                    p.vs_opponent_games = [
                        GameStat(
                            game_date=x.get("game_date"),
                            opponent_abbr=x.get("opponent_abbr"),
                            value=float(x["value"]),
                        )
                        for x in vs_lines
                        if "value" in x
                    ]

            p_over = prob_over(line=p.line, params=params)
            p.model_prob = p_over if p.side == "over" else (1.0 - p_over)

    async def _apply_ollama(self, props: list[Prop]) -> None:
        assert self._ollama is not None
        assert self._espn is not None

        if not await self._ollama.is_available():
            for p in props:
                p.notes.append("Ollama not available (skipping qualitative analysis).")
            return

        AI_PROMPT_VERSION = "v5"
        sem = asyncio.Semaphore(4)

        def _mean(xs: list[float]) -> float | None:
            if not xs:
                return None
            return sum(xs) / len(xs)

        def _hit_rate(values: list[float], *, line: float, side: str) -> float | None:
            if not values:
                return None
            if side == "over":
                hits = sum(1 for v in values if v > line)
            else:
                hits = sum(1 for v in values if v < line)
            return hits / len(values)

        async def run_one(p: Prop) -> None:
            cache_key = f"ollama:prop:{AI_PROMPT_VERSION}:{p.underdog_option_id}:{p.side}:{p.line}"
            cached = self._cache.get_json(cache_key)
            if isinstance(cached, dict) and cached:
                self._apply_ai_to_prop(p, cached)
                return

            # Injury context (team + opponent) from ESPN league injuries
            sl = self._espn.sport_league_for_scope(p.sport)
            team_inj: list[str] = []
            opp_inj: list[str] = []
            if sl is not None:
                sport_slug, league_slug = sl
                if p.team_abbr:
                    try:
                        team_inj = await self._espn.get_team_injuries_summary(
                            sport=sport_slug, league=league_slug, team_abbr=p.team_abbr, limit=5
                        )
                    except Exception:
                        team_inj = []
                if p.opponent_abbr:
                    try:
                        opp_inj = await self._espn.get_team_injuries_summary(
                            sport=sport_slug, league=league_slug, team_abbr=p.opponent_abbr, limit=5
                        )
                    except Exception:
                        opp_inj = []

            last10_vals = [g.value for g in (p.recent_games or [])]
            vs_vals = [g.value for g in (p.vs_opponent_games or [])]
            last10_avg = _mean(last10_vals)
            last10_hit = _hit_rate(last10_vals, line=p.line, side=p.side)
            vs_avg = _mean(vs_vals)
            vs_hit = _hit_rate(vs_vals, line=p.line, side=p.side)

            prompt = (
                "Analyze an Underdog Pick'em prop.\n"
                "Write a concise, logically-reasoned summary that weighs injuries/availability, matchup context, and recent form.\n"
                "Rules:\n"
                "- If injury lists are provided, mention 1-2 relevant injuries and how they impact role/usage.\n"
                "- ONLY mention injuries that appear in the provided injury report lines. Do NOT invent injuries.\n"
                "- If you mention an injury, include the exact injury line verbatim in parentheses.\n"
                "- Mention matchup context (pace/role/expected minutes, home/away, opponent).\n"
                "- Cite at least two numbers from the stats context.\n"
                "- Do NOT just restate model_prob/edge; explain *why*.\n\n"
                f"Sport: {p.sport}\n"
                f"Game: {p.game_title or 'N/A'}\n"
                f"Team vs Opp: {(p.team_abbr or '?')} vs {(p.opponent_abbr or '?')}\n"
                f"Player: {p.player_name}\n"
                f"Pick: {p.side.upper()} {p.line} {p.display_stat or p.stat}\n"
                f"ESPN_field_used: {p.stat_field}\n\n"
                "TEAM_INJURY_LINES (ESPN; allowed references):\n"
                + ("\n".join([f"- {x}" for x in team_inj]) if team_inj else "- none listed")
                + "\n\n"
                "OPP_INJURY_LINES (ESPN; allowed references):\n"
                + ("\n".join([f"- {x}" for x in opp_inj]) if opp_inj else "- none listed")
                + "\n\n"
                "Stats context:\n"
                f"- last10_avg: {last10_avg}\n"
                f"- last10_hit_rate_vs_line: {last10_hit}\n"
                f"- vs_opp_avg: {vs_avg}\n"
                f"- vs_opp_hit_rate_vs_line: {vs_hit}\n\n"
                "Recent games (most recent first):\n"
                + "\n".join(
                    [
                        f"- {g.game_date.date().isoformat() if g.game_date else 'unknown_date'} vs {g.opponent_abbr or '?'}: {g.value}"
                        for g in (p.recent_games or [])
                    ]
                )
                + "\n\n"
                "Vs this opponent (current season, if available):\n"
                + (
                    "\n".join(
                        [
                            f"- {g.game_date.date().isoformat() if g.game_date else 'unknown_date'}: {g.value}"
                            for g in (p.vs_opponent_games or [])
                        ]
                    )
                    or "- none"
                )
                + "\n\n"
                "Stats model:\n"
                f"- model_prob: {p.model_prob}\n"
                f"- implied_prob: {p.implied_prob}\n"
                f"- edge: {p.edge}\n"
                f"- volatility: {p.volatility}\n\n"
                "Return JSON only."
            )

            async with sem:
                try:
                    result = await self._ollama.analyze_prop(prompt=prompt)
                except Exception as e:
                    p.notes.append(f"Ollama analysis skipped: {e}")
                    return
            self._cache.set_json(cache_key, result, ttl_seconds=30 * 60)
            self._apply_ai_to_prop(p, result)

        await asyncio.gather(*(run_one(p) for p in props))

    @staticmethod
    def _apply_ai_to_prop(p: Prop, result: dict[str, Any]) -> None:
        summary = result.get("summary")
        bias = result.get("overall_bias")
        conf = result.get("confidence")
        tailwinds = result.get("tailwinds")
        risks = result.get("risk_factors")

        if isinstance(summary, str) and summary.strip():
            p.ai_summary = summary.strip()[:500]
        if isinstance(bias, int) and bias in (-1, 0, 1):
            p.ai_bias = bias
        if isinstance(conf, (int, float)):
            p.ai_confidence = max(0.0, min(1.0, float(conf)))
        if isinstance(tailwinds, list):
            p.ai_tailwinds = [str(x) for x in tailwinds if str(x).strip()][:8]
        if isinstance(risks, list):
            p.ai_risk_factors = [str(x) for x in risks if str(x).strip()][:8]

