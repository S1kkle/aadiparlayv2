import asyncio
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from app.clients.espn import EspnClient, _canon_name
from app.clients.ollama import OllamaClient
from app.clients.underdog import UnderdogClient
from app.db import SqliteTTLCache
from app.models.core import GameStat, Prop, SportId
from app.services.underdog_normalizer import (
    american_to_implied_prob as ud_american_to_implied_prob,
    normalize_underdog_over_under_lines,
)
from app.services.stat_model import (
    NormalParams,
    bayesian_shrink,
    compute_stat_profile,
    fit_normal_weighted,
    is_low_count_stat,
    line_percentile,
    poisson_prob_over,
    prob_over,
)


def fmtf(v: float | None, *, signed: bool = False) -> str:
    if v is None:
        return "?"
    if signed:
        return f"{v:+.3f}"
    return f"{v:.3f}"


@dataclass(frozen=True)
class RankerConfig:
    last_n: int
    w_edge: float
    w_ev: float
    w_vol: float
    w_ai: float


HOME_ADVANTAGE: dict[str, float] = {
    "NBA": 0.02,
    "NFL": 0.02,
    "NHL": 0.015,
}

NFL_STAT_KEY_MAP: dict[str, str] = {
    "passingYards": "passingYards",
    "rushingYards": "rushingYards",
    "receivingYards": "receivingYards",
    "passingTouchdowns": "passingTouchdowns",
    "receptions": "receptions",
    "interceptions": "interceptions",
}

NHL_STAT_KEY_MAP: dict[str, str] = {
    "goals": "goals",
    "assists": "assists",
    "points": "points",
    "shotsTotal": "shots",
}


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
    raw = stat.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    if sport == "NBA":
        mapping = {
            "points": ["points"],
            "rebounds": ["totalRebounds"],
            "assists": ["assists"],
            "blocks": ["blocks"],
            "steals": ["steals"],
            "turnovers": ["turnovers"],
            "three_pointers_made": ["threePointFieldGoalsMade"],
            "3_pointers_made": ["threePointFieldGoalsMade"],
            "3pm": ["threePointFieldGoalsMade"],
            "free_throws_made": ["freeThrowsMade"],
            "ftm": ["freeThrowsMade"],
        }
        return mapping.get(raw, []) or mapping.get(s, [])

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
            "completions": ["completions"],
            "passing_attempts": ["passingAttempts"],
            "rushing_attempts": ["rushingAttempts"],
        }
        return mapping.get(raw, []) or mapping.get(s, [])

    if sport == "NHL":
        mapping = {
            "goals": ["goals"],
            "assists": ["assists"],
            "points": ["points"],
            "shots": ["shotsTotal"],
            "shots_on_goal": ["shotsTotal"],
            "saves": ["saves"],
            "goals_against": ["goalsAgainst"],
            "blocked_shots": ["blockedShots"],
        }
        return mapping.get(raw, []) or mapping.get(s, [])

    if sport == "MMA":
        mapping = {
            "sig_strikes": ["sigStrikesLanded"],
            "significant_strikes": ["sigStrikesLanded"],
            "significant_strikes_landed": ["sigStrikesLanded"],
            "sig_strikes_landed": ["sigStrikesLanded"],
            "takedowns": ["takedownsLanded"],
            "takedowns_landed": ["takedownsLanded"],
            "knockdowns": ["knockDowns"],
            "total_strikes": ["totalStrikesLanded"],
            "total_strikes_landed": ["totalStrikesLanded"],
            "submissions": ["submissions"],
            "advances": ["advances"],
        }
        return mapping.get(raw, []) or mapping.get(s, [])

    return []


def _compute_confidence_tier(p: Prop) -> str:
    score = 0
    if p.recent_games and len(p.recent_games) >= 8:
        score += 1
    if abs(p.edge or 0) > 0.05:
        score += 1
    if p.model_ai_agree:
        score += 1
    if p.hit_rate_last10 is not None and p.hit_rate_last10 >= 0.6:
        score += 1
    if p.stat_consistency is not None and p.stat_consistency >= 0.6:
        score += 1
    if p.current_streak is not None and abs(p.current_streak) >= 3:
        if (p.side == "over" and p.current_streak > 0) or (p.side == "under" and p.current_streak < 0):
            score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


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
        require_ai: bool = False,
        require_ai_count: int = 10,
        on_ai_progress: Any | None = None,
        on_model_done: Any | None = None,
    ) -> list[Prop]:
        async def _emit_stage(stage: str, detail: str = "") -> None:
            if on_ai_progress is None:
                return
            try:
                await on_ai_progress({
                    "type": "stage",
                    "stage": stage,
                    "detail": detail,
                })
            except Exception:
                pass

        await _emit_stage("fetch", "Fetching props from Underdog...")

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

        await _emit_stage("fetch", f"Found {len(props)} props")

        for p in props:
            if p.american_price is None:
                p.american_price = -110
            if p.decimal_price is None:
                p.decimal_price = american_to_decimal(p.american_price)
            if p.implied_prob is None:
                p.implied_prob = ud_american_to_implied_prob(p.american_price)

        if self._espn is not None:
            unique_count = len({p.player_name for p in props})
            await _emit_stage("espn", f"Resolving ESPN data for {unique_count} players...")
            await self._apply_espn_model(props)
            modeled = sum(1 for p in props if p.model_prob is not None)
            await _emit_stage("espn", f"ESPN model applied to {modeled}/{len(props)} props")

        for p in props:
            if p.model_prob is None:
                p.model_prob = p.implied_prob
                if p.sport in ("NBA", "NFL", "NHL", "MMA"):
                    p.notes.append("Model fallback: no ESPN stat series available.")
                else:
                    p.notes.append(
                        f"Model not supported for {p.sport} yet (experimental); using implied probability."
                    )

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

        z_by_option: dict[str, tuple[float, float, float]] = {}
        for i, p in enumerate(props):
            z_by_option[p.underdog_option_id] = (edge_z[i], ev_z[i], vol_z[i])
            p.score = self._cfg.w_edge * edge_z[i] + self._cfg.w_ev * ev_z[i] - self._cfg.w_vol * vol_z[i]

        await _emit_stage("rank", "Scoring and ranking props...")

        props.sort(key=lambda p: (p.score is None, -(p.score or 0.0)))

        # --- Deduplicate over/under for same player+stat (#14) ---
        seen_player_stat: dict[tuple[str, str], str] = {}
        deduped: list[Prop] = []
        for p in props:
            key = (p.player_name, p.stat)
            if key not in seen_player_stat:
                seen_player_stat[key] = p.underdog_option_id
                deduped.append(p)
        props = deduped

        await _emit_stage("rank", f"Ranked {len(props)} unique player+stat props")

        # Assign preliminary confidence tiers
        for p in props:
            p.confidence_tier = _compute_confidence_tier(p)

        # --- Emit all stat-model props before AI ---
        if on_model_done is not None:
            try:
                await on_model_done(props)
            except Exception:
                pass

        selected: list[Prop] = []
        if require_ai and self._ollama is not None and ai_limit > 0:
            if not await self._ollama.is_available():
                raise RuntimeError("Ollama not available.")

            # --- Step 1: AI selection — let AI pick the best N from top candidates ---
            # Deduplicate candidate pool: one prop per player (best score)
            _seen_players: set[str] = set()
            _deduped_pool: list[Prop] = []
            for p in props:
                if p.player_name not in _seen_players:
                    _seen_players.add(p.player_name)
                    _deduped_pool.append(p)
            candidate_pool = _deduped_pool[:min(40, len(_deduped_pool))]

            await _emit_stage("ai_select", f"AI selecting best {require_ai_count} from {len(candidate_pool)} candidates...")

            ai_selected_indices = await self._ai_select_props(
                candidate_pool, pick_count=require_ai_count
            )

            ai_picks: list[Prop] = []
            seen_pick_players: set[str] = set()
            if ai_selected_indices:
                for idx in ai_selected_indices:
                    if 0 <= idx < len(candidate_pool):
                        p = candidate_pool[idx]
                        if p.player_name not in seen_pick_players:
                            seen_pick_players.add(p.player_name)
                            ai_picks.append(p)
            if len(ai_picks) < require_ai_count:
                for p in candidate_pool:
                    if p.player_name not in seen_pick_players:
                        seen_pick_players.add(p.player_name)
                        ai_picks.append(p)
                        if len(ai_picks) >= require_ai_count:
                            break

            await _emit_stage("ai", f"Running deep analysis on {len(ai_picks)} AI-selected props...")

            # --- Step 2: Full AI analysis on AI-selected props only ---
            async def _on_prop_done(p: Prop) -> None:
                if on_ai_progress is None:
                    return
                try:
                    ok = isinstance(p.ai_summary, str) and p.ai_summary.strip()
                    await on_ai_progress({"type": "ai_prop_done", "ok": bool(ok), "prop": p})
                except Exception:
                    return

            await self._apply_ollama(
                ai_picks,
                on_prop_done=_on_prop_done,
                per_prop_timeout_s=90,
                ollama_timeout_s=90,
            )

            if on_ai_progress is not None:
                try:
                    have_ai = sum(1 for p in ai_picks if isinstance(p.ai_summary, str) and p.ai_summary.strip())
                    await on_ai_progress({
                        "type": "ai_batch",
                        "analyzed": len(ai_picks),
                        "have_ai": have_ai,
                        "need_ai": int(require_ai_count),
                    })
                except Exception:
                    pass

            # Keep all AI-selected props; those with summaries first, then the rest
            with_ai = [p for p in ai_picks if isinstance(p.ai_summary, str) and p.ai_summary.strip()]
            without_ai = [p for p in ai_picks if not (isinstance(p.ai_summary, str) and p.ai_summary.strip())]
            selected = with_ai + without_ai
        else:
            selected = props[:max_props] if max_props > 0 else props
            if self._ollama is not None and ai_limit > 0:
                await self._apply_ollama(selected)

        # Final score (stats + AI) for selected only
        for p in selected:
            ez, evz, vz = z_by_option.get(p.underdog_option_id, (0.0, 0.0, 0.0))
            ai_signal = float((p.ai_bias or 0) * (p.ai_confidence or 0.0))
            ai_adj = float(p.ai_prob_adjustment or 0.0)
            p.score = (
                self._cfg.w_edge * ez
                + self._cfg.w_ev * evz
                - self._cfg.w_vol * vz
                + self._cfg.w_ai * (ai_signal + ai_adj * 3.0)
            )

            model_favorable = (p.edge or 0) > 0
            if p.ai_bias is not None:
                p.model_ai_agree = model_favorable and p.ai_bias == 1
            p.confidence_tier = _compute_confidence_tier(p)

        selected.sort(key=lambda p: (p.score is None, -(p.score or 0.0)))

        # Return selected (with AI) + remaining model-only props
        selected_ids = {p.underdog_option_id for p in selected}
        remaining = [p for p in props if p.underdog_option_id not in selected_ids]
        return selected + remaining

    async def recommend_parlay(
        self,
        *,
        props: list[Prop],
        legs: int,
    ) -> dict[str, Any]:
        """Pick the best N-leg parlay from the given props and generate an AI summary."""
        assert self._ollama is not None

        candidates = props[:30]
        lines: list[str] = []
        for i, p in enumerate(candidates):
            lines.append(
                f"{i+1}. {p.player_name} | {p.side.upper()} {p.line} {p.display_stat or p.stat} | "
                f"sport={p.sport} game={p.game_title or '?'} team={p.team_abbr or '?'} "
                f"opp={p.opponent_abbr or '?'} | "
                f"model_prob={p.model_prob:.3f} edge={p.edge:+.3f} ev={p.ev:+.4f} "
                f"hit_rate={p.hit_rate_str or '?'} consistency={p.stat_consistency or '?'} "
                f"streak={p.current_streak or 0} trend={p.trend_direction or '?'} "
                f"ai_bias={p.ai_bias} ai_conf={p.ai_confidence}"
            )

        prompt = (
            f"You are building a {legs}-leg parlay from Underdog Fantasy props.\n"
            f"Pick exactly {legs} legs that maximize combined edge while minimizing correlation risk.\n\n"
            "RULES:\n"
            "- Prefer picks with positive edge, high hit rate, and high consistency.\n"
            "- Avoid picking multiple props from the SAME GAME (correlation risk).\n"
            "- Avoid picking multiple props for the SAME PLAYER.\n"
            "- Prefer a mix of sports if available.\n"
            "- Consider streaks and trends when selecting.\n\n"
            "CANDIDATES:\n"
            + "\n".join(lines)
            + "\n\n"
            "Return ONLY valid JSON with keys:\n"
            '- "picks" (array of integers: the 1-indexed candidate numbers you selected)\n'
            '- "parlay_summary" (string: 3-5 sentence analysis of why these legs work together, '
            "mention combined probability, correlation risk, and key factors)\n"
            '- "combined_confidence" (float 0..1: your overall confidence in this parlay)\n'
            '- "risk_factors" (string[]: 2-3 key risks for this parlay)\n'
        )

        parlay_system = (
            "You are a sports parlay analyst. "
            "Return ONLY valid JSON with keys: "
            "picks (array of integers), parlay_summary (string 3-5 sentences), "
            "combined_confidence (float 0..1), risk_factors (string[])."
        )
        result = await self._ollama.analyze_prop(prompt=prompt, timeout_s=60, system=parlay_system)
        picks_indices: list[int] = []
        raw_picks = result.get("picks", [])
        if isinstance(raw_picks, list):
            for idx in raw_picks:
                if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(candidates):
                    picks_indices.append(int(idx) - 1)

        if len(picks_indices) != legs:
            picks_indices = list(range(min(legs, len(candidates))))

        selected_props = [candidates[i] for i in picks_indices if i < len(candidates)]

        return {
            "legs": legs,
            "props": [p.model_dump() for p in selected_props],
            "parlay_summary": result.get("parlay_summary", ""),
            "combined_confidence": result.get("combined_confidence", 0),
            "risk_factors": result.get("risk_factors", []),
            "combined_model_prob": round(
                math.prod(p.model_prob or 0.5 for p in selected_props), 6
            ) if selected_props else 0.0,
        }

    async def _ai_select_props(
        self,
        candidates: list[Prop],
        *,
        pick_count: int = 10,
    ) -> list[int]:
        """Ask AI to pick the best N props from a candidate list. Returns 0-indexed indices."""
        assert self._ollama is not None

        # Shuffle presentation order to prevent positional bias (AI picking top-listed)
        indices = list(range(len(candidates)))
        rng = random.Random(42)
        rng.shuffle(indices)
        idx_to_display: dict[int, int] = {}
        display_to_idx: dict[int, int] = {}
        for display_num, real_idx in enumerate(indices, start=1):
            idx_to_display[real_idx] = display_num
            display_to_idx[display_num] = real_idx

        ask_count = min(pick_count + 5, len(candidates))

        lines: list[str] = []
        for real_idx in indices:
            p = candidates[real_idx]
            dn = idx_to_display[real_idx]

            safety_note = ""
            if p.stat_floor is not None and p.stat_median is not None:
                if p.side == "over" and p.stat_floor > p.line:
                    safety_note = "SAFE: even floor beats line. "
                elif p.side == "over" and p.stat_median < p.line:
                    safety_note = "RISKY: median below line. "
                elif p.side == "under" and p.stat_ceiling is not None and p.stat_ceiling < p.line:
                    safety_note = "SAFE: even ceiling under line. "
                elif p.side == "under" and p.stat_median > p.line:
                    safety_note = "RISKY: median above line. "

            streak_note = ""
            if p.current_streak and abs(p.current_streak) >= 3:
                direction = "overs" if p.current_streak > 0 else "unders"
                streak_note = f"{abs(p.current_streak)}-game {direction} streak. "

            b2b_note = ""
            if p.is_b2b:
                b2b_note = "BACK-TO-BACK (fatigue risk). "
            elif p.rest_days is not None and p.rest_days >= 3:
                b2b_note = f"{p.rest_days}d rest (well-rested). "

            lines.append(
                f"#{dn}. {p.player_name} ({p.sport} | {p.team_abbr or '?'} vs {p.opponent_abbr or '?'}) | "
                f"{p.side.upper()} {p.line} {p.display_stat or p.stat}\n"
                f"    Stats: avg={fmtf(p.model_prob and (p.stat_median or 0))}, median={fmtf(p.stat_median)}, "
                f"floor={fmtf(p.stat_floor)}, ceil={fmtf(p.stat_ceiling)}, consistency={fmtf(p.stat_consistency)}\n"
                f"    Model: prob={fmtf(p.model_prob)}, implied={fmtf(p.implied_prob)}, edge={fmtf(p.edge, signed=True)}, "
                f"hit_rate={p.hit_rate_str or '?'}, trend={p.trend_direction or '?'}\n"
                f"    Context: {'HOME' if p.is_home else 'AWAY' if p.is_home is not None else '?'} "
                f"{b2b_note}{streak_note}{safety_note}"
            )

        prompt = (
            f"You are an expert sports betting analyst selecting the {ask_count} BEST prop picks from "
            f"{len(candidates)} candidates.\n\n"
            "IMPORTANT: Candidates are in RANDOM order. Do NOT favor lower-numbered candidates.\n"
            "The statistical model already ranked them, but it misses qualitative factors. "
            "YOUR VALUE is reasoning about things the model CANNOT capture:\n\n"
            "REASONING CHECKLIST — evaluate EACH candidate on:\n"
            "1. LINE vs MEDIAN/FLOOR/CEILING: Is the line realistic given the player's range? "
            "   A player with median 22 on an OVER 24.5 is risky. A player with floor 18 on OVER 15.5 is safe.\n"
            "2. MATCHUP CONTEXT: Does home/away matter for this sport? Is the opponent strong or weak defensively?\n"
            "3. FATIGUE/REST: Back-to-back games hurt production. Well-rested players perform better.\n"
            "4. STREAK & TREND: Hot streaks in the pick's direction are good. Cold streaks or downtrends are red flags.\n"
            "5. CONSISTENCY: High consistency (>60%) means the player reliably hits their average. "
            "   Low consistency means more variance — riskier.\n"
            "6. HIT RATE: How often has the player actually cleared this line recently?\n"
            "7. AVOID: Props where the line is above the ceiling, or the player is trending away from the line.\n"
            "8. DIVERSIFY: Don't load up on one sport or game. Spread picks across matchups.\n\n"
            "CANDIDATES:\n"
            + "\n".join(lines)
            + "\n\n"
            f"Select EXACTLY {ask_count} picks. For each, give a one-sentence reason WHY.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            f'  "picks": [list of {ask_count} candidate numbers (the # numbers above)],\n'
            '  "reasons": {"<number>": "<one sentence reason>", ...},\n'
            '  "reasoning": "2-3 sentence overall strategy summary"\n'
            "}"
        )

        select_system = (
            "You are a sharp sports prop analyst. Your job is to find the picks most likely to HIT, "
            "NOT just the ones with the highest model edge. Use matchup logic, line safety, consistency, "
            "and trend analysis. Candidates are in RANDOM order — do not favor lower numbers. "
            "Return ONLY valid JSON with keys: picks (array of integers), reasons (object), reasoning (string)."
        )

        cache_key = f"ai_select:v2:{ask_count}:" + ":".join(
            p.underdog_option_id for p in candidates[:10]
        )
        cached = self._cache.get_json(cache_key)
        if isinstance(cached, dict) and cached.get("picks"):
            raw = cached["picks"]
            if isinstance(raw, list):
                return [
                    display_to_idx[int(x)]
                    for x in raw
                    if isinstance(x, (int, float)) and int(x) in display_to_idx
                ]

        try:
            result = await self._ollama.analyze_prop(
                prompt=prompt, timeout_s=60, system=select_system
            )
            self._cache.set_json(cache_key, result, ttl_seconds=10 * 60)
        except Exception:
            return list(range(min(ask_count, len(candidates))))

        raw_picks = result.get("picks", [])
        if isinstance(raw_picks, list):
            real_indices = []
            for x in raw_picks:
                dn = int(x) if isinstance(x, (int, float)) else None
                if dn is not None and dn in display_to_idx:
                    real_indices.append(display_to_idx[dn])
            if real_indices:
                return real_indices[:ask_count]

        return list(range(min(ask_count, len(candidates))))

    async def _apply_espn_model(self, props: list[Prop]) -> None:
        assert self._espn is not None

        team_sport_props = [p for p in props if p.sport in ("NBA", "NFL", "NHL")]
        mma_props = [p for p in props if p.sport == "MMA"]

        if team_sport_props:
            await self._apply_espn_model_team_sports(team_sport_props)
        if mma_props:
            await self._apply_espn_model_mma(mma_props)

    async def _apply_espn_model_team_sports(self, props: list[Prop]) -> None:
        assert self._espn is not None

        unique_players: list[tuple[SportId, str]] = []
        seen: set[tuple[SportId, str]] = set()
        for p in props:
            key = (p.sport, p.player_name)
            if key in seen:
                continue
            seen.add(key)
            unique_players.append(key)

        gamelog_by_player: dict[tuple[SportId, str], dict[str, Any] | None] = {}
        position_by_player: dict[tuple[SportId, str], str | None] = {}
        athlete_id_by_player: dict[tuple[SportId, str], int | None] = {}

        sem = asyncio.Semaphore(10)

        async def fetch_one(sp: SportId, name: str) -> None:
            sl = self._espn.sport_league_for_scope(sp)
            if sl is None:
                gamelog_by_player[(sp, name)] = None
                return
            sport_slug, league_slug = sl
            team_abbr = None
            for p in props:
                if p.sport == sp and p.player_name == name and p.team_abbr:
                    team_abbr = p.team_abbr
                    break
            async with sem:
                if team_abbr:
                    athlete_id, pos = await self._espn.resolve_athlete_profile_from_team_roster(
                        sport=sport_slug, league=league_slug, team_abbr=team_abbr, full_name=name
                    )
                    position_by_player[(sp, name)] = pos
                    if athlete_id is None:
                        athlete_id = await self._espn.find_best_athlete_id(
                            sport=sport_slug, league=league_slug, full_name=name
                        )
                else:
                    athlete_id = await self._espn.find_best_athlete_id(
                        sport=sport_slug, league=league_slug, full_name=name
                    )
                    position_by_player[(sp, name)] = None
            if athlete_id is None:
                gamelog_by_player[(sp, name)] = None
                athlete_id_by_player[(sp, name)] = None
                return
            try:
                async with sem:
                    gamelog_by_player[(sp, name)] = await self._espn.fetch_gamelog(
                        sport=sport_slug, league=league_slug, athlete_id=athlete_id
                    )
                    athlete_id_by_player[(sp, name)] = athlete_id
            except Exception:
                gamelog_by_player[(sp, name)] = None
                athlete_id_by_player[(sp, name)] = athlete_id

        await asyncio.gather(*(fetch_one(sp, name) for sp, name in unique_players))

        for p in props:
            p.player_position = position_by_player.get((p.sport, p.player_name))
            p.espn_athlete_id = athlete_id_by_player.get((p.sport, p.player_name))
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

            # Recency-weighted normal fit (#1)
            params = fit_normal_weighted(series, decay=0.85)
            if params is None:
                p.notes.append("Unable to fit distribution (insufficient ESPN data).")
                continue

            # Bayesian shrinkage for small samples (#6)
            if len(series) < 5:
                shrunk_mu = bayesian_shrink(params.mu, len(series), p.line, shrinkage_k=3.0)
                params = NormalParams(mu=shrunk_mu, sigma=params.sigma)

            # Opponent adjustment using cached league snapshot (#4)
            if p.opponent_abbr and p.sport in ("NBA", "NFL", "NHL"):
                sl = self._espn.sport_league_for_scope(p.sport)
                if sl:
                    sport_slug, league_slug = sl
                    snap_key = field_used or ""
                    snap = self._espn.get_cached_league_allowed_rank_snapshot(
                        sport=sport_slug, league=league_slug, stat_key=snap_key, last_n_games=3
                    )
                    if isinstance(snap, dict) and snap.get("league_avg") and snap.get("teams"):
                        league_avg = float(snap["league_avg"])
                        teams_data = snap.get("teams", {})
                        opp_upper = p.opponent_abbr.strip().upper()
                        if isinstance(teams_data, dict) and opp_upper in teams_data:
                            opp_data = teams_data[opp_upper]
                            if isinstance(opp_data, dict) and "allowed_avg" in opp_data:
                                opp_allowed = float(opp_data["allowed_avg"])
                                if league_avg > 0:
                                    adj_factor = opp_allowed / league_avg
                                    params = NormalParams(mu=params.mu * adj_factor, sigma=params.sigma)

            # Home/away adjustment (#5)
            if p.is_home is not None and p.sport in HOME_ADVANTAGE:
                ha = HOME_ADVANTAGE[p.sport]
                factor = 1.0 + ha if p.is_home else 1.0 - ha
                params = NormalParams(mu=params.mu * factor, sigma=params.sigma)

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

            # Trend detection (#2)
            if len(series) >= 3:
                short_avg = sum(series[:3]) / 3
                full_avg = sum(series) / len(series)
                p.trend_short_avg = round(short_avg, 2)
                if full_avg > 0:
                    if short_avg > full_avg * 1.05:
                        p.trend_direction = "up"
                    elif short_avg < full_avg * 0.95:
                        p.trend_direction = "down"
                    else:
                        p.trend_direction = "flat"

            # Hit rate
            hits = sum(1 for v in series if (v > p.line if p.side == "over" else v < p.line))
            p.hit_rate_last10 = hits / len(series)
            p.hit_rate_str = f"{hits}/{len(series)}"

            # Stat profile (median, floor, ceiling, consistency, streak)
            profile = compute_stat_profile(series, line=p.line, side=p.side)
            if profile:
                p.stat_median = profile.median
                p.stat_floor = profile.floor
                p.stat_ceiling = profile.ceiling
                p.stat_consistency = profile.consistency
                p.current_streak = profile.current_streak

            # Line percentile
            p.line_percentile = line_percentile(line=p.line, params=params)

            # Average minutes (#9)
            if p.sport in ("NBA",):
                mins = self._espn.extract_stat_series(gamelog, field_name="minutes", last_n=self._cfg.last_n)
                if mins:
                    p.avg_minutes = round(sum(mins) / len(mins), 1)

            # B2B / rest days detection (#8)
            if p.scheduled_at and p.recent_games:
                last_game = p.recent_games[0].game_date
                if last_game:
                    diff = p.scheduled_at - last_game
                    days = diff.days
                    p.rest_days = max(0, days)
                    p.is_b2b = (days <= 1)

            # Model probability — Poisson for low-count stats (#3), normal otherwise
            if is_low_count_stat(field_used):
                p_over = poisson_prob_over(line=p.line, lam=params.mu)
            else:
                p_over = prob_over(line=p.line, params=params)
            p.model_prob = p_over if p.side == "over" else (1.0 - p_over)

    async def _apply_espn_model_mma(self, props: list[Prop]) -> None:
        """MMA model: uses per-fight stats from ESPN eventlog instead of gamelog."""
        assert self._espn is not None

        unique_fighters: list[str] = []
        seen: set[str] = set()
        for p in props:
            if p.player_name in seen:
                continue
            seen.add(p.player_name)
            unique_fighters.append(p.player_name)

        fight_history_by_name: dict[str, list[dict[str, Any]]] = {}
        athlete_id_by_name: dict[str, int | None] = {}
        career_stats_by_name: dict[str, dict[str, float]] = {}
        sem = asyncio.Semaphore(6)

        async def fetch_fighter(name: str) -> None:
            async with sem:
                aid = await self._espn.find_mma_athlete_id(full_name=name)
            athlete_id_by_name[name] = aid
            if aid is None:
                fight_history_by_name[name] = []
                career_stats_by_name[name] = {}
                return
            async with sem:
                fight_history_by_name[name] = await self._espn.fetch_mma_fight_history(
                    athlete_id=aid, last_n=self._cfg.last_n,
                )
            async with sem:
                career_stats_by_name[name] = await self._espn.fetch_mma_career_stats(athlete_id=aid)

        await asyncio.gather(*(fetch_fighter(n) for n in unique_fighters))

        for p in props:
            p.espn_athlete_id = athlete_id_by_name.get(p.player_name)
            fights = fight_history_by_name.get(p.player_name, [])
            if not fights:
                p.notes.append("No ESPN fight history found for athlete.")
                continue

            candidates = _stat_field_candidates(sport="MMA", stat=p.stat)
            if not candidates:
                p.notes.append(f"Stat '{p.stat}' not mapped to ESPN MMA stat fields yet.")
                continue

            series: list[float] = []
            field_used: str | None = None
            for field in candidates:
                vals = [f["stats"].get(field, None) for f in fights if isinstance(f.get("stats"), dict)]
                vals = [v for v in vals if v is not None]
                if vals:
                    series = vals
                    field_used = field
                    break
            if not series:
                p.notes.append("ESPN fight history has no values for this stat field.")
                continue

            # Recency-weighted fit (#1)
            params = fit_normal_weighted(series, decay=0.85)
            if params is None:
                p.notes.append("Unable to fit distribution (insufficient ESPN MMA data).")
                continue

            # Bayesian shrinkage for small MMA samples (#6)
            if len(series) < 5:
                shrunk_mu = bayesian_shrink(params.mu, len(series), p.line, shrinkage_k=3.0)
                params = NormalParams(mu=shrunk_mu, sigma=params.sigma)

            p.volatility = float(params.sigma)
            p.stat_field = field_used

            p.recent_games = []
            for f in fights:
                date_str = f.get("date")
                game_date = None
                if isinstance(date_str, str):
                    try:
                        game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    except ValueError:
                        pass
                stat_val = f.get("stats", {}).get(field_used)
                if stat_val is not None:
                    p.recent_games.append(GameStat(
                        game_date=game_date,
                        opponent_abbr=f.get("opponent_name"),
                        value=float(stat_val),
                    ))

            if p.opponent_abbr:
                opp_canon = _canon_name(p.opponent_abbr)
                for f in fights:
                    opp_name = f.get("opponent_name")
                    if not opp_name:
                        continue
                    if _canon_name(opp_name) == opp_canon:
                        date_str = f.get("date")
                        game_date = None
                        if isinstance(date_str, str):
                            try:
                                game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            except ValueError:
                                pass
                        stat_val = f.get("stats", {}).get(field_used)
                        if stat_val is not None:
                            p.vs_opponent_games.append(GameStat(
                                game_date=game_date,
                                opponent_abbr=opp_name,
                                value=float(stat_val),
                            ))

            career = career_stats_by_name.get(p.player_name, {})
            if career:
                p.notes.append(
                    f"Career: SLpM={career.get('strikeLPM', 'N/A')}, "
                    f"StrAcc={career.get('strikeAccuracy', 'N/A')}%, "
                    f"TDAcc={career.get('takedownAccuracy', 'N/A')}%, "
                    f"TDAvg={career.get('takedownAvg', 'N/A')}, "
                    f"SubAvg={career.get('submissionAvg', 'N/A')}"
                )

            # Trend detection (#2)
            if len(series) >= 3:
                short_avg = sum(series[:3]) / 3
                full_avg = sum(series) / len(series)
                p.trend_short_avg = round(short_avg, 2)
                if full_avg > 0:
                    if short_avg > full_avg * 1.05:
                        p.trend_direction = "up"
                    elif short_avg < full_avg * 0.95:
                        p.trend_direction = "down"
                    else:
                        p.trend_direction = "flat"

            # Hit rate
            hits = sum(1 for v in series if (v > p.line if p.side == "over" else v < p.line))
            p.hit_rate_last10 = hits / len(series)
            p.hit_rate_str = f"{hits}/{len(series)}"

            # Stat profile
            profile = compute_stat_profile(series, line=p.line, side=p.side)
            if profile:
                p.stat_median = profile.median
                p.stat_floor = profile.floor
                p.stat_ceiling = profile.ceiling
                p.stat_consistency = profile.consistency
                p.current_streak = profile.current_streak
            p.line_percentile = line_percentile(line=p.line, params=params)

            # Poisson for low-count MMA stats (#3)
            if is_low_count_stat(field_used):
                p_over = poisson_prob_over(line=p.line, lam=params.mu)
            else:
                p_over = prob_over(line=p.line, params=params)
            p.model_prob = p_over if p.side == "over" else (1.0 - p_over)

    async def _apply_ollama(
        self,
        props: list[Prop],
        *,
        on_prop_done: Any | None = None,
        per_prop_timeout_s: float = 25.0,
        ollama_timeout_s: float = 30.0,
    ) -> None:
        assert self._ollama is not None
        assert self._espn is not None

        if not await self._ollama.is_available():
            for p in props:
                p.notes.append("Ollama not available (skipping qualitative analysis).")
            return

        AI_PROMPT_VERSION = "v10"
        sem = asyncio.Semaphore(3)

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

        async def _mma_matchup_lines(p: Prop) -> list[str]:
            out: list[str] = []
            assert self._espn is not None

            for note in p.notes:
                if note.startswith("Career:"):
                    out.append(f"Fighter ({p.player_name}) {note}")
                    break

            if p.opponent_abbr:
                try:
                    opp_aid = await self._espn.find_mma_athlete_id(full_name=p.opponent_abbr)
                    if opp_aid:
                        opp_career = await self._espn.fetch_mma_career_stats(athlete_id=opp_aid)
                        if opp_career:
                            out.append(
                                f"Opponent ({p.opponent_abbr}) career: "
                                f"SLpM={opp_career.get('strikeLPM', 'N/A')}, "
                                f"StrAcc={opp_career.get('strikeAccuracy', 'N/A')}%, "
                                f"TDAcc={opp_career.get('takedownAccuracy', 'N/A')}%, "
                                f"TDAvg={opp_career.get('takedownAvg', 'N/A')}, "
                                f"SubAvg={opp_career.get('submissionAvg', 'N/A')}"
                            )

                        opp_fights = await self._espn.fetch_mma_fight_history(
                            athlete_id=opp_aid, last_n=5,
                        )
                        if opp_fights and p.stat_field:
                            opp_allowed: list[float] = []
                            for f in opp_fights:
                                v = f.get("stats", {}).get(p.stat_field)
                                if v is not None:
                                    opp_allowed.append(float(v))
                            if opp_allowed:
                                avg_opp = sum(opp_allowed) / len(opp_allowed)
                                out.append(
                                    f"Opponent ({p.opponent_abbr}) averages {avg_opp:.1f} "
                                    f"{p.stat_field} themselves (last {len(opp_allowed)} fights)."
                                )
                except Exception:
                    pass

            recent = p.recent_games or []
            if recent:
                out.append(f"Fighter has data from {len(recent)} recent fights for {p.stat_field}.")

            return out

        async def _matchup_lines(p: Prop) -> list[str]:
            if p.sport == "MMA":
                return await _mma_matchup_lines(p)
            if p.sport not in ("NBA", "NFL", "NHL"):
                return []
            if not p.opponent_abbr:
                return []
            sl = self._espn.sport_league_for_scope(p.sport)
            if sl is None:
                return []
            sport_slug, league_slug = sl

            # Map prop stat to team-total stat key for opponent defense context
            stat_key: str | None = None
            if p.sport == "NBA":
                if p.stat_field in ("assists", "blocks", "steals", "turnovers", "totalRebounds", "points"):
                    stat_key = p.stat_field
                else:
                    s = p.stat.strip().lower()
                    nba_map = {"points": "points", "assists": "assists", "rebounds": "totalRebounds",
                               "blocks": "blocks", "steals": "steals", "turnovers": "turnovers"}
                    stat_key = nba_map.get(s)
            elif p.sport == "NFL":
                stat_key = NFL_STAT_KEY_MAP.get(p.stat_field or "")
                if not stat_key:
                    s = p.stat.strip().lower()
                    nfl_map = {"passing_yards": "passingYards", "rushing_yards": "rushingYards",
                               "receiving_yards": "receivingYards", "receptions": "receptions",
                               "passing_touchdowns": "passingTouchdowns"}
                    stat_key = nfl_map.get(s) or nfl_map.get(re.sub(r"[^a-z0-9]+", "_", s).strip("_"))
            elif p.sport == "NHL":
                stat_key = NHL_STAT_KEY_MAP.get(p.stat_field or "")
                if not stat_key:
                    s = p.stat.strip().lower()
                    nhl_map = {"goals": "goals", "assists": "assists", "points": "points",
                               "shots": "shots", "shots_on_goal": "shots"}
                    stat_key = nhl_map.get(s) or nhl_map.get(re.sub(r"[^a-z0-9]+", "_", s).strip("_"))

            if not stat_key:
                return []

            out: list[str] = []
            try:
                opp_allowed = await self._espn.compute_team_allowed_average(
                    sport=sport_slug,
                    league=league_slug,
                    defense_team_abbr=p.opponent_abbr,
                    stat_key=stat_key,
                    last_n_games=5,
                )
            except Exception:
                opp_allowed = None

            if isinstance(opp_allowed, (int, float)):
                snap = self._espn.get_cached_league_allowed_rank_snapshot(
                    sport=sport_slug, league=league_slug, stat_key=stat_key, last_n_games=3
                )
                teams = snap.get("teams") if isinstance(snap, dict) else None
                league_avg = snap.get("league_avg") if isinstance(snap, dict) else None
                n = snap.get("n") if isinstance(snap, dict) else None
                if isinstance(league_avg, (int, float)) and float(league_avg) != 0.0:
                    diff = float(opp_allowed) - float(league_avg)
                    diff_pct = (diff / float(league_avg)) * 100.0
                    diff_str = f"{diff:+.2f} ({diff_pct:+.0f}%)"
                else:
                    diff_str = None

                if isinstance(teams, dict) and isinstance(teams.get(p.opponent_abbr.upper()), dict):
                    t = teams[p.opponent_abbr.upper()]
                    rank = t.get("rank")
                    if isinstance(rank, int) and isinstance(n, int):
                        out.append(
                            f"Opponent {p.opponent_abbr.upper()} allows ~{opp_allowed:.2f} {stat_key}/game (last5)"
                            + (f", {diff_str} vs league avg ~{float(league_avg):.2f}" if diff_str else "")
                            + f"; softness rank {rank}/{n}."
                        )
                    else:
                        out.append(
                            f"Opponent {p.opponent_abbr.upper()} allows ~{opp_allowed:.2f} {stat_key}/game (last5)"
                            + (f", {diff_str} vs league avg ~{float(league_avg):.2f}" if diff_str else "")
                            + "."
                        )
                else:
                    if isinstance(league_avg, (int, float)):
                        out.append(
                            f"Opponent {p.opponent_abbr.upper()} allows ~{opp_allowed:.2f} {stat_key}/game (last5); "
                            + (f"{diff_str} vs league avg ~{float(league_avg):.2f} (cached)." if diff_str else f"league avg ~{float(league_avg):.2f} (cached).")
                        )
                    else:
                        out.append(f"Opponent {p.opponent_abbr.upper()} allows ~{opp_allowed:.2f} {stat_key}/game (last5).")

            # Position-split for NBA
            if p.sport == "NBA" and p.player_position and p.player_position.strip():
                pos = p.player_position.strip().upper()
                player_stat_key = None
                if stat_key == "totalRebounds":
                    player_stat_key = "rebounds"
                elif stat_key in ("assists", "blocks", "steals", "turnovers", "points"):
                    player_stat_key = stat_key
                if player_stat_key:
                    try:
                        pos_allowed = await self._espn.compute_team_allowed_by_position_average(
                            sport=sport_slug,
                            league=league_slug,
                            defense_team_abbr=p.opponent_abbr,
                            offense_position_abbr=pos,
                            stat_key=player_stat_key,
                            last_n_games=5,
                        )
                    except Exception:
                        pos_allowed = None
                    if isinstance(pos_allowed, (int, float)):
                        out.append(
                            f"Position split (est.): {p.opponent_abbr.upper()} allows ~{pos_allowed:.2f} {player_stat_key}/game "
                            f"to {pos} (sum of opposing {pos} boxscore lines; last5)."
                        )

            return out

        async def run_one(p: Prop) -> None:
            cache_key = f"ollama:prop:{AI_PROMPT_VERSION}:{p.underdog_option_id}:{p.side}:{p.line}"
            cached = self._cache.get_json(cache_key)
            if isinstance(cached, dict) and cached:
                self._apply_ai_to_prop(p, cached)
                return

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

            matchup = await _matchup_lines(p)

            is_mma = p.sport == "MMA"

            if is_mma:
                sport_rules = (
                    "- Analyze the matchup between the two fighters (striking vs grappling, style clashes, reach/weight advantages).\n"
                    "- Consider recent fight outcomes (wins/losses, finishes vs decisions, activity level).\n"
                    "- Mention key career stats from MATCHUP_CONTEXT if available (SLpM, TDAcc, etc.).\n"
                    "- Fighters compete infrequently; note the time gap between fights if significant.\n"
                )
                context_label = f"Fight: {p.game_title or 'N/A'}\nFighter: {p.player_name} vs {p.opponent_abbr or '?'}"
                recent_label = "Recent fights (most recent first):"
                vs_label = "Previous fights vs this opponent:"
            else:
                sport_rules = (
                    "- If injury lists are provided, mention 1-2 relevant injuries and how they impact role/usage.\n"
                    "- ONLY mention injuries that appear in the provided injury report lines. Do NOT invent injuries.\n"
                    "- If you mention an injury, include the exact injury line verbatim in parentheses.\n"
                    "- Mention matchup context (pace/role/expected minutes, home/away, opponent defensive tendencies).\n"
                )
                context_label = (
                    f"Game: {p.game_title or 'N/A'}\n"
                    f"Team vs Opp: {(p.team_abbr or '?')} vs {(p.opponent_abbr or '?')}\n"
                    f"Player: {p.player_name}\n"
                    f"Player_position: {p.player_position or 'unknown'}"
                )
                recent_label = "Recent games (most recent first):"
                vs_label = "Vs this opponent (current season, if available):"

            # Enriched context lines
            context_extras: list[str] = []
            if p.is_home is not None:
                context_extras.append(f"Venue: {'HOME' if p.is_home else 'AWAY'}")
            if p.is_b2b:
                context_extras.append("Schedule: BACK-TO-BACK (2nd game in 2 days)")
            elif p.rest_days is not None and p.rest_days >= 3:
                context_extras.append(f"Schedule: {p.rest_days} days rest (well-rested)")
            if p.avg_minutes is not None:
                context_extras.append(f"Avg minutes (last {self._cfg.last_n}): {p.avg_minutes}")
            if p.trend_direction and p.trend_short_avg is not None:
                context_extras.append(
                    f"Trend: {p.trend_direction.upper()} (last3 avg {p.trend_short_avg} vs overall avg {last10_avg:.1f})"
                    if last10_avg is not None else f"Trend: {p.trend_direction.upper()} (last3 avg {p.trend_short_avg})"
                )
            if p.hit_rate_str:
                context_extras.append(f"Hit rate vs line: {p.hit_rate_str} ({p.side} {p.line})")
            if p.current_streak and p.current_streak != 0:
                streak_dir = "overs" if p.current_streak > 0 else "unders"
                context_extras.append(f"Current streak: {abs(p.current_streak)} consecutive {streak_dir}")

            context_extras_str = "\n".join(f"- {x}" for x in context_extras) if context_extras else "- none"

            # Rich stat distribution context
            dist_lines: list[str] = []
            dist_lines.append(f"recent_avg (weighted): {last10_avg}")
            if p.stat_median is not None:
                dist_lines.append(f"median: {p.stat_median}")
            if p.stat_floor is not None and p.stat_ceiling is not None:
                dist_lines.append(f"range: floor(10th pctl)={p.stat_floor}, ceiling(90th pctl)={p.stat_ceiling}")
            if p.stat_consistency is not None:
                dist_lines.append(f"consistency (% within 1 sigma): {p.stat_consistency:.0%}")
            if p.line_percentile is not None:
                pctl = p.line_percentile
                dist_lines.append(
                    f"line {p.line} sits at {pctl:.0%} of distribution "
                    f"({'above average' if pctl > 0.55 else 'below average' if pctl < 0.45 else 'near average'})"
                )
            dist_lines.append(f"volatility (sigma): {p.volatility}")
            dist_lines.append(f"recent_hit_rate_vs_line: {last10_hit}")
            dist_lines.append(f"vs_opp_avg: {vs_avg}")
            dist_lines.append(f"vs_opp_hit_rate_vs_line: {vs_hit}")

            prompt = (
                "Analyze an Underdog Pick'em prop.\n"
                "Write a concise, logically-reasoned summary that weighs matchup context and recent form.\n"
                "Rules:\n"
                + sport_rules
                + "- Cite at least two numbers from the stats context.\n"
                "- Do NOT just restate model_prob/edge; explain *why*.\n"
                "- Consider trend, streak, consistency, floor/ceiling range, and situational factors.\n"
                "- If the player has a hot/cold streak, mention it and whether you expect regression.\n"
                "- If the line is significantly above or below the player's median, note this.\n\n"
                f"Sport: {p.sport}\n"
                f"{context_label}\n"
                f"Pick: {p.side.upper()} {p.line} {p.display_stat or p.stat}\n"
                f"ESPN_field_used: {p.stat_field}\n\n"
                "SITUATIONAL_CONTEXT:\n"
                + context_extras_str
                + "\n\n"
                "MATCHUP_CONTEXT (ESPN-derived; best-effort):\n"
                + ("\n".join([f"- {x}" for x in matchup]) if matchup else "- none")
                + "\n\n"
                + (
                    "TEAM_INJURY_LINES (ESPN; allowed references):\n"
                    + ("\n".join([f"- {x}" for x in team_inj]) if team_inj else "- none listed")
                    + "\n\n"
                    "OPP_INJURY_LINES (ESPN; allowed references):\n"
                    + ("\n".join([f"- {x}" for x in opp_inj]) if opp_inj else "- none listed")
                    + "\n\n"
                    if not is_mma else ""
                )
                + "STATISTICAL_PROFILE:\n"
                + "\n".join(f"- {x}" for x in dist_lines)
                + "\n\n"
                f"{recent_label}\n"
                + "\n".join(
                    [
                        f"- {g.game_date.date().isoformat() if g.game_date else 'unknown_date'} vs {g.opponent_abbr or '?'}: {g.value}"
                        for g in (p.recent_games or [])
                    ]
                )
                + "\n\n"
                f"{vs_label}\n"
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
                "MODEL_OUTPUT:\n"
                f"- model_prob: {p.model_prob}\n"
                f"- implied_prob: {p.implied_prob}\n"
                f"- edge: {p.edge}\n\n"
                "Return JSON only."
            )

            async with sem:
                try:
                    result = await asyncio.wait_for(
                        self._ollama.analyze_prop(prompt=prompt, timeout_s=ollama_timeout_s),
                        timeout=per_prop_timeout_s,
                    )
                except asyncio.TimeoutError:
                    p.notes.append("Ollama analysis timed out (skipping qualitative analysis).")
                    if on_prop_done is not None:
                        try:
                            await on_prop_done(p)
                        except Exception:
                            pass
                    return
                except Exception as e:
                    p.notes.append(f"Ollama analysis skipped: {e}")
                    if on_prop_done is not None:
                        try:
                            await on_prop_done(p)
                        except Exception:
                            pass
                    return
            self._cache.set_json(cache_key, result, ttl_seconds=30 * 60)
            self._apply_ai_to_prop(p, result)
            if on_prop_done is not None:
                try:
                    await on_prop_done(p)
                except Exception:
                    pass

        await asyncio.gather(*(run_one(p) for p in props))

    @staticmethod
    def _apply_ai_to_prop(p: Prop, result: dict[str, Any]) -> None:
        summary = result.get("summary")
        bias = result.get("overall_bias")
        conf = result.get("confidence")
        tailwinds = result.get("tailwinds")
        risks = result.get("risk_factors")
        prob_adj = result.get("prob_adjustment")

        if isinstance(summary, str) and summary.strip():
            p.ai_summary = summary.strip()[:1000]
        if isinstance(bias, int) and bias in (-1, 0, 1):
            p.ai_bias = bias
        if isinstance(conf, (int, float)):
            p.ai_confidence = max(0.0, min(1.0, float(conf)))
        if isinstance(tailwinds, list):
            p.ai_tailwinds = [str(x) for x in tailwinds if str(x).strip()][:8]
        if isinstance(risks, list):
            p.ai_risk_factors = [str(x) for x in risks if str(x).strip()][:8]
        if isinstance(prob_adj, (int, float)):
            p.ai_prob_adjustment = max(-0.15, min(0.15, float(prob_adj)))
