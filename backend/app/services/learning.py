"""
Learning service: resolves pick outcomes from ESPN, diagnoses misses via AI,
and generates weekly improvement reports.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from app.clients.espn import EspnClient
from app.clients.groq import GroqClient
from app.clients.ollama import OllamaClient
from app.db import SqliteTTLCache

log = logging.getLogger(__name__)


class LearningService:
    def __init__(
        self,
        *,
        cache: SqliteTTLCache,
        espn: EspnClient,
        llm: OllamaClient | GroqClient,
    ) -> None:
        self._cache = cache
        self._espn = espn
        self._llm = llm

    # ── Step 1: resolve outcomes ──────────────────────────────────────

    async def resolve_outcomes(self) -> dict[str, Any]:
        """Look up actual results for past picks that haven't been resolved yet.

        Deduplication strategy:
        - Cross-history-entry: same (player, stat, line, side, scheduled-day)
          resolves once. This prevents 3 prediction runs from creating 3
          learning_log rows for the same physical prop outcome.
        - Within-history-entry: also deduped by underdog_option_id.
        """
        history = self._cache.get_history(limit=50)
        if not history:
            return {"resolved": 0, "already_done": 0, "failed_lookup": 0, "skipped_future": 0, "detail": "No history entries found."}

        existing_entries = self._cache.get_learning_entries(limit=5000)
        existing_ids: set[str] = {e["id"] for e in existing_entries}
        # Build a logical-key index from already-resolved entries so we don't
        # re-resolve the same physical prop under a different history_id.
        existing_logical: set[str] = set()
        for e in existing_entries:
            ts_raw = e.get("timestamp", "")
            day_part = (ts_raw or "")[:10]  # YYYY-MM-DD
            logical = (
                f"{e.get('player_name', '?')}::{(e.get('stat') or '').lower()}"
                f"::{float(e.get('line', 0))}::{e.get('side', '?')}::{day_part}"
            )
            existing_logical.add(logical)

        resolved_count = 0
        already_done = 0
        failed = 0
        skipped_future = 0
        deduped = 0

        for entry in history:
            hid = entry["id"]
            ts = entry["timestamp"]
            props = entry.get("props", [])
            if not isinstance(props, list):
                continue

            for p in props:
                if not isinstance(p, dict):
                    continue

                option_id = p.get("underdog_option_id", "")
                if not option_id:
                    continue
                lid = f"{hid}:{option_id}"

                if lid in existing_ids:
                    already_done += 1
                    continue

                # Cross-history-entry dedup: skip if this physical prop already
                # has a resolved row under a different history_id.
                scheduled_at = p.get("scheduled_at")
                day_part = ""
                if isinstance(scheduled_at, str):
                    day_part = scheduled_at[:10]
                logical_key = (
                    f"{p.get('player_name', '?')}::{(p.get('stat') or '').lower()}"
                    f"::{float(p.get('line', 0))}::{p.get('side', '?')}::{day_part}"
                )
                if logical_key in existing_logical:
                    deduped += 1
                    continue

                scheduled = p.get("scheduled_at")
                if scheduled:
                    try:
                        game_time = datetime.fromisoformat(str(scheduled).replace("Z", "+00:00"))
                        if game_time > datetime.now(timezone.utc) - timedelta(hours=3):
                            skipped_future += 1
                            continue
                    except Exception:
                        pass

                actual = await self._lookup_actual_value(p)
                if actual is None:
                    failed += 1
                    continue

                line = float(p.get("line", 0))
                side = p.get("side", "over")
                hit = (actual > line) if side == "over" else (actual < line)

                # Persist the historical series leading up to the prediction so that
                # calibration grid search can TRULY re-simulate the prediction under
                # candidate params, not just re-threshold a frozen model_prob.
                series_vals: list[float] = []
                recent = p.get("recent_games") or []
                if isinstance(recent, list):
                    for g in recent:
                        if isinstance(g, dict) and "value" in g:
                            try:
                                series_vals.append(float(g["value"]))
                            except (TypeError, ValueError):
                                continue

                # Online Bayesian update of the per-player posterior using
                # the freshly-resolved actual_value. Closed-form conjugate
                # update; the next time this player's prop is ranked the
                # ranker's `get_hierarchical_prior` will see a sharper
                # estimate than the static league baseline.
                try:
                    from app.services.online_priors import update_player_prior
                    update_player_prior(
                        self._cache,
                        sport=str(p.get("sport") or ""),
                        stat_field=str(p.get("stat_field") or p.get("stat") or ""),
                        player=str(p.get("player_name") or ""),
                        position=p.get("player_position"),
                        observation=float(actual),
                    )
                except Exception:
                    log.debug("online_priors update failed (non-fatal)", exc_info=True)

                # Online SGD step on the tier-logistic. One step per resolved
                # outcome — keeps the deployed model tracking drift between
                # the (slower) batch retrain cycles. The regression-guarded
                # batch retrain remains the source of truth.
                try:
                    from app.services.stat_model import (
                        online_tier_step, _tier_features_from_dict,
                    )
                    feat_dict = dict(p)
                    feat_dict["side"] = side
                    feats = _tier_features_from_dict(feat_dict)
                    online_tier_step(feats, 1 if hit else 0)
                except Exception:
                    log.debug("online_tier_step failed (non-fatal)", exc_info=True)

                self._cache.save_learning_entry({
                    "id": lid,
                    "history_id": hid,
                    "timestamp": ts,
                    "player_name": p.get("player_name", "?"),
                    "sport": p.get("sport", "?"),
                    "stat": p.get("stat", "?"),
                    "line": line,
                    "side": side,
                    "model_prob": p.get("model_prob"),
                    "implied_prob": p.get("implied_prob"),
                    "edge": p.get("edge"),
                    "ai_bias": p.get("ai_bias"),
                    "ai_confidence": p.get("ai_confidence"),
                    "actual_value": actual,
                    "hit": 1 if hit else 0,
                    "miss_reason": None,
                    "miss_category": None,
                    # Even MISSES are "resolved" — the only reason to re-resolve
                    # would be a data error. Previously this was 0 for misses,
                    # which made them eligible for re-lookup forever.
                    "resolved": 1,
                    "series_json": json.dumps(series_vals) if series_vals else None,
                    "stat_field": p.get("stat_field"),
                    "position": p.get("player_position"),
                    "decimal_price": p.get("decimal_price"),
                    "payout_multiplier": p.get("payout_multiplier"),
                    # CLV / accounting / provenance — saved at creation when
                    # available; the closing-line job will fill in close_line +
                    # clv_cents asynchronously.
                    "line_at_pick": p.get("line"),
                    "close_line": None,
                    "close_implied_prob": None,
                    "clv_cents": None,
                    "stake_amount": p.get("stake_amount"),
                    "payout_amount": p.get("payout_amount"),
                    "profit": p.get("profit"),
                    "prompt_version": p.get("prompt_version"),
                    "model_params_id": p.get("model_params_id"),
                    "entry_type": p.get("entry_type"),
                })
                existing_ids.add(lid)
                existing_logical.add(logical_key)
                resolved_count += 1

        return {
            "resolved": resolved_count,
            "already_done": already_done,
            "deduped": deduped,
            "failed_lookup": failed,
            "skipped_future": skipped_future,
        }

    async def _lookup_actual_value(self, prop: dict[str, Any]) -> float | None:
        """Fetch the player's actual stat value for the game from ESPN.

        Strategy:
        1. If espn_athlete_id + stat_field present, use direct gamelog lookup.
        2. Otherwise, search ESPN by player name and use the stat name to find a field.
        """
        sport = prop.get("sport", "")
        athlete_id = prop.get("espn_athlete_id")
        stat_field = prop.get("stat_field")
        scheduled_at = prop.get("scheduled_at")
        player_name = prop.get("player_name", "")

        sl = self._espn.sport_league_for_scope(sport)
        if sl is None:
            return None
        sport_slug, league_slug = sl

        if not athlete_id and player_name:
            try:
                athlete_id = await self._espn.find_best_athlete_id(
                    sport=sport_slug, league=league_slug, full_name=player_name
                )
            except Exception:
                pass

        if not athlete_id:
            return None

        if not stat_field:
            stat_field = self._guess_stat_field(prop.get("stat", ""), sport)
            if not stat_field:
                return None

        try:
            gamelog = await self._espn.fetch_gamelog(
                sport=sport_slug, league=league_slug, athlete_id=int(athlete_id)
            )
        except Exception:
            return None

        lines = self._espn.extract_game_stat_lines(
            gamelog, field_name=stat_field, last_n=20
        )
        if not lines:
            return None

        if scheduled_at:
            try:
                target = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
                for g in lines:
                    gd = g.get("game_date")
                    if gd and abs((gd - target).total_seconds()) < 36 * 3600:
                        return float(g["value"])
            except Exception:
                pass

        return float(lines[0]["value"]) if lines else None

    @staticmethod
    def _guess_stat_field(stat_name: str, sport: str) -> str | None:
        """Map common Underdog stat names to ESPN gamelog field names."""
        s = stat_name.lower().strip()
        mapping: dict[str, str] = {
            "points": "points",
            "pts": "points",
            "rebounds": "totalRebounds",
            "total rebounds": "totalRebounds",
            "rebs": "totalRebounds",
            "assists": "assists",
            "asts": "assists",
            "steals": "steals",
            "blocks": "blocks",
            "turnovers": "turnovers",
            "3-pointers made": "threePointFieldGoalsMade",
            "3-pt made": "threePointFieldGoalsMade",
            "three pointers made": "threePointFieldGoalsMade",
            "pts + rebs": "points",
            "pts + asts": "points",
            "pts + rebs + asts": "points",
            "fantasy points": "points",
            "passing yards": "passingYards",
            "rushing yards": "rushingYards",
            "receiving yards": "receivingYards",
            "receptions": "receptions",
            "passing touchdowns": "passingTouchdowns",
            "rush + rec yards": "rushingYards",
            "goals": "goals",
            # NHL uses ESPN field "shotsTotal" for skater shots — there is no
            # "shotsOnGoal" key in the ESPN gamelog payload. ranker.py and
            # calibration.py already canonicalise to "shots" / "shotsTotal".
            "shots": "shotsTotal",
            "shots on goal": "shotsTotal",
            "sog": "shotsTotal",
            "saves": "saves",
            "goals against": "goalsAgainst",
            # Combo NHL stats fall back to the dominant component
            "goals + assists": "points",
        }
        return mapping.get(s)

    # ── Step 2: AI miss analysis ──────────────────────────────────────

    async def analyze_misses(self) -> dict[str, Any]:
        """Run AI analysis on unresolved (missed) picks to diagnose why."""
        entries = self._cache.get_learning_entries(limit=500)
        unanalyzed = [
            e for e in entries
            if e.get("hit") == 0 and e.get("resolved") == 0
        ]

        if not unanalyzed:
            return {"analyzed": 0, "message": "No unanalyzed misses found."}

        analyzed = 0
        errors = 0
        for entry in unanalyzed:
            prompt = (
                "A sports prop pick MISSED. Analyze WHY it missed and categorize the reason.\n\n"
                f"Player: {entry['player_name']}\n"
                f"Sport: {entry['sport']}\n"
                f"Pick: {entry['side'].upper()} {entry['line']} {entry['stat']}\n"
                f"Actual result: {entry.get('actual_value', '?')}\n"
                f"Model probability: {entry.get('model_prob', '?')}\n"
                f"Implied probability: {entry.get('implied_prob', '?')}\n"
                f"Edge: {entry.get('edge', '?')}\n"
                f"AI bias: {entry.get('ai_bias', '?')}\n"
                f"AI confidence: {entry.get('ai_confidence', '?')}\n\n"
                "CATEGORIES (pick exactly one):\n"
                "- injury_impact: a player's injury or teammate injury affected minutes/usage\n"
                "- matchup_mismatch: opponent defense was tougher/easier than model expected\n"
                "- outlier_performance: unusually good or bad game (random variance)\n"
                "- blowout_game: game was a blowout so starters sat / garbage time skewed stats\n"
                "- rest_schedule: back-to-back, rest, or minutes restriction affected output\n"
                "- line_too_aggressive: the line was set too high/low relative to realistic range\n"
                "- model_overconfidence: model edge was large but wrong (systematic bias)\n"
                "- trend_reversal: player was trending one direction but reverted\n\n"
                "Return ONLY valid JSON with keys:\n"
                '- "category" (string: one of the categories above)\n'
                '- "explanation" (string: 2-3 sentences explaining why this pick missed)\n'
                '- "model_lesson" (string: what the statistical model could learn from this)\n'
                '- "prompt_lesson" (string: what the AI prompt could be improved to catch this)\n'
            )

            miss_system = (
                "You are a sports betting analyst reviewing missed picks. "
                "Return ONLY valid JSON with keys: category, explanation, model_lesson, prompt_lesson."
            )

            try:
                result = await self._llm.analyze_prop(
                    prompt=prompt, timeout_s=45, system=miss_system
                )
            except Exception as exc:
                log.warning("AI miss analysis failed for %s: %s", entry["id"], exc)
                errors += 1
                continue

            category = result.get("category", "outlier_performance")
            explanation = result.get("explanation", "")
            model_lesson = result.get("model_lesson", "")
            prompt_lesson = result.get("prompt_lesson", "")

            entry["miss_category"] = category
            entry["miss_reason"] = json.dumps({
                "explanation": explanation,
                "model_lesson": model_lesson,
                "prompt_lesson": prompt_lesson,
            })
            entry["resolved"] = 1
            self._cache.save_learning_entry(entry)
            analyzed += 1

        return {"analyzed": analyzed, "total_unanalyzed": len(unanalyzed), "errors": errors}

    # ── Step 3: weekly report ─────────────────────────────────────────

    async def generate_weekly_report(self) -> dict[str, Any]:
        """Aggregate miss patterns and generate improvement suggestions."""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        week_start = week_ago.isoformat()
        week_end = now.isoformat()

        all_entries = self._cache.get_learning_entries(resolved_only=True, limit=500)
        week_entries = [
            e for e in all_entries
            if e.get("timestamp", "") >= week_start
        ]

        if not week_entries:
            all_resolved = self._cache.get_learning_entries(resolved_only=True, limit=500)
            if all_resolved:
                week_entries = all_resolved[:50]
                week_start = week_entries[-1].get("timestamp", week_start) if week_entries else week_start
            else:
                return {
                    "id": str(uuid.uuid4()),
                    "week_start": week_start,
                    "week_end": week_end,
                    "created_at": now.isoformat(),
                    "total_picks": 0,
                    "hits": 0,
                    "misses": 0,
                    "hit_rate": 0,
                    "miss_breakdown": {},
                    "suggestions": {
                        "stat_model": ["No data available yet — run predictions and wait for games to complete, then resolve outcomes."],
                        "ai_prompt": [],
                        "general_insights": [],
                        "biggest_blind_spot": "No picks resolved yet.",
                    },
                }

        total = len(week_entries)
        hits = sum(1 for e in week_entries if e.get("hit") == 1)
        misses = total - hits
        hit_rate = hits / total if total > 0 else 0

        miss_entries = [e for e in week_entries if e.get("hit") == 0]
        category_counts: Counter[str] = Counter()
        model_lessons: list[str] = []
        prompt_lessons: list[str] = []

        for e in miss_entries:
            cat = e.get("miss_category") or "unknown"
            category_counts[cat] += 1
            reason_raw = e.get("miss_reason")
            if isinstance(reason_raw, str):
                try:
                    reason = json.loads(reason_raw)
                    ml = reason.get("model_lesson", "")
                    pl = reason.get("prompt_lesson", "")
                    if ml:
                        model_lessons.append(ml)
                    if pl:
                        prompt_lessons.append(pl)
                except Exception:
                    pass

        miss_breakdown = dict(category_counts.most_common())

        prompt = (
            "You are reviewing a week of sports prop predictions to suggest improvements.\n\n"
            f"PERFORMANCE: {hits}/{total} hit ({hit_rate:.0%}), {misses} misses\n\n"
            f"MISS CATEGORIES:\n"
            + "\n".join(f"- {cat}: {cnt} misses" for cat, cnt in category_counts.most_common())
            + "\n\n"
            "INDIVIDUAL MISS LESSONS (from AI analysis of each miss):\n"
            "Model lessons:\n"
            + "\n".join(f"- {l}" for l in model_lessons[:15])
            + "\n\n"
            "Prompt lessons:\n"
            + "\n".join(f"- {l}" for l in prompt_lessons[:15])
            + "\n\n"
            "Based on the patterns above, generate specific, actionable suggestions.\n\n"
            "Return ONLY valid JSON with keys:\n"
            '- "stat_model_suggestions" (string[]: 3-5 specific changes to the statistical model)\n'
            '- "ai_prompt_suggestions" (string[]: 3-5 specific changes to the AI analysis prompt)\n'
            '- "general_insights" (string[]: 2-3 overall observations about prediction patterns)\n'
            '- "biggest_blind_spot" (string: the single most impactful thing we\'re missing)\n'
        )

        report_system = (
            "You are a sports prediction model improvement advisor. "
            "Return ONLY valid JSON with keys: stat_model_suggestions, ai_prompt_suggestions, "
            "general_insights, biggest_blind_spot."
        )

        try:
            ai_result = await self._llm.analyze_prop(
                prompt=prompt, timeout_s=60, system=report_system
            )
        except Exception:
            ai_result = {
                "stat_model_suggestions": ["Unable to generate — AI unavailable"],
                "ai_prompt_suggestions": [],
                "general_insights": [],
                "biggest_blind_spot": "N/A",
            }

        report = {
            "id": str(uuid.uuid4()),
            "week_start": week_start,
            "week_end": week_end,
            "created_at": now.isoformat(),
            "total_picks": total,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 4),
            "miss_breakdown": miss_breakdown,
            "miss_breakdown_json": json.dumps(miss_breakdown),
            "suggestions": {
                "stat_model": ai_result.get("stat_model_suggestions", []),
                "ai_prompt": ai_result.get("ai_prompt_suggestions", []),
                "general_insights": ai_result.get("general_insights", []),
                "biggest_blind_spot": ai_result.get("biggest_blind_spot", ""),
            },
            "suggestions_json": json.dumps({
                "stat_model": ai_result.get("stat_model_suggestions", []),
                "ai_prompt": ai_result.get("ai_prompt_suggestions", []),
                "general_insights": ai_result.get("general_insights", []),
                "biggest_blind_spot": ai_result.get("biggest_blind_spot", ""),
            }),
        }

        self._cache.save_learning_report(report)
        return report
