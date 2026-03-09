"""
Learning service: resolves pick outcomes from ESPN, diagnoses misses via AI,
and generates weekly improvement reports.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from app.clients.espn import EspnClient
from app.clients.groq import GroqClient
from app.clients.ollama import OllamaClient
from app.db import SqliteTTLCache


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
        """Look up actual results for past picks that haven't been resolved yet."""
        history = self._cache.get_history(limit=50)
        resolved_count = 0
        already_done = 0
        failed = 0

        for entry in history:
            hid = entry["id"]
            ts = entry["timestamp"]
            props = entry.get("props", [])
            if not isinstance(props, list):
                continue

            for p in props:
                if not isinstance(p, dict):
                    continue
                lid = f"{hid}:{p.get('underdog_option_id', '')}"

                existing = self._cache.get_learning_entries(limit=1000)
                if any(e["id"] == lid for e in existing):
                    already_done += 1
                    continue

                scheduled = p.get("scheduled_at")
                if scheduled:
                    try:
                        game_time = datetime.fromisoformat(str(scheduled).replace("Z", "+00:00"))
                        if game_time > datetime.now(timezone.utc) - timedelta(hours=2):
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
                    "resolved": 1 if hit else 0,
                })
                resolved_count += 1

        return {
            "resolved": resolved_count,
            "already_done": already_done,
            "failed_lookup": failed,
        }

    async def _lookup_actual_value(self, prop: dict[str, Any]) -> float | None:
        """Fetch the player's actual stat value for the game from ESPN."""
        sport = prop.get("sport", "")
        athlete_id = prop.get("espn_athlete_id")
        stat_field = prop.get("stat_field")
        scheduled_at = prop.get("scheduled_at")

        if not athlete_id or not stat_field:
            return None

        sl = self._espn.sport_league_for_scope(sport)
        if sl is None:
            return None
        sport_slug, league_slug = sl

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
                    if gd and abs((gd - target).total_seconds()) < 24 * 3600:
                        return float(g["value"])
            except Exception:
                pass

        return float(lines[0]["value"]) if lines else None

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
            except Exception:
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

        return {"analyzed": analyzed, "total_unanalyzed": len(unanalyzed)}

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
            return {"error": "No resolved picks in the last 7 days."}

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

        suggestions = (
            ai_result.get("stat_model_suggestions", [])
            + ai_result.get("ai_prompt_suggestions", [])
        )

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
