"""Unified "what changed" log of model events.

Aggregates discrete events from every continuous-learning source into a
single time-ordered feed the UI can render as a changelog. Intentionally
read-only and side-effect free.

Event sources (and how they map to LogEntry kinds):

  - calibration_runs table         -> "calibration_run"
  - tier-model lineage cache       -> "tier_train"
  - learning_log (resolved + analyzed misses, grouped by day) ->
        "miss_discovery" / "resolution_batch"
  - learning_reports table         -> "weekly_report"

Each entry has a stable `id` so React can key on it efficiently and the
client can locally hide / pin individual events without re-fetching.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from app.db import SqliteTTLCache
from app.services.continuous_learning import get_tier_lineage

log = logging.getLogger(__name__)


def _safe_iso(s: Any) -> str:
    """Coerce arbitrary timestamp values to a stable ISO string. Returns
    empty string when nothing parseable is available — caller decides what
    to do with un-timestamped events."""
    if isinstance(s, str) and s.strip():
        return s.strip()
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).isoformat()
    return ""


def _calibration_event(row: dict[str, Any]) -> dict[str, Any]:
    applied = bool(row.get("applied"))
    real_acc = row.get("accuracy_real")
    synth_acc = row.get("accuracy_synthetic")
    brier = row.get("brier")
    log_loss = row.get("log_loss")
    ece = row.get("ece")
    isotonic = row.get("isotonic_json")
    has_iso = isinstance(isotonic, str) and len(isotonic) > 4

    kind_label = "Calibration adopted" if applied else "Calibration ran (not adopted)"
    summary_bits: list[str] = []
    if isinstance(synth_acc, (int, float)):
        summary_bits.append(f"synth acc {synth_acc * 100:.1f}%")
    if isinstance(real_acc, (int, float)):
        summary_bits.append(f"real acc {real_acc * 100:.1f}%")
    if isinstance(brier, (int, float)):
        summary_bits.append(f"Brier {brier:.4f}")
    if isinstance(log_loss, (int, float)):
        summary_bits.append(f"log-loss {log_loss:.4f}")
    if isinstance(ece, (int, float)):
        summary_bits.append(f"ECE {ece:.4f}")
    if has_iso:
        summary_bits.append("isotonic mapping refit")

    return {
        "id": f"calib:{row.get('id', '')}",
        "timestamp": _safe_iso(row.get("created_at")),
        "kind": "calibration_run",
        "status": "adopted" if applied else "skipped",
        "title": kind_label,
        "summary": " · ".join(summary_bits) if summary_bits else "",
        "details": {
            "source": row.get("source"),
            "total_real": row.get("total_real"),
            "total_synthetic": row.get("total_synthetic"),
            "params": row.get("params"),
            "metrics": {
                "real_accuracy": real_acc,
                "synthetic_accuracy": synth_acc,
                "brier": brier,
                "log_loss": log_loss,
                "ece": ece,
            },
            "isotonic_refit": has_iso,
        },
    }


def _tier_event(entry: dict[str, Any]) -> dict[str, Any]:
    status = (entry.get("status") or "info").strip()
    new_brier = entry.get("new_cv_brier")
    cur_brier = entry.get("current_cv_brier")
    delta = (
        new_brier - cur_brier
        if isinstance(new_brier, (int, float)) and isinstance(cur_brier, (int, float))
        else None
    )

    if status == "adopted":
        title = "Tier model upgraded"
        summary_parts: list[str] = []
        if isinstance(new_brier, (int, float)):
            summary_parts.append(f"CV-Brier {new_brier:.4f}")
        if delta is not None:
            arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "≈")
            summary_parts.append(f"{arrow} {abs(delta):.4f} vs prior")
        if isinstance(entry.get("new_since_last_train"), (int, float)):
            summary_parts.append(f"+{int(entry['new_since_last_train'])} new resolved")
        summary = " · ".join(summary_parts)
    elif status == "rejected_regression":
        title = "Tier-train rejected (worse than current)"
        summary = (
            f"new CV-Brier {new_brier:.4f} > current {cur_brier:.4f}"
            if isinstance(new_brier, (int, float)) and isinstance(cur_brier, (int, float))
            else "new model lost head-to-head"
        )
    elif status == "skipped_no_new_data":
        title = "Tier-train skipped (no new resolved data)"
        summary = (
            f"{entry.get('new_since_last_train', 0)} new since last train, "
            f"need {entry.get('min_new_resolved', '?')}"
        )
    elif status == "skipped_low_volume":
        title = "Tier-train skipped (insufficient resolved data)"
        summary = (
            f"have {entry.get('total_resolved', 0)} resolved, "
            f"need {entry.get('min_total', '?')}"
        )
    elif status == "train_failed":
        title = "Tier-train failed"
        summary = "fit returned no model — likely degenerate input"
    else:
        title = f"Tier-train {status}"
        summary = entry.get("error", "")

    return {
        "id": f"tier:{entry.get('timestamp', '')}",
        "timestamp": _safe_iso(entry.get("timestamp")),
        "kind": "tier_train",
        "status": "adopted" if status == "adopted" else (
            "rejected" if status == "rejected_regression" else "skipped"
        ),
        "title": title,
        "summary": summary,
        "details": entry,
    }


def _resolution_batch_events(
    entries: list[dict[str, Any]],
    *,
    bucket_hours: int = 6,
) -> list[dict[str, Any]]:
    """Group recently-resolved picks into time buckets so the feed shows
    "12 picks resolved (8 hits, 4 misses) — 6 hours ago" instead of
    flooding with a row per pick.

    Buckets are aligned to UTC hour boundaries; bucket size defaults to
    6 hours (matches the resolve loop cadence)."""
    out: list[dict[str, Any]] = []
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if e.get("hit") not in (0, 1):
            continue
        ts = e.get("timestamp")
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        bucket_id = int(dt.astimezone(timezone.utc).timestamp() // (bucket_hours * 3600))
        by_bucket[bucket_id].append(e)

    for bucket_id, group in by_bucket.items():
        hits = sum(1 for e in group if e.get("hit") == 1)
        misses = len(group) - hits
        # Anchor the event timestamp at the *most recent* pick in the
        # bucket so it sorts naturally with everything else in the feed.
        latest_ts = ""
        for e in group:
            if isinstance(e.get("timestamp"), str) and e["timestamp"] > latest_ts:
                latest_ts = e["timestamp"]
        sport_counter = Counter(e.get("sport", "?") for e in group)
        sport_summary = ", ".join(
            f"{sport} ×{cnt}" for sport, cnt in sport_counter.most_common(3)
        )
        rate = hits / max(1, len(group))
        out.append({
            "id": f"resolved-batch:{bucket_id}",
            "timestamp": latest_ts,
            "kind": "resolution_batch",
            "status": "info",
            "title": f"{len(group)} picks resolved",
            "summary": (
                f"{hits} hits, {misses} misses ({rate * 100:.1f}%) · {sport_summary}"
            ),
            "details": {
                "count": len(group),
                "hits": hits,
                "misses": misses,
                "hit_rate": round(rate, 4),
                "by_sport": dict(sport_counter),
                "sample_picks": [
                    {
                        "player_name": e.get("player_name"),
                        "sport": e.get("sport"),
                        "stat": e.get("stat"),
                        "side": e.get("side"),
                        "line": e.get("line"),
                        "actual_value": e.get("actual_value"),
                        "hit": e.get("hit"),
                    }
                    for e in group[:10]
                ],
            },
        })
    return out


def _miss_discovery_events(
    entries: list[dict[str, Any]],
    *,
    bucket_hours: int = 24,
) -> list[dict[str, Any]]:
    """Group analyzed misses into daily discovery batches showing which
    failure modes the LLM diagnosed. This is the closest thing to "the
    model learned X" we have — patterns of misses concentrated by
    category."""
    out: list[dict[str, Any]] = []
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if e.get("hit") != 0:
            continue
        if not e.get("miss_category"):
            continue
        ts = e.get("timestamp")
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        bucket_id = int(dt.astimezone(timezone.utc).timestamp() // (bucket_hours * 3600))
        by_bucket[bucket_id].append(e)

    for bucket_id, group in by_bucket.items():
        cats = Counter(e.get("miss_category", "unknown") for e in group)
        latest_ts = ""
        for e in group:
            if isinstance(e.get("timestamp"), str) and e["timestamp"] > latest_ts:
                latest_ts = e["timestamp"]
        top_cat, top_n = cats.most_common(1)[0]
        cat_summary = ", ".join(
            f"{cat.replace('_', ' ')} ×{cnt}" for cat, cnt in cats.most_common(5)
        )
        # Pull up to 3 representative "model_lesson" sentences when present.
        import json as _json
        lessons: list[str] = []
        for e in group[:20]:
            raw = e.get("miss_reason")
            if not isinstance(raw, str):
                continue
            try:
                parsed = _json.loads(raw)
            except (ValueError, TypeError):
                continue
            lesson = (parsed or {}).get("model_lesson")
            if isinstance(lesson, str) and lesson.strip():
                lessons.append(lesson.strip())
            if len(lessons) >= 3:
                break

        out.append({
            "id": f"miss-discovery:{bucket_id}",
            "timestamp": latest_ts,
            "kind": "miss_discovery",
            "status": "info",
            "title": f"{len(group)} misses categorized — top: {top_cat.replace('_', ' ')}",
            "summary": cat_summary,
            "details": {
                "count": len(group),
                "categories": dict(cats),
                "lessons": lessons,
                "sample_misses": [
                    {
                        "player_name": e.get("player_name"),
                        "sport": e.get("sport"),
                        "stat": e.get("stat"),
                        "side": e.get("side"),
                        "line": e.get("line"),
                        "actual_value": e.get("actual_value"),
                        "miss_category": e.get("miss_category"),
                    }
                    for e in group[:8]
                ],
            },
        })
    return out


def _weekly_report_event(report: dict[str, Any]) -> dict[str, Any]:
    suggestions = report.get("suggestions") or {}
    if not isinstance(suggestions, dict):
        suggestions = {}
    return {
        "id": f"weekly:{report.get('id', '')}",
        "timestamp": _safe_iso(report.get("created_at")),
        "kind": "weekly_report",
        "status": "info",
        "title": "Weekly report generated",
        "summary": (
            f"{report.get('total_picks', 0)} picks, "
            f"{(report.get('hit_rate', 0) or 0) * 100:.1f}% hit rate"
        ),
        "details": {
            "week_start": report.get("week_start"),
            "week_end": report.get("week_end"),
            "total_picks": report.get("total_picks"),
            "hits": report.get("hits"),
            "misses": report.get("misses"),
            "miss_breakdown": report.get("miss_breakdown"),
            "biggest_blind_spot": suggestions.get("biggest_blind_spot"),
            "stat_model_suggestions": suggestions.get("stat_model"),
            "ai_prompt_suggestions": suggestions.get("ai_prompt"),
            "general_insights": suggestions.get("general_insights"),
        },
    }


def build_learning_log(
    *,
    cache: SqliteTTLCache,
    limit: int = 60,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate every event source into a single time-ordered feed.

    Returns:
        {
          "entries": [LogEntry, ...],   # newest first, sorted by timestamp
          "totals": {"calibration_run": int, ...},
          "generated_at": iso str,
        }
    """
    accept_kinds = set(kinds) if kinds else None
    entries: list[dict[str, Any]] = []

    # Calibration runs (last 50 — gives ~3 months of weekly cadence).
    try:
        calib_history = cache.get_calibration_history(limit=50)
        for row in calib_history:
            ev = _calibration_event(row)
            if ev["timestamp"] and (accept_kinds is None or ev["kind"] in accept_kinds):
                entries.append(ev)
    except Exception:
        log.exception("learning_log: calibration history failed")

    # Tier model lineage.
    try:
        for entry in get_tier_lineage(cache, limit=50):
            ev = _tier_event(entry)
            if ev["timestamp"] and (accept_kinds is None or ev["kind"] in accept_kinds):
                entries.append(ev)
    except Exception:
        log.exception("learning_log: tier lineage failed")

    # Resolved-pick batches + miss-discovery batches share one DB read.
    try:
        learning_entries = cache.get_learning_entries(resolved_only=True, limit=2000)
    except Exception:
        log.exception("learning_log: learning entries failed")
        learning_entries = []

    if not accept_kinds or "resolution_batch" in accept_kinds:
        try:
            entries.extend(_resolution_batch_events(learning_entries))
        except Exception:
            log.exception("learning_log: resolution batches failed")
    if not accept_kinds or "miss_discovery" in accept_kinds:
        try:
            entries.extend(_miss_discovery_events(learning_entries))
        except Exception:
            log.exception("learning_log: miss discoveries failed")

    # Weekly reports.
    try:
        for r in cache.get_learning_reports(limit=20):
            ev = _weekly_report_event(r)
            if ev["timestamp"] and (accept_kinds is None or ev["kind"] in accept_kinds):
                entries.append(ev)
    except Exception:
        log.exception("learning_log: reports failed")

    entries.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    truncated = entries[: max(1, int(limit))]

    totals = Counter(e["kind"] for e in entries)

    return {
        "entries": truncated,
        "totals": dict(totals),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
