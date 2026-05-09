"""Continuous learning service.

Coordinates the *recurring* learning operations that, together, ensure the
model never stops improving:

1. **Auto-resolve outcomes** — pull final stat values from ESPN for picks
   whose games have completed. Without this, `learning_log` stops growing
   and every downstream step (calibration, tier training, miss analysis)
   starves.
2. **Auto-analyze misses** — feed unresolved-reason misses through the LLM
   so weekly reports / future calibration can stratify by failure mode.
3. **Auto-train the confidence-tier logistic** with a *no-regression* gate
   — only adopt the new tier model when its out-of-fold (CV) Brier score
   is at least as good as the currently-deployed one. Worse fits get
   logged to a lineage history but **not** activated.
4. **Persist lineage** — every train attempt (adopted or rejected) is
   stored in cache so the operator can see the trajectory of the model
   over time and roll back manually if needed.

All operations are best-effort: failures log and return a structured dict
rather than raising, so the calling background loop in `main.py` can keep
ticking forever without ever crashing the server.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.db import SqliteTTLCache

log = logging.getLogger(__name__)


# ── Cache keys (local module constants — no schema migrations needed) ──

# Number of resolved learning_log rows that existed the last time we
# successfully **trained** (regardless of adoption) the tier model. We
# only retrain when total resolved has grown by >= TIER_TRAIN_MIN_NEW.
KEY_LAST_TRAIN_RESOLVED_N = "continuous_learning:tier:last_train_resolved_n"

# Lineage of recent tier-train attempts (adopted + rejected). Capped to
# 30 entries — newest at the head. Used purely for visibility / audit.
KEY_TIER_LINEAGE = "continuous_learning:tier:lineage"

# The currently-active tier model snapshot. Mirrors what's in
# `tier_model:v1` but written here too with extra metadata so the
# rollback / regression-guard logic doesn't have to re-derive it.
KEY_ACTIVE_TIER = "continuous_learning:tier:active"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_int(cache: SqliteTTLCache, key: str, default: int = 0) -> int:
    raw = cache.get_json(key)
    if isinstance(raw, dict) and isinstance(raw.get("v"), (int, float)):
        return int(raw["v"])
    return default


def _write_int(cache: SqliteTTLCache, key: str, value: int) -> None:
    # Effectively-permanent TTL — stat-tracking values must survive
    # restarts and cache TTL sweeps.
    cache.set_json(key, {"v": int(value)}, ttl_seconds=10 * 365 * 24 * 3600)


def _read_lineage(cache: SqliteTTLCache) -> list[dict[str, Any]]:
    raw = cache.get_json(KEY_TIER_LINEAGE)
    if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        return [e for e in raw["entries"] if isinstance(e, dict)]
    return []


def _append_lineage(cache: SqliteTTLCache, entry: dict[str, Any], *, max_keep: int = 30) -> None:
    existing = _read_lineage(cache)
    existing.insert(0, entry)
    cache.set_json(
        KEY_TIER_LINEAGE,
        {"entries": existing[:max_keep]},
        ttl_seconds=10 * 365 * 24 * 3600,
    )


def _current_active_metrics(cache: SqliteTTLCache) -> dict[str, Any] | None:
    """Return the metrics dict of the currently-deployed tier model, or
    None if no model is deployed. Reads from the canonical `tier_model:v1`
    cache key written by main.py / fit_tier_logistic so we always compare
    against the live model rather than a stale local snapshot.
    """
    raw = cache.get_json("tier_model:v1")
    if not isinstance(raw, dict):
        return None
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def _cv_brier(metrics: dict[str, Any] | None) -> float | None:
    """Pull CV-Brier from a tier-model metrics dict. We treat CV-Brier as
    the canonical regression-guard metric because it's the most honest
    out-of-sample signal `fit_tier_logistic` produces.
    """
    if not isinstance(metrics, dict):
        return None
    cv = metrics.get("cv")
    if isinstance(cv, dict) and isinstance(cv.get("brier"), (int, float)):
        return float(cv["brier"])
    # Fall back to in-sample brier when CV not present (small datasets).
    if isinstance(metrics.get("brier"), (int, float)):
        return float(metrics["brier"])
    return None


def maybe_train_tier_model(
    *,
    cache: SqliteTTLCache,
    min_new_resolved: int,
    min_total: int,
    force: bool = False,
) -> dict[str, Any]:
    """Threshold-gated tier-model retrain with no-regression adoption.

    Returns a structured dict suitable for logging / API response. The dict
    always contains a `status` field — one of:

      - `skipped_low_volume`   — < `min_total` resolved overall
      - `skipped_no_new_data`  — total grew < `min_new_resolved` since last train
      - `train_failed`         — fit returned None (degenerate input)
      - `adopted`              — new model deployed (CV-Brier ≤ current)
      - `rejected_regression`  — new model trained but CV-Brier worse than current
      - `error`                — unexpected exception
    """
    from app.services.stat_model import (
        fit_tier_logistic,
        _tier_features_from_dict,
        set_tier_model,
    )

    try:
        entries = cache.get_learning_entries(resolved_only=True, limit=100000)
    except Exception as exc:
        log.exception("continuous_learning: failed to load learning entries")
        return {"status": "error", "error": str(exc)}

    rows: list[dict[str, Any]] = []
    for e in entries:
        if e.get("hit") not in (0, 1):
            continue
        try:
            feats = _tier_features_from_dict(e)
        except Exception:
            continue
        rows.append({"features": feats, "hit": int(e["hit"])})

    total_resolved = len(rows)
    last_seen = _read_int(cache, KEY_LAST_TRAIN_RESOLVED_N, 0)
    new_since = max(0, total_resolved - last_seen)

    if total_resolved < min_total:
        return {
            "status": "skipped_low_volume",
            "total_resolved": total_resolved,
            "min_total": min_total,
        }

    if not force and new_since < min_new_resolved:
        return {
            "status": "skipped_no_new_data",
            "total_resolved": total_resolved,
            "new_since_last_train": new_since,
            "min_new_resolved": min_new_resolved,
        }

    try:
        result = fit_tier_logistic(rows)
    except Exception as exc:
        log.exception("continuous_learning: fit_tier_logistic raised")
        return {"status": "error", "error": str(exc)}
    if result is None:
        return {"status": "train_failed", "total_resolved": total_resolved}

    new_metrics = result.get("metrics") or {}
    new_brier = _cv_brier(new_metrics)
    current_metrics = _current_active_metrics(cache)
    current_brier = _cv_brier(current_metrics)

    # The guard: if we *had* a model and the new one is worse OOS, reject.
    # Rolling back to current is automatic because we never overwrote
    # `tier_model:v1` until this branch.
    if (
        current_brier is not None
        and new_brier is not None
        and new_brier > current_brier + 1e-6
    ):
        # `fit_tier_logistic` already mutated the in-process module-level
        # tier weights via `set_tier_model`. Restore the previously-active
        # ones from cache so the regression we just trained doesn't
        # actually go live for the rest of this process.
        cached = cache.get_json("tier_model:v1")
        if isinstance(cached, dict) and isinstance(cached.get("weights"), list):
            try:
                set_tier_model(cached["weights"], metrics=cached.get("metrics") or {})
            except Exception:
                pass
        outcome = {
            "status": "rejected_regression",
            "total_resolved": total_resolved,
            "new_since_last_train": new_since,
            "new_cv_brier": new_brier,
            "current_cv_brier": current_brier,
            "delta": new_brier - current_brier,
        }
        _append_lineage(cache, {"timestamp": _now_iso(), **outcome})
        # IMPORTANT: still bump the high-water mark so we don't loop on
        # the same rejected dataset every cycle.
        _write_int(cache, KEY_LAST_TRAIN_RESOLVED_N, total_resolved)
        return outcome

    # Accept the new model: write to `tier_model:v1` for restart-safety
    # and bump the resolved-count high-water mark.
    cache.set_json(
        "tier_model:v1",
        {
            "weights": result["weights"],
            "feature_names": result["feature_names"],
            "metrics": new_metrics,
        },
        ttl_seconds=10 * 365 * 24 * 3600,
    )
    cache.set_json(
        KEY_ACTIVE_TIER,
        {
            "trained_at": _now_iso(),
            "total_resolved": total_resolved,
            "metrics": new_metrics,
        },
        ttl_seconds=10 * 365 * 24 * 3600,
    )
    _write_int(cache, KEY_LAST_TRAIN_RESOLVED_N, total_resolved)
    outcome = {
        "status": "adopted",
        "total_resolved": total_resolved,
        "new_since_last_train": new_since,
        "new_cv_brier": new_brier,
        "current_cv_brier": current_brier,
        "metrics": new_metrics,
    }
    _append_lineage(cache, {"timestamp": _now_iso(), **outcome})
    return outcome


def get_tier_lineage(cache: SqliteTTLCache, limit: int = 30) -> list[dict[str, Any]]:
    return _read_lineage(cache)[: max(1, int(limit))]
