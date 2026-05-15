"""Rolling-window empirical league priors.

The static `_PRIORS` dict in `stat_model.py` reflects a one-time snapshot of
league-average production. NBA pace, 3PT rate, and positional usage drift
5-10% over a season; a quarterly calibration cycle is too slow to keep up.

This module recomputes per-(sport, stat_field, position) rolling-30d means
from resolved `learning_log` entries (their `actual_value` column) and pushes
the result into `stat_model.set_dynamic_priors()`. The refresher runs once
nightly via `_dynamic_priors_refresh_loop` in main.py, plus once at startup
so a freshly-restarted server doesn't fall back to stale hardcoded priors.

Partial-pooled with the hardcoded baseline by weight n/(n+k_pool) so small
samples don't yank the prior around.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import SqliteTTLCache
from app.services.stat_model import (
    _PRIORS as STATIC_PRIORS,
    league_prior_mean,
    set_dynamic_priors,
)

log = logging.getLogger(__name__)


_CACHE_KEY = "priors:dynamic:v1"
_TTL_SECONDS = 14 * 24 * 3600  # 14 days; refresh job rewrites nightly


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def compute_dynamic_priors(
    entries: list[dict[str, Any]],
    *,
    window_days: int = 30,
    pool_k: float = 5.0,
    min_samples: int = 8,
    now: datetime | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute rolling-window empirical priors from resolved learning entries.

    For each (sport, stat_field, position) bucket with >= `min_samples`
    actual-value observations within `window_days`, blend the empirical
    mean with the static baseline using weight n / (n + pool_k). Position
    "*" is always included as the unconditional bucket.
    """
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=window_days)

    # bucket key: (sport_upper, stat_field, position_upper_or_star)
    buckets: dict[tuple[str, str, str], list[float]] = {}

    for e in entries:
        ts = _parse_iso(e.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        actual = e.get("actual_value")
        if not isinstance(actual, (int, float)):
            continue
        sport = (e.get("sport") or "").upper()
        stat_field = e.get("stat_field") or ""
        if not sport or not stat_field:
            continue
        pos_raw = (e.get("position") or "").strip().upper()
        pos_keys = ["*"]
        if pos_raw:
            pos_keys.append(pos_raw)
        for pk in pos_keys:
            buckets.setdefault((sport, stat_field, pk), []).append(float(actual))

    overlay: dict[str, dict[str, dict[str, float]]] = {}
    for (sport, stat, pos), values in buckets.items():
        if len(values) < min_samples:
            continue
        empirical = sum(values) / len(values)
        baseline = league_prior_mean(
            sport=sport, stat_field=stat,
            position=None if pos == "*" else pos,
        )
        if baseline is None:
            baseline = empirical  # no static fallback → trust empirical
        n = len(values)
        weight = n / (n + pool_k)
        blended = weight * empirical + (1.0 - weight) * baseline
        overlay.setdefault(sport, {}).setdefault(stat, {})[pos] = round(blended, 4)

    return overlay


def refresh_dynamic_priors(cache: SqliteTTLCache) -> dict[str, Any]:
    """Recompute, persist, and hot-load the dynamic-priors overlay.

    Returns a small structured dict so the background loop can log volume.
    """
    try:
        entries = cache.get_learning_entries(resolved_only=True, limit=100000)
    except Exception as exc:
        log.exception("dynamic_priors: failed to load learning entries")
        return {"status": "error", "error": str(exc)}

    overlay = compute_dynamic_priors(entries)
    set_dynamic_priors(overlay)

    n_buckets = sum(len(stat_map) for sport in overlay.values() for stat_map in sport.values())
    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_resolved_entries": len(entries),
        "n_buckets": n_buckets,
        "overlay": overlay,
    }
    try:
        cache.set_json(_CACHE_KEY, payload, ttl_seconds=_TTL_SECONDS)
    except Exception:
        log.exception("dynamic_priors: failed to persist overlay")

    if n_buckets:
        log.info(
            "Dynamic priors refreshed: %d buckets across %d sports (n_resolved=%d)",
            n_buckets,
            len(overlay),
            len(entries),
        )
    return {"status": "ok", "n_buckets": n_buckets, "n_resolved": len(entries)}


def load_persisted_dynamic_priors(cache: SqliteTTLCache) -> bool:
    """Hot-load a previously-computed overlay from the cache at startup."""
    try:
        payload = cache.get_json(_CACHE_KEY)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    overlay = payload.get("overlay")
    if not isinstance(overlay, dict):
        return False
    set_dynamic_priors(overlay)
    log.info(
        "Loaded persisted dynamic priors: %d buckets (computed_at=%s)",
        payload.get("n_buckets", 0),
        payload.get("computed_at"),
    )
    return True


# Keep ``STATIC_PRIORS`` exported for downstream introspection / debug
# endpoints — re-exporting here makes the module the canonical entry point.
__all__ = [
    "STATIC_PRIORS",
    "compute_dynamic_priors",
    "refresh_dynamic_priors",
    "load_persisted_dynamic_priors",
]
