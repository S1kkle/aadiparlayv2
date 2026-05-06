"""
Statistical Model Calibration Script (CLI)

Standalone entry point for running calibration manually.
Uses the shared calibration engine from app.services.calibration.

Usage:
    cd backend && python -m scripts.calibrate_model

Or from repo root:
    python scripts/calibrate_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _setup_import_path() -> None:
    """Allow importing from backend/app when run from repo root or scripts/."""
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if backend_dir.exists() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))


_setup_import_path()

from app.services.calibration import (
    ModelConfig,
    generate_synthetic_data,
    grid_search,
    run_backtest,
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    log("=" * 60)
    log("PROP MODEL CALIBRATION (CLI)")
    log("Target: 65%+ prediction accuracy")
    log("Using synthetic NBA player distributions")
    log("=" * 60)

    log("\nGenerating player data...")
    player_data = generate_synthetic_data(n_games=25, n_copies=2)
    log(f"  {len(player_data)} profiles, ~{sum(sum(len(v) for v in p['stats'].values()) for p in player_data)} game values")

    log("\n=== Current defaults ===")
    dcfg = ModelConfig()
    dm = run_backtest(player_data, dcfg)
    log(f"  Accuracy: {dm['accuracy']:.2%} ({dm['total']} predictions)")

    best_cfg, best_m, _ = grid_search(player_data)

    elapsed = time.time() - t_start
    log("\n" + "=" * 60)
    log(f"CALIBRATION COMPLETE in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log("=" * 60)
    log(f"\nDefault accuracy:   {dm['accuracy']:.2%}")
    log(f"Optimized accuracy: {best_m['accuracy']:.2%} ({best_m['total']} predictions)")
    if best_m.get("high_conf_total", 0) > 0:
        log(f"High-confidence:    {best_m['high_conf_accuracy']:.2%} ({best_m['high_conf_total']})")
    log(f"\nOptimal parameters:")
    params = best_cfg.to_params_dict()
    for k, v in params.items():
        log(f"  {k:20s} = {v}")

    met = best_m["accuracy"] >= 0.65
    log(f"\n*** TARGET 65% {'ACHIEVED' if met else 'NOT MET'} ***")

    results = {
        "accuracy": best_m["accuracy"],
        "default_accuracy": dm["accuracy"],
        "total_predictions": best_m["total"],
        "params": params,
    }

    out_path = Path(__file__).parent / "calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
