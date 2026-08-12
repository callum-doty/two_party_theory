#!/usr/bin/env python3
"""
Decomposes the double-oracle mixed equilibrium's ~5-11-portfolio support
into per-race statistics, so the mixture reads as "which races does
strategic randomization actually touch" rather than a probability
distribution over 433-dimensional vectors.

For each race i and each side, across that side's support portfolios
{portfolio_j} with mixture weights {p_j}:

    E[w_i]   = sum_j p_j * portfolio_j[i]
    Var[w_i] = sum_j p_j * (portfolio_j[i] - E[w_i])^2
    CV[w_i]  = sqrt(Var[w_i]) / E[w_i]      (coefficient of variation)

Races are then bucketed into three categories per side:

  - "core":       E[w_i] >= materiality threshold, CV in the LOWER half of
                  materially-funded races -- nearly every support portfolio
                  funds this race about the same amount. The committee
                  should probably just always fund these.
  - "swing":      E[w_i] >= materiality threshold, CV in the UPPER half --
                  whether/how much this race gets funded depends heavily on
                  which equilibrium portfolio gets drawn. This is where
                  strategic randomization actually happens.
  - "irrelevant": E[w_i] below the materiality threshold (1% of that side's
                  per-race cap) -- essentially never funded across the
                  support.

Materiality threshold and the core/swing median-CV split are both
data-driven (stated explicitly in the output), not tuned magic numbers --
see the module-level MATERIALITY_FRAC constant if that judgment call needs
revisiting.

Requires a completed double-oracle solve (scripts/double_oracle.py) for the
requested cycle -- game/double_oracle.py::load_solved finds it (preferring
a `_resumed` run if present, e.g. 2022's converged extended run).

Usage:
    python scripts/equilibrium_support_composition.py --cycle 2024
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from build_cycle_state import build_cycle_state  # noqa: E402
from game import double_oracle as do  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("equilibrium_support_composition")

MATERIALITY_FRAC = 0.01  # fraction of a side's per-race cap; below this, a race is "irrelevant"


def _mean_var(pool: list[np.ndarray], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(pool, axis=0)  # (n_portfolios, n_races)
    mean = np.average(stacked, axis=0, weights=weights)
    var = np.average((stacked - mean) ** 2, axis=0, weights=weights)
    return mean, var


def _classify(mean: np.ndarray, cv: np.ndarray, cap: float) -> tuple[np.ndarray, float]:
    material = mean >= MATERIALITY_FRAC * cap
    median_cv = float(np.median(cv[material])) if material.any() else float("nan")
    category = np.full(len(mean), "irrelevant", dtype=object)
    category[material & (cv <= median_cv)] = "core"
    category[material & (cv > median_cv)] = "swing"
    return category, median_cv


def main() -> None:
    parser = argparse.ArgumentParser(description="Equilibrium-support race decomposition (core/swing/irrelevant)")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=10, help="how many races to list per category in the summary")
    args = parser.parse_args()

    results_dir = REPO_ROOT / "results"
    solved = do.load_solved(results_dir, args.cycle)
    if solved is None:
        logger.error(f"No double-oracle solve found for cycle {args.cycle} -- "
                      f"run scripts/double_oracle.py --cycle {args.cycle} first.")
        sys.exit(1)
    logger.info(f"Loaded equilibrium from {solved['source_file']} "
                f"(converged={solved['converged']}, D support size={int((solved['p'] > 0).sum())}, "
                f"R support size={int((solved['q'] > 0).sum())})")

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races = state["races"]
    cap_d, cap_r = state["cap_d"], state["cap_r"]
    n = len(races)
    if any(len(dp) != n for dp in solved["d_pool"]) or any(len(rp) != n for rp in solved["r_pool"]):
        logger.error("Saved portfolio length doesn't match this cycle's current race universe -- "
                      "the double-oracle solve may predate a universe-building change. Re-run "
                      "scripts/double_oracle.py before trusting this output.")
        sys.exit(1)

    mean_d, var_d = _mean_var(solved["d_pool"], solved["p"])
    mean_r, var_r = _mean_var(solved["r_pool"], solved["q"])
    sd_d, sd_r = np.sqrt(var_d), np.sqrt(var_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_d = np.where(mean_d > 0, sd_d / np.where(mean_d > 0, mean_d, 1.0), np.nan)
        cv_r = np.where(mean_r > 0, sd_r / np.where(mean_r > 0, mean_r, 1.0), np.nan)

    cat_d, median_cv_d = _classify(mean_d, cv_d, cap_d)
    cat_r, median_cv_r = _classify(mean_r, cv_r, cap_r)

    counts_d = {c: int((cat_d == c).sum()) for c in ("core", "swing", "irrelevant")}
    counts_r = {c: int((cat_r == c).sum()) for c in ("core", "swing", "irrelevant")}
    logger.info(f"D-side: {counts_d} (materiality >= {MATERIALITY_FRAC:.0%} of cap=${cap_d/1e6:.1f}M, "
                f"median CV among funded={median_cv_d:.3f})")
    logger.info(f"R-side: {counts_r} (materiality >= {MATERIALITY_FRAC:.0%} of cap=${cap_r/1e6:.1f}M, "
                f"median CV among funded={median_cv_r:.3f})")

    # Per-race CSV -- the full table, for anyone who wants to look up a specific district.
    csv_path = results_dir / f"equilibrium_support_composition_{args.cycle}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["district_id", "state", "cook_rating", "pvi",
                    "mean_d", "sd_d", "cv_d", "category_d",
                    "mean_r", "sd_r", "cv_r", "category_r"])
        for i, r in enumerate(races):
            w.writerow([r.district_id, r.state, r.cook_rating, r.pvi,
                        f"{mean_d[i]:.2f}", f"{sd_d[i]:.2f}",
                        "" if np.isnan(cv_d[i]) else f"{cv_d[i]:.4f}", cat_d[i],
                        f"{mean_r[i]:.2f}", f"{sd_r[i]:.2f}",
                        "" if np.isnan(cv_r[i]) else f"{cv_r[i]:.4f}", cat_r[i]])
    logger.info(f"Saved per-race table -> {csv_path}")

    def _top(mean, cv, category, cat_name, by, n_top):
        idx = np.where(category == cat_name)[0]
        if len(idx) == 0:
            return []
        # CV is scale-free: a race funded by exactly ONE support portfolio (zero in the
        # rest) has CV = sqrt((1-w)/w), a function of that single portfolio's mixture
        # weight w ALONE -- unrelated to the dollar amount. With few support portfolios
        # (2024's 5), several races legitimately tie on CV this way. Break ties by mean
        # (secondary key) so the ordering surfaces the larger-dollar swing races first,
        # rather than leaving tie order to argsort's index-position default.
        if by == "cv":
            key = np.nan_to_num(cv[idx]) + 1e-9 * np.nan_to_num(mean[idx]) / (mean[idx].max() or 1.0)
        else:
            key = mean[idx]
        order = idx[np.argsort(-key)][:n_top]
        return [{"district_id": races[i].district_id, "cook_rating": races[i].cook_rating,
                  "mean": float(mean[i]), "cv": None if np.isnan(cv[i]) else float(cv[i])}
                for i in order]

    summary = {
        "cycle": args.cycle,
        "config": {"materiality_frac": MATERIALITY_FRAC,
                   "cap_fraction_d": args.cap_fraction_d, "cap_fraction_r": args.cap_fraction_r},
        "equilibrium_source": solved["source_file"],
        "d_side": {
            "counts": counts_d, "cap": cap_d, "median_cv_among_funded": median_cv_d,
            "top_core_by_mean": _top(mean_d, cv_d, cat_d, "core", "mean", args.top_n),
            "top_swing_by_cv": _top(mean_d, cv_d, cat_d, "swing", "cv", args.top_n),
        },
        "r_side": {
            "counts": counts_r, "cap": cap_r, "median_cv_among_funded": median_cv_r,
            "top_core_by_mean": _top(mean_r, cv_r, cat_r, "core", "mean", args.top_n),
            "top_swing_by_cv": _top(mean_r, cv_r, cat_r, "swing", "cv", args.top_n),
        },
    }
    json_path = results_dir / f"equilibrium_support_composition_{args.cycle}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"Saved summary -> {json_path}")

    logger.info(f"D-side top swing races (highest CV among materially-funded): "
                f"{[r['district_id'] for r in summary['d_side']['top_swing_by_cv'][:5]]}")
    logger.info(f"R-side top swing races (highest CV among materially-funded): "
                f"{[r['district_id'] for r in summary['r_side']['top_swing_by_cv'][:5]]}")


if __name__ == "__main__":
    main()
