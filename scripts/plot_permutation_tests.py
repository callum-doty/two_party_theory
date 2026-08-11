#!/usr/bin/env python3
"""
Plot the two permutation-test null distributions against the real observed
values (docs/paper1_draft.md Section 9.2, docs/full_derivation.md Part III
Section III.21).

Left panel: the Spearman rho permutation test. 2000 random reassignments of
DCCC's observed spending across the 53 competitive races, breaking any link
to MSG, vs. the real rho = -0.582. None of the 2000 shuffles are as extreme.

Right panel: the allocation-efficiency permutation test -- the stronger,
assumption-lighter check. 2000 random reshuffles of DCCC's own dollars
across the same races, vs. DCCC's actual E[Seats] and the model optimizer's
E[Seats]. DCCC sits at the very bottom of its own null distribution; the
model sits at (or past) the very top.

Reads outputs/permutation_null_spearman{suffix}.csv, outputs/permutation_null_allocation{suffix}.csv,
and outputs/permutation_tests{suffix}.json, all written by scripts/run_backtest.py.
Run that script first if any is missing.

Usage:
    python scripts/plot_permutation_tests.py                # 2024 (default)
    python scripts/plot_permutation_tests.py --cycle 2022    # 2022 OOS

Output: outputs/permutation_tests_null_distributions{suffix}.png
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

_COLOR_NULL = "#7f7f7f"
_COLOR_OBS = "#1f4e9c"
_COLOR_MODEL = "#1a7a3c"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot permutation-test null distributions")
    parser.add_argument("--cycle", type=int, default=2024,
                        help="Election cycle to plot (default: 2024). Use 2022 for the OOS run.")
    args = parser.parse_args()
    suffix = f"_{args.cycle}" if args.cycle != 2024 else ""

    outputs = config.outputs_path()

    rho_path = outputs / f"permutation_null_spearman{suffix}.csv"
    alloc_path = outputs / f"permutation_null_allocation{suffix}.csv"
    summary_path = outputs / f"permutation_tests{suffix}.json"
    for p in (rho_path, alloc_path, summary_path):
        if not p.exists():
            sys.exit(f"{p} not found. Run `python scripts/run_backtest.py --cycle {args.cycle}"
                      f"{' --processed-dir data/processed_oos_2020' if args.cycle == 2022 else ''}` first.")

    null_rhos = pd.read_csv(rho_path)["null_rho"].to_numpy()
    null_seats = pd.read_csv(alloc_path)["null_expected_seats"].to_numpy()
    with open(summary_path) as f:
        summary = json.load(f)
    sp = summary["spearman_efficiency"]
    al = summary["allocation_efficiency"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel 1: Spearman rho permutation null ──────────────────────────────
    ax1.hist(null_rhos, bins=40, color=_COLOR_NULL, alpha=0.7,
              edgecolor="white", linewidth=0.4,
              label=f"Null: DCCC spending randomly\nreassigned (n={sp['n_permutations']:,})")
    ax1.axvline(sp["rho"], color=_COLOR_OBS, linewidth=2.2,
                label=f"Observed ρ = {sp['rho']:.3f}")
    ax1.axvline(-sp["rho"], color=_COLOR_OBS, linewidth=1.2, linestyle=":",
                alpha=0.6, label="Mirror (|ρ| threshold)")
    ax1.set_xlabel("Spearman ρ (DCCC spending vs. MSG)")
    ax1.set_ylabel("Count")
    n_null_extreme = int(np.sum(np.abs(null_rhos) >= abs(sp["rho"])))
    ax1.set_title(
        f"{n_null_extreme} of {sp['n_permutations']:,} shuffles reached |ρ| ≥ {abs(sp['rho']):.3f}\n"
        f"(permutation p = {sp['p_value_permutation']:.3g}, "
        f"asymptotic p = {sp['p_value_asymptotic']:.2g})"
    )
    ax1.legend(loc="upper left", fontsize=8, frameon=True)

    # ── Panel 2: allocation-efficiency permutation null ─────────────────────
    ax2.hist(null_seats, bins=40, color=_COLOR_NULL, alpha=0.7,
              edgecolor="white", linewidth=0.4,
              label=f"Null: DCCC's own dollars\nrandomly reshuffled (n={al['n_permutations']:,})")
    ax2.axvline(al["dccc_expected_seats"], color=_COLOR_OBS, linewidth=2.2,
                label=f"DCCC actual = {al['dccc_expected_seats']:.2f}")
    ax2.axvline(al["model_expected_seats"], color=_COLOR_MODEL, linewidth=2.2,
                label=f"Model optimizer = {al['model_expected_seats']:.2f}")
    ax2.set_xlabel("E[Seats] under the shuffled allocation")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"P(shuffle ≥ DCCC) = {al['p_value_dccc_below_null']:.0%}   "
        f"P(shuffle ≥ model) = {al['p_value_model_exceeds_null']:.0%}"
    )
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    cycle_label = f"{args.cycle} OOS" if args.cycle == 2022 else str(args.cycle)
    fig.suptitle(
        f"Permutation tests ({cycle_label}): DCCC's actual allocation vs. "
        f"2,000 random reshuffles of its own dollars",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    save_path = outputs / f"permutation_tests_null_distributions{suffix}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()
