#!/usr/bin/env python3
"""
Unified-Theta figure (docs/methodology.md's "Unified sequential decision
value" section): Theta_full vs. its flex-only/info-only counterfactual
components, one bar group per (cycle, side), plus the interaction term
left over when the two counterfactuals don't sum to the full value.
Consumes results/theta_unified.json only.

Output: figures/static/theta_unified_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "figures" / "static"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

FULL_COLOR = "#333333"
FLEX_COLOR = "#6a3d9a"
INFO_COLOR = "#33a02c"
INTERACTION_COLOR = "#bbbbbb"


def main() -> None:
    data = json.load(open(RESULTS / "theta_unified.json"))

    rows = []
    for cycle_str, cycle_data in data.items():
        for side in ("D", "R"):
            r = cycle_data[side]
            rows.append(dict(
                label=f"{cycle_str} {side}",
                theta_full=r["theta_full"], flex=r["theta_flex_only"],
                info=r["theta_info_only"], interaction=r["interaction"],
            ))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    width = 0.2
    ax.bar(x - 1.5 * width, [r["theta_full"] for r in rows], width, color=FULL_COLOR, label="Theta_full (unified Bellman)")
    ax.bar(x - 0.5 * width, [r["flex"] for r in rows], width, color=FLEX_COLOR, label="Theta_flex_only (info frozen)")
    ax.bar(x + 0.5 * width, [r["info"] for r in rows], width, color=INFO_COLOR, label="Theta_info_only (opponent capital frozen)")
    ax.bar(x + 1.5 * width, [r["interaction"] for r in rows], width, color=INTERACTION_COLOR, label="Interaction (full - flex - info)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows])
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("Expected seats, $1M delta")
    ax.set_title("Unified Theta: one Bellman value, decomposed by counterfactual", fontsize=12.5, loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=False)

    fig.text(0.5, -0.03,
              "One decision rule (noisy-V_uni x retention, realized at TRUE PSV) used at every date and in every counterfactual -- "
              "unlike the earlier two-module Theta, flex-only and info-only are computed on the SAME Bellman recursion, differing only in which state variable is frozen.",
              ha="center", fontsize=7.3, color="#9a9a9a")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "theta_unified_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
