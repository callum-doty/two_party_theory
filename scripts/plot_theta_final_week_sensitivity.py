#!/usr/bin/env python3
"""
Final-week sensitivity figure (docs/methodology.md's "K-expansion" section):
Theta_full vs. Theta_full_excl_final_week, K=3 and K~8 pools side by side.
Consumes results/theta_final_week_sensitivity.json only.

Output: figures/static/theta_final_week_sensitivity.png
"""
from __future__ import annotations

import json
import textwrap
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

FULL_COLOR = "#e66101"       # orange: includes the mechanical final-week floor
TRIMMED_COLOR = "#5e3c99"    # purple: excludes it -- the more defensible "genuine timing" number


def main() -> None:
    data = json.load(open(RESULTS / "theta_final_week_sensitivity.json"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pool_key, title in zip(axes, ("curve_K3", "primary_K8"), ("K=3 (top-leverage curve)", "K~8 (primary pool)")):
        pool = data[pool_key]
        keys = sorted(pool.keys())
        labels = [k.replace("_", " ") for k in keys]
        full_vals = [pool[k]["theta_full"] for k in keys]
        trimmed_vals = [pool[k]["theta_full_excl_final_week"] for k in keys]
        mechanical = [pool[k]["theta_full_realized_at_days_before"] == 7 for k in keys]

        x = np.arange(len(keys))
        width = 0.35
        ax.bar(x - width / 2, full_vals, width, color=FULL_COLOR, label="Theta_full (all 8 dates)")
        ax.bar(x + width / 2, trimmed_vals, width, color=TRIMMED_COLOR, label="Theta_full, excl. final week")
        for xi, val, is_mech in zip(x, full_vals, mechanical):
            if is_mech:
                ax.text(xi - width / 2, val + 0.003, "mechanical\nfloor", ha="center", fontsize=6.5, color=FULL_COLOR)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.set_title(title, fontsize=11, loc="left")
        if ax is axes[0]:
            ax.set_ylabel("Expected seats, $1M delta")
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle("How much of Theta_full is genuine mid-season timing value, vs. the mechanical final-week floor?",
                  fontsize=12, x=0.02, ha="left", fontweight="bold")
    caption = ("Every race's retention converges to ~100% at the final reference date by construction (strategic_window.py) -- a race can dominate Theta_full there purely by "
               "having the largest raw V_uni in the K-pool, independent of whether it was ever genuinely contested earlier. Excluding that date isolates the timing value that "
               "actually depends on the opponent's mid-season capital constraints.")
    fig.text(0.5, 0.14, textwrap.fill(caption, width=100), ha="center", va="top", fontsize=7.3, color="#9a9a9a")

    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "theta_final_week_sensitivity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
