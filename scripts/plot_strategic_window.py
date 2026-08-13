#!/usr/bin/env python3
"""
Strategic-window figure (docs/methodology.md's "Strategic window" section).
Consumes results/strategic_window_{cycle}.json only -- no recomputation.

One panel per cycle: retention (%) vs. days before Election Day (x-axis
inverted so it reads chronologically left-to-right), 80%/100% reference
lines, each race's T_i^80 marked where it lands at a non-trivial (not
day-one, not last-week) point.

Output: figures/static/strategic_window_summary.png
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

D_COLOR = "#2e6da4"
R_COLOR = "#c0392b"
GRAY = "#9a9a9a"
INK = "#333333"


def load(cycle: int) -> dict:
    return json.load(open(RESULTS / f"strategic_window_{cycle}.json"))


def panel(ax, data: dict, cycle: int) -> None:
    days_before = data["days_before"]
    labels: list[tuple[str, float, float, str]] = []
    for key, color in (("strategic_window_D", D_COLOR), ("strategic_window_R", R_COLOR)):
        rows = data[key]
        districts = sorted({r["district_id"] for r in rows})
        for did in districts:
            # rows are already in chronological (days_before) order per
            # district -- compute_strategic_window.py appends them that way
            # and json round-trips list order, so no re-sort is needed.
            sub = [r for r in rows if r["district_id"] == did]
            y = [r["retention_rate"] * 100 for r in sub]
            x = days_before[: len(y)]
            ax.plot(x, y, color=color, linewidth=1.6, marker="o", markersize=3.5, alpha=0.85)
            # Label at the FIRST (leftmost, full-flexibility) point, not the
            # last -- every line's endpoint mechanically converges toward
            # ~100% in the final week (see caption), so end-labels crowd
            # together there; start-of-season retention is both the more
            # spread-out and the more informative value to show.
            labels.append((did, x[0], y[0], color))

    ax.axhline(100, color=GRAY, linewidth=0.9, linestyle=":")
    ax.axhline(80, color=GRAY, linewidth=0.9, linestyle="--")
    ax.text(days_before[-1] + 2, 101, "100%", fontsize=7.5, color=GRAY, va="bottom", ha="right")
    ax.text(days_before[-1] + 2, 81, "80% threshold", fontsize=7.5, color=GRAY, va="bottom", ha="right")
    ax.set_xlim(days_before[0] + 22, days_before[-1] - 2)  # left > right: inverted natively, no separate invert_xaxis() needed
    ax.set_xlabel("Days before Election Day")
    ax.set_ylabel("Retention: PSV / V_uni (%)")
    ax.set_title(f"{cycle}", fontsize=11, color=INK, loc="left", fontweight="bold")

    ys = [l[2] for l in labels]
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    min_gap = 0.045 * y_span
    labels.sort(key=lambda l: l[2])
    adjusted_y = []
    prev = None
    for _, _, y, _ in labels:
        y_adj = y if prev is None else max(y, prev + min_gap)
        adjusted_y.append(y_adj)
        prev = y_adj
    for (text, x, y, color), y_adj in zip(labels, adjusted_y):
        if abs(y_adj - y) > 1e-6:
            ax.plot([x, x + 4], [y, y_adj], color=color, linewidth=0.5, alpha=0.5)
        # x-axis is inverted (0 on the right), so a LARGER x pushes the
        # label further left/outward from the first data point.
        ax.text(x + 5, y_adj, text, fontsize=7.2, color=color, va="center", ha="left")


def main() -> None:
    data_2024 = load(2024)
    data_2022 = load(2022)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    panel(axes[0], data_2024, 2024)
    panel(axes[1], data_2022, 2022)

    d_line = plt.Line2D([0], [0], color=D_COLOR, linewidth=1.6, marker="o", markersize=4, label="Democratic move (R's real committed capital constrains its response)")
    r_line = plt.Line2D([0], [0], color=R_COLOR, linewidth=1.6, marker="o", markersize=4, label="Republican move (D's real committed capital constrains its response)")
    fig.legend(handles=[d_line, r_line], loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02),
               fontsize=9, frameon=False)

    fig.suptitle("Strategic window: when does an opportunity become hard to neutralize?",
                  fontsize=13.5, color=INK, y=1.13, fontweight="bold")
    fig.text(0.5, -0.03,
              "tau=0: opponent's response uses its REAL, tier-pooled committed capital as of the SAME date the move is made (estimation.commitment_timing). delta=\\$1M. Exact SLSQP throughout.\n"
              "Note: retention converges to exactly 100% in the final week for every race, mechanically -- by then the opponent has almost no flexible budget left for ANYTHING, not because this race specifically became durable.",
              ha="center", fontsize=7.5, color=GRAY)

    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "strategic_window_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
