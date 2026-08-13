#!/usr/bin/env python3
"""
Value-of-waiting figure (docs/methodology.md's "Value of waiting" section).
Consumes results/value_of_waiting.json only -- no recomputation.

Horizontal bar chart: net_waiting_value per race, both cycles, colored by
side. Races whose T80 was already the first date (no genuine waiting
occurred -- the statistic degenerates to "was this race the best
immediate choice") are hatched, to keep them visually distinct from races
where delay was actually tested.

Output: figures/static/value_of_waiting_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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


def main() -> None:
    data = json.load(open(RESULTS / "value_of_waiting.json"))
    window = {
        2024: json.load(open(RESULTS / "strategic_window_2024.json")),
        2022: json.load(open(RESULTS / "strategic_window_2022.json")),
    }

    rows = []
    for cycle in (2024, 2022):
        earliest_date = window[cycle]["strategic_window_D"][0]["ref_date"]
        for side, color in (("D", D_COLOR), ("R", R_COLOR)):
            for r in data[str(cycle)][side]:
                genuine = r["T80"] is not None and r["T80"] != earliest_date
                rows.append(dict(
                    label=f"{r['district_id']} '{str(cycle)[2:]} ({side})",
                    value=r["net_waiting_value"], color=color, genuine=genuine,
                ))
    rows.sort(key=lambda r: r["value"])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        hatch = None if r["genuine"] else "///"
        ax.barh(y, r["value"], color=r["color"], alpha=0.85 if r["genuine"] else 0.35,
                edgecolor=r["color"], hatch=hatch, height=0.65)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_xlabel("Net value of waiting (expected seats, $1M delta)")
    ax.set_title("Value of waiting: durability gained vs. the best immediate alternative",
                  fontsize=12, color=INK, loc="left", fontweight="bold")

    d_patch = plt.Rectangle((0, 0), 1, 1, color=D_COLOR, alpha=0.85, label="D-side move")
    r_patch = plt.Rectangle((0, 0), 1, 1, color=R_COLOR, alpha=0.85, label="R-side move")
    faded_patch = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY, hatch="///",
                                 label="no genuine wait (already durable at day 1)")
    ax.legend(handles=[d_patch, r_patch, faded_patch], loc="lower right", fontsize=8.5, frameon=False)

    fig.text(0.5, -0.01,
              "net_waiting_value = PSV at this race's own T_i^80 minus the best value achievable RIGHT NOW (this race or the best of the other 5 pre-screened candidates on the same side).\n"
              "Positive = worth holding capital in reserve for this specific race. Negative = better to act now on whatever is already best. Faded bars: T_i^80 was already day 1, so no delay was actually tested.",
              ha="center", fontsize=7.5, color=GRAY)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "value_of_waiting_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
