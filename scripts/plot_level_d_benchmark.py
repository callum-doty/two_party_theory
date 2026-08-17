#!/usr/bin/env python3
"""
Level D benchmark figure for the final paper (2026-08-17): the paradox
underlying this project's "convention/institutional equilibrium"
interpretation -- observed DCCC/NRCC spending is closest, by L1 distance,
to a simple Cook-rating competitiveness heuristic, and roughly tied for
FARTHEST from both the one-sided optimizer and the game-theoretic mixed
equilibrium, on both cycles (docs/methodology.md's "Level D five-way
benchmark" section). The strategies that are farthest in allocation space
also score BEST in expected seats -- exactly the pattern expected if
committees follow an evolved heuristic rather than solving the
underlying optimization problem, and that heuristic still captures most
of the achievable value.

Consumes results/level_d_benchmark_{cycle}.json only, no new computation.

Output: figures/static/level_d_benchmark_summary.png
"""
from __future__ import annotations

import json
import textwrap
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

STRATEGY_LABELS = {
    "cook_heuristic": "Cook heuristic",
    "equal": "Equal allocation",
    "random_feasible": "Random feasible",
    "one_sided_optimizer": "One-sided optimizer",
    "mixed_equilibrium": "Mixed equilibrium",
}
STRATEGY_COLORS = {
    "cook_heuristic": "#1a9850",
    "equal": "#999999",
    "random_feasible": "#bbbbbb",
    "one_sided_optimizer": "#fc8d59",
    "mixed_equilibrium": "#d73027",
}
MARKERS = {
    "cook_heuristic": "o", "equal": "s", "random_feasible": "^",
    "one_sided_optimizer": "D", "mixed_equilibrium": "*",
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    for ax, cycle in zip(axes, (2022, 2024)):
        d = json.load(open(RESULTS / f"level_d_benchmark_{cycle}.json"))
        rows = d["rows"]
        observed_seats = rows["observed"]["e_seats_d"]

        for name in ("cook_heuristic", "equal", "random_feasible", "one_sided_optimizer", "mixed_equilibrium"):
            r = rows[name]
            l1_total_m = (r["l1_d"] + r["l1_r"]) / 1e6
            ax.scatter([l1_total_m], [r["e_seats_d"]], s=170, marker=MARKERS[name],
                       color=STRATEGY_COLORS[name], edgecolors="#333333", linewidths=0.8, zorder=3,
                       label=STRATEGY_LABELS[name])
            ax.annotate(STRATEGY_LABELS[name], (l1_total_m, r["e_seats_d"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=8, color="#333333")

        ax.scatter([0], [observed_seats], s=220, marker="X", color="black", zorder=4, label="Observed")
        ax.annotate("Observed", (0, observed_seats), textcoords="offset points", xytext=(8, -12),
                    fontsize=8.5, fontweight="bold")

        ax.set_xlabel("L1 distance from observed allocation ($M)")
        ax.set_ylabel("E[D seats]")
        ax.set_title(f"{cycle}", fontsize=12.5, loc="left", fontweight="bold")

    fig.suptitle("Observed spending is closest to a simple heuristic, and farthest strategies score only marginally better",
                  fontsize=12.5, x=0.02, ha="left", fontweight="bold", y=1.03)
    caption = ("Each point: one allocation strategy's distance from what DCCC/NRCC actually did (x-axis) against its modeled expected D seats (y-axis). "
               "On both cycles, the Cook-rating heuristic sits closest to observed AND scores nearly as well as the strategies that are farthest away -- "
               "the one-sided optimizer and the game-theoretic mixed equilibrium buy only a fraction of a seat over a much simpler rule of thumb.")
    fig.text(0.5, 0.12, textwrap.fill(caption, width=110), ha="center", va="top", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "level_d_benchmark_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
