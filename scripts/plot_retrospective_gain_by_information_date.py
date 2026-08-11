#!/usr/bin/env python3
"""
Plot the information-date decomposition of Paper I's retrospective seat-gain
finding (scripts/decompose_retrospective_gain_by_information_date.py).

Two panels, 2024 (primary) and 2022 (OOS), sharing a y-axis. Each panel
plots two series against days-before-election:

  - "own-environment" (dashed, muted): the checkpoint-informed recommendation
    evaluated inside its own reconstructed, immature-opponent-spending world.
    Kept on the chart as a labeled cautionary series, not deleted -- it's the
    naive comparison, and the gap between it and the real-world series *is*
    the finding (an artifact of comparing against two different opponent-
    spending environments, not the model picking worse races).
  - "real-world" (solid, primary): the same checkpoint decision evaluated
    against the real, final candidate floors and opponent spending -- the
    fair, decision-relevant comparison, analogous to how Paper I's own
    retrospective counterfactual is evaluated.

A horizontal reference line marks the full-hindsight headline (+2.83 for
2024, +3.22 for 2022) and a zero line marks break-even against DCCC's real
final outcome.

Reads outputs/retrospective_gain_by_information_date_{cycle}.csv, written by
decompose_retrospective_gain_by_information_date.py. Run that script first
(both cycles) if either CSV is missing.

Usage:
    python scripts/plot_retrospective_gain_by_information_date.py

Output: outputs/retrospective_gain_by_information_date.png
"""

from __future__ import annotations
import sys
from pathlib import Path

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

_COLOR_OWN_ENV = "#9a988f"     # muted grey -- the flawed/naive comparison
_COLOR_REAL_WORLD = "#1f4e9c"  # blue -- the fair, decision-relevant comparison
_COLOR_HINDSIGHT = "#1a7a3c"   # green -- the validated full-hindsight reference

FULL_HINDSIGHT = {2024: 2.83, 2022: 3.22}


def _plot_panel(ax, df: pd.DataFrame, cycle: int) -> None:
    days = df["days_before_election"]

    ax.plot(days, df["gain_vs_dccc_final"], "o--", color=_COLOR_OWN_ENV,
             linewidth=1.3, markersize=4.5, label="own-environment eval\n(naive, flawed)")
    ax.plot(days, df["real_world_gain_vs_dccc_final"], "o-", color=_COLOR_REAL_WORLD,
             linewidth=2.0, markersize=5.5, label="real-world eval\n(fair comparison)")

    hindsight = FULL_HINDSIGHT[cycle]
    ax.axhline(hindsight, color=_COLOR_HINDSIGHT, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text(days.min(), hindsight, f"  full-hindsight +{hindsight:.2f}",
            color=_COLOR_HINDSIGHT, fontsize=9, va="bottom", ha="left")
    # Also mark it as the effective t=0 point of both series.
    ax.plot([0], [hindsight], "o", color=_COLOR_HINDSIGHT, markersize=6, zorder=5)

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="-", alpha=0.6)

    ax.invert_xaxis()  # time flows left (early cycle) -> right (Election Day)
    ax.set_xlabel("Days before Election Day")
    ax.set_title(f"{cycle} {'(primary)' if cycle == 2024 else '(OOS)'}", fontsize=11.5)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    outputs = config.outputs_path()

    dfs = {}
    for cycle in (2024, 2022):
        path = outputs / f"retrospective_gain_by_information_date_{cycle}.csv"
        if not path.exists():
            raise SystemExit(
                f"Missing {path} -- run "
                f"`python scripts/decompose_retrospective_gain_by_information_date.py "
                f"--cycle {cycle}"
                + (" --processed-dir data/processed_oos_2020`" if cycle == 2022 else "`")
            )
        dfs[cycle] = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, cycle in zip(axes, (2024, 2022)):
        _plot_panel(ax, dfs[cycle], cycle)

    axes[0].set_ylabel("Expected-seat gain vs. DCCC's real final allocation")
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Does the retrospective seat-gain finding hold up in real time?\n"
        "Model recommendation at each historical checkpoint vs. DCCC's actual final outcome",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()

    out_path = outputs / "retrospective_gain_by_information_date.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
