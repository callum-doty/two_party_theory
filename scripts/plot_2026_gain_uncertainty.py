#!/usr/bin/env python3
"""
Visualize the scoped joint uncertainty simulation
(simulate_2026_gain_uncertainty.py) -- the distribution of the combined
2026 gain estimate across 1000 draws jointly varying the forecast-model
bootstrap, the floor-maturity threshold, and generic-ballot uncertainty.

Usage:
    python scripts/plot_2026_gain_uncertainty.py

Output: outputs/gain_uncertainty_2026.png
"""
from __future__ import annotations

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

C_MODEL = "#2a9d4f"
C_ZERO = "#c0392b"


def main() -> None:
    outputs = config.outputs_path()
    df = pd.read_csv(outputs / "simulate_2026_gain_uncertainty.csv")
    summary = json.load(open(outputs / "simulate_2026_gain_uncertainty_summary.json"))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df["gain"], bins=40, color=C_MODEL, alpha=0.8, zorder=3)

    ax.axvline(summary["mean"], color="#1a1a1a", linewidth=1.8, linestyle="--", zorder=4)
    ax.text(summary["mean"], ax.get_ylim()[1] * 0.97, f"  mean +{summary['mean']:.2f}",
            fontsize=9.5, va="top")
    ax.axvspan(summary["p5"], summary["p95"], alpha=0.10, color=C_MODEL, zorder=1)
    ax.axvline(0, color=C_ZERO, linewidth=1.2, zorder=2)

    ax.set_xlabel("Simulated gain: E[Seats | model-optimal] - E[Seats | forecasted-DCCC]")
    ax.set_ylabel(f"Draws (of {summary['n_draws']})")
    ax.set_title(
        f"2026 gain under joint uncertainty (forecast-model bootstrap + maturity threshold + GB volatility)\n"
        f"+{summary['mean']:.2f} +/- {summary['sd']:.2f} (1 SD)  |  90% scenario range "
        f"[+{summary['p5']:.2f}, +{summary['p95']:.2f}]  |  {100-summary['pct_leq_zero']:.0f}% of draws > 0",
        fontsize=10.5,
    )
    ax.grid(axis="y", color="#e5e4df", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02,
              "Optimizer allocation held fixed at the point estimate across all draws (re-solving under "
              "each scenario is not tractable) -- a scenario range under this approximation, not a full CI.",
              ha="center", fontsize=8.5, color="#888888")

    fig.tight_layout()
    out_path = outputs / "gain_uncertainty_2026.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
