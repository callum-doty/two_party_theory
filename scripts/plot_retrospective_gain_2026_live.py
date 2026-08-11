#!/usr/bin/env python3
"""
Visualize the 2026 live analog of the real-world seat-gain decomposition
(scripts/decompose_retrospective_gain_2026_live.py).

Two panels:
  Left  -- DCCC-observed (real pattern, scaled to the full budget) vs.
           model-optimal expected seats, both evaluated against today's real
           floors, with the resulting gain annotated. Styled to match the
           other 2026-live figures (plot_allocator_comparison_2026.py's
           color conventions: model green, DCCC/actual blue).
  Right -- how thin the sample behind the left panel actually is: L_t
           committed so far vs. the full budget, and how many of 434 races
           have received any money at all. This is the caveat that matters
           most for reading the left panel honestly, so it's placed directly
           next to it rather than left to a caption.

Reads outputs/retrospective_gain_2026_live.csv, written by
decompose_retrospective_gain_2026_live.py. Run that script first if missing.

Usage:
    python scripts/plot_retrospective_gain_2026_live.py

Output: outputs/retrospective_gain_2026_live.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.ledger import RealizedSpendCommitmentSource
from datetime import date

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

C_MODEL = "#2a9d4f"    # matches plot_allocator_comparison_2026.py
C_DCCC = "#1a6faf"
C_CAVEAT = "#c0392b"
C_MUTED = "#888888"

# 2022/2024 analogy context (decompose_retrospective_gain_by_information_date.py's
# own validated output, not recomputed here) -- the real-world gain at the
# closest comparable checkpoint in each historical cycle.
ANALOGY = [
    ("2024", 113, 1.978, 2.825),
    ("2022", 116, 2.250, 3.223),
]


def main() -> None:
    outputs = config.outputs_path()
    path = outputs / "retrospective_gain_2026_live.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run decompose_retrospective_gain_2026_live.py first.")
    row = pd.read_csv(path).iloc[0]

    # Re-derive the committed-races count directly (not stored in the CSV) --
    # the single number that most explains why the left panel's gain is
    # noisy, so it's worth getting fresh rather than hardcoding.
    races = build_universe(cycle=2026)
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, date.fromisoformat(row["as_of"]), races)
    n_races = len(races)
    n_committed = sum(1 for v in committed.values() if v > 0)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Left: DCCC-observed (scaled) vs. model-optimal expected seats ──────
    bars_x = [0, 1]
    bars_y = [row["dccc_observed_scaled_expected_seats"], row["model_optimal_expected_seats"]]
    colors = [C_DCCC, C_MODEL]
    labels = ["DCCC-observed\n(real pattern, scaled)", "Model-optimal\n(same budget)"]

    ax_l.bar(bars_x, bars_y, color=colors, width=0.55, zorder=3)
    ymin = min(bars_y) - 8
    ax_l.set_ylim(ymin, max(bars_y) + 4)
    ax_l.set_xticks(bars_x)
    ax_l.set_xticklabels(labels, fontsize=10)
    ax_l.set_ylabel("Expected seats (of 434 modeled races)")
    for x, y in zip(bars_x, bars_y):
        ax_l.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=10.5, fontweight="bold")

    gain = row["gain"]
    ax_l.annotate(
        "", xy=(1, bars_y[1] - 1.2), xytext=(0, bars_y[0] - 1.2),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.3,
                         connectionstyle="arc3,rad=-0.25"),
    )
    ax_l.text(0.5, min(bars_y) - 3.2, f"gain = {gain:+.1f} seats",
               ha="center", fontsize=11, color="#333333")

    ax_l.set_title(
        f"2026 live, as of {row['as_of']} ({int(row['days_before_election'])}d to Election Day)\n"
        "PROSPECTIVE -- cycle in progress, not a validated outcome",
        fontsize=10.5,
    )
    ax_l.grid(axis="y", color="#e5e4df", linewidth=0.7, zorder=0)
    ax_l.set_axisbelow(True)

    # ── Right: how thin is the sample behind the left panel? Both bars share
    # a 0-100 x-axis: one is a literal %, the other is races-with-any-$ as a
    # % of the full universe -- a shared, directly comparable scale. ────────
    pct = row["pct_budget_committed"]
    frac_races = 100 * n_committed / n_races

    ax_r.barh([1], [100], color="#e8e6df", height=0.45, zorder=2)
    ax_r.barh([1], [pct], color=C_CAVEAT, height=0.45, zorder=3)
    ax_r.text(2, 1, f"{pct:.2f}% of ${row['budget_2026']/1e6:.0f}M budget committed",
              va="center", fontsize=9.5, color="#333333")

    ax_r.barh([0], [100], color="#e8e6df", height=0.45, zorder=2)
    ax_r.barh([0], [frac_races], color=C_CAVEAT, height=0.45, zorder=3)
    ax_r.text(2, 0, f"{n_committed} of {n_races} races have received any $ yet",
              va="center", fontsize=9.5, color="#333333")

    ax_r.set_xlim(0, 100)
    ax_r.set_ylim(-0.6, 1.6)
    ax_r.set_yticks([])
    ax_r.set_xlabel("% of full picture observed so far")
    ax_r.set_title("Why the left panel is noisy right now", fontsize=10.5)
    ax_r.spines["left"].set_visible(False)

    fig.suptitle(
        "2026 live seat-gain read is not yet comparable to the validated 2022/2024 figures",
        fontsize=12.5, y=1.02,
    )

    context_lines = "  |  ".join(
        f"{cyc} @ {d}d out: +{g:.2f} ({100*g/h:.0f}% of eventual +{h:.2f})"
        for cyc, d, g, h in ANALOGY
    )
    fig.text(0.5, -0.04,
              f"For reference, not comparable directly: validated real-world gain at a similar "
              f"horizon in the historical cycles -- {context_lines}",
              ha="center", fontsize=8.5, color=C_MUTED)

    fig.tight_layout()
    out_path = outputs / "retrospective_gain_2026_live.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
