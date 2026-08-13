#!/usr/bin/env python3
"""
Two-panel figure for the response-delay findings (docs/methodology.md's
"Response delay: does the opportunity survive a non-instantaneous
opponent?" section). Consumes results/response_delay_{cycle}.json and
estimation.commitment_timing only -- no recomputation.

  A: retention (%) vs. response delay tau, one line per race, both cycles
     -- the headline result: retention rises with tau for races that start
     BELOW 100% retention, but falls for races that start ABOVE 100%.
  B: the underlying mechanism -- real, dated commitment-fraction curves
     (what fraction of each committee's eventual national-committee-own IE
     spend is already locked in, as a function of days after Sept 1).

Output: figures/static/response_delay_summary.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datetime import date, timedelta  # noqa: E402

from estimation.commitment_timing import commitment_fraction_as_of, commitment_fraction_curve  # noqa: E402

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

RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "figures" / "static"
D_COLOR = "#2e6da4"
R_COLOR = "#c0392b"
GRAY = "#9a9a9a"
INK = "#333333"


def load(cycle: int) -> dict:
    return json.load(open(RESULTS / f"response_delay_{cycle}.json"))


def panel_retention(ax) -> None:
    labels: list[tuple[str, float, float, str]] = []
    for cycle, dash in ((2024, "-"), (2022, (0, (4, 2)))):
        data = load(cycle)
        for side, key, color in (("D", "response_delay_D", D_COLOR), ("R", "response_delay_R", R_COLOR)):
            rows = data[key]
            districts = sorted({r["district_id"] for r in rows})
            for did in districts:
                sub = sorted((r for r in rows if r["district_id"] == did), key=lambda r: r["tau_days"])
                x = [r["tau_days"] for r in sub]
                y = [r["retention_rate"] * 100 for r in sub]
                ax.plot(x, y, color=color, linestyle=dash, linewidth=1.6, marker="o", markersize=3.5, alpha=0.85)
                labels.append((f"{did} '{str(cycle)[2:]}", x[-1], y[-1], color))

    # Greedy monotonic label stacking in DATA space (not per-neighbor point
    # offsets, which under-corrects once 3+ labels cluster together and one
    # exits the cluster still carrying its neighbors' accumulated offset --
    # found in the first render: PA-12/CT-05 still overlapped despite a
    # collision check, because PA-12 inherited WI-03/WI-01/FL-27's stacked
    # offset even though PA-12 itself was far enough from CT-05 to not need
    # it). Each label's TEXT position is pushed up just enough to clear the
    # previous one; the dot/line stays at the true data point via a
    # separate connecting line when the label had to move.
    ys = [l[2] for l in labels]
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    min_gap = 0.038 * y_span
    labels.sort(key=lambda l: l[2])
    adjusted_y = []
    prev = None
    for _, _, y, _ in labels:
        y_adj = y if prev is None else max(y, prev + min_gap)
        adjusted_y.append(y_adj)
        prev = y_adj
    for (text, x, y, color), y_adj in zip(labels, adjusted_y):
        if abs(y_adj - y) > 1e-6:
            ax.plot([x, x + 0.9], [y, y_adj], color=color, linewidth=0.5, alpha=0.5)
        ax.text(x + 1.0, y_adj, text, fontsize=6.8, color=color, va="center")
    ax.set_xlim(right=ax.get_xlim()[1] + 3.5)

    ax.axhline(100, color=GRAY, linewidth=0.9, linestyle=":")
    ax.text(0.3, 101, "100% (fully retained)", fontsize=7.5, color=GRAY, va="bottom")
    ax.set_xlabel("Opponent's response delay, tau (days after Sept 1)")
    ax.set_ylabel("Retention: PSV / V_uni (%)")
    ax.set_title("Retention rises with delay for suppressed races,\nfalls for the >100% \"reshuffling bonus\" races", fontsize=10.5, color=INK, loc="left")


def panel_commitment(ax) -> None:
    taus = list(range(0, 29))
    for cycle, dash in ((2024, "-"), (2022, (0, (4, 2)))):
        for party, color in (("D", D_COLOR), ("R", R_COLOR)):
            curve = commitment_fraction_curve(cycle, party)
            t0 = date(cycle, 9, 1)
            y = [commitment_fraction_as_of(cycle, party, t0 + timedelta(days=t), curve) * 100 for t in taus]
            ax.plot(taus, y, color=color, linestyle=dash, linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Days after Sept 1")
    ax.set_ylabel("% of eventual national-committee IE spend already committed")
    ax.set_title("The mechanism: real, dated commitment curves\n(solid = 2024, dashed = 2022)", fontsize=10.5, color=INK, loc="left")

    d_line = plt.Line2D([0], [0], color=D_COLOR, linewidth=1.8, label="DCCC")
    r_line = plt.Line2D([0], [0], color=R_COLOR, linewidth=1.8, label="NRCC")
    ax.legend(handles=[d_line, r_line], loc="upper left", fontsize=8.5, frameon=False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panel_retention(axes[0])
    panel_commitment(axes[1])

    fig.suptitle("Response delay: does locked-capital friction preserve strategic leverage?",
                 fontsize=13, color=INK, y=1.05, fontweight="bold")
    fig.text(0.5, -0.02,
              "Reference date t0 = Sept 1; tau = additional days before the opponent's best response is computed, using its REAL locked capital as of t0+tau "
              "(estimation.commitment_timing, dated FEC data). delta = \\$1M. Exact SLSQP throughout.",
              ha="center", fontsize=7.5, color=GRAY)

    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "response_delay_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
