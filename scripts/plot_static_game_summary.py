#!/usr/bin/env python3
"""
Headline static-game summary figure for the final paper (2026-08-17):
one-shot unilateral exploitability (docs/methodology.md's "Corrected
headline exploitability" section) alongside the double-oracle mixed
equilibrium's support composition (docs/methodology.md's "Equilibrium
support composition" section) -- the two numbers that anchor the paper's
"reciprocal optimization competes away most apparent opportunity, but not
into one stable deterministic portfolio" claim.

Consumes only already-computed, already-reported numbers (no new solves):
the exploitability table is transcribed from docs/methodology.md (the
underlying run is `compute_exploitability.py`, not re-invoked here);
equilibrium support composition reads directly from
results/equilibrium_support_composition_{cycle}.json.

Output: figures/static/static_game_summary.png
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

D_COLOR = "#3288bd"
R_COLOR = "#d73027"
CORE_COLOR = "#4d4d4d"
SWING_COLOR = "#fdae61"
IRRELEVANT_COLOR = "#e0e0e0"

# Transcribed from docs/methodology.md's "Corrected headline exploitability" table (2026-08-12).
EXPLOITABILITY = {
    2022: dict(regret_d=3.03, regret_r=2.41, e_total=5.44, pct=2.53),
    2024: dict(regret_d=2.84, regret_r=2.30, e_total=5.14, pct=2.37),
}


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: one-shot unilateral exploitability ---
    cycles = [2022, 2024]
    x = np.arange(len(cycles))
    width = 0.32
    regret_d = [EXPLOITABILITY[c]["regret_d"] for c in cycles]
    regret_r = [EXPLOITABILITY[c]["regret_r"] for c in cycles]
    ax1.bar(x - width / 2, regret_d, width, color=D_COLOR, label="RegretD (D's unrealized gain)")
    ax1.bar(x + width / 2, regret_r, width, color=R_COLOR, label="RegretR (R's unrealized gain)")
    for xi, c in zip(x, cycles):
        total = EXPLOITABILITY[c]["e_total"]
        pct = EXPLOITABILITY[c]["pct"]
        ax1.text(xi, max(regret_d[list(x).index(xi)], regret_r[list(x).index(xi)]) + 0.15,
                  f"E={total:.2f} seats\n({pct:.1f}% of D total)", ha="center", fontsize=8, color="#333333")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(c) for c in cycles])
    ax1.set_ylabel("Expected seats, one-shot best response")
    ax1.set_title("A. Static exploitability is real but small\n(~2.4-2.5% of the D-seat total)", fontsize=11, loc="left")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax1.set_ylim(0, 4.2)

    # --- Panel B: equilibrium support composition (core/swing/irrelevant) ---
    rows = []
    for cycle in cycles:
        d = json.load(open(RESULTS / f"equilibrium_support_composition_{cycle}.json"))
        for side in ("d_side", "r_side"):
            c = d[side]["counts"]
            rows.append((cycle, "D" if side == "d_side" else "R", c["core"], c["swing"], c["irrelevant"]))

    labels = [f"{c} {s}" for c, s, *_ in rows]
    core = [r[2] for r in rows]
    swing = [r[3] for r in rows]
    total_funded = [r[2] + r[3] for r in rows]  # irrelevant races omitted from the stacked bar (dominates visually, not the point)
    xb = np.arange(len(rows))
    ax2.bar(xb, core, color=CORE_COLOR, label="Core (funded ~identically by every equilibrium portfolio)")
    ax2.bar(xb, swing, bottom=core, color=SWING_COLOR, label="Swing (funded only by SOME equilibrium portfolios)")
    for xi, tf, row in zip(xb, total_funded, rows):
        ax2.text(xi, tf + 1.5, f"{tf}/433\nfunded", ha="center", fontsize=7.5, color="#555555")
    ax2.set_xticks(list(xb))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Number of races")
    ax2.set_ylim(0, 100)
    ax2.set_title("B. The equilibrium is a DISTRIBUTION over portfolios,\nnot one deterministic allocation", fontsize=11, loc="left")
    ax2.legend(loc="upper right", fontsize=7.8, frameon=False)

    fig.suptitle("Reciprocal optimization competes away most apparent unilateral opportunity, but not into a single stable portfolio",
                  fontsize=12.5, x=0.02, ha="left", fontweight="bold", y=1.02)
    caption = ("Panel A: one-shot best-response regret if the opponent were held passive. Panel B: the double-oracle mixed equilibrium's support, decomposed per race -- "
               "most of the 433-race universe (not shown) is never funded in any equilibrium portfolio; among races that ARE funded, roughly half are 'swing' -- "
               "funded only when a particular equilibrium portfolio happens to be drawn, not a fixed target list.")
    fig.text(0.5, 0.13, textwrap.fill(caption, width=110), ha="center", va="top", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "static_game_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
