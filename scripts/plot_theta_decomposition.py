#!/usr/bin/env python3
"""
Theta decomposition figure (docs/methodology.md's "Information option
value" section): information option value vs. strategic flexibility option
value, side by side, for the same 12 candidates. Consumes results/
information_value.json and results/value_of_waiting.json only.

Output: figures/static/theta_decomposition_summary.png
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

STRATEGIC_COLOR = "#6a3d9a"   # purple: distinct from the D/R blue-red used elsewhere, since this compares TWO KINDS of value, not two sides
INFO_COLOR = "#33a02c"        # green
GRAY = "#9a9a9a"
INK = "#333333"


def main() -> None:
    info = json.load(open(RESULTS / "information_value.json"))
    waiting = json.load(open(RESULTS / "value_of_waiting.json"))

    rows = []
    for cycle_str, cycle_data in info.items():
        cycle = int(cycle_str)
        for side in ("D", "R"):
            info_val = cycle_data[side]["info_option_value"]
            # info_option_value is computed ONCE per side (the V_uni-anchored
            # pick), not per race -- broadcast it onto the anchor race only,
            # and onto the other candidates as "not applicable" (they were
            # never the zero-noise pick, so the noise-driven mispick question
            # doesn't apply to them individually the way net_waiting_value
            # does per race).
            anchor_district = cycle_data[side]["best_true_district"]
            for wr in waiting[cycle_str][side]:
                did = wr["district_id"]
                net_wait = wr["net_waiting_value"]
                this_info = info_val if did == anchor_district else None
                rows.append(dict(
                    label=f"{did} '{str(cycle)[2:]} ({side})",
                    strategic=net_wait, info=this_info,
                    is_anchor=(did == anchor_district),
                ))

    rows.sort(key=lambda r: (r["strategic"] is None, r["strategic"] or 0))
    fig, ax = plt.subplots(figsize=(10, 7))
    ys = np.arange(len(rows))
    bar_h = 0.38
    for y, r in zip(ys, rows):
        if r["strategic"] is not None:
            ax.barh(y + bar_h / 2, r["strategic"], height=bar_h, color=STRATEGIC_COLOR, alpha=0.85)
        if r["info"] is not None:
            ax.barh(y - bar_h / 2, r["info"], height=bar_h, color=INFO_COLOR, alpha=0.85)
            ax.text(max(r["info"], 0) + 0.001, y - bar_h / 2, " info-value anchor race",
                    fontsize=6.5, color=INFO_COLOR, va="center")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_xlabel("Expected seats, $1M delta")
    ax.set_title("Theta decomposition: strategic flexibility vs. information option value",
                  fontsize=12.5, color=INK, loc="left", fontweight="bold")

    s_patch = plt.Rectangle((0, 0), 1, 1, color=STRATEGIC_COLOR, alpha=0.85,
                             label="Strategic flexibility (net waiting value, per race)")
    i_patch = plt.Rectangle((0, 0), 1, 1, color=INFO_COLOR, alpha=0.85,
                             label="Information (per SIDE, shown on its V_uni-anchor race)")
    ax.legend(handles=[s_patch, i_patch], loc="lower right", fontsize=8.5, frameon=False)

    fig.text(0.5, -0.02,
              "Information option value is computed once per side (the cost of a noisy generic-ballot-driven mispick relative to a perfectly-informed V_uni-based decision), not per race -- "
              "shown on the race that decision rule actually anchors to. It is near zero in 3 of 4 sides because the V_uni gap between the top candidate and the runner-up is too large for plausible national-environment noise to overturn; "
              "the one exception (2022 R-side) has closely-matched V_uni scores, where noise CAN flip the pick.",
              ha="center", fontsize=7.3, color=GRAY)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "theta_decomposition_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
