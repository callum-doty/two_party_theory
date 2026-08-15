#!/usr/bin/env python3
"""
Final verdict figure for the K=15-20 + weekly-resolution stress test
(2026-08-14): AZ-09's smooth weekly retention trajectory (survives every
check applied this session) against FL-02's sharp, late discontinuity
(technically clears the "exclude only the final date" rule, but fails a
smoothness/character test the way NC-06 and CT-02 already failed one at
coarser resolution -- a jump concentrated in the single week before the
mechanical floor is not distinguishable in kind from the floor itself).

Consumes results/strategic_window_union_{cycle}_excl_redistricting_weekly.json.

Output: figures/static/weekly_stress_test_verdict.png
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

GREEN = "#1a9850"
RED = "#d73027"
INK = "#333333"
BG = "#f2f2f2"


def trajectory(cycle: int, side: str, district: str) -> list[tuple[int, float, float]]:
    w = json.load(open(RESULTS / f"strategic_window_union_{cycle}_excl_redistricting_weekly.json"))
    key = "strategic_window_D" if side == "D" else "strategic_window_R"
    rows = [r for r in w[key] if r["district_id"] == district]
    days = w["days_before"]
    return [(d, r["PSV"], r["retention_rate"]) for d, r in zip(days, rows)]


def main() -> None:
    az09 = trajectory(2024, "R", "AZ-09")
    fl02 = trajectory(2022, "R", "FL-02")

    fig, ax = plt.subplots(figsize=(10, 6))

    for traj, color, label in ((az09, GREEN, "AZ-09 '24 (R) -- smooth, gradual, survives every check"),
                                (fl02, RED, "FL-02 '22 (R) -- flat, then a sharp jump in the final 1-2 weeks")):
        days = [t[0] for t in traj]
        retention = [t[2] * 100 for t in traj]
        ax.plot(days, retention, "-", color=color, linewidth=2.4, marker="o", markersize=4.5, label=label)

    ax.axhline(80, color=INK, linewidth=1.0, linestyle=":", alpha=0.6)
    ax.text(122, 82, "80% durability threshold", fontsize=8, color=INK, alpha=0.75)
    ax.axvspan(0, 9, color=BG, zorder=0)
    ax.set_xlim(126, 4)
    ax.text(4.5, ax.get_ylim()[1] * 0.95, "final 1-2\nweeks", fontsize=7.5, color="#999999", ha="center", va="top")

    ax.set_xlabel("Days before Election Day (weekly grid)")
    ax.set_ylabel("Retention (PSV / V_uni)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_title("The verdict: a genuine mid-season signal looks like AZ-09, not FL-02",
                  fontsize=13, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)

    fig.text(0.5, -0.02,
              "Both races technically clear the '80% retention, holds afterward' bar used to compute Theta -- but only AZ-09 gets there gradually, while the opponent still has real "
              "flexible money left. FL-02 sits flat and heavily countered for the whole season, then jumps only once the opponent has nearly run out of money to respond with -- "
              "the same character as the mechanical final-week floor, just one data point earlier. A genuine timing signal should look like the green line, not the red one.",
              ha="center", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "weekly_stress_test_verdict.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
