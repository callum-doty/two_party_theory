#!/usr/bin/env python3
"""
Conditional value-of-waiting figure, race by race (2026-08-14): visualizes
the corrected claim from docs/methodology.md's "Widening the action space"
and "Final-week sensitivity" sections -- a moderate MID-SEASON wait (weeks,
not days) can convert an easily-countered opportunity into a durable one,
and it is worth doing only when nothing better is available today.

Pure post-processing of results already on disk (strategic_window_{cycle}.json,
value_of_waiting.json) -- no new solves. For each of the 12 K=3-pool
candidate races, recomputes V_wait EXCLUDING the mechanical final-week date
(the point every race converges toward ~100% retention by construction,
per strategic_window.py's own methodology) so "value of waiting" reflects
genuine mid-season timing, not the trivial end-of-cycle floor.

Three categories emerge, not two:
  - GENUINE DURABLE + WORTH IT: retention crosses 80%+ mid-season AND that
    beats the best immediate alternative (CT-02, FL-22, NV-01).
  - GENUINE DURABLE BUT NOT WORTH IT: retention crosses 80%+ mid-season but
    a better opportunity was ALREADY available the whole time (AZ-02) --
    the direct illustration of "conditional on nothing better available."
  - GROWS BUT NEVER GENUINELY DURABLE: value rises with waiting but stays
    well under 80% retention until the mechanical final-week jump (NC-06)
    -- a race that LOOKS like a waiting success only if the final-week
    artifact is not excluded; shown as its own category rather than
    grouped with the genuine cases.
  - ALREADY DURABLE DAY ONE: nothing to wait for (flat).

Usage:
    python scripts/plot_conditional_waiting_value.py
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

GREEN = "#1a9850"    # genuine + worth it
ORANGE = "#e08214"   # genuine but NOT worth it (better alternative already available)
GRAY_MECH = "#8073ac"  # grows but never genuinely durable except mechanically
FLAT = "#b0b0b0"     # already durable day one, nothing to wait for
INK = "#333333"
BG = "#f2f2f2"

RETENTION_DURABLE = 0.80


def load_data() -> list[dict]:
    w = {2024: json.load(open(RESULTS / "strategic_window_2024.json")),
         2022: json.load(open(RESULTS / "strategic_window_2022.json"))}
    vow = json.load(open(RESULTS / "value_of_waiting.json"))

    out = []
    for cycle in (2024, 2022):
        window = w[cycle]
        days = window["days_before"]
        for side, key in (("D", "strategic_window_D"), ("R", "strategic_window_R")):
            by_district: dict[str, list[dict]] = {}
            for row in window[key]:
                by_district.setdefault(row["district_id"], []).append(row)
            for did, rows in by_district.items():
                excl_final = rows[:-1]
                best_excl = max(excl_final, key=lambda r: r["PSV"])
                vow_row = next(r for r in vow[str(cycle)][side] if r["district_id"] == did)
                already_durable = rows[0]["retention_rate"] >= RETENTION_DURABLE
                genuinely_durable = (not already_durable) and best_excl["retention_rate"] >= RETENTION_DURABLE
                idx = rows.index(best_excl)
                net_genuine = best_excl["PSV"] - vow_row["best_immediate"]

                if already_durable:
                    category = "flat"
                elif genuinely_durable and net_genuine > 0:
                    category = "genuine_worth_it"
                elif genuinely_durable:
                    category = "genuine_not_worth_it"
                else:
                    category = "grows_never_durable"

                out.append(dict(
                    cycle=cycle, side=side, district=did, label=f"{did} '{str(cycle)[2:]} ({side})",
                    V_now=rows[0]["PSV"], best_immediate=vow_row["best_immediate"], best_alt=vow_row["best_alt_district"],
                    V_wait_genuine=best_excl["PSV"], genuine_days_out=days[idx], genuine_retention=best_excl["retention_rate"],
                    net_genuine=net_genuine, category=category,
                    trajectory=[(d, r["PSV"], r["retention_rate"]) for d, r in zip(days, rows)],
                ))
    return out


def main() -> None:
    data = load_data()
    label_by_cat = {
        "genuine_worth_it": "Converts to durable mid-season, AND worth waiting for",
        "genuine_not_worth_it": "Converts to durable mid-season, but NOT worth waiting for (better option already available)",
        "grows_never_durable": "Value grows with waiting, but never genuinely durable (only 'durable' via the mechanical final-week floor)",
        "flat": "Already durable from day one -- nothing to wait for",
    }
    color_by_cat = {"genuine_worth_it": GREEN, "genuine_not_worth_it": ORANGE,
                    "grows_never_durable": GRAY_MECH, "flat": FLAT}

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=(0.85, 1.35), hspace=0.32)

    # --- Panel A: retention trajectories for the 5 illustrative (non-flat) races ---
    ax1 = fig.add_subplot(gs[0])
    illustrative = [r for r in data if r["category"] != "flat"]
    illustrative.sort(key=lambda r: {"genuine_worth_it": 0, "genuine_not_worth_it": 1, "grows_never_durable": 2}[r["category"]])
    genuine_worth_it_shades = ["#1a9850", "#66bd63", "#00441b"]  # distinguish CT-02/FL-22/NV-01 within the shared green category
    shade_i = 0
    for r in illustrative:
        days = [t[0] for t in r["trajectory"]]
        retention = [t[2] * 100 for t in r["trajectory"]]
        if r["category"] == "genuine_worth_it":
            color = genuine_worth_it_shades[shade_i % len(genuine_worth_it_shades)]
            shade_i += 1
        else:
            color = color_by_cat[r["category"]]
        style = "-" if r["category"] != "grows_never_durable" else "--"
        ax1.plot(days, retention, style, color=color, linewidth=2.2, marker="o", markersize=4, label=r["label"])
        # mark the genuine (non-mechanical) peak used for V_wait_genuine
        gday, gpsv, gret = next(t for t in r["trajectory"] if t[0] == r["genuine_days_out"])
        ax1.scatter([gday], [gret * 100], s=90, facecolors="white", edgecolors=color, linewidths=2.2, zorder=5)

    ax1.axhline(RETENTION_DURABLE * 100, color=INK, linewidth=1.0, linestyle=":", alpha=0.6)
    ax1.text(122, RETENTION_DURABLE * 100 + 2, "80% durability threshold", fontsize=8, color=INK, alpha=0.75)
    ax1.axvspan(0, 8.5, color=BG, zorder=0)
    ax1.set_xlim(126, 4)
    ax1.text(4.2, ax1.get_ylim()[1] * 0.97, "mechanical\nfinal-week\nfloor zone\n(excluded from\n\"genuine\" value)",
              fontsize=7, color="#999999", ha="center", va="top")
    ax1.set_xlabel("Days before Election Day")
    ax1.set_ylabel("Retention (PSV / V_uni)")
    ax1.set_title("A. How each race's durability evolves over the season", fontsize=11.5, loc="left", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7.8, frameon=False)
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")

    # --- Panel B: per-race conditional value dumbbell ---
    ax2 = fig.add_subplot(gs[1])
    order = {"genuine_worth_it": 0, "genuine_not_worth_it": 1, "grows_never_durable": 2, "flat": 3}
    data.sort(key=lambda r: (order[r["category"]], -r["net_genuine"]))
    ys = np.arange(len(data))

    for y, r in zip(ys, data):
        color = color_by_cat[r["category"]]
        ax2.plot([r["V_now"], r["V_wait_genuine"]], [y, y], color=color, linewidth=2.5, alpha=0.85, zorder=2)
        ax2.scatter([r["V_now"]], [y], s=55, facecolors="white", edgecolors=color, linewidths=2, zorder=3)
        ax2.scatter([r["V_wait_genuine"]], [y], s=75, facecolors=color, edgecolors=color, linewidths=1, zorder=3)
        ax2.scatter([r["best_immediate"]], [y], marker="D", s=45, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)

    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([r["label"] for r in data], fontsize=9)
    ax2.axvline(0, color=INK, linewidth=0.8)
    ax2.set_xlabel("Expected seats, $1M delta")
    ax2.set_title("B. Deploy now vs. wait -- and does waiting beat the best alternative available today?",
                   fontsize=11.5, loc="left", fontweight="bold")

    # category background bands + labels
    cat_order = ["genuine_worth_it", "genuine_not_worth_it", "grows_never_durable", "flat"]
    start = 0
    xlim = ax2.get_xlim()
    for cat in cat_order:
        n = sum(1 for r in data if r["category"] == cat)
        if n == 0:
            continue
        ax2.axhspan(start - 0.5, start + n - 0.5, color=color_by_cat[cat], alpha=0.06, zorder=0)
        start += n

    hollow = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK, markersize=7, label="V_now (deploy today)")
    filled = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markeredgecolor=INK, markersize=7, label="V_wait (genuine, excl. mechanical floor)")
    diamond = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="none", markeredgecolor=INK, markersize=7, label="best_immediate (best alternative use of the same $1M, today)")
    ax2.legend(handles=[hollow, filled, diamond], loc="lower right", fontsize=8, frameon=False)

    fig.suptitle("Conditional value of waiting, race by race (2022 & 2024, K=3 top-leverage candidates per side)",
                  fontsize=13.5, x=0.02, ha="left", fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
              "Waiting pays off only where the connecting line's endpoint (dark dot) clears BOTH the 80% durability line in panel A AND the diamond (best_immediate) in panel B. "
              "CT-02, FL-22, and NV-01 clear both. AZ-02 becomes durable but a better option (NY-20) was already sitting there the whole season. "
              "NC-06's apparent +0.155-seat 'win' reported by a naive value-of-waiting calculation is almost entirely the mechanical final-week floor -- excluded here, it barely clears 39% retention.",
              ha="center", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "conditional_waiting_value_by_race.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
