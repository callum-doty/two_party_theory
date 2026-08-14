#!/usr/bin/env python3
"""
FINAL conditional value-of-waiting figure (2026-08-14), after the K=15-20
stress test: supersedes plot_conditional_waiting_value.py's K=3 version.
The stress test (widen the candidate pool via rank_candidate_races.py's
union screen, then check both the mechanical-final-week and
redistricting-flagged-baseline confounds) overturned two of that figure's
three headline "genuine mid-season wait" stories:

  - CT-02 (2024 D): looked like the clearest genuine-wait case at K=3.
    At K~17 (redistricting-flagged excluded), FL-27 -- already >100%
    retained on day one -- dominates the ENTIRE season; waiting for CT-02
    specifically is strictly worse than just deploying to FL-27 today.
  - NC-06 (2024 R): already downgraded to "mostly mechanical floor" by
    theta_final_week_sensitivity.py. Fully debunked here: NC-06 is one of
    project_spec's 13 redistricting-flagged districts with an
    already-documented unstable baseline (the "$5M threshold jump"
    finding elsewhere in this project).
  - FL-07 (2022 R): the K~8 pool's headline "discovery" turned out to be
    almost entirely the mechanical floor (+0.0622 -> +0.0037 excl. final
    week). At K~17, it is ALSO dominated by TN-09, which is durable from
    day one -- FL-07 was never a genuine opportunity at any pool size
    tested.

Only ONE genuine, doubly-corrected mid-season wait case survives: AZ-09
(2024 R), net +0.0490, rising smoothly from 60% to >100% retention by 30
days out -- not redistricting-flagged, not a mechanical-floor artifact,
and it beats every alternative available today INCLUDING the best
immediate option (NM-01).

Consumes results/conditional_waiting_value_union_clean.json (run
analyze_conditional_waiting_value_union.py first).

Output: figures/static/conditional_waiting_value_final.png
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

GREEN = "#1a9850"     # genuine, doubly-corrected, worth waiting for
TEAL = "#3288bd"      # already durable / dominant -- deploy now, nothing to gain from waiting
GRAY = "#8073ac"      # debunked by the stress test (was a story at smaller K, dominated or artifact at K~17)
INK = "#333333"


def main() -> None:
    data = json.load(open(RESULTS / "conditional_waiting_value_union_clean.json"))

    # Curated rows: the surviving genuine case, the dominant "deploy now" case per pool,
    # and the three previously-reported "story" races the stress test overturned.
    rows_spec = [
        ("2024_R", "AZ-09", "genuine"),
        ("2024_R", "NM-01", "genuine"),
        ("2024_D", "FL-27", "dominant"),
        ("2022_D", "FL-02", "dominant"),
        ("2022_R", "TN-09", "dominant"),
        ("2024_D", "CT-02", "debunked"),
        ("2024_R", "FL-22", "debunked"),
        ("2024_R", "NV-01", "debunked"),
        ("2022_R", "FL-07", "debunked"),
    ]
    color_by_cat = {"genuine": GREEN, "dominant": TEAL, "debunked": GRAY}
    label_by_cat = {
        "genuine": "Genuine mid-season wait, survives full stress test (K~17, excl. redistricting-flagged, excl. mechanical floor)",
        "dominant": "Already durable / best option from day one -- deploy now, nothing to gain from waiting",
        "debunked": "Looked like a genuine wait story at smaller K -- dominated or artifact once the pool widened",
    }

    rows = []
    for key, district, cat in rows_spec:
        cycle, side = key.split("_")
        r = next(x for x in data[key] if x["district"] == district)
        rows.append(dict(label=f"{district} '{cycle[2:]} ({side})", category=cat, **r))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    order = {"genuine": 0, "dominant": 1, "debunked": 2}
    rows.sort(key=lambda r: (order[r["category"]], -r["net_genuine"]))
    ys = np.arange(len(rows))

    for y, r in zip(ys, rows):
        color = color_by_cat[r["category"]]
        ax.plot([r["V_now"], r["V_wait_genuine"]], [y, y], color=color, linewidth=2.5, alpha=0.85, zorder=2)
        ax.scatter([r["V_now"]], [y], s=55, facecolors="white", edgecolors=color, linewidths=2, zorder=3)
        ax.scatter([r["V_wait_genuine"]], [y], s=75, facecolors=color, edgecolors=color, linewidths=1, zorder=3)
        ax.scatter([r["best_immediate"]], [y], marker="D", s=45, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)
        ax.text(max(r["V_now"], r["V_wait_genuine"], r["best_immediate"]) + 0.004, y,
                f"net {r['net_genuine']:+.3f}", fontsize=7.5, color=color, va="center")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Expected seats, $1M delta")
    ax.set_title("After the K~15-20 stress test: only one genuine mid-season wait case survives",
                  fontsize=13, loc="left", fontweight="bold")

    start = 0
    for cat in ("genuine", "dominant", "debunked"):
        n = sum(1 for r in rows if r["category"] == cat)
        if n:
            ax.axhspan(start - 0.5, start + n - 0.5, color=color_by_cat[cat], alpha=0.06, zorder=0)
            start += n

    hollow = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK, markersize=7, label="V_now (deploy today)")
    filled = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markeredgecolor=INK, markersize=7, label="V_wait (genuine, excl. mechanical floor)")
    diamond = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="none", markeredgecolor=INK, markersize=7, label="best_immediate (best alternative available today, same pool)")
    cat_patches = [plt.Rectangle((0, 0), 1, 1, color=color_by_cat[c], alpha=0.5, label=label_by_cat[c]) for c in ("genuine", "dominant", "debunked")]
    ax.legend(handles=[hollow, filled, diamond] + cat_patches, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7.8, frameon=False, handletextpad=0.6)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.set_ylim(-0.5, len(rows) - 0.5)

    fig.text(0.5, -0.02,
              "Widening the candidate pool from K=3 to a principled K~15-20 union screen, then removing districts with a redistricting-flagged (unreliable) baseline, "
              "overturns most of the earlier 'genuine mid-season wait' stories: FL-27 and TN-09 dominate their pools from day one (nothing to wait for), CT-02/FL-22/NV-01/FL-07 "
              "are all beaten by a better option already sitting there. AZ-09 is the one case that survives every check.",
              ha="center", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "conditional_waiting_value_final.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
