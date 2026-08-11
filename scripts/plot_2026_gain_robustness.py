#!/usr/bin/env python3
"""
Visualize the three robustness checks on the 2026 live +7.9-seat read:
  - Generic-ballot sensitivity (robustness_2026_live_gain.py)
  - Bootstrap CI on the thin DCCC-observed sample (robustness_2026_live_gain.py)
  - Per-race / per-category "feature importance" decomposition
    (decompose_2026_gain_by_race.py)

Four panels:
  Top-left:     gain vs. generic ballot, historical range, live estimate marked.
  Top-right:    bootstrap distribution of the gain, point estimate marked.
  Bottom-left:  gain contribution by Cook rating category.
  Bottom-right: gain contribution by incumbency status, with the committed-
                vs-zero-committed split called out as text (18 vs 416 races
                is too lopsided a split to read as a bar chart fairly).

Reads the three CSVs written by robustness_2026_live_gain.py and
decompose_2026_gain_by_race.py. Run those first if any is missing.

Usage:
    python scripts/plot_2026_gain_robustness.py

Output: outputs/gain_robustness_2026.png
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
C_DCCC = "#1a6faf"
C_CAVEAT = "#c0392b"
C_MUTED = "#888888"
C_LIVE = "#8e44ad"


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run the script that produces it first.")
    return pd.read_csv(path)


def main() -> None:
    outputs = config.outputs_path()
    gb_df = _require(outputs / "robustness_2026_gb_sensitivity.csv")
    boot_df = _require(outputs / "robustness_2026_bootstrap_gains.csv")
    cat_df = _require(outputs / "gain_decomposition_2026_by_category.csv")

    point_gain = gb_df.loc[gb_df["gb"] == 5.02, "gain"].iloc[0]

    fig, ((ax_gb, ax_boot), (ax_cook, ax_incumb)) = plt.subplots(2, 2, figsize=(13, 10))

    # ── Top-left: GB sensitivity ────────────────────────────────────────────
    gb_sorted = gb_df.sort_values("gb")
    ax_gb.plot(gb_sorted["gb"], gb_sorted["gain"], "o-", color=C_MODEL, linewidth=2, markersize=7)
    live_row = gb_sorted[gb_sorted["gb"] == 5.02].iloc[0]
    ax_gb.plot([live_row["gb"]], [live_row["gain"]], "o", color=C_LIVE, markersize=11, zorder=5)
    ax_gb.annotate("2026 live\n(D+5.02)", (live_row["gb"], live_row["gain"]),
                    xytext=(live_row["gb"] - 1, live_row["gain"] + 0.55),
                    fontsize=9, color=C_LIVE, ha="center")
    ax_gb.axhline(0, color="#444444", linewidth=0.7, alpha=0.5)
    ax_gb.set_xlabel("Generic ballot (D − R, points)")
    ax_gb.set_ylabel("Gain (seats)")
    ax_gb.set_title("Gain is NOT primarily driven by the favorable\nnational environment", fontsize=11)
    ax_gb.grid(axis="y", color="#e5e4df", linewidth=0.7, zorder=0)
    ax_gb.set_axisbelow(True)
    ax_gb.text(0.02, 0.03,
               f"Even at 2014's D−5.8 (worst historical\nenvironment for Dems): still +{gb_sorted['gain'].iloc[0]:.2f}",
               transform=ax_gb.transAxes, fontsize=8.5, color=C_MUTED, va="bottom")

    # ── Top-right: bootstrap CI ─────────────────────────────────────────────
    ax_boot.hist(boot_df["boot_gain"], bins=40, color=C_DCCC, alpha=0.75, zorder=3)
    p5, p95 = np.percentile(boot_df["boot_gain"], [5, 95])
    ax_boot.axvline(point_gain, color=C_CAVEAT, linewidth=2, linestyle="--", zorder=4)
    ax_boot.text(point_gain, ax_boot.get_ylim()[1] * 0.92, f"  point estimate\n  +{point_gain:.2f}",
                 color=C_CAVEAT, fontsize=9)
    ax_boot.axvspan(p5, p95, alpha=0.12, color=C_MODEL, zorder=1)
    ax_boot.set_xlabel("Bootstrapped gain (seats)")
    ax_boot.set_ylabel("Resamples (of 1000)")
    ax_boot.set_title("Bootstrap of the 18 committed races:\nstable WITHIN this sample, not proof the sample is representative",
                       fontsize=10.5)
    ax_boot.grid(axis="y", color="#e5e4df", linewidth=0.7, zorder=0)
    ax_boot.set_axisbelow(True)

    # ── Bottom-left: by Cook rating ─────────────────────────────────────────
    cook = cat_df[cat_df["dimension"] == "cook_rating"].copy()
    order = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
    cook["category"] = pd.Categorical(cook["category"], categories=order, ordered=True)
    cook = cook.sort_values("category")
    colors_cook = ["#1a6faf" if "D" in c else ("#888888" if c == "Toss-Up" else "#c0392b")
                   for c in cook["category"]]
    bars = ax_cook.barh(cook["category"].astype(str), cook["total_delta_seats"], color=colors_cook, zorder=3)
    for b, n in zip(bars, cook["count"]):
        ax_cook.text(b.get_width() + 0.03, b.get_y() + b.get_height() / 2, f"n={n}",
                     va="center", fontsize=8, color=C_MUTED)
    ax_cook.set_xlabel("Seats contributed")
    ax_cook.set_title("Gain by Cook rating: concentrated in Likely R --\nthe SAME concentration Paper II already flagged", fontsize=10.5)
    ax_cook.grid(axis="x", color="#e5e4df", linewidth=0.7, zorder=0)
    ax_cook.set_axisbelow(True)
    ax_cook.invert_yaxis()

    # ── Bottom-right: by incumbency, with committed-status called out ──────
    incumb = cat_df[cat_df["dimension"] == "incumb_status"].sort_values("total_delta_seats", ascending=True)
    ax_incumb.barh(incumb["category"], incumb["total_delta_seats"], color=C_MODEL, zorder=3)
    for i, (cat, val, n) in enumerate(zip(incumb["category"], incumb["total_delta_seats"], incumb["count"])):
        ax_incumb.text(val + 0.05, i, f"n={n}", va="center", fontsize=8, color=C_MUTED)
    ax_incumb.set_xlabel("Seats contributed")
    ax_incumb.set_title("Gain by incumbency: challengers dominate", fontsize=10.5)
    ax_incumb.grid(axis="x", color="#e5e4df", linewidth=0.7, zorder=0)
    ax_incumb.set_axisbelow(True)

    committed = cat_df[cat_df["dimension"] == "committed_status"]
    committed_pct = committed.set_index("category")["total_delta_seats"]
    zero_val = committed_pct.get("416 zero-committed races", 0.0)
    comm_val = committed_pct.get("18 currently-committed races", 0.0)
    ax_incumb.text(
        0.98, 0.03,
        f"98.8% of the gain (+{zero_val:.2f} of +{zero_val+comm_val:.2f}) comes from the\n"
        f"416 races DCCC hasn't committed money to yet --\n"
        f"only +{comm_val:.2f} from the 18 it has (same 'selection\n"
        f"dominates' pattern as Paper I's 2022/2024 finding)",
        transform=ax_incumb.transAxes, fontsize=8.5, color=C_CAVEAT, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="#fdf2f2", edgecolor=C_CAVEAT, linewidth=0.6),
    )

    fig.suptitle("Why is the 2026 live gain +7.9 seats? Three robustness checks",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()

    out_path = outputs / "gain_robustness_2026.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
