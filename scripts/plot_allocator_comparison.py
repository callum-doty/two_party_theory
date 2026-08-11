#!/usr/bin/env python3
"""
Four-way allocator spending comparison for competitive races.

Compares, for each of the 53 competitive races (Toss-Up / Lean D / Lean R),
how much money each allocation strategy directs to that race:

  1. DCCC observed      — actual 2024 spending
  2. Model optimizer    — nonlinear SLSQP optimizer recommendation
  3. Cook-implied       — proportional to Cook win probability
  4. Uniform            — equal share across all competitive races

Races are sorted by DCCC observed spending (descending) so the DCCC's
concentration pattern is immediately visible against the alternatives.

Output: outputs/allocator_spending_by_race.png
(Renamed 2026-07-22 from outputs/allocator_comparison.png -- that filename
collided with make_charts.py's aggregate E[Seats] bar chart, a different
chart documented under the same name in FINDINGS.md Section 11. Whichever
script ran last was silently overwriting the other's output.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import config


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(ROOT / "outputs" / "race_table_baseline.csv")

COMP_RATINGS = config.competitive_ratings()   # ['Toss-Up', 'Lean D', 'Lean R']
COOK_MAP     = config.cook_win_probs()

comp = df[df["cook_rating"].isin(COMP_RATINGS)].copy()
budget = df["d_total"].sum()   # total D budget across all 433 races


# ── Compute allocations ($M) ──────────────────────────────────────────────────

# 1. DCCC observed
comp["dccc_m"] = comp["d_total"] / 1e6

# 2. Model optimizer
comp["model_m"] = comp["recommended_share"] * budget / 1e6

# 3. Cook-implied: proportional to Cook win probability within competitive set
comp["cook_prob"] = comp["cook_rating"].map(COOK_MAP)
cook_total = comp["cook_prob"].sum()
comp["cook_share_comp"] = comp["cook_prob"] / cook_total   # fraction within competitive

# 4. Uniform: equal share across competitive races
n_comp = len(comp)
comp["uniform_share_comp"] = 1.0 / n_comp

# Normalise Cook and Uniform to the DCCC competitive total so the comparison
# is about *distribution pattern*, not how much each strategy reserves for
# non-competitive races.  DCCC and Model are shown in actual dollars.
dccc_comp_total = comp["dccc_m"].sum()
comp["cook_m"]    = comp["cook_share_comp"]    * dccc_comp_total
comp["uniform_m"] = comp["uniform_share_comp"] * dccc_comp_total

# Sort by Δ = model − DCCC (ascending: largest cuts left, largest additions right)
comp["delta_m"] = comp["model_m"] - comp["dccc_m"]
comp = comp.sort_values("delta_m", ascending=True).reset_index(drop=True)
comp["rank"] = comp.index

# Abbreviated race labels
comp["label"] = comp["district_id"].str.replace("-", "‑")  # non-breaking hyphen


# ── Colour palette ────────────────────────────────────────────────────────────

C_DCCC    = "#1a6faf"   # blue
C_MODEL   = "#2a9d4f"   # green
C_COOK    = "#e07b39"   # orange
C_UNIFORM = "#888888"   # grey

OUTCOME_EDGE = {"D": "#1a6faf", "R": "#c0392b", None: "#aaaaaa"}


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 6))

x = comp["rank"].values

# ── Zone shading (cuts / neutral / additions) ──────────────────────────────
cut_boundary = (comp["delta_m"] < -0.5).sum()   # races where model removes >$0.5M
add_boundary = (comp["delta_m"] < 0.5).sum()    # races where model adds >$0.5M

ax.axvspan(-1, cut_boundary - 0.5,  alpha=0.06, color="#c0392b", zorder=0)
ax.axvspan(cut_boundary - 0.5, add_boundary - 0.5, alpha=0.04, color="#888888", zorder=0)
ax.axvspan(add_boundary - 0.5, n_comp, alpha=0.06, color="#2a9d4f", zorder=0)

ymax = comp[["dccc_m","model_m"]].max().max()
for label, x_pos, ha in [
    ("← cuts", cut_boundary / 2, "center"),
    ("neutral", (cut_boundary + add_boundary) / 2, "center"),
    ("additions →", (add_boundary + n_comp) / 2, "center"),
]:
    ax.text(x_pos - 0.5, ymax * 1.04, label,
            fontsize=8, color="#555555", ha=ha, va="bottom", style="italic")

# Lines connecting DCCC to model (show direction of reallocation)
for i, row in comp.iterrows():
    lo, hi = sorted([row["dccc_m"], row["model_m"]])
    color = C_MODEL if row["model_m"] > row["dccc_m"] else C_DCCC
    ax.plot([row["rank"], row["rank"]], [lo, hi],
            color=color, lw=0.6, alpha=0.35, zorder=1)

# Four series
ax.scatter(x, comp["cook_m"],    color=C_COOK,    s=28, zorder=3,
           marker="D", alpha=0.85, label="Cook-implied")
ax.scatter(x, comp["uniform_m"], color=C_UNIFORM, s=22, zorder=2,
           marker="s", alpha=0.7,  label=f"Uniform (\\${comp['uniform_m'].iloc[0]:.1f}M each)")
ax.scatter(x, comp["model_m"],   color=C_MODEL,   s=38, zorder=4,
           marker="^", alpha=0.9,  label="Model optimizer (nonlinear)")
ax.scatter(x, comp["dccc_m"],    color=C_DCCC,    s=38, zorder=5,
           marker="o", alpha=0.9,  label="DCCC observed")

# Outcome ring (D=blue edge, R=red edge) on DCCC dots
for _, row in comp.iterrows():
    ec = OUTCOME_EDGE.get(row.get("outcome"), "#aaaaaa")
    ax.scatter(row["rank"], row["dccc_m"],
               s=100, facecolors="none", edgecolors=ec, lw=1.4, zorder=6)

# Annotate a handful of key races
ANNOTATE = {"VA-10", "PA-07", "NC-06", "NC-14", "TX-15", "NJ-07"}
for _, row in comp.iterrows():
    if row["district_id"] in ANNOTATE:
        y_val = max(row["dccc_m"], row["model_m"]) + 0.5
        ax.text(row["rank"], y_val, row["district_id"],
                fontsize=7.5, ha="center", va="bottom",
                color="#333333", rotation=0)

# X-axis: race labels rotated
ax.set_xticks(x)
ax.set_xticklabels(comp["label"], rotation=90, fontsize=6.5)

ax.set_xlabel(r"Competitive race  (sorted by $\Delta$ = Optimizer $-$ DCCC,  cuts $\leftarrow$ | $\rightarrow$ additions)", fontsize=10)
ax.set_ylabel("Spending ($M)", fontsize=11)
ax.set_title(
    "Competitive Race Spending: Four Allocation Strategies (2024 House, 53 races)\n"
    r"Sorted by $\Delta$ = Optimizer $-$ DCCC.  "
    "Cook-implied and Uniform scaled to DCCC competitive total for comparability.",
    fontsize=10,
)

# Legend: allocators
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

# Outcome legend (ring colour)
d_patch = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                         markeredgecolor=OUTCOME_EDGE["D"], markeredgewidth=1.4,
                         markersize=9, label="Outcome: D won")
r_patch = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                         markeredgecolor=OUTCOME_EDGE["R"], markeredgewidth=1.4,
                         markersize=9, label="Outcome: R won")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [d_patch, r_patch],
          labels=ax.get_legend_handles_labels()[1] + ["Outcome: D won", "Outcome: R won"],
          loc="upper right", fontsize=8.5, framealpha=0.92)

ax.grid(axis="y", alpha=0.2)
ax.set_xlim(-1, n_comp)
ax.set_ylim(-0.5, comp["dccc_m"].max() * 1.12)

fig.tight_layout()
out = ROOT / "outputs" / "allocator_spending_by_race.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# Summary table
print("\nAllocation totals to competitive races ($M):")
for col, label in [("dccc_m","DCCC observed"), ("model_m","Model optimizer"),
                   ("cook_m","Cook-implied (scaled)"), ("uniform_m","Uniform (scaled)")]:
    print(f"  {label:<28}: ${comp[col].sum():.1f}M")
print(f"\nTotal D budget (all races):     ${budget/1e6:.1f}M")
print(f"DCCC competitive total:         ${dccc_comp_total:.1f}M")
print(f"Model competitive total (raw):  ${comp['model_m'].sum():.1f}M")
print(f"Competitive races: {n_comp}")
