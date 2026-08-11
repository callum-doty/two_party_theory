#!/usr/bin/env python3
"""
Holistic backtest summary chart.

Three-panel figure aligned with the paper's framework and the backtest findings:

  A (left, tall):   Efficiency test — rank-rank scatter of DCCC spending vs MSG
                    for the 53 competitive races.  Spearman ρ = -0.582.
  B (top-right):    Allocator comparison — expected-seat gain vs DCCC baseline
                    for four strategies, 2024 and 2022 shown together.
  C (bottom-right): Cross-cycle validation — Spearman ρ and seat-gain metrics
                    for 2024 (primary) and 2022 (out-of-sample).

Output: outputs/backtest_summary.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"

# ── colour palette (consistent with other charts) ─────────────────────────────
COOK_COLOR = {
    "Safe D":   "#1a3a5c",
    "Likely D": "#2e6da4",
    "Lean D":   "#5b9bd5",
    "Toss-Up":  "#e6a817",
    "Lean R":   "#e87c7c",
    "Likely R": "#c0392b",
    "Safe R":   "#7b0000",
}
COMP_RATINGS = ["Lean D", "Toss-Up", "Lean R"]

C_2024  = "#1f4e79"   # dark navy  — primary
C_2022  = "#7bafd4"   # light blue — OOS
C_MODEL = "#1a7a40"   # green      — model optimizer
C_NULL  = "#9467bd"   # purple     — null equal-weight
C_COOK  = "#8c6d3f"   # tan        — Cook-implied
C_REF   = "#aaaaaa"   # gray       — reference / DCCC baseline


# ── data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(OUT / "race_table_baseline.csv")
comp = df[df["cook_rating"].isin(COMP_RATINGS)].copy()

# Ranks: higher rank = more spending / higher MSG
comp["spend_rank"] = comp["d_total"].rank(method="average")
comp["msg_rank"]   = comp["msg_i_per_1m"].rank(method="average")
n_comp = len(comp)

# Brier scores (computed from data, not hardcoded — same COOK_PROB mapping as make_charts.py)
COOK_PROB = {
    "Safe D": 0.97, "Likely D": 0.85, "Lean D": 0.70,
    "Toss-Up": 0.50,
    "Lean R": 0.30, "Likely R": 0.15, "Safe R": 0.03,
}
_scored = df[df["outcome"].notna()].copy()
_scored["outcome_bin"] = (_scored["outcome"] == "D").astype(int)
_scored["cook_prob"] = _scored["cook_rating"].map(COOK_PROB)
model_brier = float(((_scored["p_win"] - _scored["outcome_bin"]) ** 2).mean())
cook_brier = float(((_scored["cook_prob"] - _scored["outcome_bin"]) ** 2).mean())
brier_pct_better = (cook_brier - model_brier) / cook_brier * 100

rho, pval = scipy_stats.spearmanr(comp["d_total"], comp["msg_i_per_1m"])

# 2024 allocator data
alloc_24 = pd.read_csv(OUT / "allocator_comparison_table.csv")
dccc_24  = alloc_24.loc[alloc_24["allocator"] == "DCCC observed", "expected_seats"].iloc[0]

def gain(alloc_df, baseline):
    return {
        "Model optimizer":     alloc_df.loc[alloc_df["allocator"] == "Model optimizer",  "expected_seats"].iloc[0] - baseline,
        "Null (equal-weight)": alloc_df.loc[alloc_df["allocator"] == "Null (equal-weight)", "expected_seats"].iloc[0] - baseline,
        "Cook-implied":        alloc_df.loc[alloc_df["allocator"] == "Cook-implied",      "expected_seats"].iloc[0] - baseline,
    }

gains_24 = gain(alloc_24, dccc_24)

alloc_22  = pd.read_csv(OUT / "allocator_comparison_table_2022.csv")
dccc_22   = alloc_22.loc[alloc_22["allocator"] == "DCCC observed", "expected_seats"].iloc[0]
gains_22  = gain(alloc_22, dccc_22)

# Aggregate stats for cross-cycle panel
agg_24 = pd.read_csv(OUT / "aggregate_summary_baseline.csv").iloc[0]
agg_22 = pd.read_csv(OUT / "aggregate_summary_baseline_2022.csv").iloc[0]


# ── figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(
    2, 2,
    width_ratios=[1.45, 1],
    height_ratios=[1.05, 1],
    hspace=0.42,
    wspace=0.32,
    left=0.07, right=0.97,
    top=0.90,  bottom=0.08,
)
ax_scatter = fig.add_subplot(gs[:, 0])   # full left column
ax_alloc   = fig.add_subplot(gs[0, 1])  # top-right
ax_cycle   = fig.add_subplot(gs[1, 1])  # bottom-right


# ══ Panel A: rank-rank scatter ════════════════════════════════════════════════

for rating in COMP_RATINGS:
    sub = comp[comp["cook_rating"] == rating]
    ax_scatter.scatter(
        sub["spend_rank"], sub["msg_rank"],
        color=COOK_COLOR[rating], s=60, alpha=0.85,
        label=rating, zorder=3,
    )

# Efficient-allocation reference: positive diagonal
diag = np.array([1, n_comp])
ax_scatter.plot(diag, diag, "--", color=C_REF, lw=1.2, label="Efficient allocation (ρ = +1.0)", zorder=2)

# Regression line through the data (negative slope)
m, b, *_ = scipy_stats.linregress(comp["spend_rank"], comp["msg_rank"])
x_fit = np.linspace(1, n_comp, 100)
ax_scatter.plot(x_fit, m * x_fit + b, "-", color="#c0392b", lw=2, alpha=0.7,
                label=f"Observed trend (ρ = {rho:.3f})", zorder=4)

# Label a handful of notable races
label_races = {
    "NC-06":  "NC-06\n(R won)",
    "NC-14":  "NC-14\n(R won)",
    "CA-45":  "CA-45",
    "PA-07":  "PA-07",
    "NY-22":  "NY-22",
}
for did, label in label_races.items():
    row = comp[comp["district_id"] == did]
    if row.empty:
        continue
    x, y = float(row["spend_rank"].iloc[0]), float(row["msg_rank"].iloc[0])
    ax_scatter.annotate(
        label, (x, y),
        xytext=(8, 4), textcoords="offset points",
        fontsize=8.5, color="#333333",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
    )

ax_scatter.set_xlabel("DCCC Observed Spending Rank\n(1 = least spent → 53 = most spent)", fontsize=10.5)
ax_scatter.set_ylabel("Marginal Seat Gain Rank\n(1 = lowest MSG → 53 = highest MSG)", fontsize=10.5)
ax_scatter.set_xlim(0, n_comp + 2)
ax_scatter.set_ylim(0, n_comp + 2)
ax_scatter.set_xticks([1, 10, 20, 30, 40, 53])
ax_scatter.set_yticks([1, 10, 20, 30, 40, 53])

# Under efficient allocation, annotation goes in top-right; observed trend goes
# from top-left (high MSG, low spend) to bottom-right (low MSG, high spend).
ax_scatter.text(
    0.04, 0.96,
    "← Under-funded\n   (High MSG, Low Spend)",
    transform=ax_scatter.transAxes,
    fontsize=8.5, va="top", ha="left", color="#1a7a40",
    style="italic",
)
ax_scatter.text(
    0.96, 0.04,
    "Over-funded →\n(Low MSG, High Spend)",
    transform=ax_scatter.transAxes,
    fontsize=8.5, va="bottom", ha="right", color="#c0392b",
    style="italic",
)

rho_22   = float(agg_22["spearman_rho"])
n_22     = int(agg_22["n_competitive"])
p_22     = float(agg_22["spearman_p_value"])
p22_str  = "p < 0.001" if p_22 < 0.001 else f"p = {p_22:.3f}"

# Spearman annotation box
ax_scatter.text(
    0.04, 0.06,
    f"Spearman ρ = {rho:.3f}   (p < 0.001, n = {n_comp})\n"
    f"2022 OOS: ρ = {rho_22:.3f}  ({p22_str}, n = {n_22})",
    transform=ax_scatter.transAxes,
    fontsize=9.5, va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8),
)

ax_scatter.set_title(
    "A.  Efficiency Test: DCCC Spends More Where Marginal Seat Gain Is Lower",
    fontsize=11.5, fontweight="bold", loc="left", pad=10,
)
ax_scatter.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)


# ══ Panel B: allocator comparison ════════════════════════════════════════════

strategies = ["Model optimizer", "Null (equal-weight)", "Cook-implied"]
labels     = ["Model optimizer", "Null\n(equal-weight)", "Cook-implied"]
y_pos      = [2, 1, 0]

# Plot 2024 and 2022 side by side
bar_h = 0.28
for i, (strat, lbl, y) in enumerate(zip(strategies, labels, y_pos)):
    g24 = gains_24[strat]
    g22 = gains_22[strat]

    col = C_MODEL if strat == "Model optimizer" else (C_NULL if "Null" in strat else C_COOK)

    ax_alloc.barh(y + bar_h/2 + 0.02, g24, height=bar_h, color=col, alpha=0.9,
                  label="2024 (primary)" if i == 0 else None)
    ax_alloc.barh(y - bar_h/2 - 0.02, g22, height=bar_h, color=col, alpha=0.40,
                  hatch="///", label="2022 (OOS)" if i == 0 else None)

    # Value annotations
    x_off_24 = 0.12 if g24 >= 0 else -0.12
    x_off_22 = 0.12 if g22 >= 0 else -0.12
    ax_alloc.text(g24 + x_off_24, y + bar_h/2 + 0.02, f"{g24:+.2f}",
                  va="center", ha="left" if g24 >= 0 else "right", fontsize=9, color=col)
    ax_alloc.text(g22 + x_off_22, y - bar_h/2 - 0.02, f"{g22:+.2f}",
                  va="center", ha="left" if g22 >= 0 else "right", fontsize=9, color=col, alpha=0.7)

ax_alloc.axvline(0, color="#333333", lw=0.9, zorder=5)
ax_alloc.set_yticks(y_pos)
ax_alloc.set_yticklabels(labels, fontsize=10)
ax_alloc.set_xlabel("Expected Seat Gain vs. DCCC Observed Allocation", fontsize=9.5)
ax_alloc.set_xlim(-1.2, 7.5)
ax_alloc.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax_alloc.legend(frameon=False, fontsize=9, loc="lower right")
ax_alloc.set_title(
    "B.  Seat Gain vs. DCCC Baseline\n(same total budget, reallocated)",
    fontsize=10.5, fontweight="bold", loc="left", pad=8,
)
ax_alloc.text(
    0.97, 0.97,
    f"2024 DCCC baseline: {dccc_24:.1f} seats\n2022 DCCC baseline: {dccc_22:.1f} seats",
    transform=ax_alloc.transAxes,
    fontsize=8.5, va="top", ha="right", color="#555555",
)


# ══ Panel C: cross-cycle validation ══════════════════════════════════════════

metrics = {
    "Spearman ρ\n(× 10)":         (abs(float(agg_24["spearman_rho"])) * 10,
                                    abs(float(agg_22["spearman_rho"])) * 10),
    "Model gain\nvs. DCCC (seats)": (gains_24["Model optimizer"],
                                     gains_22["Model optimizer"]),
    "Null gain\nvs. DCCC (seats)":  (gains_24["Null (equal-weight)"],
                                     gains_22["Null (equal-weight)"]),
}

x_pos = np.arange(len(metrics))
bw = 0.3
metric_labels = list(metrics.keys())
vals_24 = [v[0] for v in metrics.values()]
vals_22 = [v[1] for v in metrics.values()]

bars_24 = ax_cycle.bar(x_pos - bw/2, vals_24, bw, color=C_2024,  alpha=0.9, label="2024 (primary)")
bars_22 = ax_cycle.bar(x_pos + bw/2, vals_22, bw, color=C_2022, alpha=0.75,
                       hatch="///", label="2022 (OOS)", edgecolor=C_2022)

# Annotate bars
for bar, val in zip(bars_24, vals_24):
    ax_cycle.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=C_2024, fontweight="bold")
for bar, val in zip(bars_22, vals_22):
    ax_cycle.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color="#4a7ca0", fontweight="bold")

ax_cycle.set_xticks(x_pos)
ax_cycle.set_xticklabels(metric_labels, fontsize=9.5)
ax_cycle.set_ylabel("Value (see metric labels)", fontsize=9)
ax_cycle.set_ylim(0, max(max(vals_24), max(vals_22)) * 1.25)
ax_cycle.axhline(0, color="#bbbbbb", lw=0.7)
ax_cycle.legend(frameon=False, fontsize=9, loc="upper right")
ax_cycle.set_title(
    "C.  Cross-Cycle Validation (2022 Fully Out-of-Sample)\n"
    "Direction and magnitude consistent across cycles",
    fontsize=10.5, fontweight="bold", loc="left", pad=8,
)

# Add note that ρ is scaled for readability
ax_cycle.text(
    x_pos[0], vals_24[0] * 0.48,
    "×10 scale",
    ha="center", fontsize=7.5, color="#666666", style="italic",
)

# Brier score callout as text in the panel
ax_cycle.text(
    0.03, 0.05,
    f"Model Brier: {model_brier:.4f}   Cook Brier: {cook_brier:.4f}\n"
    f"(model +{brier_pct_better:.0f}% better calibrated, 2024)",
    transform=ax_cycle.transAxes,
    fontsize=8.5, va="bottom", ha="left", color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", lw=0.7),
)


# ── figure-level title and footnote ──────────────────────────────────────────

fig.suptitle(
    "DCCC Spending Allocation Efficiency — 2024 House Cycle",
    fontsize=14, fontweight="bold", y=0.97,
)
fig.text(
    0.5, 0.01,
    "Sources: FEC bulk filings, MIT Election Lab, Cook Political Report, RealClearPolitics. "
    "Estimation panel: 2012–2022. 2022 OOS uses 2012–2020 panel only. "
    "α₅ (indiv_share) zeroed out — see §5.4.",
    ha="center", fontsize=8, color="#666666",
)

save_path = OUT / "backtest_summary.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved → {save_path}")
