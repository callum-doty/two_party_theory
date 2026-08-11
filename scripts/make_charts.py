#!/usr/bin/env python3
"""
Generate publication-quality charts from backtest outputs.

Charts produced:
  1. msg_efficiency.png        — MSG vs D spend (Spearman efficiency test)
  2. model_calibration.png     — P_win bins vs actual win rate
  3. spending_by_cook.png      — D/R spend distribution by Cook rating
  4. allocator_comparison.png  — E[Seats] comparison across allocators
  5. allocation_shift.png      — Recommended vs DCCC allocation shift
  6. spending_ratio_vs_pvi.png — D share of spend vs district lean
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

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

OUT  = Path(__file__).parent.parent / "outputs"
DATA = OUT / "race_table_baseline.csv"
AGG  = OUT / "aggregate_summary_baseline.csv"
ALLOC_TABLE = OUT / "allocator_comparison_table.csv"

df = pd.read_csv(DATA)

# Load comparison table produced by run_backtest.py (authoritative; avoids hardcoding)
_alloc_df = pd.read_csv(ALLOC_TABLE) if ALLOC_TABLE.exists() else None
df["d_total_m"] = df["d_total"] / 1e6
df["r_total_m"] = df["r_total"] / 1e6
df["outcome_bin"] = (df["outcome"] == "D").astype(int)

COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
COOK_COLOR = {
    "Safe D":   "#1a3a5c",
    "Likely D": "#2e6da4",
    "Lean D":   "#5b9bd5",
    "Toss-Up":  "#e6a817",
    "Lean R":   "#e87c7c",
    "Likely R": "#c0392b",
    "Safe R":   "#7b0000",
}
COOK_PROB = {
    "Safe D": 0.97, "Likely D": 0.85, "Lean D": 0.70,
    "Toss-Up": 0.50,
    "Lean R": 0.30, "Likely R": 0.15, "Safe R": 0.03,
}
COMPETITIVE = {"Toss-Up", "Lean D", "Lean R"}


# ─── 1. MSG Efficiency Scatter ──────────────────────────────────────────────

comp = df[df["cook_rating"].isin(COMPETITIVE)].copy()
rho, pval = scipy_stats.spearmanr(comp["d_total"], comp["msg_i_per_1m"])

fig, ax = plt.subplots(figsize=(8, 5.5))
for rating in ["Lean D", "Toss-Up", "Lean R"]:
    sub = comp[comp["cook_rating"] == rating]
    ax.scatter(sub["d_total_m"], sub["msg_i_per_1m"],
               c=COOK_COLOR[rating], s=65, alpha=0.82, linewidths=0.4,
               edgecolors="white", label=rating, zorder=3)

ax.set_xlabel("Total Democratic Spending ($M)", fontsize=12)
ax.set_ylabel("Marginal Seat Gain per $1M (MSG)", fontsize=12)
ax.set_title("DCCC Spends More Where Returns Are Lower\n(Competitive races, 2024 House)",
             fontsize=13, fontweight="bold")

note = f"Spearman ρ = {rho:.2f}  (p = {pval:.4f})\nn = {len(comp)} competitive races"
ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

ax.legend(title="Cook Rating", frameon=False, fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
fig.tight_layout()
fig.savefig(OUT / "msg_efficiency.png", bbox_inches="tight")
plt.close(fig)
print("✓ msg_efficiency.png")


# ─── 2. Model Calibration ───────────────────────────────────────────────────

rows = df[df["outcome"].notna()].copy()
rows["cook_prob"] = rows["cook_rating"].map(COOK_PROB)
bins = np.linspace(0, 1, 11)
bin_mids = (bins[:-1] + bins[1:]) / 2

model_rates, model_ns, model_xs = [], [], []
cook_rates, cook_ns, cook_xs = [], [], []

for lo, hi, mid in zip(bins[:-1], bins[1:], bin_mids):
    for col, rates, ns, xs in [
        ("p_win",     model_rates, model_ns, model_xs),
        ("cook_prob", cook_rates,  cook_ns,  cook_xs),
    ]:
        mask = (rows[col] >= lo) & (rows[col] < hi)
        grp = rows[mask]
        if len(grp) >= 3:
            rates.append(grp["outcome_bin"].mean())
            ns.append(len(grp))
            xs.append(mid)

model_brier = float(((rows["p_win"] - rows["outcome_bin"]) ** 2).mean())
cook_brier = float(((rows["cook_prob"] - rows["outcome_bin"]) ** 2).mean())

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1.2, label="Perfect calibration")
ax.plot(cook_xs, cook_rates, "s--", color="#c0392b", lw=1.5,
        ms=8, alpha=0.8, label=f"Cook Rating  (Brier = {cook_brier:.4f})")
ax.scatter(model_xs, model_rates,
           s=[max(30, n * 3) for n in model_ns],
           c="#2e6da4", zorder=4, edgecolors="white", linewidths=0.5,
           label=f"Model P_win  (Brier = {model_brier:.4f})")
ax.plot(model_xs, model_rates, "-", color="#2e6da4", lw=1.5, alpha=0.7)

ax.set_xlabel("Predicted Win Probability", fontsize=12)
ax.set_ylabel("Actual Democratic Win Rate", fontsize=12)
ax.set_title("Model Calibration vs Cook Political Report\n(2024 House races with outcomes)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "model_calibration.png", bbox_inches="tight")
plt.close(fig)
print("✓ model_calibration.png")


# ─── 3. Spending Distribution by Cook Rating ────────────────────────────────

order = [c for c in COOK_ORDER if c in df["cook_rating"].values]
colors = [COOK_COLOR[c] for c in order]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel A: median D and R spend by category
d_med = [df[df["cook_rating"] == c]["d_total_m"].median() for c in order]
r_med = [df[df["cook_rating"] == c]["r_total_m"].median() for c in order]
y = np.arange(len(order))
bw = 0.35
axes[0].barh(y + bw/2, d_med, bw, color=colors, alpha=0.9, label="Democrat")
axes[0].barh(y - bw/2, r_med, bw, color="#c0392b", alpha=0.45, label="Republican")
axes[0].set_yticks(y)
axes[0].set_yticklabels(order)
axes[0].set_xlabel("Median Total Spending ($M)", fontsize=11)
axes[0].set_title("Median Party vs R Spending\nby Cook Rating", fontsize=12, fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
axes[0].legend(frameon=False, fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Panel B: box plots for competitive races
comp_cats = ["Lean D", "Toss-Up", "Lean R"]
comp_d = [df[df["cook_rating"] == c]["d_total_m"].values for c in comp_cats]
comp_r = [df[df["cook_rating"] == c]["r_total_m"].values for c in comp_cats]

pos_d = [0.0, 1.1, 2.2]
pos_r = [0.45, 1.55, 2.65]
bp_d = axes[1].boxplot(comp_d, vert=True, patch_artist=True,
                       positions=pos_d, widths=0.35,
                       medianprops=dict(color="white", lw=2),
                       flierprops=dict(marker="o", ms=3, alpha=0.35))
bp_r = axes[1].boxplot(comp_r, vert=True, patch_artist=True,
                       positions=pos_r, widths=0.35,
                       medianprops=dict(color="white", lw=2),
                       flierprops=dict(marker="o", ms=3, alpha=0.35))
for patch, cat in zip(bp_d["boxes"], comp_cats):
    patch.set_facecolor(COOK_COLOR[cat])
    patch.set_alpha(0.85)
for patch in bp_r["boxes"]:
    patch.set_facecolor("#c0392b")
    patch.set_alpha(0.45)
for bp in [bp_d, bp_r]:
    for elem in ["whiskers", "caps"]:
        for line in bp[elem]:
            line.set_color("#555555")

mid_ticks = [(pos_d[i] + pos_r[i]) / 2 for i in range(3)]
axes[1].set_xticks(mid_ticks)
axes[1].set_xticklabels(comp_cats)
axes[1].set_xlabel("Cook Rating (Competitive)", fontsize=11)
axes[1].set_ylabel("Total Spending ($M)", fontsize=11)
axes[1].set_title("D vs R Spending Distribution\nCompetitive Races", fontsize=12, fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
legend_els = [Patch(fc="#2e6da4", label="Democrat"), Patch(fc="#c0392b", alpha=0.6, label="Republican")]
axes[1].legend(handles=legend_els, frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "spending_by_cook.png", bbox_inches="tight")
plt.close(fig)
print("✓ spending_by_cook.png")


# ─── 4. Allocator Comparison Bar Chart ──────────────────────────────────────

if _alloc_df is not None:
    # Read authoritative values from run_backtest.py output
    _label_map = {
        "DCCC observed":     "DCCC\nObserved",
        "Null (equal-weight)": "Null\n(Equal)",
        "Cook-implied":      "Cook\n-Implied",
        "Model optimizer":   "Model\nOptimizer",
    }
    _order = ["DCCC observed", "Cook-implied", "Null (equal-weight)", "Model optimizer"]
    allocators = [_label_map[k] for k in _order]
    seats      = [float(_alloc_df.loc[_alloc_df["allocator"] == k, "expected_seats"].iloc[0])
                  for k in _order]
else:
    allocators = ["DCCC\nObserved", "Cook\n-Implied", "Null\n(Equal)", "Model\nOptimizer"]
    seats = [df["p_win"].sum()] * 4  # fallback: all equal, signals missing data

bar_colors = ["#2e6da4", "#aaaaaa", "#888888", "#1a5c2b"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(allocators, seats, color=bar_colors, edgecolor="white",
              linewidth=0.5, width=0.55, zorder=3)

baseline = seats[0]
for bar, s in zip(bars, seats):
    delta = s - baseline
    delta_str = f"\n({'+' if delta >= 0 else ''}{delta:.2f})" if delta != 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, s + 0.04,
            f"{s:.2f}{delta_str}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold")

ax.axhline(baseline, color="#2e6da4", lw=1.2, ls="--", alpha=0.5, zorder=2)
ax.set_ylabel("Expected Democratic Seats", fontsize=12)
ax.set_title("Expected Seats by Allocation Strategy\n(2024 House — model win probabilities)",
             fontsize=13, fontweight="bold")

# Dynamic y-axis: full range from below floor to above model bar
_y_lo = min(seats) - 0.5
_y_hi = max(seats) + 0.8
ax.set_ylim(_y_lo, _y_hi)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.grid(axis="y", lw=0.5, alpha=0.35, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "allocator_comparison.png", bbox_inches="tight")
plt.close(fig)
print("✓ allocator_comparison.png")


# ─── 5. Allocation Shift ────────────────────────────────────────────────────

df["diff_pct"] = df["difference"] * 100
sig = df[df["diff_pct"].abs() > 0.05].copy()
sig = sig.sort_values("diff_pct")

# Limit display: top 20 positive, top 20 negative
n_each = 20
display = pd.concat([sig.head(n_each), sig.tail(n_each)]).drop_duplicates()
display = display.sort_values("diff_pct")

fig, ax = plt.subplots(figsize=(9, max(7, len(display) * 0.24)))
bar_c = [COOK_COLOR.get(r, "#888888") for r in display["cook_rating"]]
ypos = np.arange(len(display))
ax.barh(ypos, display["diff_pct"], color=bar_c, alpha=0.85,
        edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", lw=0.8)

labels = [f"{row['district_id']}  {row['cook_rating'][:7]}"
          for _, row in display.iterrows()]
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Recommended − Observed Share (% of total budget)", fontsize=11)
ax.set_title("Model Optimal vs DCCC Allocation\n(Top/bottom 20 races by shift magnitude)",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.2f}%"))

legend_els = [Patch(facecolor=COOK_COLOR[c], label=c)
              for c in COOK_ORDER if c in display["cook_rating"].values]
ax.legend(handles=legend_els, frameon=False, fontsize=8.5, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "allocation_shift.png", bbox_inches="tight")
plt.close(fig)
print("✓ allocation_shift.png")


# ─── 6. Spending Ratio vs PVI ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for rating in COOK_ORDER:
    sub = df[df["cook_rating"] == rating]
    if len(sub) < 1:
        continue
    ax.scatter(sub["pvi"], sub["ratio"], c=COOK_COLOR[rating],
               s=28, alpha=0.65, edgecolors="none", zorder=3)

ax.axhline(0.5, color="#555555", lw=1.0, ls="--", alpha=0.6)
ax.axvline(0, color="#aaaaaa", lw=0.8, ls=":", alpha=0.5)
ax.text(0.5, 0.52, "Equal spend", transform=ax.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=9, color="#555555")
ax.set_xlabel("Cook PVI  (D+ →)", fontsize=12)
ax.set_ylabel("Democratic Share of Total Spending", fontsize=12)
ax.set_title("Spending Parity vs District Lean\n(All 433 races, 2024 House)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_xlim(-36, 44)
ax.set_ylim(-0.02, 1.02)

legend_els = [Patch(facecolor=COOK_COLOR[c], label=c)
              for c in COOK_ORDER if c in df["cook_rating"].values]
ax.legend(handles=legend_els, frameon=False, fontsize=9, ncol=2,
          title="Cook Rating", title_fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "spending_ratio_vs_pvi.png", bbox_inches="tight")
plt.close(fig)
print("✓ spending_ratio_vs_pvi.png")

print(f"\nAll 6 charts written to {OUT}/")
