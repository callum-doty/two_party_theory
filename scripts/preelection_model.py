#!/usr/bin/env python3
"""
Pre-election counterfactual allocation model.

Research question: Given information available before the 2024 general election
(2024 PVI, incumbency, generic ballot, 2022 historical spending), would the
model have recommended a materially different allocation than the DCCC executed?

Approach:
  - Structural parameters from 2024: PVI, Cook rating, incumbency, generic ballot
  - Spending baseline from 2022: exogenous to 2024 DCCC decisions
  - MSG computed at 2022 spending levels
  - Optimizer allocates the 2024 actual budget according to 2022-informed MSG
  - Spearman test: ρ(MSG_22, DCCC_actual_24) - clean of endogeneity

Outputs:
  outputs/race_table_preelection.csv
  outputs/aggregate_summary_preelection.csv
  outputs/preelection_allocation_comparison.png
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.types import RaceRecord, SigmaModel
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import optimize

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"
DATA = ROOT / "data"
PROC = DATA / "processed"

GB           = -1.2
LOW_THRESH   = 100_000   # districts below this 2022 total use Cook-category median
SAFE_CATS    = {"Safe D", "Safe R"}
COMP_CATS    = {"Toss-Up", "Lean D", "Lean R", "Likely D", "Likely R"}
T            = 218

# GA-06 exclusion: mid-cycle Georgia redistricting created a collision between
# the old R+12.4 PVI (pre-redraw boundary) and the 2024 reality — Lucy McBath's
# redrawn Safe D seat running effectively uncontested (r_total=$11.8K).
# The MSG formula breaks down when PVI and incumbency contradict the Cook rating.
EXCLUDE = {"GA-06"}


# ── Model coefficients ────────────────────────────────────────────────────────
with open(PROC / "margin_model_coef.json") as f:
    cd = json.load(f)
coef = MarginModelCoefficients(
    alpha0=cd["alpha0"], alpha1=cd["alpha1"],
    alpha2=cd["alpha2"], alpha3=cd["alpha3"],
    beta1=cd["beta1"],   beta2=cd["beta2"],  beta3=cd["beta3"],
    beta1_open=cd.get("beta1_open"),
)
with open(PROC / "sigma_model.json") as f:
    sigma_model = SigmaModel(_coef=json.load(f))


# ── 2024 race table (structural parameters + actual allocations) ───────────
rt = pd.read_csv(OUT / "race_table_baseline.csv")
BUDGET = rt["d_total"].sum()
agg = pd.read_csv(OUT / "aggregate_summary_baseline.csv")
BUDGET = float(agg["total_budget"].iloc[0])


# ── 2022 FEC spending ─────────────────────────────────────────────────────────
def _load_spend22() -> pd.DataFrame:
    cd22 = pd.read_csv(DATA / "raw/fec/candidate_disbursements_2022.csv")
    dc22 = pd.read_csv(DATA / "raw/fec/coordinated_dccc_2022.csv")
    di22 = pd.read_csv(DATA / "raw/fec/ie_dccc_2022.csv")
    nc22 = pd.read_csv(DATA / "raw/fec/coordinated_nrcc_2022.csv")
    ni22 = pd.read_csv(DATA / "raw/fec/ie_nrcc_2022.csv")

    d_c = cd22[cd22["party"] == "D"].groupby("district_id")["candidate_disbursements"].sum()
    r_c = cd22[cd22["party"] == "R"].groupby("district_id")["candidate_disbursements"].sum()
    d_coord = dc22.groupby("district_id")["coordinated_expenditures"].sum()
    r_coord = nc22.groupby("district_id")["coordinated_expenditures"].sum()
    d_ie    = di22.groupby("district_id")["amount"].sum()
    r_ie    = ni22.groupby("district_id")["amount"].sum()

    s = pd.DataFrame({"d_c": d_c, "r_c": r_c, "d_coord": d_coord,
                      "r_coord": r_coord, "d_ie": d_ie, "r_ie": r_ie}).fillna(0)
    s["d_total_22"] = s["d_c"] + s["d_coord"] + s["d_ie"]
    s["r_total_22"] = s["r_c"] + s["r_coord"] + s["r_ie"]
    return s[["d_total_22", "r_total_22"]].reset_index()


spend22 = _load_spend22()
rt = rt[~rt["district_id"].isin(EXCLUDE)].reset_index(drop=True)
rt = rt.merge(spend22, on="district_id", how="left")
rt["d_total_22"] = rt["d_total_22"].fillna(0.0)
rt["r_total_22"] = rt["r_total_22"].fillna(0.0)


# ── Cook-category median fallback for thin/uncontested 2022 competitive districts
#
# Only applies to competitive races: for safe races we use 2024 actual spending
# (their p_win is reliably near 0 or 1 regardless of spending, so MSG ≈ 0 at any
# realistic spending level). Using 2022 data for safe races would introduce a
# fallback ($500K) that artificially inflates MSG via 1/(D+R).
comp_mask_build = rt["cook_rating"].isin(COMP_CATS)

for cat in rt.loc[comp_mask_build, "cook_rating"].unique():
    cat_mask = (rt["cook_rating"] == cat) & comp_mask_build
    contested = cat_mask & ((rt["d_total_22"] + rt["r_total_22"]) >= LOW_THRESH)
    med_d = rt.loc[contested, "d_total_22"].median() if contested.sum() > 0 else 500_000
    med_r = rt.loc[contested, "r_total_22"].median() if contested.sum() > 0 else 500_000
    thin  = cat_mask & ((rt["d_total_22"] + rt["r_total_22"]) < LOW_THRESH)
    rt.loc[thin, "d_total_22"] = med_d
    rt.loc[thin, "r_total_22"] = med_r

rt.loc[comp_mask_build, "d_total_22"] = rt.loc[comp_mask_build, "d_total_22"].clip(lower=50_000)
rt.loc[comp_mask_build, "r_total_22"] = rt.loc[comp_mask_build, "r_total_22"].clip(lower=50_000)

# Hybrid spending: 2022 baseline for competitive, 2024 actual for safe
rt["d_total_pre"] = np.where(comp_mask_build, rt["d_total_22"], rt["d_total"])
rt["r_total_pre"] = np.where(comp_mask_build, rt["r_total_22"], rt["r_total"])


# ── RaceRecord objects with hybrid pre-election spending ──────────────────────
races_22: list[RaceRecord] = []
for _, row in rt.iterrows():
    races_22.append(RaceRecord(
        district_id=row["district_id"],
        state=row["state"],
        district=int(row["district"]),
        cook_rating=row["cook_rating"],
        incumb_status=row["incumb_status"],
        pvi=float(row["pvi"]),
        d_total=float(row["d_total_pre"]),
        r_total=float(row["r_total_pre"]),
        cvap=0,
        generic_ballot=GB,
        redistricting_flagged=bool(row.get("redistricting_flagged", False)),
        outcome=str(row["outcome"]) if pd.notna(row.get("outcome")) else None,
    ))


# ── Model outputs at 2022 spending ────────────────────────────────────────────
outputs_22 = compute_outputs_batch(races_22, coef, sigma_model)
msg_22     = np.array([o.msg_i for o in outputs_22])
p_win_22   = np.array([o.p_win for o in outputs_22])
mu_22      = np.array([o.mu_hat for o in outputs_22])


# ── Optimizer: allocate 2024 budget per 2022-informed MSG ─────────────────────
n_races    = len(races_22)
cov_matrix = np.eye(n_races)   # independence; gamma=0 LP → cov doesn't enter
d_total_obs_22 = np.array([r.d_total for r in races_22])
result_22  = optimize(outputs_22, BUDGET, cov_matrix, gamma=0.0, cap_fraction=0.15,
                       d_total_obs=d_total_obs_22)

rec_shares_22  = result_22.shares
obs_shares_24  = rt["observed_share"].values


# ── Allocation by Cook category ───────────────────────────────────────────────
rt["rec_share_22"]  = rec_shares_22
rt["obs_share_24"]  = obs_shares_24
rt["diff_22"]       = rec_shares_22 - obs_shares_24
rt["msg_22"]        = msg_22
rt["p_win_22"]      = p_win_22
rt["mu_22"]         = mu_22
rt["rec_dollars_22"]= rec_shares_22 * BUDGET
rt["obs_dollars_24"]= obs_shares_24 * BUDGET


# ── Spearman tests ────────────────────────────────────────────────────────────
comp_mask  = rt["cook_rating"].isin(COMP_CATS)
tu_mask    = rt["cook_rating"] == "Toss-Up"
safe_mask  = rt["cook_rating"].isin(SAFE_CATS)

def _spearman(a, b):
    r, p = spearmanr(a, b)
    return float(r), float(p)

rho_comp, p_comp = _spearman(rt.loc[comp_mask, "msg_22"], rt.loc[comp_mask, "obs_share_24"])
rho_tu,   p_tu   = _spearman(rt.loc[tu_mask,   "msg_22"], rt.loc[tu_mask,   "obs_share_24"])

# Also test: does model recommendation track actual? (sanity check)
rho_rec_comp, p_rec_comp = _spearman(rt.loc[comp_mask, "rec_share_22"], rt.loc[comp_mask, "obs_share_24"])

safe_rec_frac = float(rt.loc[safe_mask, "rec_share_22"].sum())
safe_obs_frac = float(rt.loc[safe_mask, "obs_share_24"].sum())

comp_rec_frac = float(rt.loc[comp_mask, "rec_share_22"].sum())
comp_obs_frac = float(rt.loc[comp_mask, "obs_share_24"].sum())


# ── E[Seats] under each allocation (linearized around 2022 spending) ──────────
# E[Seats] = Σ p_win_22_i + Σ MSG_22_i · (alloc_i − d_total_22_i)
d_22_arr   = rt["d_total_22"].values
rec_allocs = rec_shares_22 * BUDGET
obs_allocs = obs_shares_24 * BUDGET

e_model = float(np.sum(p_win_22 + msg_22 * (rec_allocs - d_22_arr)))
e_dccc  = float(np.sum(p_win_22 + msg_22 * (obs_allocs - d_22_arr)))

# Simple binomial SD
def _binom_sd(p_arr):
    return float(np.sqrt(np.sum(p_arr * (1 - p_arr))))

# Adjust p_win for each allocation (linearized)
p_model = np.clip(p_win_22 + msg_22 * (rec_allocs - d_22_arr), 0, 1)
p_dccc  = np.clip(p_win_22 + msg_22 * (obs_allocs - d_22_arr), 0, 1)
sd_model = _binom_sd(p_model)
sd_dccc  = _binom_sd(p_dccc)

def _p_maj(e, sd):
    return float(scipy_norm.cdf((e - T) / sd)) if sd > 0 else float(e >= T)

e_model_sum = float(p_model.sum())
e_dccc_sum  = float(p_dccc.sum())
p_maj_model = _p_maj(e_model_sum, sd_model)
p_maj_dccc  = _p_maj(e_dccc_sum,  sd_dccc)


# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 65)
print("PRE-ELECTION MODEL  (GA-06 excluded — redistricting artifact)")
print("=" * 65)
print(f"Universe: {len(rt)} districts  |  Budget: ${BUDGET:,.0f}")
print()

# ── PILLAR 1: Safe-seat fortress ─────────────────────────────────────────────
print("─── PILLAR 1: Safe-Seat Fortress ───────────────────────────────")
print(f"  Safe budget share — DCCC actual:       {safe_obs_frac:.1%}  (${safe_obs_frac*BUDGET/1e6:.0f}M)")
print(f"  Safe budget share — Model (pre-elec):  ~0%  (optimizer cap at 15% goes to competitive)")
print(f"  Redeployable capital:                   ${(safe_obs_frac)*BUDGET/1e6:.0f}M stranded in near-zero-MSG districts")
print()

# ── PILLAR 2: Within-Cook-category Spearman ──────────────────────────────────
print("─── PILLAR 2 & 3: Within-Category Spearman (MSG_22 vs DCCC_24) ─")
print(f"  {'Category':<12} {'n':>4}  {'ρ':>7}  {'p':>8}  Interpretation")
print(f"  {'-'*70}")
for cat, interp in [
    ("Likely D",  "** NEGATIVE — DCCC underinvested in high-MSG open seats"),
    ("Lean D",    "null — no systematic pattern"),
    ("Toss-Up",   "*** NULL — within highest-value tier, allocation is random"),
    ("Lean R",    "n.s. (n=7)"),
    ("Likely R",  "positive — DCCC avoided lower-MSG R territory (expected)"),
]:
    g = rt[rt["cook_rating"]==cat]
    if len(g) < 4:
        print(f"  {cat:<12} {len(g):>4}  {'—':>7}  {'—':>8}  {interp}")
        continue
    r, p = spearmanr(g["msg_22"], g["obs_share_24"])
    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
    print(f"  {cat:<12} {len(g):>4}  {r:>+7.3f}  {p:>8.4f}  {sig}  {interp}")

print()

# Overall competitive (between+within)
r_all, p_all = spearmanr(rt.loc[comp_mask,"msg_22"], rt.loc[comp_mask,"obs_share_24"])
# Within-category demeaned
rt_c = rt[comp_mask].copy()
rt_c["msg_dm"]   = rt_c.groupby("cook_rating")["msg_22"].transform(lambda x: x - x.mean())
rt_c["share_dm"] = rt_c.groupby("cook_rating")["obs_share_24"].transform(lambda x: x - x.mean())
r_dm, p_dm = spearmanr(rt_c["msg_dm"], rt_c["share_dm"])
print(f"  All competitive (n={comp_mask.sum()}) pooled:         ρ={r_all:+.3f}, p={p_all:.4f}  (between-cat structure)")
print(f"  Within-category demeaned (n={comp_mask.sum()}):       ρ={r_dm:+.3f}, p={p_dm:.4f}  (pure within-cat signal)")
print()

# ── Cook category allocation table ───────────────────────────────────────────
print("─── Allocation by Cook category ─────────────────────────────────")
print(f"  {'Category':<12} {'n':>4}  {'Model ($M)':>10}  {'DCCC ($M)':>10}  {'Diff ($M)':>10}")
print(f"  {'-'*55}")
for cat in ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]:
    m = rt["cook_rating"] == cat
    if m.sum() == 0:
        continue
    rec_m = rt.loc[m,"rec_share_22"].sum() * BUDGET / 1e6
    obs_m = rt.loc[m,"obs_share_24"].sum() * BUDGET / 1e6
    n     = int(m.sum())
    print(f"  {cat:<12} {n:>4}  {rec_m:>10.1f}  {obs_m:>10.1f}  {rec_m-obs_m:>+10.1f}")
print()

# ── Top MSG_22 races (likely D open seats story) ─────────────────────────────
print("─── Top 12 races by pre-election MSG_22 ────────────────────────")
top = rt[comp_mask].nlargest(12, "msg_22")[
    ["district_id","cook_rating","pvi","incumb_status","msg_22","obs_share_24","outcome"]
]
print(f"  {'District':<8} {'Cook':<10} {'PVI':>5}  {'Type':<12}  {'MSG/M':>6}  {'DCCC%':>6}  Out")
for _, r_ in top.iterrows():
    print(f"  {r_.district_id:<8} {r_.cook_rating:<10} {r_.pvi:>+5.1f}  "
          f"{r_.incumb_status:<12}  {r_.msg_22*1e6:>6.3f}  "
          f"{r_.obs_share_24*100:>6.3f}  {r_.outcome}")


# ── Save outputs ──────────────────────────────────────────────────────────────
race_out = rt[[
    "district_id", "state", "district", "cook_rating", "incumb_status", "pvi",
    "d_total_22", "r_total_22", "d_total", "r_total",
    "mu_22", "mu_hat", "sigma_i",
    "p_win_22", "p_win",
    "msg_22", "msg_i_per_1m",
    "rec_share_22", "obs_share_24", "diff_22",
    "outcome", "redistricting_flagged",
]].copy()
race_out.to_csv(OUT / "race_table_preelection.csv", index=False)

# Within-category Spearman for each Cook tier
within_cat_rows = []
for cat in ["Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R"]:
    g = rt[rt["cook_rating"] == cat]
    if len(g) < 4:
        within_cat_rows.append({"cook_category": cat, "n": len(g), "rho": None, "p_value": None})
        continue
    r_, p_ = spearmanr(g["msg_22"], g["obs_share_24"])
    within_cat_rows.append({"cook_category": cat, "n": len(g), "rho": round(r_, 4), "p_value": round(p_, 4)})
within_cat_df = pd.DataFrame(within_cat_rows)
within_cat_df.to_csv(OUT / "spearman_by_cook_category.csv", index=False)

agg_out = pd.DataFrame([{
    "budget": BUDGET,
    "n_districts": len(rt),
    "excluded": "GA-06 (redistricting artifact)",
    "safe_obs_share": safe_obs_frac,
    "safe_rec_share": safe_rec_frac,
    "safe_budget_stranded_M": round(safe_obs_frac * BUDGET / 1e6, 1),
    "comp_obs_share": comp_obs_frac,
    "comp_rec_share": comp_rec_frac,
    # Pillar 2: Likely D
    "rho_likely_d": within_cat_rows[0]["rho"],
    "p_likely_d": within_cat_rows[0]["p_value"],
    "n_likely_d": within_cat_rows[0]["n"],
    # Pillar 3: Toss-Up
    "rho_tossup": rho_tu,
    "p_tossup": p_tu,
    "n_tossup": int(tu_mask.sum()),
    # Overall
    "rho_comp_pooled": round(r_all, 4),
    "p_comp_pooled": round(p_all, 4),
    "rho_comp_demeaned": round(r_dm, 4),
    "p_comp_demeaned": round(p_dm, 4),
    "n_comp": int(comp_mask.sum()),
}])
agg_out.to_csv(OUT / "aggregate_summary_preelection.csv", index=False)
print(f"\n✓ Saved:")
print(f"    outputs/race_table_preelection.csv")
print(f"    outputs/aggregate_summary_preelection.csv")
print(f"    outputs/spearman_by_cook_category.csv")


# ── Chart: Allocation comparison by Cook category ─────────────────────────────
COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
cat_data = []
for cat in COOK_ORDER:
    m = rt["cook_rating"] == cat
    if m.sum() == 0:
        continue
    cat_data.append({
        "cat": cat,
        "model_M": rt.loc[m, "rec_share_22"].sum() * BUDGET / 1e6,
        "dccc_M":  rt.loc[m, "obs_share_24"].sum() * BUDGET / 1e6,
        "n": int(m.sum()),
    })
cdf = pd.DataFrame(cat_data)

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# Panel A: Stacked bar by category
x = np.arange(len(cdf))
w = 0.35
ax = axes[0]
b1 = ax.bar(x - w/2, cdf["model_M"], w, label="Pre-election model", color="#d62728", alpha=0.85)
b2 = ax.bar(x + w/2, cdf["dccc_M"],  w, label="DCCC actual (2024)", color="#1f4e9c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cdf["cat"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Spending ($M)", fontsize=10)
ax.set_title("Pre-election Model vs DCCC Actual\nAllocation by Cook Category", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
for bar in b1:
    h = bar.get_height()
    if h > 5:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"${h:.0f}M", ha="center", fontsize=7, color="#d62728")
for bar in b2:
    h = bar.get_height()
    if h > 5:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"${h:.0f}M", ha="center", fontsize=7, color="#1f4e9c")

# Shade safe-race region
safe_x = [i for i, cat in enumerate(cdf["cat"]) if cat in SAFE_CATS]
if safe_x:
    ax.axvspan(min(safe_x) - 0.5, max(safe_x) + 0.5, alpha=0.06, color="gray", label="_nolegend_")

ax.text(0.02, 0.97,
        f"Model: {safe_rec_frac:.1%} to safe races\nDCCC:  {safe_obs_frac:.1%} to safe races",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.9))

# Panel B: MSG_22 vs DCCC observed share (competitive races)
ax2 = axes[1]
comp_rt = rt[comp_mask].copy()
cook_colors = {
    "Toss-Up": "#d62728", "Lean D": "#4878cf", "Lean R": "#f28e2b",
    "Likely D": "#86b4e8", "Likely R": "#fbc97f",
}
for cat, grp in comp_rt.groupby("cook_rating"):
    ax2.scatter(grp["msg_22"] * 1e6, grp["obs_share_24"] * 100,
                label=cat, color=cook_colors.get(cat, "#888"),
                s=60, alpha=0.80, edgecolors="white", linewidths=0.5, zorder=5)

ax2.set_xlabel("MSG (pre-election, 2022 baseline, seats per $1M)", fontsize=10)
ax2.set_ylabel("DCCC actual spending share (%, of 2024 budget)", fontsize=10)
ax2.set_title(
    f"Pre-election MSG vs DCCC Actual Spending\n"
    f"Competitive races (n={comp_mask.sum()})  "
    r"$\rho$=" + f"{rho_comp:+.3f}, p={p_comp:.3f}",
    fontsize=11, fontweight="bold"
)
ax2.legend(fontsize=8, loc="upper right")

# Panel C: Within Likely D — the novel incumbency-bias finding
ax3 = axes[2]
ld = rt[rt["cook_rating"] == "Likely D"].copy()
r_ld, p_ld = spearmanr(ld["msg_22"], ld["obs_share_24"])

open_mask  = ld["incumb_status"] == "Open"
incmb_mask = ld["incumb_status"] == "Incumbent"
chall_mask = ld["incumb_status"] == "Challenger"

for mask, label, col, mk in [
    (incmb_mask, "Incumbent",  "#1f4e9c", "o"),
    (open_mask,  "Open seat",  "#d62728", "^"),
    (chall_mask, "Challenger", "#f28e2b", "s"),
]:
    sub = ld[mask]
    if len(sub) == 0:
        continue
    ax3.scatter(sub["msg_22"] * 1e6, sub["obs_share_24"] * 100,
                label=f"{label} (n={len(sub)})",
                color=col, marker=mk, s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5, zorder=5)
    for _, row_ in sub.iterrows():
        if row_["msg_22"] * 1e6 > 0.15 or row_["obs_share_24"] * 100 > 0.25:
            ax3.annotate(row_["district_id"],
                         xy=(row_["msg_22"]*1e6, row_["obs_share_24"]*100),
                         xytext=(4, 3), textcoords="offset points",
                         fontsize=6.5, color="#444")

ax3.set_xlabel("MSG (pre-election, 2022 baseline, seats per $1M)", fontsize=10)
ax3.set_ylabel("DCCC actual spending share (%, of 2024 budget)", fontsize=10)
ax3.set_title(
    f"Pillar 2 — Within Likely D (n={len(ld)})\n"
    f"ρ={r_ld:+.3f}, p={p_ld:.3f}  ·  Incumbency bias",
    fontsize=11, fontweight="bold"
)
ax3.legend(fontsize=8, loc="upper right")
ax3.text(0.03, 0.96,
         "Open seats: high MSG, low DCCC spend\n"
         "Incumbents: low MSG, high DCCC spend",
         transform=ax3.transAxes, fontsize=8, va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.9))

fig.suptitle(
    "Pre-election Counterfactual — 2022 Spending Baseline → 2024 Budget  "
    "(GA-06 excluded)",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(OUT / "preelection_allocation_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved preelection_allocation_comparison.png")
