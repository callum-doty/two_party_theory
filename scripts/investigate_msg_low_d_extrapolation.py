#!/usr/bin/env python3
"""
Investigation (not a calibration change) into whether the margin model's beta1
log-ratio elasticity -- which drives MSG's spike at near-zero D in the live 2026
universe, outputs/msg_low_leverage_check_fig.png -- is well-supported by data,
in two parts:

Part 1 (D-VALUE range check): does today's live 2026 candidate-only floor
(cand_d_total) fall inside or outside the D_total range the 2012-2022
estimation panel actually observed, by tier? Reconstructs estimate_from_panel()'s
exact row set. RESULT: not an extrapolation issue -- the historical panel goes
even lower than today's live floors in every tier that matters (Safe R min
observed D_total was $40; Likely R was $74).

Part 2 (beta_RC IDENTIFICATION composition -- the real finding): beta1 IS
beta_RC (src/backtest/model/margin.py's docstring), estimated from exactly 118
repeat-challenger pairs (src/backtest/estimation/beta_rc.py), each necessarily
a race where D is CHALLENGING an R incumbent (by identify_repeat_pairs()'s
design -- see FINDINGS.md Section 4.4). Bucketing those 118 pairs by the
PVI-derived tier of the district they occurred in shows the sample is
overwhelmingly Safe R (85/118, 72%), while the competitive tiers the model's
headline recommendation is actually ABOUT (Toss-Up/Lean D/Lean R combined)
contribute only 14 pairs. A formal interaction test (does beta1 differ in
Safe R vs. elsewhere?) is NOT statistically significant (p=0.61) -- but with
n=118 total and n=14 in the competitive tiers, this test is underpowered to
detect anything but a large difference; "not significant" here means "cannot
confirm a difference exists," not "confirmed there isn't one."

Output: outputs/msg_low_d_extrapolation_check.csv (Part 1),
        outputs/beta_rc_tier_composition.csv (Part 2), printed summary.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import config
from backtest.data import elections, fec, incumbency
from backtest.data.pvi import load_pvi, derive_rating
from backtest.data.universe import build_universe
from backtest.estimation import beta_rc as beta_rc_module

COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
COMPETITIVE = ["Toss-Up", "Lean D", "Lean R"]


def build_estimation_panel() -> pd.DataFrame:
    """Reproduce estimate_from_panel()'s row set exactly (same filters), plus a
    tier column, so the comparison is apples-to-apples with what beta1 actually saw."""
    cycles = config.panel_cycles()

    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)
    panel_spend = pd.concat([fec.build_total_spend(c) for c in cycles], ignore_index=True)
    panel_incumb = pd.concat([incumbency.load_incumbency(c) for c in cycles], ignore_index=True)
    panel_pvi = pd.concat([load_pvi(c).assign(cycle=c) for c in cycles], ignore_index=True)

    df = (
        panel_results
        .merge(panel_spend, on=["district_id", "cycle"])
        .merge(panel_incumb, on=["district_id", "cycle"])
        .merge(panel_pvi, on=["district_id", "cycle"])
    )
    # Same filters as estimate_from_panel(): drop zero-total-spend and zero-ratio rows.
    df = df[(df["d_total"] + df["r_total"]) > 0]
    df["ratio"] = df["d_total"] / (df["d_total"] + df["r_total"])
    df = df[df["ratio"] > 0]

    df["tier"] = [derive_rating(pvi, status) for pvi, status in
                  zip(df["pvi"], df["incumb_status"])]
    return df


def live_2026_floors() -> pd.DataFrame:
    """The live 2026 universe's candidate-only floor (cand_d_total) by tier --
    the D value MSG is evaluated at in outputs/msg_low_leverage_check_fig.png,
    before any party dollar is deployed."""
    races = build_universe(cycle=2026)
    return pd.DataFrame([{"district_id": r.district_id, "tier": r.cook_rating,
                           "cand_d_total": r.cand_d_total} for r in races])


def beta_rc_pairs_with_tier() -> pd.DataFrame:
    """Reconstruct identify_repeat_pairs()'s exact 118-pair sample and attach
    each pair's district tier (from the district's PVI at cycle_tm1, with the
    fixed Challenger adjustment -- these pairs are by construction always a
    D-challenger-vs-R-incumbent race)."""
    cycles = config.panel_cycles()
    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)
    panel_spend = pd.concat([fec.build_total_spend(c) for c in cycles], ignore_index=True)
    panel_incumb = pd.concat([incumbency.load_incumbency(c) for c in cycles], ignore_index=True)
    panel_pvi = pd.concat([load_pvi(c).assign(cycle=c) for c in cycles], ignore_index=True)

    pairs = beta_rc_module.identify_repeat_pairs(panel_results, panel_spend, panel_incumb)
    pvi_lookup = panel_pvi.set_index(["district_id", "cycle"])["pvi"].to_dict()
    pairs["pvi_prev"] = [pvi_lookup.get((d, c)) for d, c in
                          zip(pairs["district_id"], pairs["cycle_tm1"])]
    pairs["tier"] = [derive_rating(p, "Challenger") if p is not None else None
                      for p in pairs["pvi_prev"]]
    return pairs


def fit_beta_rc(pairs: pd.DataFrame) -> tuple[float, float, int]:
    X = sm.add_constant(pairs["delta_log_ratio"])
    y = pairs["delta_margin"]
    f = sm.OLS(y, X).fit(cov_type="HC3")
    return float(f.params["delta_log_ratio"]), float(f.bse["delta_log_ratio"]), len(pairs)


def beta_rc_composition_check(pairs: pd.DataFrame) -> dict:
    """Part 2: tier composition of beta_RC's identifying sample, plus a
    formal interaction test for whether beta1 differs in Safe R vs. elsewhere."""
    counts = pairs["tier"].value_counts().reindex(COOK_ORDER).fillna(0).astype(int)

    b_all, se_all, n_all = fit_beta_rc(pairs)
    b_safer, se_safer, n_safer = fit_beta_rc(pairs[pairs["tier"] == "Safe R"])
    b_comp, se_comp, n_comp = fit_beta_rc(pairs[pairs["tier"].isin(COMPETITIVE)])

    pairs = pairs.copy()
    pairs["is_safe_r"] = (pairs["tier"] == "Safe R").astype(float)
    pairs["dlr_x_safer"] = pairs["delta_log_ratio"] * pairs["is_safe_r"]
    X = sm.add_constant(pairs[["delta_log_ratio", "is_safe_r", "dlr_x_safer"]])
    y = pairs["delta_margin"]
    f_int = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "tier_counts": counts.to_dict(),
        "pct_safe_r": float(counts["Safe R"] / len(pairs)),
        "pct_competitive": float(counts[COMPETITIVE].sum() / len(pairs)),
        "beta_rc_all": b_all, "se_all": se_all, "n_all": n_all,
        "beta_rc_safe_r_only": b_safer, "se_safe_r_only": se_safer, "n_safe_r_only": n_safer,
        "beta_rc_competitive_only": b_comp, "se_competitive_only": se_comp, "n_competitive_only": n_comp,
        "interaction_coef": float(f_int.params["dlr_x_safer"]),
        "interaction_p_value": float(f_int.pvalues["dlr_x_safer"]),
    }


def main():
    panel = build_estimation_panel()
    live = live_2026_floors()

    print("=== Part 1: does today's live D-value fall outside the historical panel's range? ===")
    print(f"Estimation panel: {len(panel)} race-cycle observations, "
          f"{config.panel_cycles()} (post-filter)\n")

    rows = []
    for tier in COOK_ORDER:
        hist = panel[panel["tier"] == tier]["d_total"]
        live_floor = live[live["tier"] == tier]["cand_d_total"]
        if len(hist) == 0:
            continue
        rows.append({
            "tier": tier,
            "n_historical_obs": len(hist),
            "hist_d_total_min": hist.min(),
            "hist_d_total_p5": hist.quantile(0.05),
            "hist_d_total_p25": hist.quantile(0.25),
            "hist_d_total_median": hist.median(),
            "n_live_2026_races": len(live_floor),
            "live_cand_floor_min": live_floor.min() if len(live_floor) else np.nan,
            "live_cand_floor_median": live_floor.median() if len(live_floor) else np.nan,
            "pct_live_below_hist_p5": float((live_floor < hist.quantile(0.05)).mean()) if len(live_floor) else np.nan,
            "pct_live_below_hist_min": float((live_floor < hist.min()).mean()) if len(live_floor) else np.nan,
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if abs(x) > 1 else f"{x:.3f}")
    print(out.to_string(index=False))

    out.to_csv(ROOT / "outputs/msg_low_d_extrapolation_check.csv", index=False)
    print(f"\nSaved -> outputs/msg_low_d_extrapolation_check.csv")

    print("\n=== Headline check ===")
    for _, r in out.iterrows():
        if r["tier"] in ("Safe R", "Likely R", "Safe D", "Likely D"):
            print(f"  {r['tier']}: {r['pct_live_below_hist_min']:.0%} of live 2026 races have a candidate "
                  f"floor BELOW every historical observation in this tier's estimation data "
                  f"(min historical D_total was ${r['hist_d_total_min']:,.0f}, "
                  f"live median floor is ${r['live_cand_floor_median']:,.0f})")
    print("-> Part 1 conclusion: NOT an extrapolation issue. The historical panel goes even\n"
          "   lower than today's live floors in every tier that matters.")

    print("\n=== Part 2: is beta_RC's identifying sample representative of the tiers it's applied to? ===")
    pairs = beta_rc_pairs_with_tier()
    comp = beta_rc_composition_check(pairs)

    counts_df = pd.Series(comp["tier_counts"]).reindex(COOK_ORDER)
    print(f"\n118 repeat-challenger pairs (all D-challenger-vs-R-incumbent, by construction), by tier:")
    print(counts_df.to_string())
    print(f"\n  {comp['pct_safe_r']:.0%} of pairs are Safe R.")
    print(f"  {comp['pct_competitive']:.0%} of pairs are in the competitive tiers (Toss-Up/Lean D/Lean R) "
          f"-- the tiers the model's headline efficiency claim is actually about.")

    print(f"\nSplit-sample beta_RC estimates (informal comparison, small-n, wide/overlapping CIs):")
    print(f"  All 118 pairs (production value):  {comp['beta_rc_all']:.3f} (SE={comp['se_all']:.3f})")
    print(f"  Safe-R-only (n={comp['n_safe_r_only']}):             {comp['beta_rc_safe_r_only']:.3f} (SE={comp['se_safe_r_only']:.3f})")
    print(f"  Competitive-only (n={comp['n_competitive_only']}):        {comp['beta_rc_competitive_only']:.3f} (SE={comp['se_competitive_only']:.3f})")

    print(f"\nFormal interaction test (does beta1 differ in Safe R vs. elsewhere?):")
    print(f"  interaction coef = {comp['interaction_coef']:.3f}, p = {comp['interaction_p_value']:.3f}")
    print(f"  -> NOT statistically significant, but n=118 total / n={comp['n_competitive_only']} in the\n"
          f"     competitive tiers is underpowered to detect anything but a large difference.\n"
          f"     'Not significant' means 'cannot confirm a difference exists,' not 'confirmed uniform.'")

    pd.DataFrame([comp]).to_csv(ROOT / "outputs/beta_rc_tier_composition.csv", index=False)
    print(f"\nSaved -> outputs/beta_rc_tier_composition.csv")


if __name__ == "__main__":
    main()
