#!/usr/bin/env python3
"""
Does WHEN a dollar is spent matter, independent of the FINAL cumulative
D/R ratio? Motivated by a real objection to the 1-year-horizon counterfactual
(docs/theta_followup_plan.md Section 7): the Bellman/LSM machinery inherits
Paper I's margin model, which was fit on cycle-cumulative D_total/R_total
(src/backtest/data/fec.py:build_total_spend()) with no timing information at
all -- a dollar spent in January and a dollar spent in October are
indistinguishable to it, as long as the FINAL ratio is the same. If real
campaign spending's persuasive effect decays over time (as the campaign
finance literature broadly holds), that is a real, missing mechanism that
would explain why the model recommends full deployment even a year out,
where a real committee would not. This script checks two versions of that
hypothesis against actual data, on the 2022/2024 held-out cycles already
used for Section 5.4/5.5's Validation A.

KNOWN DATA CONSTRAINT, stated up front (dynamic/simulate.py's own
documented gap, `_static_floor_totals`'s docstring): candidate-committee and
coordinated-expenditure spend have NO per-filing-date source anywhere in
this repository -- only independent-expenditure (IE) spend is genuinely
date-bucketed (`fec.cumulative_ie_as_of`). Every point-in-time
reconstruction below therefore holds candidate+coordinated spend at its
FULL, FINAL cycle value at every snapshot date and only varies the IE
component. This is not a new limitation introduced here -- it is the same
constraint Validation A (`scripts/validate_state_simulator.py`) already
operates under -- but it means both tests below are checks of IE-spend
timing specifically, not all-source spend timing. Reported honestly as a
partial, not complete, test of the timing-decay hypothesis.

Test 1 (extends Validation A across the cycle, not just at September 1):
  Reconstruct each competitive race's mu_i at several snapshot dates spread
  across the cycle, Spearman-correlate each snapshot against realized
  November margin. If early-cycle financial state predicts the outcome
  about as well as late-cycle state, that's evidence AGAINST a strong
  timing/decay story (early money is "just as good" as late money at
  forecasting, consistent with the static model's assumption). If early
  snapshots correlate much more weakly, that's *consistent with* decay --
  but confounded with the alternative explanation that races simply become
  more differentiated/informative later in the cycle for reasons having
  nothing to do with money decaying (more is known, more has happened).

Test 2 (sharper, decay-specific): does a race's front-loading intensity --
  how much of a party's total-cycle IE spend happened before a mid-cycle
  cutoff, relative to the opponent's -- predict a systematic RESIDUAL after
  controlling for the FINAL, full-cycle D/R ratio (i.e. Paper I's standard,
  unmodified prediction)? This isolates timing from "how much was
  eventually spent in total," which Test 1 cannot fully separate. If
  front-loaded D spend predicts a NEGATIVE residual (D underperforms what
  the final-ratio-only model expects), that is direct evidence that early
  money "counts for less" than the model assumes -- i.e. decay.

Output: outputs/spending_timing_decay_check.json
"""

from __future__ import annotations
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data import elections, fec
from backtest.data.universe import build_universe
from backtest.model.win_prob import compute_outputs_batch
from backtest.dynamic.simulate import (
    _static_floor_totals, _reconstruct_races_at,
    _has_dated_candidate_panel, _candidate_fallback_totals,
)

from validate_state_simulator import load_coef_and_sigma, CYCLE_CONFIG, COMPETITIVE

ROOT = Path(__file__).parent.parent
MID_CUTOFF = {2022: date(2022, 9, 1), 2024: date(2024, 9, 1)}   # Test 2's front/back split


def snapshot_dates(cycle: int) -> list[date]:
    election_day = CYCLE_CONFIG[cycle]["election_day"]
    # Roughly monthly from ~9 months out to ~3 weeks out.
    offsets_days = [270, 240, 210, 180, 150, 120, 90, 60, 30]
    return [election_day - timedelta(days=d) for d in offsets_days]


# ─── Test 1: correlation strength across the cycle ─────────────────────────

def test1_correlation_by_snapshot(cycle: int) -> list[dict]:
    cfg = CYCLE_CONFIG[cycle]
    coef, sigma_model = load_coef_and_sigma(cfg["processed_dir"])
    base_races = build_universe(cycle=cycle)
    static_totals = _static_floor_totals(cycle)
    results = elections.load_results(cycle)
    realized = dict(zip(results["district_id"], results["margin_pp"]))
    use_dated_candidate_spend = _has_dated_candidate_panel(cycle)
    candidate_fallback_totals = None if use_dated_candidate_spend else _candidate_fallback_totals(cycle)

    out = []
    election_day = cfg["election_day"]
    for i, snap_date in enumerate(snapshot_dates(cycle)):
        races = _reconstruct_races_at(
            period_index=i, period_date=snap_date, cycle=cycle,
            base_races=base_races, static_totals=static_totals,
            use_dated_candidate_spend=use_dated_candidate_spend,
            candidate_fallback_totals=candidate_fallback_totals,
        )
        outputs = compute_outputs_batch(races, coef, sigma_model)
        tier_by_district = {r.district_id: r.cook_rating for r in races}
        rows = [
            {"mu": o.mu_hat, "realized": realized[o.district_id]}
            for o in outputs
            if o.district_id in realized and tier_by_district.get(o.district_id) in COMPETITIVE
        ]
        df = pd.DataFrame(rows)
        rho, p = stats.spearmanr(df["mu"], df["realized"])
        out.append({
            "cycle": cycle, "snapshot_date": str(snap_date),
            "days_before_election": (election_day - snap_date).days,
            "n_competitive": len(df), "spearman_rho": float(rho), "p_value": float(p),
        })
    return out


# ─── Test 2: front-loading residual test ───────────────────────────────────

def test2_frontload_residual(cycle: int) -> dict:
    cfg = CYCLE_CONFIG[cycle]
    coef, sigma_model = load_coef_and_sigma(cfg["processed_dir"])
    base_races = build_universe(cycle=cycle)   # final, full-cycle D/R totals
    outputs = compute_outputs_batch(base_races, coef, sigma_model)
    mu_final = {o.district_id: o.mu_hat for o in outputs}
    tier_by_district = {r.district_id: r.cook_rating for r in base_races}

    results = elections.load_results(cycle)
    realized = dict(zip(results["district_id"], results["margin_pp"]))

    ie_total = fec.load_independent_expenditures(cycle)
    ie_early = fec.cumulative_ie_as_of(cycle, MID_CUTOFF[cycle])

    def _party_totals(df, col):
        return df.groupby(["district_id", "party"])[col].sum()

    tot = _party_totals(ie_total, "ie_net")
    early = _party_totals(ie_early, "ie_net")

    rows = []
    for did in mu_final:
        if did not in realized or tier_by_district.get(did) not in COMPETITIVE:
            continue
        try:
            d_tot, r_tot = tot.get((did, "D"), 0.0), tot.get((did, "R"), 0.0)
            d_early, r_early = early.get((did, "D"), 0.0), early.get((did, "R"), 0.0)
        except KeyError:
            continue
        # Require meaningful IE activity on both sides to compute a share at all.
        if d_tot < 10_000 or r_tot < 10_000:
            continue
        d_early_share = d_early / d_tot
        r_early_share = r_early / r_tot
        frontload_asymmetry = d_early_share - r_early_share   # >0: D more front-loaded than R
        residual = realized[did] - mu_final[did]               # + means D overperformed the final-ratio prediction
        rows.append({
            "district_id": did, "tier": tier_by_district[did],
            "d_early_share": d_early_share, "r_early_share": r_early_share,
            "frontload_asymmetry": frontload_asymmetry, "residual": residual,
            "mu_final": mu_final[did], "realized": realized[did],
        })

    df = pd.DataFrame(rows)
    if len(df) < 8:
        return {"cycle": cycle, "n": len(df), "note": "too few races with two-sided IE activity"}

    rho, p = stats.spearmanr(df["frontload_asymmetry"], df["residual"])
    slope, intercept, r_value, p_lin, se = stats.linregress(df["frontload_asymmetry"], df["residual"])
    return {
        "cycle": cycle, "n": len(df), "mid_cutoff": str(MID_CUTOFF[cycle]),
        "spearman_rho_frontload_vs_residual": float(rho), "spearman_p": float(p),
        "ols_slope": float(slope), "ols_slope_se": float(se), "ols_p": float(p_lin),
        "interpretation": (
            "negative slope/rho = front-loading D relative to R predicts D UNDERPERFORMING "
            "the final-ratio-only prediction -- consistent with decay (early money counting "
            "for less than the static model assumes). positive = no decay signal, or reverse."
        ),
        "rows": rows,
    }


def main():
    all_results = {"test1_correlation_by_snapshot": [], "test2_frontload_residual": {}}
    for cycle in [2022, 2024]:
        print(f"=== Test 1: correlation strength across the cycle, {cycle} ===")
        t1 = test1_correlation_by_snapshot(cycle)
        all_results["test1_correlation_by_snapshot"].extend(t1)
        for r in t1:
            print(f"  {r['snapshot_date']} ({r['days_before_election']}d before E-day): "
                  f"n={r['n_competitive']}, rho={r['spearman_rho']:.3f} (p={r['p_value']:.4f})")

        print(f"\n=== Test 2: front-load residual test, {cycle} ===")
        t2 = test2_frontload_residual(cycle)
        all_results["test2_frontload_residual"][cycle] = t2
        if "note" in t2:
            print(f"  {t2['note']}")
        else:
            print(f"  n={t2['n']}, rho(frontload, residual)={t2['spearman_rho_frontload_vs_residual']:+.3f} "
                  f"(p={t2['spearman_p']:.4f}), OLS slope={t2['ols_slope']:+.4f} (p={t2['ols_p']:.4f})")
        print()

    out_path = ROOT / "outputs/spending_timing_decay_check.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
