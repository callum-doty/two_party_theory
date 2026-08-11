#!/usr/bin/env python3
"""
Characterizes the mechanism behind Paper III Section 8.2's "methodological
confound" -- the LP allocator (optimize(), linearized MSG objective) gives
15.0% of the live 2026 party budget to Safe R/Likely R, dramatically
different from the nonlinear optimizer's (optimize_nonlinear()) 65.6% --
which was checked once, confirmed real, and never reproduced as a script or
explained mechanically. No prior script computed this; this is the first.

Runs BOTH allocators on the IDENTICAL inputs `_solve_one_period()`
(src/backtest/dynamic/horizon.py) feeds optimize_nonlinear() -- same
races_with_floor (cand_d_total + L_t baked in), same ModelOutputs (msg_i
frozen at that floor point), same party_budget, same cap_fraction -- so any
difference is attributable to the optimizer's mechanics, not a setup
mismatch.

Hypothesis being tested: optimize()'s objective (`maximize msg @ s`) treats
each race's msg_i as a FIXED per-dollar constant -- it never re-evaluates
msg as dollars are added, so it has no mechanism to express diminishing
returns. It should therefore behave like a greedy knapsack: rank races by
(frozen) msg_i, fill the highest-msg race to its cap, then the next, until
the budget is exhausted. If msg_i's ranking is NOT dominated by Safe
R/Likely R (despite figure 6 showing MSG spikes at extreme P_win), the LP's
allocation would concentrate in whichever tier happens to hold the single
highest-msg races -- which may not be Safe R/Likely R at all.

Output: outputs/lp_vs_nonlinear_mechanism.csv, printed summary.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.ledger import RealizedSpendCommitmentSource, CapitalLedger
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import optimize, optimize_nonlinear
from run_backtest import load_processed_artifacts, build_dummy_factor_model
from plot_2026_live_allocation import BUDGET_2026

COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
CAP_FRACTION = 0.15


def main():
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    from datetime import datetime, timezone
    as_of = datetime.now(timezone.utc).date()
    commitment_source = RealizedSpendCommitmentSource(cycle=2026, party="D")
    ledger = CapitalLedger.build(0, BUDGET_2026, commitment_source, as_of, races)

    races_with_floor = ledger.apply_to_races(races)
    outputs = compute_outputs_batch(races_with_floor, coef, sigma_model)
    floor = ledger.deployable_floor_for(races)
    party_budget = ledger.deployable_total
    total_d_spend = ledger.total_budget + float(floor.sum())

    print(f"party_budget (F0) = ${party_budget:,.0f}, cap per race = ${CAP_FRACTION * party_budget:,.0f}\n")

    # --- Nonlinear optimizer (Paper II's actual live run) ---
    nl_result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=total_d_spend,
        cov_matrix=cov_matrix, gamma=0.0, cap_fraction=CAP_FRACTION,
        party_budget=party_budget,
    )

    # --- LP allocator, on the IDENTICAL inputs (frozen msg_i, same floor/budget) ---
    lp_result = optimize(
        outputs, budget=total_d_spend, cov_matrix=cov_matrix, gamma=0.0,
        cap_fraction=CAP_FRACTION, floor_allocations=floor, party_budget=party_budget,
    )

    tiers = np.array([r.cook_rating for r in races])
    df = pd.DataFrame({
        "district_id": [r.district_id for r in races],
        "tier": tiers,
        "msg_i_per_1m": [o.msg_i * 1e6 for o in outputs],
        "floor": floor,
        "nl_party_alloc": np.maximum(nl_result.allocations - floor, 0.0),
        "lp_party_alloc": np.maximum(lp_result.allocations - floor, 0.0),
    })

    print("=== Tier-level party-budget share: LP vs. nonlinear, identical inputs ===\n")
    tier_summary = df.groupby("tier").agg(
        n=("district_id", "count"),
        nl_total=("nl_party_alloc", "sum"),
        lp_total=("lp_party_alloc", "sum"),
    ).reindex(COOK_ORDER)
    tier_summary["nl_share"] = tier_summary["nl_total"] / tier_summary["nl_total"].sum()
    tier_summary["lp_share"] = tier_summary["lp_total"] / tier_summary["lp_total"].sum()
    print(tier_summary.to_string(float_format=lambda x: f"{x:,.3f}" if x < 2 else f"{x:,.0f}"))

    safe_r_likely_r_nl = tier_summary.loc[["Safe R", "Likely R"], "nl_share"].sum()
    safe_r_likely_r_lp = tier_summary.loc[["Safe R", "Likely R"], "lp_share"].sum()
    print(f"\nSafe R + Likely R: nonlinear={safe_r_likely_r_nl:.1%}, LP={safe_r_likely_r_lp:.1%}")

    # --- Mechanism check: how many races does the LP actually fund, and where do they rank by msg? ---
    n_lp_funded = int((df["lp_party_alloc"] > 1.0).sum())
    n_lp_capped = int((df["lp_party_alloc"] >= CAP_FRACTION * party_budget * 0.999).sum())
    print(f"\n=== Mechanism: LP concentration ===")
    print(f"LP funds {n_lp_funded}/{len(df)} races with >$1 (nonlinear funds "
          f"{int((df['nl_party_alloc'] > 1.0).sum())}/{len(df)})")
    print(f"LP hits the per-race cap (${CAP_FRACTION * party_budget:,.0f}) on {n_lp_capped} races")

    top_lp = df.sort_values("lp_party_alloc", ascending=False).head(10)
    print(f"\nTop 10 LP recipients (msg_i frozen at floor point, ranked by raw MSG):")
    print(top_lp[["district_id", "tier", "msg_i_per_1m", "floor", "lp_party_alloc"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}" if x < 1000 else f"{x:,.0f}"))

    df.to_csv(ROOT / "outputs/lp_vs_nonlinear_mechanism.csv", index=False)
    print(f"\nSaved -> outputs/lp_vs_nonlinear_mechanism.csv")


if __name__ == "__main__":
    main()
