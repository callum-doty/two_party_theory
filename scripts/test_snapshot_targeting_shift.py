#!/usr/bin/env python3
"""
Premise check for an adaptive-reallocation channel (docs/theta_followup_plan.md
Section 10's proposed next step): if the DCCC's real, final-cycle party budget
had to be allocated using ONLY the information available at an early
mid-cycle snapshot, versus a late one, would the NONLINEAR optimizer
(optimize_nonlinear() -- the one that actually respects diminishing returns,
unlike the LP's knapsack-degenerate optimize(), per paper3_draft.md Section
8.2's addendum) meaningfully change WHICH races it targets?

If yes: waiting to let idiosyncratic uncertainty resolve, then re-targeting,
would have been a real, exploitable improvement in these two historical
cycles -- concrete motivation (and a template) for building an adaptive-
reallocation channel into the Bellman machinery. If no: the same allocator,
even with more information, converges on nearly the same targets regardless
of when it's asked, and an adaptive-reallocation build would not be worth
the engineering cost.

Design: hold the REAL, final-cycle total budget and DCCC party budget fixed
(so this isolates "does information change targeting," not "does the amount
of money available change targeting"). Candidate-committee floors are
frozen at their real final value at every snapshot regardless (this repo's
known data constraint -- no per-filing-date source for candidate/coordinated
spend, docs/theta_followup_plan.md Section 10.1). Only the opponent's (R)
side of the ratio genuinely varies by snapshot, via dated IE spend
(fec.cumulative_ie_as_of) -- the same constraint Section 10's tests already
disclosed. Two snapshots per cycle, chosen using Section 10's own finding
about WHERE real differentiation happens (correlation with realized outcome
was flat from 270 to ~90 days out, then moved only in the final 60-30 day
window): 60 days out and 14 days out.

Output: outputs/snapshot_targeting_shift_check.json
"""

from __future__ import annotations
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data.universe import build_universe
from backtest.optimizer.allocator import optimize_nonlinear
from backtest.dynamic.simulate import (
    _static_floor_totals, _reconstruct_races_at, _has_dated_candidate_panel,
    _candidate_fallback_totals,
)

from validate_state_simulator import load_coef_and_sigma, CYCLE_CONFIG

ROOT = Path(__file__).parent.parent
CAP_FRACTION = 0.15   # matches run_backtest.py's baseline cap regime
GAMMA = 0.0            # pure E[Seats] max, matches run_backtest.py's gate baseline
ETA = 0.0              # retrospective mode, matches run_backtest.py's default
SNAPSHOT_OFFSETS_DAYS = {"60d_out": 60, "14d_out": 14}
TOP_N = 20


def run_snapshot(cycle: int, label: str, offset_days: int,
                  coef, sigma_model, final_races, static_totals, budget, party_budget, cov_matrix):
    election_day = CYCLE_CONFIG[cycle]["election_day"]
    snap_date = election_day - timedelta(days=offset_days)
    use_dated_candidate_spend = _has_dated_candidate_panel(cycle)
    candidate_fallback_totals = None if use_dated_candidate_spend else _candidate_fallback_totals(cycle)
    races = _reconstruct_races_at(
        period_index=0, period_date=snap_date, cycle=cycle,
        base_races=final_races, static_totals=static_totals,
        use_dated_candidate_spend=use_dated_candidate_spend,
        candidate_fallback_totals=candidate_fallback_totals,
    )
    result = optimize_nonlinear(
        races, coef, sigma_model, budget, cov_matrix, GAMMA, CAP_FRACTION,
        party_budget=party_budget, eta=ETA,
    )
    party_alloc = np.maximum(result.allocations - np.array([r.cand_d_total for r in races]), 0.0)
    df = pd.DataFrame({
        "district_id": [r.district_id for r in races],
        "tier": [r.cook_rating for r in races],
        "party_alloc": party_alloc,
    }).sort_values("party_alloc", ascending=False).reset_index(drop=True)
    return {
        "cycle": cycle, "label": label, "snapshot_date": str(snap_date),
        "days_before_election": offset_days,
        "expected_seats": result.expected_seats, "status": result.status,
        "n_corner_solutions": result.n_corner_solutions,
        "top_n_districts": df.head(TOP_N)["district_id"].tolist(),
        "alloc_by_district": dict(zip(df["district_id"], df["party_alloc"])),
    }


def compare(cycle: int, snap_a: dict, snap_b: dict) -> dict:
    top_a, top_b = set(snap_a["top_n_districts"]), set(snap_b["top_n_districts"])
    overlap = top_a & top_b
    jaccard = len(overlap) / len(top_a | top_b)

    districts = sorted(set(snap_a["alloc_by_district"]) & set(snap_b["alloc_by_district"]))
    a_vec = np.array([snap_a["alloc_by_district"][d] for d in districts])
    b_vec = np.array([snap_b["alloc_by_district"][d] for d in districts])
    rho, p = stats.spearmanr(a_vec, b_vec)

    return {
        "cycle": cycle,
        "top_n": TOP_N,
        "jaccard_overlap_top_n": jaccard,
        "only_in_a": sorted(top_a - top_b), "only_in_b": sorted(top_b - top_a),
        "spearman_rho_full_allocation": float(rho), "spearman_p": float(p),
        "expected_seats_a": snap_a["expected_seats"], "expected_seats_b": snap_b["expected_seats"],
    }


def main():
    all_results = {"snapshots": [], "comparisons": []}
    for cycle in [2022, 2024]:
        cfg = CYCLE_CONFIG[cycle]
        coef, sigma_model = load_coef_and_sigma(cfg["processed_dir"])
        final_races = build_universe(cycle=cycle)
        static_totals = _static_floor_totals(cycle)
        n = len(final_races)
        budget = sum(r.d_total for r in final_races)
        party_budget = sum(r.d_total - r.cand_d_total for r in final_races)
        cov_matrix = np.eye(n) * 1e-6   # unused: GAMMA=0 means the variance term never enters the objective

        print(f"=== {cycle}: budget=${budget:,.0f}, party_budget=${party_budget:,.0f} ===")
        snaps = {}
        for label, offset in SNAPSHOT_OFFSETS_DAYS.items():
            print(f"  running optimize_nonlinear() at {label} ({offset}d before election)...")
            snaps[label] = run_snapshot(cycle, label, offset, coef, sigma_model,
                                         final_races, static_totals, budget, party_budget, cov_matrix)
            print(f"    E[Seats]={snaps[label]['expected_seats']:.2f}, "
                  f"top-5: {snaps[label]['top_n_districts'][:5]}")
            all_results["snapshots"].append(snaps[label])

        cmp = compare(cycle, snaps["60d_out"], snaps["14d_out"])
        all_results["comparisons"].append(cmp)
        print(f"  Jaccard overlap (top {TOP_N}): {cmp['jaccard_overlap_top_n']:.2f}")
        print(f"  Spearman rho (full allocation vector, 60d vs 14d): {cmp['spearman_rho_full_allocation']:+.3f}")
        print(f"  Only in 60d-out top-{TOP_N}: {cmp['only_in_a']}")
        print(f"  Only in 14d-out top-{TOP_N}: {cmp['only_in_b']}")
        print()

    out_path = ROOT / "outputs/snapshot_targeting_shift_check.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
