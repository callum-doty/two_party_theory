#!/usr/bin/env python3
"""
Decompose Paper I's selection gain by viability-flag status
(scripts/audit_omitted_information.py's output) -- the follow-up the audit's
headcount alone doesn't answer: how much of the +2.56 (2024) / +2.69 (2022)
selection gain actually comes from the 56/60 flagged (publicly-explicable)
races versus the 8/12 unexplained ones, and does the headline gain survive
if flagged races are excluded or penalized rather than just counted.

Five eligible-universe scenarios, all using the identical optimizer
(optimize_nonlinear, unmodified) and the identical DCCC-observed baseline --
only which races the optimizer is permitted to fund changes:

  A. All races (Paper I's own headline, reproduced as a sanity check)
  B. Flagged (explicable) races hard-excluded -- fixed_zero_mask=True for
     the 56/60 races with >=1 viability flag; every other race (already-
     funded, plus the unexplained residual) stays eligible.
  C. Only the unexplained residual is newly-fundable -- every zero-funded
     race EXCEPT the unexplained ones is excluded; already-funded races
     can still be rebalanced (this isolates the unexplained residual's own
     contribution from the intensity component, not conflates them).
  D. Flagged races soft-penalized, not excluded -- floor_maturity (reused
     as a graduated viability-discount multiplier, 1 - 0.25*n_flags per
     flagged race, floored at 0.25) applied to their persuasion ceiling;
     every race stays eligible, but flagged ones are structurally
     discouraged in proportion to how many flags they tripped.
  E. Material unexplained-only (the 5/6 races that are both unexplained
     AND not a marginal/rounding-sized recommendation) -- the narrowest,
     sharpest cut: everything else zero-funded is excluded.

Also reports $ allocated to flagged vs. unflagged races under the
unrestricted (scenario A) optimal allocation.

Usage:
    python scripts/decompose_selection_gain_by_viability.py --cycle 2024
    python scripts/decompose_selection_gain_by_viability.py --cycle 2022 --processed-dir data/processed_oos_2020
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import optimize_nonlinear
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose selection gain by viability-flag status")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    cycle = args.cycle
    suffix = f"_{cycle}" if cycle != 2024 else ""

    beta_rc, coef, sigma_model = load_processed_artifacts(processed)
    races = build_universe(cycle=cycle)
    budget = sum(r.d_total for r in races)
    party_budget = sum(r.d_total - r.cand_d_total for r in races)

    outputs = compute_outputs_batch(races, coef, sigma_model)
    dccc_expected_seats = float(sum(o.p_win for o in outputs))

    factor_model = build_dummy_factor_model(races, config.generic_ballot_for_cycle(cycle))
    cov_matrix = factor_model.race_covariance()
    cap_baseline = config.optimizer_cfg()["cap_regimes"][-1]

    audit_path = config.outputs_path() / f"omitted_information_audit{suffix}.csv"
    summary_path = config.outputs_path() / f"omitted_information_audit_summary{suffix}.json"
    audit = pd.read_csv(audit_path)
    summary = json.load(open(summary_path))

    flagged_ids = set(audit.loc[audit["explicable_by_observables"], "district_id"])
    unexplained_ids = set(audit.loc[~audit["explicable_by_observables"], "district_id"])
    material_unexplained_ids = set(summary["unexplained_and_material_district_ids"])
    n_flags_by_id = dict(zip(audit["district_id"], audit["n_flags"]))

    party_observed = np.array([r.d_total - r.cand_d_total for r in races])
    zero_funded_mask = party_observed <= (1e-3 * party_budget)
    district_ids = [r.district_id for r in races]

    def run(label: str, fixed_zero_mask=None, floor_maturity=None):
        result = optimize_nonlinear(
            races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline,
            party_budget=party_budget, eta=args.eta,
            fixed_zero_mask=fixed_zero_mask, floor_maturity=floor_maturity,
        )
        gain = result.expected_seats - dccc_expected_seats
        print(f"  {label}: E[Seats]={result.expected_seats:.3f}  gain={gain:+.3f}")
        return result, gain

    print(f"Cycle {cycle} -- DCCC observed E[Seats]={dccc_expected_seats:.3f}\n")

    # ── A. All races ────────────────────────────────────────────────────────
    result_a, gain_a = run("A. All races (Paper I headline)")

    # ── B. Flagged races hard-excluded ──────────────────────────────────────
    mask_b = np.array([did in flagged_ids for did in district_ids])
    _, gain_b = run("B. Flagged (explicable) races excluded", fixed_zero_mask=mask_b)

    # ── C. Only the unexplained residual is newly-fundable ─────────────────
    mask_c = np.array([
        zero_funded_mask[i] and district_ids[i] not in unexplained_ids
        for i in range(len(races))
    ])
    _, gain_c = run("C. Only unexplained residual newly-fundable", fixed_zero_mask=mask_c)

    # ── D. Flagged races soft-penalized ─────────────────────────────────────
    floor_maturity_d = np.array([
        max(1.0 - 0.25 * n_flags_by_id.get(did, 0), 0.25) if did in flagged_ids else 1.0
        for did in district_ids
    ])
    _, gain_d = run("D. Flagged races soft-penalized (graduated by n_flags)", floor_maturity=floor_maturity_d)

    # ── E. Material unexplained-only ────────────────────────────────────────
    mask_e = np.array([
        zero_funded_mask[i] and district_ids[i] not in material_unexplained_ids
        for i in range(len(races))
    ])
    _, gain_e = run("E. Material unexplained-only (5-6 races)", fixed_zero_mask=mask_e)

    # ── $ allocated to flagged vs. unflagged under the unrestricted optimum ─
    floors = np.array([r.cand_d_total for r in races])
    model_party_a = result_a.allocations - floors
    flagged_idx = np.array([did in flagged_ids for did in district_ids])
    unexplained_idx = np.array([did in unexplained_ids for did in district_ids])
    dollars_flagged = float(model_party_a[flagged_idx].sum())
    dollars_unexplained = float(model_party_a[unexplained_idx].sum())
    print(f"\n$ allocated under scenario A -- flagged races: ${dollars_flagged:,.0f}  "
          f"unexplained races: ${dollars_unexplained:,.0f}")

    # ── Summary table ────────────────────────────────────────────────────────
    table = pd.DataFrame([
        {"scenario": "A. All races", "modeled_gain": gain_a},
        {"scenario": "B. Flagged (publicly viable) excluded", "modeled_gain": gain_b},
        {"scenario": "C. Unexplained residual only, newly-fundable", "modeled_gain": gain_c},
        {"scenario": "D. Flagged races soft-penalized", "modeled_gain": gain_d},
        {"scenario": "E. Material unexplained-only (5-6 races)", "modeled_gain": gain_e},
    ])
    table["pct_of_headline"] = (table["modeled_gain"] / gain_a * 100).round(1)
    print("\n" + table.round(3).to_string(index=False))

    out_path = config.outputs_path() / f"selection_gain_by_viability{suffix}.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
