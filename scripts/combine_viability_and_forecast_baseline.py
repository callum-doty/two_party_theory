#!/usr/bin/env python3
"""
Closes the loose thread flagged after the last round of results: Section
9's viability-flag decomposition (naive headline vs. DCCC-observed) and
Section 12's forecast-based baseline (model-optimal vs. forecasted-DCCC)
were never combined. This does that, for the two cycles with both a real
omitted-information audit and a real leave-one-cycle-out forecast: 2022 and
2024.

Three scenarios per cycle, all evaluated against the SAME forecasted-DCCC
baseline (held fixed -- it represents DCCC's realistic behavior, viability
concerns and all, since real DCCC behavior already implicitly reflects
whatever institutional knowledge produced those flags in the first place;
only what the OPTIMIZER is allowed to recommend varies):

  A. All races eligible (Section 12's already-reported forecast-baseline gain)
  B. Flagged (publicly-explicable) races hard-excluded from the optimizer
  D. Flagged races soft-penalized (graduated by n_flags, Section 9's exact treatment)

If B collapses toward zero the way it did against the real-DCCC baseline
(Section 9: 78-85% of headline gone), the forecast-baseline residual
(+1.22/+1.76) is ALSO mostly viability-driven, not a more robust finding.
If B stays substantially positive, the forecast-baseline result survives
the same scrutiny the naive headline didn't.

Usage:
    python scripts/combine_viability_and_forecast_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import optimize_nonlinear, nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore
from build_dccc_forecast_training_data import COOK_ORDINAL  # type: ignore
from fit_dccc_forecast_model import FEATURES, _prep, fit_hurdle, predict_shares  # type: ignore

CYCLES = {2024: config.processed_path(), 2022: Path("data/processed_oos_2020")}


def main() -> None:
    train_all = _prep(pd.read_csv(config.processed_path() / "dccc_forecast_training_data.csv"))

    all_rows = []
    for cycle, processed in CYCLES.items():
        suffix = f"_{cycle}" if cycle != 2024 else ""
        beta_rc, coef, sigma_model = load_processed_artifacts(processed)
        races = build_universe(cycle=cycle)
        budget = sum(r.d_total for r in races)
        party_budget = sum(r.d_total - r.cand_d_total for r in races)
        factor_model = build_dummy_factor_model(races, config.generic_ballot_for_cycle(cycle))
        cov_matrix = factor_model.race_covariance()
        cap_baseline = config.optimizer_cfg()["cap_regimes"][-1]

        # ── Leave-this-cycle-out forecast (identical to fit_dccc_forecast_model.py's LOCO fold) ──
        train_df = train_all[train_all["cycle"] != cycle]
        test_df = train_all[train_all["cycle"] == cycle].set_index("district_id").loc[
            [r.district_id for r in races]].reset_index()
        sel_model, inten_model = fit_hurdle(train_df)
        raw_pred = predict_shares(sel_model, inten_model, test_df)
        forecast_dollars = raw_pred / raw_pred.sum() * party_budget
        forecast_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, forecast_dollars)

        # ── Viability flags from the omitted-information audit ─────────────
        audit = pd.read_csv(config.outputs_path() / f"omitted_information_audit{suffix}.csv")
        flagged_ids = set(audit.loc[audit["explicable_by_observables"], "district_id"])
        n_flags_by_id = dict(zip(audit["district_id"], audit["n_flags"]))
        district_ids = [r.district_id for r in races]

        # A: all races, vs forecast (should match Section 12's reported figure)
        result_a = optimize_nonlinear(
            races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline, party_budget=party_budget,
        )
        gain_a = result_a.expected_seats - forecast_seats

        # B: flagged races hard-excluded, vs forecast
        mask_b = np.array([did in flagged_ids for did in district_ids])
        result_b = optimize_nonlinear(
            races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline, party_budget=party_budget,
            fixed_zero_mask=mask_b,
        )
        gain_b = result_b.expected_seats - forecast_seats

        # D: flagged races soft-penalized, vs forecast
        floor_maturity_d = np.array([
            max(1.0 - 0.25 * n_flags_by_id.get(did, 0), 0.25) if did in flagged_ids else 1.0
            for did in district_ids
        ])
        result_d = optimize_nonlinear(
            races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline, party_budget=party_budget,
            floor_maturity=floor_maturity_d,
        )
        gain_d = result_d.expected_seats - forecast_seats

        print(f"\nCycle {cycle} (forecast E[Seats]={forecast_seats:.3f}):")
        print(f"  A. All races vs. forecast:              {gain_a:+.3f}")
        print(f"  B. Flagged races excluded vs. forecast: {gain_b:+.3f}  ({100*gain_b/gain_a:.1f}% of A)")
        print(f"  D. Flagged soft-penalized vs. forecast: {gain_d:+.3f}  ({100*gain_d/gain_a:.1f}% of A)")

        all_rows.append({"cycle": cycle, "scenario": "A_all_races_vs_forecast", "gain": gain_a})
        all_rows.append({"cycle": cycle, "scenario": "B_flagged_excluded_vs_forecast", "gain": gain_b,
                          "pct_of_A": 100 * gain_b / gain_a})
        all_rows.append({"cycle": cycle, "scenario": "D_flagged_penalized_vs_forecast", "gain": gain_d,
                          "pct_of_A": 100 * gain_d / gain_a})

    df = pd.DataFrame(all_rows)
    out_path = config.outputs_path() / "combined_viability_forecast_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
