#!/usr/bin/env python3
"""
Check a set of flagged newly-funded races against the persuasion ceiling
directly, rather than just noting they exist. Two selection modes:

  --pvi-threshold X   (default 15.0) -- flag newly-funded races with |PVI| > X
  --district-ids A,B  -- flag an explicit, caller-supplied list of newly-funded
                          district IDs instead (e.g. the "unexplained and
                          material" residual from audit_omitted_information.py)

Two independent checks, both against real pipeline machinery (no
reimplementation of the ceiling math):

1. Saturation: for each flagged race, at the model's own recommended
   spending level, how much of the race's ceiling headroom C_i is actually
   used (`1 - exp(-delta/C_i)`)? Saturation near 1 means the recommendation
   is essentially determined by the cap's asymptote, not by the underlying
   regression -- a real reason to distrust the pick regardless of the
   candidate-quality flags. Saturation near 0 means the ceiling isn't doing
   much for this race and the flag is likely coincidental.

2. c_max robustness: does each flagged race still get selected (positive
   model-recommended party $) across the same 7-point c_max sweep
   {3,5,7,10,15,20,30} Paper I's own Appendix E.1 sweep uses? A race whose
   selection depends on how generous c_max is would be a ceiling-calibration
   artifact; a race selected at every tested c_max is not.

Usage:
    python scripts/check_deep_pvi_ceiling.py --cycle 2022 --processed-dir data/processed_oos_2020
    python scripts/check_deep_pvi_ceiling.py --cycle 2024 --district-ids PA-01,AZ-02,MT-01,CA-09,CO-04
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import predict, predict_floor_margin
from backtest.model.win_prob import compute_outputs_batch
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import optimize_nonlinear
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

SWEEP_C_MAX = [3, 5, 7, 10, 15, 20, 30]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check flagged newly-funded races against the persuasion ceiling")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--pvi-threshold", type=float, default=15.0)
    parser.add_argument("--district-ids", type=str, default=None,
                        help="Comma-separated list of district IDs to check instead of "
                             "the |PVI|>threshold selection, e.g. PA-01,AZ-02,MT-01")
    parser.add_argument("--label", type=str, default=None,
                        help="Output-filename tag (default: 'deep_pvi' or 'flagged' "
                             "depending on selection mode)")
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    cycle = args.cycle

    beta_rc, coef, sigma_model = load_processed_artifacts(processed)
    races = build_universe(cycle=cycle)
    budget = sum(r.d_total for r in races)
    party_budget = sum(r.d_total - r.cand_d_total for r in races)
    cand_floors = np.array([r.cand_d_total for r in races])
    r_totals = np.array([r.r_total for r in races])

    factor_model = build_dummy_factor_model(races, config.generic_ballot_2024())
    cov_matrix = factor_model.race_covariance()
    cap_baseline = config.optimizer_cfg()["cap_regimes"][-1]
    c_max_default = config.persuasion_ceiling_c_max()

    primary_result = optimize_nonlinear(
        races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline,
        party_budget=party_budget, eta=args.eta,
    )
    model_party = primary_result.allocations - cand_floors

    party_observed = np.array([r.d_total - r.cand_d_total for r in races])
    zero_funded_mask = party_observed <= (1e-3 * party_budget)
    newly_funded_mask = zero_funded_mask & (model_party > 1e-3 * party_budget)

    if args.district_ids:
        requested = [d.strip().upper() for d in args.district_ids.split(",") if d.strip()]
        id_to_idx = {r.district_id: i for i, r in enumerate(races)}
        missing = [d for d in requested if d not in id_to_idx]
        if missing:
            raise SystemExit(f"District ID(s) not found in the {cycle} universe: {missing}")
        not_newly_funded = [d for d in requested if not newly_funded_mask[id_to_idx[d]]]
        if not_newly_funded:
            print(f"Warning: {not_newly_funded} are not in the newly-funded set (DCCC zero, "
                  f"model positive) for cycle {cycle} -- checking them anyway, but the "
                  "ceiling-saturation framing assumes a newly-funded race.")
        flagged_idx = np.array([id_to_idx[d] for d in requested])
        label = args.label or "flagged"
        print(f"{len(flagged_idx)} explicitly-requested races (cycle {cycle}, c_max={c_max_default}): "
              f"{requested}\n")
    else:
        pvi_arr = np.array([r.pvi for r in races])
        deep_pvi_mask = newly_funded_mask & (np.abs(pvi_arr) > args.pvi_threshold)
        flagged_idx = np.where(deep_pvi_mask)[0]
        label = args.label or "deep_pvi"
        print(f"{len(flagged_idx)} newly-funded races with |PVI| > {args.pvi_threshold} "
              f"(cycle {cycle}, c_max={c_max_default}):\n")

    if len(flagged_idx) == 0:
        print("Nothing to check -- the selection is empty, so there's no ceiling-saturation "
              "or c_max-robustness question to ask.")
        return

    # ── Check 1: saturation at the model's recommended spending level ──────
    rows = []
    for i in flagged_idx:
        race = races[i]
        mu_floor = predict_floor_margin(
            pvi=race.pvi, incumb_status=race.incumb_status,
            generic_ballot=race.generic_ballot, cand_d_total=race.cand_d_total,
            r_total=race.r_total, coef=coef, cvap=race.cvap, indiv_share=race.indiv_share,
        )
        sigma_i = sigma_model.predict(abs(race.pvi), race.incumb_status, race.generic_ballot)
        phi0 = float(norm.cdf(mu_floor / sigma_i))
        C_i = float(ceiling_mod.ceiling(mu_floor, sigma_i, c_max_default))

        d_recommended = float(primary_result.allocations[i])
        ratio_recommended = np.clip(d_recommended / (d_recommended + race.r_total), 1e-6, 1 - 1e-6)
        mu_raw = predict(
            pvi=race.pvi, incumb_status=race.incumb_status, generic_ballot=race.generic_ballot,
            ratio=ratio_recommended, coef=coef, total_spend=d_recommended + race.r_total,
            cvap=race.cvap, indiv_share=race.indiv_share,
        )
        mu_capped, grad_factor = ceiling_mod.apply(mu_raw, mu_floor, sigma_i, c_max_default)
        delta = max(mu_raw - mu_floor, 0.0)
        saturation = 1.0 - float(grad_factor)  # grad_factor == decay == exp(-delta/C_i)

        rows.append({
            "district_id": race.district_id,
            "cook_rating": race.cook_rating,
            "pvi": round(race.pvi, 2),
            "phi0_at_floor": round(phi0, 4),
            "sigma_i": round(sigma_i, 3),
            "C_i_ceiling_pp": round(C_i, 4),
            "mu_floor_pp": round(mu_floor, 3),
            "mu_raw_uncapped_pp": round(mu_raw, 3),
            "mu_capped_pp": round(mu_capped, 3),
            "delta_raw_pp": round(delta, 3),
            "saturation_pct": round(saturation * 100, 1),
            "model_party_dollars": round(float(model_party[i]), 0),
        })

    sat_df = pd.DataFrame(rows).sort_values("saturation_pct", ascending=False).reset_index(drop=True)
    print("Check 1: ceiling saturation at the model's recommended spending level")
    print(sat_df.to_string(index=False))
    print(
        "\nsaturation_pct near 100 = recommendation is essentially the ceiling's own "
        "asymptote (mu_floor + C_i), not the underlying regression; near 0 = ceiling "
        "barely touched, recommendation reflects the raw structural estimate almost "
        "unchanged.\n"
    )

    # ── Check 2: c_max robustness sweep ─────────────────────────────────────
    print(f"Check 2: c_max robustness sweep {SWEEP_C_MAX} "
          f"(re-solving optimize_nonlinear at each point, this takes a while)")
    flagged_ids = [races[i].district_id for i in flagged_idx]
    sweep_rows = {did: [] for did in flagged_ids}

    for cm in SWEEP_C_MAX:
        config._cfg["persuasion_ceiling"]["c_max"] = cm
        res = optimize_nonlinear(
            races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline,
            party_budget=party_budget, eta=args.eta,
        )
        party_at_cm = res.allocations - cand_floors
        for i in flagged_idx:
            did = races[i].district_id
            party_i = float(party_at_cm[i])
            sweep_rows[did].append({
                "c_max": cm,
                "party_dollars": round(party_i, 0),
                "selected": bool(party_i > 1e-3 * party_budget),
            })
        print(f"  c_max={cm:>4}: "
              + ", ".join(f"{races[i].district_id}={'Y' if party_at_cm[i] > 1e-3*party_budget else 'N'}"
                           f"(${party_at_cm[i]:,.0f})" for i in flagged_idx))
    config._cfg["persuasion_ceiling"]["c_max"] = c_max_default

    print("\nSelected at every tested c_max (robust, not a ceiling artifact):")
    robust, fragile = [], []
    for did in flagged_ids:
        all_selected = all(r["selected"] for r in sweep_rows[did])
        (robust if all_selected else fragile).append(did)
    print(f"  {robust if robust else '(none)'}")
    print("Selection depends on c_max (ceiling-sensitive -- treat with more caution):")
    print(f"  {fragile if fragile else '(none)'}")

    sweep_long = pd.DataFrame(
        [{"district_id": did, **row} for did, rows_ in sweep_rows.items() for row in rows_]
    )
    sat_path = config.outputs_path() / f"{label}_ceiling_saturation_{cycle}.csv"
    sweep_path = config.outputs_path() / f"{label}_ceiling_check_{cycle}.csv"
    sat_df.to_csv(sat_path, index=False)
    sweep_long.to_csv(sweep_path, index=False)
    print(f"\nSaved: {sat_path}")
    print(f"Saved: {sweep_path}")


if __name__ == "__main__":
    main()
