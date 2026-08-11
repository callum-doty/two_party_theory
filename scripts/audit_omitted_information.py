#!/usr/bin/env python3
"""
Omitted-information audit of the newly-funded races behind Paper I's
selection-vs-intensity decomposition (Section 8: 64 of 433 races, 91% of
the model-implied gain, receive zero DCCC party dollars but positive
party dollars under the model-optimal reallocation).

Question this answers: for each such race, does *public* data already
contain a plausible reason a sophisticated committee might have declined
to fund it (weak fundraising traction, being financially overwhelmed by
the opponent, thin grassroots support, extreme structural PVI), or does
it have a real candidate floor and no visible public-data explanation?

This does not resolve whether the model is right -- it cannot, from
observational data alone (Paper I's own stated limitation). It sorts the
64 races into "explicable by observables already in this pipeline" versus
"no visible explanation," which is the evidence needed to judge whether
the 2.83-seat gain is more likely a real institutional blind spot or a
sign the model is missing state variables DCCC has and this repo doesn't.

Reuses the exact same estimation artifacts, universe, and optimizer call
run_backtest.py uses for the primary (γ=0, 15% cap) allocation, so the
race set produced here is identical to the one behind Paper I Table 5.

Usage:
    python scripts/audit_omitted_information.py
    python scripts/audit_omitted_information.py --cycle 2022 --processed-dir data/processed_oos_2020
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data.universe import build_universe, competitive_subset
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import optimize_nonlinear
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Omitted-information audit of newly-funded races")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    cycle = args.cycle
    suffix = f"_{cycle}" if cycle != 2024 else ""
    out_dir = config.outputs_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproduce the exact primary allocation run_backtest.py Table 5 uses ──
    beta_rc, coef, sigma_model = load_processed_artifacts(processed)
    races = build_universe(cycle=cycle)
    budget = sum(r.d_total for r in races)
    party_budget = sum(r.d_total - r.cand_d_total for r in races)
    cand_floors = np.array([r.cand_d_total for r in races])

    outputs = compute_outputs_batch(races, coef, sigma_model)
    factor_model = build_dummy_factor_model(races, config.generic_ballot_2024())
    cov_matrix = factor_model.race_covariance()

    opt_cfg = config.optimizer_cfg()
    cap_baseline = opt_cfg["cap_regimes"][-1]

    primary_result = optimize_nonlinear(
        races, coef, sigma_model, budget, cov_matrix, 0.0, cap_baseline,
        party_budget=party_budget, eta=args.eta,
    )

    # ── Identify the newly-funded set: DCCC zero, model positive ──────────
    party_observed = np.array([r.d_total - r.cand_d_total for r in races])
    zero_funded_mask = party_observed <= (1e-3 * party_budget)
    model_party = primary_result.allocations - cand_floors
    newly_funded_mask = zero_funded_mask & (model_party > 1e-3 * party_budget)

    n_zero_funded = int(zero_funded_mask.sum())
    n_newly_funded = int(newly_funded_mask.sum())
    print(f"DCCC zero-funded races: {n_zero_funded}")
    print(f"Of those, model-selected (newly funded) races: {n_newly_funded}")

    # ── Cohort benchmarks: what a "typical funded, viable" race looks like ──
    # Reference cohort = competitive races (Toss-Up/Lean D/Lean R), the
    # segment DCCC actually engages with, so "weak relative to a race DCCC
    # would normally fund" is measured against DCCC's own revealed standard,
    # not an arbitrary external threshold.
    competitive = competitive_subset(races)
    comp_floors = np.array([r.cand_d_total for r in competitive])
    comp_indiv_share = np.array([r.indiv_share for r in competitive])

    floor_p25 = _percentile(comp_floors, 25)
    indiv_share_p25 = _percentile(comp_indiv_share, 25)

    print(f"Competitive-race cand_d_total p25 (weak-fundraising threshold): ${floor_p25:,.0f}")
    print(f"Competitive-race indiv_share p25 (low-grassroots threshold): {indiv_share_p25:.3f}")

    # ── Build the per-race audit table ─────────────────────────────────────
    rows = []
    for i, (race, out) in enumerate(zip(races, outputs)):
        if not newly_funded_mask[i]:
            continue

        model_party_i = float(model_party[i])
        model_party_share = model_party_i / party_budget if party_budget > 0 else 0.0
        outspent_ratio = (race.r_total / race.cand_d_total) if race.cand_d_total > 0 else float("inf")

        # Candidate-quality / viability flags: these answer "does public data
        # already contain a reason a committee might rationally have skipped
        # this race" -- the omitted-information question. marginal_model_conviction
        # is deliberately excluded from this set: it says the model itself only
        # weakly wants this race, which is a claim about how much weight the
        # model's own pick deserves, not a claim about a DCCC-observable reason
        # to skip it. Reported separately as a materiality cross-tab instead.
        flags = {
            "weak_fundraising": bool(race.cand_d_total < floor_p25),
            "low_grassroots_support": bool(race.indiv_share < indiv_share_p25),
            "severely_outspent": bool(outspent_ratio > 3.0),
            "deep_structural_pvi": bool(abs(race.pvi) > 15.0),
        }
        n_flags = sum(flags.values())
        marginal_model_conviction = bool(model_party_share < 0.005)

        rows.append({
            "district_id": race.district_id,
            "cook_rating": race.cook_rating,
            "incumb_status": race.incumb_status,
            "pvi": round(race.pvi, 2),
            "cand_d_total": round(race.cand_d_total, 0),
            "r_total": round(race.r_total, 0),
            "outspent_ratio": round(outspent_ratio, 2) if np.isfinite(outspent_ratio) else None,
            "indiv_share": round(race.indiv_share, 3),
            "msg_i_per_1m": round(out.msg_i * 1e6, 6),
            "model_recommended_party_dollars": round(model_party_i, 0),
            "model_party_share_pct": round(model_party_share * 100, 3),
            **flags,
            "n_flags": n_flags,
            "explicable_by_observables": n_flags >= 1,
            "marginal_model_conviction": marginal_model_conviction,
        })

    audit_df = pd.DataFrame(rows).sort_values(
        "model_recommended_party_dollars", ascending=False
    ).reset_index(drop=True)

    csv_path = out_dir / f"omitted_information_audit{suffix}.csv"
    audit_df.to_csv(csv_path, index=False)
    print(f"\nPer-race audit table → {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    n_explicable = int(audit_df["explicable_by_observables"].sum())
    n_unexplained = n_newly_funded - n_explicable

    flag_prevalence = {
        flag: int(audit_df[flag].sum())
        for flag in ["weak_fundraising", "low_grassroots_support", "severely_outspent",
                     "deep_structural_pvi"]
    }

    unexplained = audit_df[~audit_df["explicable_by_observables"]].sort_values(
        "model_recommended_party_dollars", ascending=False
    )
    n_unexplained_and_material = int(
        (~audit_df["explicable_by_observables"] & ~audit_df["marginal_model_conviction"]).sum()
    )

    summary = {
        "cycle": cycle,
        "n_zero_funded_dccc": n_zero_funded,
        "n_newly_funded_by_model": n_newly_funded,
        "n_explicable_by_observables": n_explicable,
        "n_unexplained": n_unexplained,
        "pct_unexplained": round(100 * n_unexplained / n_newly_funded, 1) if n_newly_funded else None,
        "n_unexplained_and_material": n_unexplained_and_material,
        "flag_prevalence": flag_prevalence,
        "n_marginal_model_conviction": int(audit_df["marginal_model_conviction"].sum()),
        "cohort_thresholds": {
            "competitive_cand_d_total_p25": floor_p25,
            "competitive_indiv_share_p25": indiv_share_p25,
        },
        "unexplained_district_ids": unexplained["district_id"].tolist(),
        "unexplained_and_material_district_ids": audit_df[
            ~audit_df["explicable_by_observables"] & ~audit_df["marginal_model_conviction"]
        ]["district_id"].tolist(),
    }

    summary_path = out_dir / f"omitted_information_audit_summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    print(f"\n{n_newly_funded} newly-funded races: "
          f"{n_explicable} explicable by at least one candidate-quality/viability flag, "
          f"{n_unexplained} with no visible public-data explanation "
          f"({summary['pct_unexplained']}%).")
    print("\nCandidate-quality/viability flag prevalence (of {} races):".format(n_newly_funded))
    for flag, count in flag_prevalence.items():
        print(f"  {flag}: {count} ({100*count/n_newly_funded:.1f}%)")
    print(f"  marginal_model_conviction (<0.5% of party budget; reported separately, "
          f"not a viability flag): {summary['n_marginal_model_conviction']} "
          f"({100*summary['n_marginal_model_conviction']/n_newly_funded:.1f}%)")

    print(f"\nUnexplained races (no viability flag), largest model-recommended $ first:")
    print(unexplained[["district_id", "cook_rating", "pvi", "cand_d_total",
                        "r_total", "indiv_share", "model_recommended_party_dollars",
                        "marginal_model_conviction"]]
          .to_string(index=False))
    print(f"\nOf those, {n_unexplained_and_material} are ALSO not a marginal/rounding-sized "
          f"model recommendation (>=0.5% of party budget) -- the sharpest subset: real candidate "
          f"floor, no visible viability flag, and the model itself is not weakly hedging on it.")


if __name__ == "__main__":
    main()
