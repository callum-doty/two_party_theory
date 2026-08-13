#!/usr/bin/env python3
"""
Information option value (research-discussion follow-up, 2026-08-13):
completes the Theta = information option value + strategic flexibility
option value decomposition that game/strategic_window.py / compute_value_
of_waiting.py left half-built. Pure Monte Carlo over a closed-form payoff
evaluation -- NO best-response solves, so unlike every other script in
this project's dynamic-extension work, this one runs in seconds, not
minutes, and needs no background job.

Uses the SAME 12 candidates and the SAME earliest reference date (120 days
before Election Day, ~full opponent flexibility) that compute_value_of_
waiting.py already used for its own V_now/best_immediate comparison -- so
the two Theta components are computed on directly comparable footing and
can be added together honestly.

Usage:
    python scripts/compute_information_value.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import information_value as iv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_information_value")

N_DRAWS = 5000


def main() -> None:
    all_results = {}
    for cycle in (2024, 2022):
        window = json.load(open(REPO_ROOT / "results" / f"strategic_window_{cycle}.json"))
        earliest_date = window["strategic_window_D"][0]["ref_date"]
        earliest_days_before = window["days_before"][0]
        logger.info(f"=== {cycle}: reference date {earliest_date} ({earliest_days_before} days before election) ===")

        state = build_cycle_state(cycle, 0.15, 0.15)
        races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
        cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
        d0 = np.array([r.d_total for r in races])
        r0 = np.array([r.r_total for r in races])
        floors_d = np.array([r.cand_d_total for r in races])
        party_d_obs = np.maximum(d0 - floors_d, 0.0)
        party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
        idx_by_district = {r.district_id: i for i, r in enumerate(races)}

        cycle_out = {}
        for side, key in (("D", "strategic_window_D"), ("R", "strategic_window_R")):
            rows = window[key]
            districts = sorted({r["district_id"] for r in rows})
            candidate_indices = [idx_by_district[d] for d in districts]
            psv_true_at_t = {
                idx_by_district[d]: next(r["PSV"] for r in rows if r["district_id"] == d and r["ref_date"] == earliest_date)
                for d in districts
            }
            cap_fraction = 0.15
            budget = budget_d if side == "D" else budget_r

            result = iv.information_option_value(
                races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs,
                candidate_indices=candidate_indices, delta=1_000_000.0,
                cap_fraction=cap_fraction, budget=budget,
                cycle=cycle, days_before=earliest_days_before, side=side,
                psv_true_at_t=psv_true_at_t, n_draws=N_DRAWS,
            )
            idx_to_district = {i: d for d, i in idx_by_district.items()}
            result["pick_frequency"] = {idx_to_district[int(i)]: c for i, c in result["pick_frequency"].items()}
            result["best_true_district"] = idx_to_district[result["best_true_idx"]]
            result["global_best_psv_district"] = idx_to_district[result["global_best_psv_idx"]]

            logger.info(f"--- {side}-side ---")
            logger.info(f"  sigma_eps (generic-ballot std at this horizon) = {result['sigma_eps']:.3f} pts")
            logger.info(f"  V_uni-rule zero-noise pick: {result['best_true_district']} (PSV={result['best_true_immediate']:+.4f})")
            logger.info(f"  globally best-PSV race: {result['global_best_psv_district']} (PSV={result['global_best_psv_value']:+.4f})  "
                        f"disagree={result['v_uni_rule_disagrees_with_psv_best']}")
            logger.info(f"  E[realized value under noisy date-t info] = {result['e_realized_value_under_noise']:+.4f}")
            logger.info(f"  info_option_value = {result['info_option_value']:+.4f}")
            logger.info(f"  pick frequency across {N_DRAWS} draws: {result['pick_frequency']}")

            cycle_out[side] = result
        all_results[cycle] = cycle_out

    out_path = REPO_ROOT / "results" / "information_value.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
