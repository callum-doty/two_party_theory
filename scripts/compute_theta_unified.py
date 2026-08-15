#!/usr/bin/env python3
"""
Unified Theta: one Bellman value V_t(X_t) = max(V_deploy_t, V_wait_t),
replacing the previous practice of computing strategic-flexibility value
(value_of_waiting.py, decision rule = rank by PSV) and information value
(compute_information_value.py, decision rule = rank by V_uni) separately
and describing their sum as Theta. game/unified_theta.py has the full
derivation; this script is pure post-processing of results already on
disk (results/strategic_window_{cycle}.json) plus the same closed-form,
no-solve Monte Carlo compute_information_value.py already uses -- no new
best-response solves.

For each (cycle, side), backward-induces V_t across the same 8 reference
dates strategic_window.py already swept, under three regimes:
  - full:      both information noise and opponent-commitment maturation active
  - flex_only: information noise held at zero (perfect information throughout)
  - info_only: opponent commitments frozen at the earliest (most-flexible) date's level

Theta_full is reported alongside Theta_flex_only + Theta_info_only WITHOUT
asserting they are equal -- the gap between them is the interaction the two
counterfactuals leave out, reported explicitly.

Usage:
    python scripts/compute_theta_unified.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from estimation.gb_uncertainty import residual_gb_std  # noqa: E402
from game import unified_theta as ut  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_theta_unified")

N_DRAWS = 3000
CAP_FRACTION = 0.15


def _run_side(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs,
              budget, idx_by_district, cycle: int, side: str, rows: list[dict],
              election_day: date, delta: float, seed: int, exclude_redistricting: bool = False) -> dict:
    districts = sorted({r["district_id"] for r in rows})
    if exclude_redistricting:
        flagged = {r.district_id for r in races if r.redistricting_flagged}
        dropped = sorted(set(districts) & flagged)
        if dropped:
            logger.info(f"  --exclude-redistricting: dropping {dropped} from the candidate pool")
        districts = [d for d in districts if d not in flagged]
    candidate_indices = [idx_by_district[d] for d in districts]
    dates_str = []
    seen = set()
    for r in rows:
        if r["ref_date"] not in seen:
            seen.add(r["ref_date"])
            dates_str.append(r["ref_date"])
    days_before = [(election_day - date.fromisoformat(d)).days for d in dates_str]
    logger.info(f"{cycle} {side}-side: {len(districts)} candidates, dates {list(zip(dates_str, days_before))}")

    psv_by_date = {d: {} for d in dates_str}
    retention_by_date = {d: {} for d in dates_str}
    for r in rows:
        i = idx_by_district[r["district_id"]]
        psv_by_date[r["ref_date"]][i] = r["PSV"]
        retention_by_date[r["ref_date"]][i] = r["retention_rate"]

    cap = CAP_FRACTION * budget
    rng = np.random.default_rng(seed)

    def sampler(eps: float) -> dict[int, float]:
        return ut.noisy_v_uni_all(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs,
                                   candidate_indices, delta, cap, epsilon=eps, side=side)

    v_uni_true = sampler(0.0)  # zero-noise V_uni, reused by the flex-only regime at every date

    earliest_date = dates_str[0]
    retention_frozen = retention_by_date[earliest_date]
    psv_frozen = psv_by_date[earliest_date]  # opponent commitments frozen -> PSV itself doesn't move with t

    deploy_full, deploy_flex, deploy_info = {}, {}, {}
    for d, dbefore in zip(dates_str, days_before):
        sigma_eps = residual_gb_std(cycle, dbefore) if dbefore > 0 else 0.0

        deploy_full[d] = ut.expected_deploy_value(
            candidate_indices, psv_by_date[d], retention_by_date[d], rng, sigma_eps, sampler, N_DRAWS)

        _, flex_val = ut.deploy_value(candidate_indices, v_uni_true, retention_by_date[d], psv_by_date[d])
        deploy_flex[d] = flex_val

        deploy_info[d] = ut.expected_deploy_value(
            candidate_indices, psv_frozen, retention_frozen, rng, sigma_eps, sampler, N_DRAWS)

        logger.info(f"    t={dbefore:>3}d ({d}): sigma_eps={sigma_eps:.3f}  "
                    f"deploy_full={deploy_full[d]:+.4f}  deploy_flex_only={deploy_flex[d]:+.4f}  "
                    f"deploy_info_only={deploy_info[d]:+.4f}")

    full = ut.solve_bellman(dates_str, deploy_full)
    flex = ut.solve_bellman(dates_str, deploy_flex)
    info = ut.solve_bellman(dates_str, deploy_info)

    theta_full = full["theta_t0"]
    theta_flex_only = flex["theta_t0"]
    theta_info_only = info["theta_t0"]
    interaction = theta_full - (theta_flex_only + theta_info_only)

    logger.info(f"  Theta_full={theta_full:+.4f}  Theta_flex_only={theta_flex_only:+.4f}  "
                f"Theta_info_only={theta_info_only:+.4f}  "
                f"sum={theta_flex_only + theta_info_only:+.4f}  interaction={interaction:+.4f}")

    return dict(
        cycle=cycle, side=side, districts=districts, dates=dates_str, days_before=days_before,
        deploy_value_full=deploy_full, deploy_value_flex_only=deploy_flex, deploy_value_info_only=deploy_info,
        V_full=full["V"], V_flex_only=flex["V"], V_info_only=info["V"],
        theta_full=theta_full, theta_flex_only=theta_flex_only, theta_info_only=theta_info_only,
        theta_sum_of_parts=theta_flex_only + theta_info_only, interaction=interaction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Bellman Theta from a strategic-window candidate pool")
    parser.add_argument("--pool", choices=["curve", "primary", "union", "union_weekly_clean"], default="curve",
                         help="'curve' (default): K=3/side, reads strategic_window_{cycle}.json, writes "
                              "theta_unified.json. 'primary': K~8/side, reads strategic_window_expanded_{cycle}.json, "
                              "writes theta_unified_expanded.json. 'union': K~15-20/side, reads "
                              "strategic_window_union_{cycle}.json (run compute_strategic_window.py --pool union "
                              "first), writes theta_unified_union.json. 'union_weekly_clean': the K=15-20 stress "
                              "test's SECOND check -- redistricting-flagged districts already excluded AND the "
                              "18-point weekly date grid (run compute_strategic_window.py --pool union "
                              "--exclude-redistricting --weekly first), reads "
                              "strategic_window_union_{cycle}_excl_redistricting_weekly.json, writes "
                              "theta_unified_union_weekly_clean.json.")
    parser.add_argument("--exclude-redistricting", action="store_true",
                         help="Drop candidates flagged RaceRecord.redistricting_flagged (NC-06/13/14/etc. -- "
                              "documented elsewhere in this project as having a less certain baseline) from the "
                              "candidate pool before running the Bellman recursion. Appends '_excl_redistricting' "
                              "to the output filename so it never clobbers the unfiltered run. Not needed (and a "
                              "no-op on top of) '--pool union_weekly_clean', which already excludes them upstream.")
    args = parser.parse_args()
    window_names = {"curve": "strategic_window_{}.json", "primary": "strategic_window_expanded_{}.json",
                     "union": "strategic_window_union_{}.json",
                     "union_weekly_clean": "strategic_window_union_{}_excl_redistricting_weekly.json"}
    out_names = {"curve": "theta_unified.json", "primary": "theta_unified_expanded.json", "union": "theta_unified_union.json",
                 "union_weekly_clean": "theta_unified_union_weekly_clean.json"}
    window_name, out_name = window_names[args.pool], out_names[args.pool]
    if args.exclude_redistricting and args.pool != "union_weekly_clean":
        out_name = out_name.replace(".json", "_excl_redistricting.json")

    election_day_map = {2022: date(2022, 11, 8), 2024: date(2024, 11, 5)}
    delta = 1_000_000.0
    all_results = {}

    for cycle in (2024, 2022):
        window_path = REPO_ROOT / "results" / window_name.format(cycle)
        if not window_path.exists():
            raise SystemExit(f"{window_path} not found -- run "
                              f"compute_strategic_window.py --cycle {cycle} --pool {args.pool} first")
        window = json.load(open(window_path))

        state = build_cycle_state(cycle, CAP_FRACTION, CAP_FRACTION)
        races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
        cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
        d0 = np.array([r.d_total for r in races])
        r0 = np.array([r.r_total for r in races])
        floors_d = np.array([r.cand_d_total for r in races])
        party_d_obs = np.maximum(d0 - floors_d, 0.0)
        party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
        idx_by_district = {r.district_id: i for i, r in enumerate(races)}
        election_day = election_day_map[cycle]

        logger.info(f"=== {cycle} ===")
        cycle_out = {}
        cycle_out["D"] = _run_side(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs,
                                    budget_d, idx_by_district, cycle, "D", window["strategic_window_D"],
                                    election_day, delta, seed=cycle * 10 + 1,
                                    exclude_redistricting=args.exclude_redistricting)
        cycle_out["R"] = _run_side(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs,
                                    budget_r, idx_by_district, cycle, "R", window["strategic_window_R"],
                                    election_day, delta, seed=cycle * 10 + 2,
                                    exclude_redistricting=args.exclude_redistricting)
        all_results[cycle] = cycle_out

    out_path = REPO_ROOT / "results" / out_name
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
