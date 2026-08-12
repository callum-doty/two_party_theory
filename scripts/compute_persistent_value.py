#!/usr/bin/env python3
"""
Persistent strategic value for the top-|Z| candidate races of one cycle
(project_spec.md Sections 13-14) -- headline output #4: does each race's
apparent MSG-based opportunity survive the opponent's full optimal response?

Uses the isolated baseline (U_D(D, BR_R(D_observed)), computed once and
reused across every race) rather than the literal spec formula's raw
observed-R baseline -- see game/persistent_value.py's module docstring for
why: when observed spending is itself far from either side's unilateral
optimum, the raw-observed-baseline version comes back dominated by the
shared RegretD/RegretR term instead of each race's own signal. Pass
--baseline observed to get the literal spec formula instead.

Expensive: one BR_R (or BR_D) full-universe solve per candidate race, plus
one shared baseline solve. Defaults to 6 races per side.

Usage:
    python scripts/compute_persistent_value.py --cycle 2024
    python scripts/compute_persistent_value.py --cycle 2024 --n-races 10 --delta 250000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import best_response as br  # noqa: E402
from game import exploitability, payoff  # noqa: E402
from game import persistent_value as pv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_persistent_value")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute persistent strategic value for top-surplus races")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--n-races", type=int, default=6, help="Per side.")
    parser.add_argument("--delta", type=float, default=100_000.0)
    parser.add_argument("--baseline", choices=["isolated", "observed"], default="isolated")
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])

    surplus = exploitability.race_level_surplus(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        args.cap_fraction_d, args.cap_fraction_r,
    )
    # Restricted to races with real current party spend before ranking by
    # |Z| -- see src/validation/historical_backtest.py's inline comment for
    # why an unrestricted top-|Z| selection is dominated by the low-spend
    # MSG artifact at $0-party-spend races (the majority of races, once
    # DCCC/NRCC budgets are correctly scoped to control-only money) and
    # produces numerically unstable PSV/retention there.
    min_party_spend = 10_000.0
    funded_d = np.where(surplus["party_d_obs"] > min_party_spend)[0]
    funded_r = np.where(surplus["party_r_obs"] > min_party_spend)[0]
    top_d_idx = funded_d[np.argsort(-np.abs(surplus["Z_D"][funded_d]))[: args.n_races]]
    top_r_idx = funded_r[np.argsort(-np.abs(surplus["Z_R"][funded_r]))[: args.n_races]]

    baseline_d, baseline_r = None, None
    if args.baseline == "isolated":
        logger.info("Computing shared isolated baselines: U_D(D, BR_R(D)) and U_R(BR_D(D), R)…")
        arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
        party_d_obs = np.maximum(d0 - np.array([r.cand_d_total for r in races]), 0.0)
        party_r_obs = np.maximum(r0 - cand_r_total, 0.0)

        res_r_star = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                              budget_r=budget_r, cap_fraction_r=args.cap_fraction_r)
        baseline_d = float(payoff.p_win_shared(party_d_obs, res_r_star.party, arrays).sum())

        res_d_star = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                              budget_d=budget_d, cap_fraction_d=args.cap_fraction_d)
        e_d_at_d_star = float(payoff.p_win_shared(res_d_star.party, party_r_obs, arrays).sum())
        baseline_r = float(state["n_races"]) - e_d_at_d_star
        logger.info(f"baseline U_D = {baseline_d:.3f} (observed {surplus['p_win_obs'].sum():.3f}), "
                    f"baseline U_R = {baseline_r:.3f}")

    logger.info(f"Computing PSV^D for {len(top_d_idx)} races (delta=${args.delta:,.0f})…")
    psv_d = [
        pv.persistent_strategic_value_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=args.delta, cap_fraction_d=args.cap_fraction_d,
            cap_fraction_r=args.cap_fraction_r, baseline_e_seats=baseline_d,
        )
        for i in top_d_idx
    ]
    for row in psv_d:
        logger.info(f"  {row['district_id']}: V_uni={row['V_uni']:+.4f}  PSV={row['PSV']:+.4f}  "
                    f"retention={row['retention_rate']:.1%}")

    logger.info(f"Computing PSV^R for {len(top_r_idx)} races (delta=${args.delta:,.0f})…")
    psv_r = [
        pv.persistent_strategic_value_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=args.delta, cap_fraction_d=args.cap_fraction_d,
            cap_fraction_r=args.cap_fraction_r, baseline_e_seats_r=baseline_r,
        )
        for i in top_r_idx
    ]
    for row in psv_r:
        logger.info(f"  {row['district_id']}: V_uni={row['V_uni']:+.4f}  PSV={row['PSV']:+.4f}  "
                    f"retention={row['retention_rate']:.1%}")

    out = {
        "cycle": args.cycle, "delta": args.delta, "baseline": args.baseline,
        "persistent_value_D": psv_d, "persistent_value_R": psv_r,
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"persistent_value_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
