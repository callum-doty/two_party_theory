#!/usr/bin/env python3
"""
Solve the two-sided Nash equilibrium between DCCC and NRCC spending
allocation (src/backtest/optimizer/nash.py), replacing the reduced-form
eta scalar (FINDINGS.md Section 10.5's "left as future work") with NRCC
modeled as its own constrained optimizer.

NRCC's own candidate-committee floor (cand_r_total) is not a RaceRecord
field -- only D's own floor (cand_d_total) and R's TOTAL spend (r_total)
are -- so it's loaded here directly from the same FEC candidate-
disbursements source RaceRecord's own cand_d_total is built from
(src/backtest/data/fec.py's load_candidate_disbursements()), filtered to
party="R" instead of "D".

Usage:
    python scripts/solve_nash_equilibrium.py --cycle 2024
    python scripts/solve_nash_equilibrium.py --cycle 2022 --processed-dir data/processed_oos_2020
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data import fec
from backtest.data.universe import build_universe
from backtest.optimizer.nash import find_nash_equilibrium_multi_start

from run_backtest import load_processed_artifacts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("solve_nash_equilibrium")


def load_cand_r_total(races, cycle: int) -> np.ndarray:
    """R-party candidate-committee disbursements per race, aligned to
    races' order. Districts with no R candidate row (uncontested/missing
    data) get 0.0, matching how cand_d_total is built for the D side in
    universe.py (a missing candidate is a real absence of that side's own
    committee spending, not an error)."""
    disb = fec.load_candidate_disbursements(cycle)
    r_disb = disb[disb["party"] == "R"].set_index("district_id")["candidate_disbursements"]
    return np.array([float(r_disb.get(r.district_id, 0.0)) for r in races])


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the DCCC/NRCC Nash equilibrium")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--damping-theta", type=float, default=1.0,
                         help="1.0 = undamped Gauss-Seidel; <1.0 averages each round's "
                              "best response with the prior round, to stabilize cycling.")
    parser.add_argument("--max-rounds", type=int, default=100)
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    out_dir = config.outputs_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading estimation artifacts…")
    _beta_rc, coef, sigma_model = load_processed_artifacts(processed)

    logger.info(f"Building {args.cycle} race universe…")
    races = build_universe(cycle=args.cycle)
    n = len(races)
    logger.info(f"Universe: {n} races")

    cand_r_total = load_cand_r_total(races, args.cycle)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])

    party_budget_d = float(np.sum(d0 - floors_d))
    party_budget_r = float(np.sum(r0 - cand_r_total))
    logger.info(f"DCCC party budget: ${party_budget_d:,.0f}")
    logger.info(f"NRCC party budget: ${party_budget_r:,.0f} "
                f"(R total observed ${r0.sum():,.0f} minus R candidate-committee floor "
                f"${cand_r_total.sum():,.0f})")
    if party_budget_r <= 0:
        logger.warning("NRCC party budget is non-positive -- cand_r_total likely exceeds "
                        "observed r_total for this cycle's data; R's best response will be "
                        "degenerate (zero budget). Check load_candidate_disbursements(cycle) "
                        "coverage before trusting this run's equilibrium.")

    logger.info(f"Solving Nash equilibrium (damping_theta={args.damping_theta}, "
                f"max_rounds={args.max_rounds})…")
    result = find_nash_equilibrium_multi_start(
        races, coef, sigma_model, cand_r_total, party_budget_d, party_budget_r,
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        damping_theta=args.damping_theta, max_rounds=args.max_rounds,
    )

    logger.info(f"Converged: {result.converged} in {result.n_iterations} rounds")
    logger.info(f"E[D seats] = {result.e_seats_d:.2f}, E[R seats] = {result.e_seats_r:.2f}, "
                f"sum = {result.e_seats_d + result.e_seats_r:.2f} (n_races = {n})")
    logger.info(f"Multi-start agreement: {result.multi_start_agreement}")
    if not result.multi_start_agreement["agree_within_tolerance"]:
        logger.warning("Best-response dynamics from different starting points did NOT converge "
                        "to the same fixed point -- this game may have multiple equilibria or "
                        "the dynamics may be cycling. Reported here honestly, not resolved by "
                        "picking one candidate silently. Consider a lower --damping-theta.")

    out = {
        "cycle": args.cycle,
        "n_races": n,
        "party_budget_d": party_budget_d,
        "party_budget_r": party_budget_r,
        "cap_fraction_d": args.cap_fraction_d,
        "cap_fraction_r": args.cap_fraction_r,
        "damping_theta": args.damping_theta,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "e_seats_d": result.e_seats_d,
        "e_seats_r": result.e_seats_r,
        "multi_start_agreement": result.multi_start_agreement,
        "history": result.history,
        "per_race": [
            {
                "district_id": races[i].district_id,
                "cook_rating": races[i].cook_rating,
                "party_d": float(result.party_d[i]),
                "party_r": float(result.party_r[i]),
                "d_total_equilibrium": float(floors_d[i] + result.party_d[i]),
                "r_total_equilibrium": float(cand_r_total[i] + result.party_r[i]),
                "d_total_observed": float(d0[i]),
                "r_total_observed": float(r0[i]),
            }
            for i in range(n)
        ],
    }
    out_path = out_dir / f"nash_equilibrium_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
