#!/usr/bin/env python3
"""
Response-delay sweep (research-discussion follow-up, second major piece
alongside the locked-capital best response in src/game/best_response.py):
for the same top candidates already identified in compute_strategic_
leverage.py, hold delta fixed at $1M and vary how much of the opponent's
budget is still flexible when it responds -- tau days after a fixed
reference decision date t0, using REAL dated FEC data (estimation.
commitment_timing) rather than a synthetic commitment schedule.

t0 = September 1 of the cycle year, chosen by checking the actual
commitment-fraction curves first (not arbitrarily): by that date only
1-6% of either committee's eventual national-committee-own IE spending has
happened yet in every cycle/party combination checked, and by t0+28 days
commitment has risen to a meaningful 17-58% without fully saturating --
the range where a response-delay effect, if real, should be visible.

Usage:
    python scripts/compute_response_delay.py --cycle 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import payoff  # noqa: E402
from game import response_delay as rd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_response_delay")

DELTA = 1_000_000.0
TAUS = [0, 7, 14, 21, 28]


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-delay leverage sweep (exact SLSQP, locked capital)")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    prior_path = REPO_ROOT / "results" / f"strategic_leverage_{args.cycle}.json"
    if not prior_path.exists():
        raise SystemExit(f"{prior_path} not found -- run compute_strategic_leverage.py --cycle {args.cycle} first")
    prior = json.load(open(prior_path))
    top_d = sorted({r["district_id"] for r in prior["leverage_D_curve"]})
    top_r = sorted({r["district_id"] for r in prior["leverage_R_curve"]})
    logger.info(f"Cycle {args.cycle}: reusing top-3 D candidates {top_d} and R candidates {top_r}")

    t0 = date(args.cycle, 9, 1)
    logger.info(f"t0 = {t0}, tau = {TAUS} days")

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
    n = state["n_races"]
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    floors_d = np.array([r.cand_d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    idx_by_district = {r.district_id: i for i, r in enumerate(races)}
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    d_rows: list[dict] = []
    for did in top_d:
        i = idx_by_district[did]
        logger.info(f"D-side {did}: sweeping response delay…")
        rows = rd.leverage_by_response_delay_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, delta=DELTA, cycle=args.cycle, t0=t0, taus=TAUS,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, arrays=arrays,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        )
        for row in rows:
            logger.info(f"    tau={row['tau_days']:>2}d ({row['as_of_date']}, R commitment={row['commitment_fraction_r']:.1%}): "
                        f"PSV={row['PSV']:+.4f} leverage={row['leverage_seats_per_million']:+.4f} "
                        f"retention={row['retention_rate']:.1%} reshuffle=${row['reshuffle_l1']:,.0f}")
        d_rows.extend(rows)

    r_rows: list[dict] = []
    for did in top_r:
        i = idx_by_district[did]
        logger.info(f"R-side {did}: sweeping response delay…")
        rows = rd.leverage_by_response_delay_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, delta=DELTA, cycle=args.cycle, t0=t0, taus=TAUS,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, arrays=arrays, n_races=n,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        )
        for row in rows:
            logger.info(f"    tau={row['tau_days']:>2}d ({row['as_of_date']}, D commitment={row['commitment_fraction_d']:.1%}): "
                        f"PSV={row['PSV']:+.4f} leverage={row['leverage_seats_per_million']:+.4f} "
                        f"retention={row['retention_rate']:.1%} reshuffle=${row['reshuffle_l1']:,.0f}")
        r_rows.extend(rows)

    out = dict(cycle=args.cycle, delta=DELTA, t0=str(t0), taus=TAUS,
               response_delay_D=d_rows, response_delay_R=r_rows)
    out_path = REPO_ROOT / "results" / f"response_delay_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
