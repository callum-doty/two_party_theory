#!/usr/bin/env python3
"""
Build one cycle's race universe + both sides' budgets -- the (i, t) unit of
analysis (project_spec.md Section 3), frozen at the final-cycle information
date for the initial static project.

Shared loader used by every other scripts/*.py in this project: reuses
backtest.data.universe.build_universe (race state X_i) and
backtest.data.fec.load_candidate_disbursements (R's own candidate-committee
floor, needed to split R's observed total into floor + party money the same
way D's side already is -- see solve_nash_equilibrium.py's original
docstring on why this isn't a RaceRecord field).

Usage:
    python scripts/build_cycle_state.py --cycle 2024
    python scripts/build_cycle_state.py --cycle 2024 --cap-fraction-d 0.15 --cap-fraction-r 0.15
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

from backtest.data import fec
from backtest.data.universe import build_universe

import solve_bellman_lsm as lsm  # noqa: E402 -- reuse its real-coefficient loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("build_cycle_state")


def load_cand_r_total(races, cycle: int) -> np.ndarray:
    """R-party candidate-committee disbursements per race, aligned to
    races' order -- R's own floor, mirroring how cand_d_total is built for D."""
    disb = fec.load_candidate_disbursements(cycle)
    r_disb = disb[disb["party"] == "R"].set_index("district_id")["candidate_disbursements"]
    return np.array([float(r_disb.get(r.district_id, 0.0)) for r in races])


def build_cycle_state(cycle: int, cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15) -> dict:
    """Returns races, coef, sigma_model, cand_r_total, and both sides'
    budgets/caps -- everything scripts/*.py need for this cycle."""
    coef, sigma_model = lsm.load_coef_and_sigma()
    races = build_universe(cycle=cycle)
    n = len(races)

    cand_r_total = load_cand_r_total(races, cycle)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])

    budget_d = float(np.sum(d0 - floors_d))
    budget_r = float(np.sum(r0 - cand_r_total))

    return dict(
        cycle=cycle, races=races, coef=coef, sigma_model=sigma_model,
        cand_r_total=cand_r_total, n_races=n,
        budget_d=budget_d, budget_r=budget_r,
        cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
        cap_d=cap_fraction_d * budget_d, cap_r=cap_fraction_r * budget_r,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and summarize one cycle's race universe/state")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    logger.info(f"Cycle {args.cycle}: {state['n_races']} races, "
                f"DCCC budget ${state['budget_d']:,.0f}, NRCC budget ${state['budget_r']:,.0f}")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cycle_state_{args.cycle}.json"
    summary = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in state.items() if k not in ("races", "coef", "sigma_model")
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
