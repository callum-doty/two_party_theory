#!/usr/bin/env python3
"""
Build one cycle's race universe + both sides' budgets -- the (i, t) unit of
analysis (project_spec.md Section 3), frozen at the final-cycle information
date for the initial static project.

Shared loader used by every other scripts/*.py in this project: reuses
backtest.data.universe.build_universe (race state X_i), then applies
estimation.control_provenance.apply_control_floor to redefine each race's
floor as ALL non-national-committee-controlled money (candidate + state
party + outside IE), not just candidate money -- so `party_d = d_total -
cand_d_total` and `party_r = r_total - cand_r_total` recover x_D / x_R
(DCCC's / NRCC's own controllable money) rather than "all money the
candidate itself didn't raise," which would let BR_D/BR_R "reallocate"
super-PAC dollars neither committee actually controls. See
control_provenance.py's module docstring for the full accounting identity
and the empirical size of the correction (NRCC's own money is $48.4M in
2024, not the $132M "everything non-candidate" figure this function
returned before 2026-08-11).

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

from backtest.data.universe import build_universe
from estimation.control_provenance import apply_control_floor

import solve_bellman_lsm as lsm  # noqa: E402 -- reuse its real-coefficient loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("build_cycle_state")


def build_cycle_state(cycle: int, cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15) -> dict:
    """Returns races, coef, sigma_model, cand_r_total, and both sides'
    budgets/caps -- everything scripts/*.py need for this cycle. races'
    cand_d_total and the returned cand_r_total are CONTROL floors (spec
    Section 5's D_i_bar/R_i_bar upper-bound accounting), not raw candidate
    disbursements -- see control_provenance.py."""
    coef, sigma_model = lsm.load_coef_and_sigma()
    races = build_universe(cycle=cycle)
    races, cand_r_total = apply_control_floor(races, cycle)
    n = len(races)

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
