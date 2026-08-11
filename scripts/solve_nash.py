#!/usr/bin/env python3
"""
The strategic equilibrium (D*, R*) for one cycle (project_spec.md
Sections 11-12), via multi-start damped Gauss-Seidel best-response dynamics.

Renamed/adapted successor to the old project's scripts/solve_nash_equilibrium.py,
now built on this project's own src/game/equilibrium.py rather than calling
backtest.optimizer.nash directly -- same validated solver underneath.

Usage:
    python scripts/solve_nash.py --cycle 2024
    python scripts/solve_nash.py --cycle 2024 --damping-theta 0.5 --max-rounds 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import equilibrium  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("solve_nash")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the two-sided Nash equilibrium")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--damping-theta", type=float, default=1.0,
                         help="1.0 = undamped Gauss-Seidel; <1.0 stabilizes cycling dynamics.")
    parser.add_argument("--max-rounds", type=int, default=100)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]

    logger.info(f"Solving Nash equilibrium (damping_theta={args.damping_theta}, "
                f"max_rounds={args.max_rounds})…")
    result = equilibrium.solve_nash(
        races, coef, sigma_model, state["cand_r_total"], state["budget_d"], state["budget_r"],
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        damping_theta=args.damping_theta, max_rounds=args.max_rounds,
    )

    logger.info(f"Converged: {result.converged} in {result.n_iterations} rounds")
    logger.info(f"E[D seats]={result.e_seats_d:.2f}, E[R seats]={result.e_seats_r:.2f}, "
                f"sum={result.e_seats_d + result.e_seats_r:.2f} (n_races={state['n_races']})")
    if not result.multi_start_agreement["agree_within_tolerance"]:
        logger.warning("Best-response dynamics from different starting points did NOT agree -- "
                        "possible multiple equilibria or cycling. See multi_start_agreement in output.")

    out = {
        "cycle": args.cycle, "n_races": state["n_races"],
        "budget_d": state["budget_d"], "budget_r": state["budget_r"],
        "cap_fraction_d": args.cap_fraction_d, "cap_fraction_r": args.cap_fraction_r,
        "damping_theta": args.damping_theta,
        "converged": result.converged, "n_iterations": result.n_iterations,
        "e_seats_d": result.e_seats_d, "e_seats_r": result.e_seats_r,
        "multi_start_agreement": result.multi_start_agreement,
        "history": result.history,
        "party_d": result.party_d.tolist(), "party_r": result.party_r.tolist(),
        "district_id": [r.district_id for r in races],
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nash_equilibrium_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
