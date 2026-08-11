#!/usr/bin/env python3
"""
Live 2026 two-player analysis (project_spec.md Section 16) -- explicitly
SECONDARY until the historical game (run_historical_backtest.py, cycles
2022/2024) is validated. Reports exploitability decomposed by side, never
a one-sided "Democrats can gain X seats" framing:

    "Under the estimated two-player model, the current spending profile has
    X expected-seat exploitability, with Y attributable to Democratic
    deviation opportunities and Z to Republican deviation opportunities."

Usage:
    python scripts/run_live_2026.py
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
from game import equilibrium, exploitability  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("run_live_2026")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 2026 two-player exploitability snapshot")
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--damping-theta", type=float, default=0.5)
    parser.add_argument("--max-rounds", type=int, default=40)
    args = parser.parse_args()

    state = build_cycle_state(2026, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]

    exploit = exploitability.exploitability(
        races, coef, sigma_model, state["cand_r_total"], state["budget_d"], state["budget_r"],
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
    )
    logger.info(
        "Under the estimated two-player model, the current spending profile has "
        f"{exploit['exploitability']:+.3f} expected-seat exploitability, with "
        f"{exploit['regret_D']:+.3f} attributable to Democratic deviation opportunities "
        f"and {exploit['regret_R']:+.3f} to Republican deviation opportunities."
    )

    nash = equilibrium.solve_nash(
        races, coef, sigma_model, state["cand_r_total"], state["budget_d"], state["budget_r"],
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        damping_theta=args.damping_theta, max_rounds=args.max_rounds,
    )
    logger.info(f"Current strategic equilibrium: E[D seats]={nash.e_seats_d:.2f}, "
                f"E[R seats]={nash.e_seats_r:.2f} (converged={nash.converged})")

    out = {
        "as_of": "live 2026 snapshot -- SECONDARY to historical validation (spec Section 16)",
        "n_races": state["n_races"],
        "exploitability": exploit,
        "nash_equilibrium": {
            "e_seats_d": nash.e_seats_d, "e_seats_r": nash.e_seats_r,
            "converged": nash.converged, "n_iterations": nash.n_iterations,
        },
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_2026.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
