#!/usr/bin/env python3
"""
Run the full per-cycle pipeline (src/validation/historical_backtest.run_cycle)
across historical cycles and report whether the near-zero aggregate Nash
result replicates (project_spec.md Sections 15, 26-27): "A result replicated
in both 2022 and 2024 is substantially stronger than a live-cycle-only
finding."

Expensive -- each cycle runs the full best-response + damped Nash + PSV
pipeline (tens of minutes per cycle on the 433-race 2024 universe; see
historical_backtest.py's own module docstring). Not run as part of
scaffolding.

Usage:
    python scripts/run_historical_backtest.py --cycles 2022,2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from validation.historical_backtest import run_cycle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("run_historical_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the historical two-player backtest across cycles")
    parser.add_argument("--cycles", type=str, default="2022,2024")
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--damping-theta", type=float, default=0.5)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--n-psv-races", type=int, default=6)
    args = parser.parse_args()

    cycles = [int(c) for c in args.cycles.split(",")]
    results_dir = REPO_ROOT / "results"
    summary = []
    for cycle in cycles:
        logger.info(f"=== Cycle {cycle} ===")
        state = build_cycle_state(cycle, args.cap_fraction_d, args.cap_fraction_r)
        out = run_cycle(
            cycle, state["races"], state["coef"], state["sigma_model"], state["cand_r_total"],
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            damping_theta=args.damping_theta, max_rounds=args.max_rounds,
            n_psv_races=args.n_psv_races, results_dir=results_dir,
        )
        summary.append({
            "cycle": cycle,
            "exploitability": out["exploitability"]["exploitability"],
            "nash_converged": out["nash"]["converged"],
            "l1_distance_observed_to_nash": out["l1_distance_observed_to_nash"],
        })

    logger.info("=== Cross-cycle summary ===")
    for row in summary:
        logger.info(f"  {row['cycle']}: E={row['exploitability']:+.3f} seats, "
                    f"Nash converged={row['nash_converged']}, "
                    f"L1 distance to Nash=${row['l1_distance_observed_to_nash']:,.0f}")


if __name__ == "__main__":
    main()
