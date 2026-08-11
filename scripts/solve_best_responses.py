#!/usr/bin/env python3
"""
BR_D(R_observed) and BR_R(D_observed) for one cycle (project_spec.md
Section 9) -- the one-shot unilateral best responses that
compute_exploitability.py's RegretD/RegretR are built from.

Usage:
    python scripts/solve_best_responses.py --cycle 2024
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
from game import best_response as br  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("solve_best_responses")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one-shot unilateral BR_D/BR_R")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    d0 = [r.d_total for r in races]
    r0 = [r.r_total for r in races]

    logger.info("Solving BR_D(R_observed)…")
    res_d = br.br_d(races, coef, sigma_model, total_r=r0, budget_d=state["budget_d"],
                     cap_fraction_d=args.cap_fraction_d)
    logger.info(f"BR_D: E[D seats] = {res_d.e_seats_own:.3f}")

    logger.info("Solving BR_R(D_observed)…")
    res_r = br.br_r(races, coef, sigma_model, total_d=d0, cand_r_total=state["cand_r_total"],
                     budget_r=state["budget_r"], cap_fraction_r=args.cap_fraction_r)
    logger.info(f"BR_R: E[R seats] (self-scored via R's own search objective) = {res_r.e_seats_own:.3f}")

    out = {
        "cycle": args.cycle,
        "BR_D": {"party": res_d.party.tolist(), "e_seats_own": res_d.e_seats_own, "status": res_d.status},
        "BR_R": {"party": res_r.party.tolist(), "e_seats_own": res_r.e_seats_own, "status": res_r.status},
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_responses_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
