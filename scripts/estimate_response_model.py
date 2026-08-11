#!/usr/bin/env python3
"""
Response-model estimation entry point (project_spec.md Section 4) and the
D/R symmetry test (Section 19).

The margin/spending-response model itself is REUSED unchanged (spec
Section 4: "the existing research already represents race state using
expected margin, uncertainty, ..."); re-estimating it is scripts/
run_estimation.py's job (carried over unmodified from the old project). This
script does two things specific to the new project:

  1. Points at that existing estimation entry point rather than duplicating it.
  2. Runs the symmetry test spec Section 19 requires before assuming a
     mirrored D-side elasticity is a valid stand-in for an R-side one:
     "Do not simply mirror the Democratic response curve onto Republicans
     without testing it."

SYMMETRY TEST STATUS: NOT YET IMPLEMENTED. beta_D (coef.beta1/beta2/beta3,
the D-side spending elasticity backtest.estimation.beta_rc already fits) has
no independently-estimated beta_R counterpart anywhere in the old codebase
-- game/gradients.py's MSG^R and the R-side best response
(backtest/optimizer/nash.py) both use D's OWN calibrated ceiling/elasticity
mirrored onto R, a documented modeling assumption, not a tested one. Testing
beta_D == beta_R properly requires fitting the SAME two-stage regression
backtest.estimation.beta_rc.py uses, but with the outcome flipped to R's own
margin/spending relationship (R's own IV strategy, R's own controls) --
nontrivial new estimation work, out of scope for scaffolding. This function
is a placeholder that raises NotImplementedError with that pointer, rather
than silently returning a fabricated result.

Usage:
    python scripts/estimate_response_model.py --cycle 2024   # points to run_estimation.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("estimate_response_model")


def test_d_r_symmetry(*args, **kwargs):
    """beta_D == beta_R? (spec Section 19). See module docstring: this
    needs a new R-side two-stage elasticity fit, not yet built."""
    raise NotImplementedError(
        "R-side elasticity is not independently estimated anywhere in this project yet -- "
        "see this module's docstring for what fitting one would require before this "
        "test can run. Until then, treat game/gradients.py's mirrored-ceiling MSG^R as an "
        "assumption, not a validated symmetric response curve."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-model estimation (reused) + D/R symmetry test")
    parser.add_argument("--run-symmetry-test", action="store_true")
    args = parser.parse_args()

    logger.info("The spending-response model is reused unchanged from the old project. "
                "Run scripts/run_estimation.py to (re-)fit beta_D/sigma/etc.")
    if args.run_symmetry_test:
        test_d_r_symmetry()


if __name__ == "__main__":
    main()
