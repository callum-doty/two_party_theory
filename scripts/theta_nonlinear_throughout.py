#!/usr/bin/env python3
"""
"Option B" of the LP-vs-nonlinear reduced-scope comparison requested in
external review of Paper III: the full backward induction, at every
period (not just t=0 as in "Option A", scripts/theta_lp_vs_nonlinear_deploy_branch.py),
using the true nonlinear allocator instead of the fast LP allocator for
the deploy branch.

Option A (t=0 only, holding the wait branch's future LP-based deploy
decisions fixed) found the nonlinear allocator gives a consistently,
substantially higher deploy value than the LP allocator (+4.5 to +7.9
expected seats across 27 comparisons) -- large enough that naively
subtracting it from the reported Theta(0) would flip the sign in all
three scenarios. But that test only upgrades ONE side of the comparison;
this script gives the wait branch the same upgrade at every period it
might deploy, for a fair answer.

K_PATHS is drastically reduced from the headline K=2,000 (nonlinear calls
run 43s-3,600s each, vs. ~11ms for the LP -- a full K=2,000 run is not
tractable). This trades Monte Carlo precision for tractability; the
question being asked (does the sign of Theta(0) survive a fair allocator
comparison) needs a large effect to matter, not fine precision.

Run on eta_bootstrap_all_cycles only (the scenario Option A checked most
thoroughly, 25 draws) -- running all three scenarios would triple the cost.

Pairing fix (2026-07-30, after external review flagged the first version of
this script was not actually using common random numbers): run_lsm() draws
its idiosyncratic-epsilon, R-reaction-noise, and G_t random walk from a
single MODULE-LEVEL `lsm.RNG` object that is mutated (consumed) in place.
The first version of this script called run_lsm() twice in sequence without
resetting RNG between calls -- the second call (LP-throughout) continued
consuming the SAME RNG stream where the first call (nonlinear-throughout)
left off, so the two runs saw DIFFERENT simulated state paths despite using
the same eta/resid draws. That confounds the allocator's own effect with
ordinary simulation noise across two different sets of paths, exactly the
opposite of what a paired comparison needs. All three of RNG's consumption
points (r_paths reaction noise, g_paths, eps_cum) occur before
_deploy_value is ever called and do not depend on use_nonlinear_allocator,
so resetting `lsm.RNG = np.random.default_rng(SAME_SEED)` immediately
before each run_lsm() call makes both calls draw an identical sequence of
random numbers in an identical order -- identical d_paths, r_paths,
mu_paths, eps_cum, and g_paths for both allocators, isolating the
allocator's own effect the way Delta_allocator = Theta_nonlinear(0) -
Theta_LP(0) is supposed to.

Output: outputs/theta_nonlinear_throughout.json
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
K_PATHS_REDUCED = 15


STATE_SEED = 20260730   # shared seed for d/r/eps/G_t paths -- reset before EACH run_lsm() call
ETA_BOOTSTRAP_SEED = 20260729   # separate seed for the eta/resid bootstrap draw (same for both allocators too)


def main():
    lsm.K_PATHS = K_PATHS_REDUCED
    print(f"K_PATHS={lsm.K_PATHS} (reduced from headline 2000), "
          f"N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    # eta/resid draws: drawn ONCE, shared by both calls below (already correct
    # in the first version of this script -- this part was never the bug).
    eta_rng = np.random.default_rng(ETA_BOOTSTRAP_SEED)
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)

    print("\n=== Nonlinear-throughout (Option B, paired) ===")
    lsm.RNG = np.random.default_rng(STATE_SEED)   # reset BEFORE this call
    t0 = time.time()
    res_nonlinear = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                                 "eta_bootstrap_all_cycles_nonlinear",
                                 eta_summary=boot_summary, use_nonlinear_allocator=True)
    elapsed_nonlinear = time.time() - t0
    print(f"  -> nonlinear-throughout wall time: {elapsed_nonlinear/3600:.2f} hours")

    print("\n=== LP-throughout, same K, same eta draws, SAME state-path RNG seed (paired) ===")
    lsm.RNG = np.random.default_rng(STATE_SEED)   # reset to the IDENTICAL seed -- this is the fix
    t0 = time.time()
    res_lp = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                          "eta_bootstrap_all_cycles_lp_same_k",
                          eta_summary=boot_summary, use_nonlinear_allocator=False)
    elapsed_lp = time.time() - t0
    print(f"  -> LP-throughout (same reduced K) wall time: {elapsed_lp/3600:.4f} hours")

    theta0_nonlinear = res_nonlinear["theta_by_period"][0]["mean_theta"]
    theta0_lp_same_k = res_lp["theta_by_period"][0]["mean_theta"]
    print(f"\nTheta(0) nonlinear-throughout, paired (K={K_PATHS_REDUCED}): {theta0_nonlinear:+.4f}")
    print(f"Theta(0) LP-throughout, paired (K={K_PATHS_REDUCED}, same eta draws + same state RNG): {theta0_lp_same_k:+.4f}")
    print(f"Delta_allocator = Theta_nonlinear(0) - Theta_LP(0) = {theta0_nonlinear - theta0_lp_same_k:+.4f}")
    print(f"Theta(0) LP-throughout (K=2000, headline, for reference): +4.692")

    out_path = ROOT / "outputs/theta_nonlinear_throughout.json"
    with open(out_path, "w") as f:
        json.dump({
            "k_paths": K_PATHS_REDUCED,
            "paired": True,
            "state_seed": STATE_SEED,
            "eta_bootstrap_seed": ETA_BOOTSTRAP_SEED,
            "nonlinear_throughout": res_nonlinear,
            "lp_throughout_same_k": res_lp,
            "delta_allocator_theta0": theta0_nonlinear - theta0_lp_same_k,
            "nonlinear_wall_seconds": elapsed_nonlinear,
            "lp_wall_seconds": elapsed_lp,
        }, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
