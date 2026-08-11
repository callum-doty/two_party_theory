#!/usr/bin/env python3
"""
Item (5) of Section 8.9's investigation plan: a validated fast surrogate
for optimize_nonlinear() that preserves diminishing returns, unlike the LP
allocator, but runs at LP speed rather than SLSQP speed (40s-3,600s/call).

Key structural fact this exploits, confirmed by reading _reactive_r()
before assuming it: R_i(party_i) depends ONLY on race i's own party
spending, not on any other race's allocation. This means the TRUE
objective sum_i Phi(mu_i'(party_i)/sigma_i) -- even with opponent reaction
and the persuasion ceiling both included -- is fully SEPARABLE across
races, subject only to the budget and per-race cap constraints. A
separable-concave resource-allocation problem of this form has a classic,
exactly-optimal solution once each race's payoff function is replaced by
its piecewise-linear concave envelope: sort every (race, segment) pair by
marginal slope, descending, and allocate greedily until the budget is
exhausted (a discrete water-filling algorithm). This is NOT a heuristic
approximation to the piecewise-linear relaxation -- it is the exact
optimum of that relaxation -- and it runs in O(n_races * n_grid_points *
log(...)) time (a sort), not an iterative nonlinear solve.

Validation (not assumed, checked): compare the surrogate's allocation,
objective value, funded-race count, and concentration against the TRUE
optimize_nonlinear() at the same states already used for the period
decomposition (scripts/theta_lp_vs_nonlinear_period_decomposition.py),
before this surrogate is trusted for anything.

Output: outputs/theta_concave_surrogate_validation.json
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
from backtest.optimizer.allocator import optimize_nonlinear, _precompute_race_arrays
from concave_surrogate import race_payoff_at_party, surrogate_allocate as _surrogate_allocate_core
import theta_lp_vs_nonlinear_period_decomposition as pd

N_GRID = 40   # per-race breakpoints from 0 to cap


def surrogate_allocate(races, coef, sigma_model, budget, cap_fraction, eta, n_grid=N_GRID):
    """Thin wrapper adding wall-clock timing around the shared core (which
    validate_at_period below needs to report the speedup); the underlying
    algorithm is defined once, in concave_surrogate.py, so both this
    validation script and solve_bellman_lsm.py's use_surrogate_allocator
    branch use the identical implementation."""
    t0 = time.time()
    party, _arrays = _surrogate_allocate_core(races, coef, sigma_model, budget, cap_fraction, eta, n_grid)
    elapsed = time.time() - t0
    return party, elapsed


def validate_at_period(tstep: int):
    state = pd.build_state()
    d_t, r_t, mu_t = pd.state_at_period(state, tstep)
    n = state["n"]
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    eta_arr = state["eta_arr"]

    import dataclasses
    races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                    d_total=float(d_t[i])) for i, r in enumerate(races)]

    t0 = time.time()
    res_nl = optimize_nonlinear(races_t, coef, sigma_model, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                                 gamma=0.0, cap_fraction=0.15, party_budget=lsm.F0, eta=eta_arr)
    nl_elapsed = time.time() - t0
    arrays_nl = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr)
    party_nl = np.maximum(res_nl.allocations - d_t, 0.0)
    nl_value = float(race_payoff_at_party(party_nl, arrays_nl).sum())

    arrays_sur = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr)
    party_sur, sur_elapsed = surrogate_allocate(races_t, coef, sigma_model, lsm.F0, 0.15, eta_arr)
    sur_value = float(race_payoff_at_party(party_sur, arrays_sur).sum())

    def stats(party):
        n_funded = int(np.sum(party > 1.0))
        top5 = np.sort(party)[::-1][:5].sum()
        total = party.sum()
        return n_funded, float(top5 / total) if total > 0 else 0.0

    nl_funded, nl_conc = stats(party_nl)
    sur_funded, sur_conc = stats(party_sur)

    result = {
        "period": tstep, "days_remaining": (lsm.N_PERIODS - tstep) * lsm.PERIOD_DAYS,
        "nonlinear_value": nl_value, "surrogate_value": sur_value,
        "value_diff_surrogate_minus_nonlinear": sur_value - nl_value,
        "nonlinear_elapsed_s": nl_elapsed, "surrogate_elapsed_s": sur_elapsed,
        "nonlinear_n_funded": nl_funded, "nonlinear_top5_share": nl_conc,
        "surrogate_n_funded": sur_funded, "surrogate_top5_share": sur_conc,
    }
    print(f"  t={tstep} ({result['days_remaining']}d left): "
          f"nonlinear={nl_value:.4f} ({nl_elapsed:.1f}s, n_funded={nl_funded}, top5={nl_conc:.2%})  "
          f"surrogate={sur_value:.4f} ({sur_elapsed:.3f}s, n_funded={sur_funded}, top5={sur_conc:.2%})  "
          f"diff={sur_value - nl_value:+.4f}")
    return result


def main():
    print("Validating concave-envelope greedy surrogate against optimize_nonlinear()...")
    results = [validate_at_period(t) for t in [0, 2, 4, 6]]
    out_path = Path(__file__).parent.parent / "outputs/theta_concave_surrogate_validation.json"
    with open(out_path, "w") as f:
        json.dump({"n_grid": N_GRID, "results": results}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
