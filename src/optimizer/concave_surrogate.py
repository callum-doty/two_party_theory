#!/usr/bin/env python3
"""
The validated fast concave surrogate (project_spec.md Section 12: "may
provide a fast alternative after symmetrical validation for both players").

race_payoff_at_party/build_concave_segments/greedy_allocate/surrogate_allocate
are carried over UNCHANGED from scripts/concave_surrogate.py -- the D-side
water-filling solve, validated there to within 0.11-0.19 expected seats of
optimize_nonlinear()'s true optimum (>99.9% of optimal value) at ~2,000-
2,700x the speed. That validation was one-sided (D against a fixed/reactive
R); this project's game treats R as its own optimizing player, so an R-side
mirror is added below (race_payoff_at_party_r / surrogate_allocate_r, reusing
build_concave_segments/greedy_allocate via an injected payoff function).

R-SIDE STATUS: NOT YET validated against optimize R's own nonlinear solve
(backtest.optimizer.nash.best_response("R", ...)) the way the D-side was --
spec Section 12's "symmetrical validation for both players" is still
outstanding. Treat surrogate_allocate_r as an untested fast-path candidate,
not yet a benchmarked substitute, until src/validation exercises it the same
way theta_concave_surrogate.py exercised the D-side.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from backtest.optimizer.allocator import _apply_ceiling, _precompute_race_arrays, _reactive_r
from backtest.optimizer.nash import _r_mu_and_grad, _r_precompute

N_GRID_DEFAULT = 40


def race_payoff_at_party(party: np.ndarray, arrays: dict) -> np.ndarray:
    """f_i(party_i) = Phi(mu_i'(party_i)/sigma_i) -- D-side, vectorized."""
    d = np.maximum(arrays["floors"] + party, 1.0)
    r = _reactive_r(party, arrays)
    t_ = d + r
    ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
    log_ratio = np.log(ratio)
    log_total_pv = np.log(t_ / arrays["cvap"])
    mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
    mu_capped, _ = _apply_ceiling(mu_raw, arrays)
    return norm.cdf(mu_capped / arrays["sigma"])


def race_payoff_at_party_r(party_r: np.ndarray, arrays: dict) -> np.ndarray:
    """f_i(party_r_i) = 1 - Phi(mu_capped_r/sigma_i) -- R-side, mirrored
    ceiling (backtest.optimizer.nash._r_mu_and_grad), vectorized."""
    mu_capped, _ = _r_mu_and_grad(party_r, arrays)
    return 1.0 - norm.cdf(mu_capped / arrays["sigma"])


def build_concave_segments(arrays: dict, cap: np.ndarray, n_grid: int = N_GRID_DEFAULT,
                            payoff_fn=race_payoff_at_party):
    """Per-race piecewise-linear concave envelope over party$ in [0, cap_i]."""
    n = len(cap)
    grid_fracs = np.linspace(0.0, 1.0, n_grid + 1)
    xs = grid_fracs[None, :] * cap[:, None]
    fs = np.zeros_like(xs)
    for k in range(n_grid + 1):
        fs[:, k] = payoff_fn(xs[:, k], arrays)

    race_idx_list, width_list, slope_list, xstart_list = [], [], [], []
    for i in range(n):
        x_i, f_i = xs[i], fs[i]
        pts_x = [x_i[0]]
        pts_f = [f_i[0]]
        for k in range(1, len(x_i)):
            while len(pts_x) >= 2:
                slope_prev = (pts_f[-1] - pts_f[-2]) / (pts_x[-1] - pts_x[-2])
                slope_new = (f_i[k] - pts_f[-1]) / (x_i[k] - pts_x[-1])
                if slope_new >= slope_prev - 1e-15:
                    pts_x.pop(); pts_f.pop()
                else:
                    break
            pts_x.append(x_i[k]); pts_f.append(f_i[k])
        for k in range(len(pts_x) - 1):
            w = pts_x[k + 1] - pts_x[k]
            if w <= 0:
                continue
            slope = (pts_f[k + 1] - pts_f[k]) / w
            race_idx_list.append(i); width_list.append(w)
            slope_list.append(slope); xstart_list.append(pts_x[k])

    return (np.array(race_idx_list), np.array(width_list),
            np.array(slope_list), np.array(xstart_list))


def greedy_allocate(race_idx, width, slope, xstart, n_races, budget) -> np.ndarray:
    """Exact optimum of the piecewise-linear-concave relaxation."""
    order = np.argsort(-slope)
    party = np.zeros(n_races)
    remaining = budget
    for k in order:
        if remaining <= 0:
            break
        take = min(width[k], remaining)
        party[race_idx[k]] += take
        remaining -= take
    return party


def surrogate_allocate(races, coef, sigma_model, budget, cap_fraction, eta, n_grid=N_GRID_DEFAULT):
    """D-side surrogate. eta plays the OLD project's reactive-response role
    (backtest.optimizer.allocator._reactive_r) -- pass eta=0.0 when calling
    this within the two-player game (R is a decision variable there, not a
    mechanical reaction; see game/gradients.py's module docstring)."""
    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=eta)
    n = len(races)
    cap = cap_fraction * budget * np.ones(n)
    race_idx, width, slope, xstart = build_concave_segments(arrays, cap, n_grid)
    party = greedy_allocate(race_idx, width, slope, xstart, n, budget)
    return party, arrays


def surrogate_allocate_r(races, coef, sigma_model, cand_r_total, d_current, budget, cap_fraction,
                          n_grid=N_GRID_DEFAULT):
    """R-side surrogate (NOT YET validated -- see module docstring)."""
    arrays = _r_precompute(races, coef, sigma_model, cand_r_total, d_current=d_current)
    n = len(races)
    cap = cap_fraction * budget * np.ones(n)
    race_idx, width, slope, xstart = build_concave_segments(
        arrays, cap, n_grid, payoff_fn=race_payoff_at_party_r
    )
    party = greedy_allocate(race_idx, width, slope, xstart, n, budget)
    return party, arrays
