#!/usr/bin/env python3
"""
Core logic for the validated fast concave surrogate (Section 8.9 item (5)),
factored into a standalone module with NO dependency on solve_bellman_lsm
so that BOTH solve_bellman_lsm.py (the new use_surrogate_allocator branch)
and theta_concave_surrogate.py (the validation script) can import it
without a circular import.

See theta_concave_surrogate.py's module docstring for the full derivation
and validation results: the true per-race objective is separable (R_i
depends only on race i's own party spend, confirmed by reading
_reactive_r() before assuming it), so the piecewise-linear concave
envelope of each race's payoff curve can be solved EXACTLY via a greedy
water-filling sort rather than an iterative nonlinear solve. Validated at
4 states to be within 0.11-0.19 expected seats of optimize_nonlinear()'s
true optimum (out of ~235-240, i.e. >99.9% of optimal value), at roughly
2,000-2,700x the speed.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

from backtest.optimizer.allocator import _precompute_race_arrays, _reactive_r, _apply_ceiling

N_GRID_DEFAULT = 40


def race_payoff_at_party(party: np.ndarray, arrays: dict) -> np.ndarray:
    """f_i(party_i) = Phi(mu_i'(party_i)/sigma_i), vectorized over races."""
    d = np.maximum(arrays["floors"] + party, 1.0)
    r = _reactive_r(party, arrays)
    t_ = d + r
    ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
    log_ratio = np.log(ratio)
    log_total_pv = np.log(t_ / arrays["cvap"])
    mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
    mu_capped, _ = _apply_ceiling(mu_raw, arrays)
    return norm.cdf(mu_capped / arrays["sigma"])


def build_concave_segments(arrays: dict, cap: np.ndarray, n_grid: int = N_GRID_DEFAULT):
    """Per-race piecewise-linear concave envelope over party$ in [0, cap_i]."""
    n = len(cap)
    grid_fracs = np.linspace(0.0, 1.0, n_grid + 1)
    xs = grid_fracs[None, :] * cap[:, None]
    fs = np.zeros_like(xs)
    for k in range(n_grid + 1):
        fs[:, k] = race_payoff_at_party(xs[:, k], arrays)

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
    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=eta)
    n = len(races)
    cap = cap_fraction * budget * np.ones(n)
    race_idx, width, slope, xstart = build_concave_segments(arrays, cap, n_grid)
    party = greedy_allocate(race_idx, width, slope, xstart, n, budget)
    return party, arrays
