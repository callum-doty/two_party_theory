#!/usr/bin/env python3
"""
Option A of the LP-vs-nonlinear reduced-scope comparison requested in
external review of Paper III (both the original review and its follow-up):
does the deploy branch's value at t=0 change materially if the fast LP
allocator (optimize(), used throughout solve_bellman_lsm.py's Monte Carlo
for tractability) is replaced by the true nonlinear allocator
(optimize_nonlinear()) for the one-time "deploy now" decision only, holding
the wait branch (still LP-based throughout its recursion) fixed?

Why this doesn't need a Monte Carlo experiment at all: at t=0, every
simulated path is in an IDENTICAL state (d_paths[:,0,:]=floor_arr,
r_paths[:,0,:]=r0_arr, eps_cum[:,0,:]=0 for every path -- no randomness has
been realized yet). The t=0 deploy-branch comparison is therefore a single,
exact computation, not a K-path Monte Carlo estimate -- scoping this out
(see conversation) originally assumed it would need K nonlinear calls at
~40s each; it needs only one call per eta scenario instead.

eta granularity: solve_bellman_lsm.py's LP branch uses a per-race (per-tier)
eta array. optimize_nonlinear()/_precompute_race_arrays() previously only
supported a scalar eta (a single "if eta == 0.0" comparison would raise on
an array); allocator.py was patched (np.all(eta == 0.0)) to accept either,
so this comparison uses the SAME eta granularity on both sides -- a scalar
eta would reintroduce exactly the mis-specification Paper III Section 4.1
argues against.

eta_bootstrap_all_cycles draws a per-(path, tier) eta combination
independently, so unlike the two single-cycle brackets there is no single
"the" eta array for that scenario -- N_BOOTSTRAP_SAMPLES representative
draws are used instead of the full K_PATHS=2000, since each requires its
own ~40s nonlinear call.

Output: outputs/lp_vs_nonlinear_deploy_branch.json, printed summary.
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
from backtest.optimizer.allocator import (
    optimize, optimize_nonlinear, _precompute_race_arrays, _reactive_r, _apply_ceiling,
)
from simulate_and_validate import remaining_variance

ROOT = Path(__file__).parent.parent
N_BOOTSTRAP_SAMPLES = 25


def build_t0_state():
    """Replicates solve_bellman_lsm.run_lsm()'s t=0 construction exactly
    (mu_struct evaluated at the candidate-only floor, before any DCCC
    deployment) -- NOT compute_outputs_batch()'s default mu_hat, which uses
    race.d_total (observed totals, likely already including some historical
    party spend) rather than the pre-decision floor."""
    coef, sigma_model = lsm.load_coef_and_sigma()
    races = lsm.build_universe(cycle=2026)
    n = len(races)

    pvi_arr = np.array([r.pvi for r in races])
    incumb_arr = [r.incumb_status for r in races]
    is_incumb_arr = np.array([1.0 if s == "Incumbent" else 0.0 for s in incumb_arr])
    floor_arr = np.array([r.cand_d_total for r in races])
    r0_arr = np.array([r.r_total for r in races])
    tiers = [r.cook_rating for r in races]
    gb_national = races[0].generic_ballot

    if coef.beta1_open is not None:
        is_open_arr = np.array([1.0 if s == "Open" else 0.0 for s in incumb_arr])
        beta1_eff_arr = np.where(is_open_arr > 0, coef.beta1_open, coef.beta1)
    else:
        beta1_eff_arr = np.full(n, coef.beta1)

    from backtest.model.win_prob import compute_outputs_batch
    outputs0 = compute_outputs_batch(races, coef, sigma_model)
    sigma_arr = np.array([o.sigma_i for o in outputs0])

    t0 = np.maximum(floor_arr + r0_arr, 1.0)
    ratio0 = np.clip(floor_arr / t0, 1e-6, 1 - 1e-6)
    c_arr = beta1_eff_arr + coef.beta2 * np.abs(pvi_arr) + coef.beta3 * is_incumb_arr
    mu0 = (coef.alpha0 + coef.alpha1 * pvi_arr + coef.alpha2 * is_incumb_arr
           + coef.alpha3 * gb_national + c_arr * np.log(ratio0))

    trickle_per_day = lsm.load_trickle_rate_per_day(tiers)
    d_terminal = floor_arr + trickle_per_day * lsm.PERIOD_DAYS * lsm.N_PERIODS

    v_remaining_0 = remaining_variance(sigma_arr, lsm.N_PERIODS * lsm.PERIOD_DAYS)
    widened_sigma = np.sqrt(v_remaining_0)   # corrected (2026-07-28/29) formula, no + sigma_i^2

    return dict(coef=coef, sigma_model=sigma_model, races=races, n=n,
                pvi_arr=pvi_arr, incumb_arr=incumb_arr, is_incumb_arr=is_incumb_arr,
                floor_arr=floor_arr, r0_arr=r0_arr, tiers=tiers, gb_national=gb_national,
                beta1_eff_arr=beta1_eff_arr, sigma_arr=sigma_arr, mu0=mu0,
                d_terminal=d_terminal, widened_sigma=widened_sigma)


def _mu_struct_at(state, d, r):
    t_ = np.maximum(d + r, 1.0)
    ratio = np.clip(d / t_, 1e-6, 1 - 1e-6)
    c_arr = state["beta1_eff_arr"] + state["coef"].beta2 * np.abs(state["pvi_arr"]) + \
        state["coef"].beta3 * state["is_incumb_arr"]
    return (state["coef"].alpha0 + state["coef"].alpha1 * state["pvi_arr"]
            + state["coef"].alpha2 * state["is_incumb_arr"]
            + state["coef"].alpha3 * state["gb_national"] + c_arr * np.log(ratio))


def deploy_value_lp(state, eta_arr: np.ndarray) -> tuple[float, float]:
    """Exact replica of solve_bellman_lsm._deploy_value's LP branch at t=0."""
    from backtest.types import ModelOutputs
    coef, races, n = state["coef"], state["races"], state["n"]
    mu_t, sigma_arr = state["mu0"], state["sigma_arr"]
    d_t, r_t = state["floor_arr"], state["r0_arr"]

    p_win0 = norm.cdf(mu_t / sigma_arr)
    phi0 = norm.pdf(mu_t / sigma_arr)
    grad = np.array([
        lsm.margin_gradient(coef, state["pvi_arr"][i], state["incumb_arr"][i], d_t[i], r_t[i], eta_arr[i])
        for i in range(n)
    ])
    msg = phi0 / sigma_arr * grad
    outs = [ModelOutputs(district_id=races[i].district_id, ratio=d_t[i] / (d_t[i] + r_t[i]),
                          mu_hat=mu_t[i], sigma_i=sigma_arr[i], p_win=p_win0[i], msg_i=msg[i])
            for i in range(n)]
    t0 = time.time()
    res = optimize(outs, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                    gamma=0.0, cap_fraction=0.15, floor_allocations=d_t, party_budget=lsm.F0,
                    d_total_obs=d_t)
    elapsed = time.time() - t0
    delta_s = np.maximum(res.allocations - d_t, 0.0)
    delta_mu = grad * delta_s

    r_terminal_expected = np.maximum(r_t + eta_arr * (state["d_terminal"] - d_t), 1.0)
    trickle_drift = _mu_struct_at(state, state["d_terminal"], r_terminal_expected) - _mu_struct_at(state, d_t, r_t)

    deployed_mu = mu_t + delta_mu + trickle_drift
    value = norm.cdf(deployed_mu / state["widened_sigma"]).sum()
    return float(value), elapsed


def deploy_value_nonlinear(state, eta_arr: np.ndarray) -> tuple[float, float]:
    """Same construction, but the allocation comes from optimize_nonlinear()
    -- the TRUE, diminishing-returns-respecting objective solve -- using the
    same per-race eta array the LP branch uses (not a scalar).

    Bug found via smoke test before trusting any number (2026-07-29): a
    first version of this function reused margin_gradient()'s LOCAL LINEAR
    gradient (evaluated at the floor) and multiplied it by the nonlinear
    allocator's delta_s, exactly mirroring the LP branch's shortcut. That
    shortcut is only valid for the LP branch because the LP's own objective
    IS that same uncapped linear approximation, so its chosen delta_s stays
    self-consistent with what scores it. The nonlinear allocator instead
    maximizes the persuasion-ceiling-capped, alpha4-inclusive objective
    (_p_win_vec/_apply_ceiling below) -- reapplying an uncapped linear
    gradient to its (much larger, since it isn't knapsack-constrained by a
    fake linear objective) chosen allocations bypassed the ceiling entirely
    and inflated deploy value by ~46 seats out of 434, an obviously
    implausible number that this smoke test caught before any Theta
    comparison was reported. Fixed by reusing the allocator's own internal
    ceiling-respecting math directly, so what gets scored here is exactly
    what optimize_nonlinear() optimized -- no separate re-derivation to
    silently drift out of sync with it."""
    coef, sigma_model, races, n = state["coef"], state["sigma_model"], state["races"], state["n"]
    mu_t = state["mu0"]
    d_t, r_t = state["floor_arr"], state["r0_arr"]

    t0 = time.time()
    res = optimize_nonlinear(races, coef, sigma_model, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                              gamma=0.0, cap_fraction=0.15, party_budget=lsm.F0, eta=eta_arr)
    elapsed = time.time() - t0

    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=eta_arr)
    party = np.maximum(res.allocations - arrays["floors"], 0.0)
    d = np.maximum(arrays["floors"] + party, 1.0)
    r = _reactive_r(party, arrays)
    t_ = d + r
    ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
    log_ratio = np.log(ratio)
    log_total_pv = np.log(t_ / arrays["cvap"])
    mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
    mu_capped, _ = _apply_ceiling(mu_raw, arrays)

    r_terminal_expected = np.maximum(r_t + eta_arr * (state["d_terminal"] - d_t), 1.0)
    trickle_drift = _mu_struct_at(state, state["d_terminal"], r_terminal_expected) - _mu_struct_at(state, d_t, r_t)

    deployed_mu = mu_capped + trickle_drift
    value = norm.cdf(deployed_mu / state["widened_sigma"]).sum()
    return float(value), elapsed


def eta_array_for_tiers(eta_by_tier: dict, tiers: list[str]) -> np.ndarray:
    return np.array([eta_by_tier.get(t, 0.0) for t in tiers])


def main():
    print("Building t=0 state (deterministic across all Monte Carlo paths)...")
    state = build_t0_state()
    tiers = state["tiers"]
    results = {}

    for label, fit_cycle in [("eta_fit_2022", 2022), ("eta_fit_2024", 2024)]:
        eta_by_tier, _ = lsm.fit_eta_and_resid(fit_cycle)
        eta_arr = eta_array_for_tiers(eta_by_tier, tiers)
        v_lp, t_lp = deploy_value_lp(state, eta_arr)
        v_nl, t_nl = deploy_value_nonlinear(state, eta_arr)
        print(f"[{label}] LP deploy(0)={v_lp:.4f} ({t_lp:.3f}s)  "
              f"nonlinear deploy(0)={v_nl:.4f} ({t_nl:.2f}s)  "
              f"diff={v_nl - v_lp:+.4f} seats  implied Theta(0) shift={-(v_nl - v_lp):+.4f}")
        results[label] = {"v_lp": v_lp, "v_nonlinear": v_nl, "diff": v_nl - v_lp,
                           "lp_call_seconds": t_lp, "nonlinear_call_seconds": t_nl}

    print(f"\n=== eta_bootstrap_all_cycles: {N_BOOTSTRAP_SAMPLES} representative draws ===")
    rng = np.random.default_rng(20260729)
    tiers_per_race = tiers
    eta_paths, _, _ = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, N_BOOTSTRAP_SAMPLES, rng)
    diffs = []
    for k in range(N_BOOTSTRAP_SAMPLES):
        eta_arr = eta_paths[k]
        v_lp, t_lp = deploy_value_lp(state, eta_arr)
        v_nl, t_nl = deploy_value_nonlinear(state, eta_arr)
        diffs.append(v_nl - v_lp)
        print(f"  draw {k}: LP={v_lp:.4f} nonlinear={v_nl:.4f} diff={v_nl - v_lp:+.4f} ({t_nl:.2f}s)")
    diffs = np.array(diffs)
    print(f"  -> mean diff={diffs.mean():+.4f}, SD={diffs.std():.4f}, "
          f"min={diffs.min():+.4f}, max={diffs.max():+.4f}")
    results["eta_bootstrap_all_cycles"] = {
        "n_samples": N_BOOTSTRAP_SAMPLES, "mean_diff": float(diffs.mean()),
        "sd_diff": float(diffs.std()), "min_diff": float(diffs.min()), "max_diff": float(diffs.max()),
        "all_diffs": [float(d) for d in diffs],
    }

    out_path = ROOT / "outputs/lp_vs_nonlinear_deploy_branch.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
