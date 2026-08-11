#!/usr/bin/env python3
"""
Item (4) of Section 8.9's investigation plan: locate WHERE the LP-vs-
nonlinear reversal comes from, by comparing the two allocators directly on
identical states at every period of the live 98-day horizon -- not the full
K-path Monte Carlo (expensive, hours per replicate), but a much cheaper,
targeted diagnostic that needs only a handful of nonlinear calls.

States compared: a single, deterministic "representative" trajectory --
candidate spending D_i,t grows via the calibrated trickle rate only (no
idiosyncratic epsilon, no G_t walk, no R-reaction noise -- matching Section
8.7's mechanism-decomposition Scenario B construction), with R_i,t reacting
deterministically to that trickle via eta (eta_bootstrap_all_cycles' mean
per-tier rate). This is not a stochastic simulation and does not attempt to
reproduce Theta itself; it isolates how the ALLOCATION and DEPLOY VALUE
each allocator would produce differ as the state evolves along one
realistic path, which is exactly the mechanism question item (4) asks.

Reports, for both allocators, at each period:
  - deploy value (Phi((mu+delta_mu)/widened_sigma).sum())
  - number of races funded above a $1 threshold
  - concentration: share of total party budget in the top 5 funded races
  - average and max per-race party allocation

Continuation value and "how often the top-funded races change across
paths" are NOT computed here -- both require either the full backward
induction's regression fit or multiple stochastic paths, neither of which
this single-trajectory diagnostic has. This script answers a narrower but
still useful question: does the nonlinear allocator's advantage over LP
grow, shrink, or stay roughly constant as Election Day approaches?

Output: outputs/lp_vs_nonlinear_period_decomposition.json
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
import dataclasses

ROOT = Path(__file__).parent.parent
PERIODS_TO_CHECK = [0, 2, 4, 6]   # 98, 70, 42, 14 days remaining -- 4 points, not all 8, to bound cost


def build_state():
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

    trickle_per_day = lsm.load_trickle_rate_per_day(tiers)
    # Bootstrap mean per-tier eta, matching the eta_bootstrap_all_cycles scenario Section 8.9 investigates:
    rng = np.random.default_rng(0)
    eta_paths, _, _ = lsm.bootstrap_eta_resid_paths(lsm.BOOTSTRAP_CYCLES, tiers, 500, rng)
    eta_mean_by_tier_arr = eta_paths.mean(axis=0)   # (n,) per-race mean eta from the bootstrap distribution

    return dict(coef=coef, sigma_model=sigma_model, races=races, n=n,
                pvi_arr=pvi_arr, incumb_arr=incumb_arr, is_incumb_arr=is_incumb_arr,
                floor_arr=floor_arr, r0_arr=r0_arr, tiers=tiers, gb_national=gb_national,
                beta1_eff_arr=beta1_eff_arr, sigma_arr=sigma_arr,
                trickle_per_day=trickle_per_day, eta_arr=eta_mean_by_tier_arr)


def _mu_struct_at(state, d, r):
    t_ = np.maximum(d + r, 1.0)
    ratio = np.clip(d / t_, 1e-6, 1 - 1e-6)
    c_arr = state["beta1_eff_arr"] + state["coef"].beta2 * np.abs(state["pvi_arr"]) + \
        state["coef"].beta3 * state["is_incumb_arr"]
    return (state["coef"].alpha0 + state["coef"].alpha1 * state["pvi_arr"]
            + state["coef"].alpha2 * state["is_incumb_arr"]
            + state["coef"].alpha3 * state["gb_national"] + c_arr * np.log(ratio))


def state_at_period(state, tstep: int):
    """Deterministic trickle-only D/R at period tstep (no stochastic terms)."""
    days_elapsed = tstep * lsm.PERIOD_DAYS
    d_t = state["floor_arr"] + state["trickle_per_day"] * days_elapsed
    delta_d = state["trickle_per_day"] * days_elapsed
    r_t = np.maximum(state["r0_arr"] + state["eta_arr"] * delta_d, 1.0)
    mu_t = _mu_struct_at(state, d_t, r_t)
    return d_t, r_t, mu_t


def _allocation_stats(alloc: np.ndarray, floors: np.ndarray, label: str, elapsed: float) -> dict:
    party = np.maximum(alloc - floors, 0.0)
    n_funded = int(np.sum(party > 1.0))
    top5 = np.sort(party)[::-1][:5].sum()
    total = party.sum()
    concentration_top5 = float(top5 / total) if total > 0 else 0.0
    return {
        "n_funded_races": n_funded,
        "concentration_top5_share": concentration_top5,
        "avg_party_alloc_funded": float(party[party > 1.0].mean()) if n_funded > 0 else 0.0,
        "max_party_alloc": float(party.max()),
        "call_seconds": elapsed,
    }


def compare_at_period(state, tstep: int) -> dict:
    d_t, r_t, mu_t = state_at_period(state, tstep)
    n = state["n"]
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    sigma_arr = state["sigma_arr"]
    days_remaining = (lsm.N_PERIODS - tstep) * lsm.PERIOD_DAYS
    widened_sigma = np.sqrt(np.maximum(remaining_variance(sigma_arr, days_remaining), 1e-6))
    eta_arr = state["eta_arr"]

    # --- LP branch (mirrors _deploy_value's LP path exactly) ---
    from backtest.types import ModelOutputs
    p_win0 = norm.cdf(mu_t / sigma_arr)
    phi0 = norm.pdf(mu_t / sigma_arr)
    grad = np.array([lsm.margin_gradient(coef, state["pvi_arr"][i], state["incumb_arr"][i],
                                          d_t[i], r_t[i], eta_arr[i]) for i in range(n)])
    msg = phi0 / sigma_arr * grad
    outs = [ModelOutputs(district_id=races[i].district_id, ratio=d_t[i] / (d_t[i] + r_t[i]),
                          mu_hat=mu_t[i], sigma_i=sigma_arr[i], p_win=p_win0[i], msg_i=msg[i])
            for i in range(n)]
    t0 = time.time()
    res_lp = optimize(outs, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                       gamma=0.0, cap_fraction=0.15, floor_allocations=d_t, party_budget=lsm.F0,
                       d_total_obs=d_t)
    lp_elapsed = time.time() - t0
    delta_s_lp = np.maximum(res_lp.allocations - d_t, 0.0)
    deployed_mu_lp = mu_t + grad * delta_s_lp
    lp_value = float(norm.cdf(deployed_mu_lp / widened_sigma).sum())
    lp_stats = _allocation_stats(res_lp.allocations, d_t, "lp", lp_elapsed)

    # --- Nonlinear branch (ceiling-respecting, matching the run_lsm fix) ---
    races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                    d_total=float(d_t[i])) for i, r in enumerate(races)]
    t0 = time.time()
    res_nl = optimize_nonlinear(races_t, coef, sigma_model, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                                 gamma=0.0, cap_fraction=0.15, party_budget=lsm.F0, eta=eta_arr)
    nl_elapsed = time.time() - t0
    arrays = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr)
    party_nl = np.maximum(res_nl.allocations - d_t, 0.0)
    d = np.maximum(arrays["floors"] + party_nl, 1.0)
    r = _reactive_r(party_nl, arrays)
    t_ = d + r
    ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
    log_ratio = np.log(ratio)
    log_total_pv = np.log(t_ / arrays["cvap"])
    mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
    mu_capped, _ = _apply_ceiling(mu_raw, arrays)
    nl_value = float(norm.cdf(mu_capped / widened_sigma).sum())
    nl_stats = _allocation_stats(res_nl.allocations, d_t, "nonlinear", nl_elapsed)

    result = {
        "period": tstep, "days_remaining": days_remaining,
        "lp_deploy_value": lp_value, "nonlinear_deploy_value": nl_value,
        "deploy_value_diff": nl_value - lp_value,
        "lp": lp_stats, "nonlinear": nl_stats,
    }
    print(f"  t={tstep} ({days_remaining}d left): LP deploy={lp_value:.3f} "
          f"(n_funded={lp_stats['n_funded_races']}, top5_share={lp_stats['concentration_top5_share']:.2%})  "
          f"nonlinear deploy={nl_value:.3f} (n_funded={nl_stats['n_funded_races']}, "
          f"top5_share={nl_stats['concentration_top5_share']:.2%})  diff={nl_value - lp_value:+.3f} "
          f"({nl_elapsed:.1f}s)")
    return result


def main():
    print("Building deterministic-trickle representative trajectory...")
    state = build_state()
    results = []
    for tstep in PERIODS_TO_CHECK:
        results.append(compare_at_period(state, tstep))

    out_path = ROOT / "outputs/lp_vs_nonlinear_period_decomposition.json"
    with open(out_path, "w") as f:
        json.dump({"periods_checked": PERIODS_TO_CHECK, "results": results}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
