#!/usr/bin/env python3
"""
Paper III revision, next-step item 5: the surrogate's validation so far
(scripts/theta_concave_surrogate.py) checks only 4 deterministic,
trickle-only period states, all drawn from a single representative
trajectory with no idiosyncratic shocks, no R-reaction noise, and a single
calibration scenario's eta. That is a narrow slice of the state space the
live backward induction actually visits. This script broadens validation
along exactly the dimensions a reviewer would ask about: reporting period,
calibration scenario, and genuinely random stochastic realizations (so
states range from noncompetitive-leaning to unusually competitive
configurations, not just the deterministic trend line).

Method: rather than re-running a full K-path Monte Carlo (expensive) just to
harvest individual (path, period) states, each sampled state is built
directly from the SAME marginal distributions run_lsm()'s own path
simulation uses -- the trickle-driven deterministic D_t/R_t base, plus a
single draw of (a) R's cumulative reaction noise, N(0, tstep * resid_std^2)
(the sum of `tstep` iid per-period residual draws), and (b) idiosyncratic
mu's cumulative resolved shock, N(0, sigma_i^2 - remaining_variance(sigma_i,
days_remaining)) (Appendix B.2's telescoping decomposition run in reverse:
this is exactly the resolved-to-date share of each race's idiosyncratic
budget). Because both are sums of independent Gaussian increments, drawing
the cumulative total directly has the identical marginal distribution as
stepping through run_lsm's own period-by-period simulation would produce at
that same period -- this is a faithful shortcut, not an approximation to a
different process. Each state is then scored under BOTH optimize_nonlinear()
(true objective) and the concave-envelope surrogate, at that period's
correct widened_sigma, exactly as the backward induction itself would.

Reports: mean/max/percentile absolute objective-value error, an allocation-
distance measure (L1 distance between the two allocators' chosen per-race
DCCC spend, as a fraction of the total reserve), and n_funded comparison --
stratified by calibration scenario and by a "competitiveness" tercile (the
fraction of races within NEAR_THRESHOLD_MARGIN_PP of a toss-up at that
sampled state, which varies genuinely across samples because idiosyncratic
shocks and eta draws differ, not something these 4 original states could
vary since they shared one deterministic trajectory).

Output: outputs/theta_surrogate_broad_validation.json
"""

from __future__ import annotations
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
from backtest.optimizer.allocator import optimize_nonlinear, _precompute_race_arrays, _reactive_r, _apply_ceiling
from backtest.model.win_prob import compute_outputs_batch
from concave_surrogate import surrogate_allocate
from simulate_and_validate import remaining_variance

ROOT = Path(__file__).parent.parent
SCENARIOS = ["eta_bootstrap_all_cycles", "eta_fit_2022", "eta_fit_2024"]
N_PER_SCENARIO = 16
MASTER_SEED = 20260803


def build_static():
    coef, sigma_model = lsm.load_coef_and_sigma()
    races = lsm.build_universe(cycle=2026)
    n = len(races)
    pvi_arr = np.array([r.pvi for r in races])
    incumb_arr = [r.incumb_status for r in races]
    is_incumb_arr = np.array([1.0 if s == "Incumbent" else 0.0 for s in incumb_arr])
    floor_arr = np.array([r.cand_d_total for r in races])
    r0_arr = np.array([r.r_total for r in races])
    tiers = [r.cook_rating for r in races]
    is_comp = np.array([t in lsm.COMPETITIVE for t in tiers])
    gb_national = races[0].generic_ballot
    if coef.beta1_open is not None:
        is_open_arr = np.array([1.0 if s == "Open" else 0.0 for s in incumb_arr])
        beta1_eff_arr = np.where(is_open_arr > 0, coef.beta1_open, coef.beta1)
    else:
        beta1_eff_arr = np.full(n, coef.beta1)
    outputs0 = compute_outputs_batch(races, coef, sigma_model)
    sigma_arr = np.array([o.sigma_i for o in outputs0])
    trickle_per_day = lsm.load_trickle_rate_per_day(tiers)
    return dict(coef=coef, sigma_model=sigma_model, races=races, n=n, pvi_arr=pvi_arr,
                incumb_arr=incumb_arr, is_incumb_arr=is_incumb_arr, floor_arr=floor_arr,
                r0_arr=r0_arr, tiers=tiers, is_comp=is_comp, gb_national=gb_national,
                beta1_eff_arr=beta1_eff_arr, sigma_arr=sigma_arr, trickle_per_day=trickle_per_day)


def _mu_struct_at(s, d, r):
    t_ = np.maximum(d + r, 1.0)
    ratio = np.clip(d / t_, 1e-6, 1 - 1e-6)
    c_arr = s["beta1_eff_arr"] + s["coef"].beta2 * np.abs(s["pvi_arr"]) + s["coef"].beta3 * s["is_incumb_arr"]
    return (s["coef"].alpha0 + s["coef"].alpha1 * s["pvi_arr"] + s["coef"].alpha2 * s["is_incumb_arr"]
            + s["coef"].alpha3 * s["gb_national"] + c_arr * np.log(ratio))


def build_eta_cache(tiers: list[str]):
    """Precompute every per-cycle eta/resid fit ONCE (fit_eta_and_resid() re-reads
    and re-processes the raw multi-cycle IE panel from disk on every call, which
    is expensive -- calling it per-sample inside the validation loop below made a
    first version of this script effectively re-load all 7 historical cycles for
    every one of 48 samples). Returns a dict scenario -> ready-to-slice arrays/
    per-tier draw pools, reused across every sample."""
    eta_2022, resid_2022 = lsm.fit_eta_and_resid(2022)
    eta_2024, resid_2024 = lsm.fit_eta_and_resid(2024)
    per_cycle_fits = {c: lsm.fit_eta_and_resid(c) for c in lsm.BOOTSTRAP_CYCLES}
    eta_by_tier_cycle = {t: [] for t in lsm.TIERS}
    resid_by_tier_cycle = {t: [] for t in lsm.TIERS}
    for c in lsm.BOOTSTRAP_CYCLES:
        eta_c, resid_c = per_cycle_fits[c]
        for t in lsm.TIERS:
            if t in eta_c:
                eta_by_tier_cycle[t].append(eta_c[t])
                resid_by_tier_cycle[t].append(resid_c[t])
    return {
        "eta_fit_2022": (np.array([eta_2022.get(t, 0.0) for t in tiers]),
                         np.array([resid_2022.get(t, 0.0) for t in tiers])),
        "eta_fit_2024": (np.array([eta_2024.get(t, 0.0) for t in tiers]),
                         np.array([resid_2024.get(t, 0.0) for t in tiers])),
        "bootstrap_pools": (eta_by_tier_cycle, resid_by_tier_cycle),
    }


def eta_resid_for_scenario(scenario: str, tiers: list[str], rng: np.random.Generator, cache: dict):
    if scenario == "eta_bootstrap_all_cycles":
        eta_by_tier_cycle, resid_by_tier_cycle = cache["bootstrap_pools"]
        n = len(tiers)
        eta_arr = np.zeros(n)
        resid_arr = np.zeros(n)
        for t in lsm.TIERS:
            idx = [i for i, race_tier in enumerate(tiers) if race_tier == t]
            available_eta = np.array(eta_by_tier_cycle[t])
            available_resid = np.array(resid_by_tier_cycle[t])
            if not idx or len(available_eta) == 0:
                continue
            draw = rng.integers(0, len(available_eta))
            for i in idx:
                eta_arr[i] = available_eta[draw]
                resid_arr[i] = available_resid[draw]
        return eta_arr, resid_arr
    return cache[scenario]


def sample_random_state(s, tstep: int, eta_arr: np.ndarray, resid_std_arr: np.ndarray,
                         rng: np.random.Generator):
    days_elapsed = tstep * lsm.PERIOD_DAYS
    delta_d = s["trickle_per_day"] * days_elapsed
    d_t = s["floor_arr"] + delta_d
    r_reaction_noise = rng.normal(0.0, resid_std_arr * np.sqrt(max(tstep, 0)))
    r_t = np.maximum(s["r0_arr"] + eta_arr * delta_d + r_reaction_noise, 1.0)
    days_remaining = (lsm.N_PERIODS - tstep) * lsm.PERIOD_DAYS
    resolved_var = np.maximum(s["sigma_arr"] ** 2 - remaining_variance(s["sigma_arr"], days_remaining), 0.0)
    eps_cum = rng.normal(0.0, np.sqrt(resolved_var))
    mu_t = _mu_struct_at(s, d_t, r_t) + eps_cum
    return d_t, r_t, mu_t


def compare_state(s, tstep: int, eta_arr: np.ndarray, d_t: np.ndarray, r_t: np.ndarray) -> dict:
    n = s["n"]
    races, coef, sigma_model = s["races"], s["coef"], s["sigma_model"]
    sigma_arr = s["sigma_arr"]
    days_remaining = (lsm.N_PERIODS - tstep) * lsm.PERIOD_DAYS
    widened_sigma = np.sqrt(np.maximum(remaining_variance(sigma_arr, days_remaining), 1e-6))

    races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                    d_total=float(d_t[i])) for i, r in enumerate(races)]

    def _mu_capped_from_party(party, arrays):
        d = np.maximum(arrays["floors"] + party, 1.0)
        r = _reactive_r(party, arrays)
        t_ = d + r
        log_ratio = np.log(np.clip(d / t_, 1e-15, 1 - 1e-15))
        log_total_pv = np.log(t_ / arrays["cvap"])
        mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
        mu_capped, _ = _apply_ceiling(mu_raw, arrays)
        return mu_capped

    t0 = time.time()
    res_nl = optimize_nonlinear(races_t, coef, sigma_model, budget=lsm.F0, cov_matrix=np.eye(n) * 1e-6,
                                 gamma=0.0, cap_fraction=0.15, party_budget=lsm.F0, eta=eta_arr)
    nl_elapsed = time.time() - t0
    arrays_nl = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr)
    party_nl = np.maximum(res_nl.allocations - d_t, 0.0)
    nl_value = float(norm.cdf(_mu_capped_from_party(party_nl, arrays_nl) / widened_sigma).sum())

    t0 = time.time()
    party_sur, arrays_sur = surrogate_allocate(races_t, coef, sigma_model, lsm.F0, 0.15, eta_arr)
    sur_elapsed = time.time() - t0
    sur_value = float(norm.cdf(_mu_capped_from_party(party_sur, arrays_sur) / widened_sigma).sum())

    alloc_l1 = float(np.abs(party_nl - party_sur).sum())
    return dict(
        period=tstep, days_remaining=int(days_remaining),
        nonlinear_value=nl_value, surrogate_value=sur_value,
        objective_error=sur_value - nl_value, abs_objective_error=abs(sur_value - nl_value),
        allocation_l1_frac_budget=alloc_l1 / lsm.F0,
        n_funded_nonlinear=int(np.sum(party_nl > 1.0)), n_funded_surrogate=int(np.sum(party_sur > 1.0)),
        nonlinear_elapsed_s=nl_elapsed, surrogate_elapsed_s=sur_elapsed,
    )


def main():
    print("Building static universe...")
    s = build_static()
    print("Precomputing eta/resid fits once (all 7 historical cycles, cached for every sample)...")
    eta_cache = build_eta_cache(s["tiers"])
    master_rng = np.random.default_rng(MASTER_SEED)

    all_results = []
    for scenario in SCENARIOS:
        print(f"\n=== scenario={scenario} ===")
        for j in range(N_PER_SCENARIO):
            state_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
            tstep = int(state_rng.integers(0, lsm.N_PERIODS + 1))
            eta_arr, resid_std_arr = eta_resid_for_scenario(scenario, s["tiers"], state_rng, eta_cache)
            d_t, r_t, mu_t = sample_random_state(s, tstep, eta_arr, resid_std_arr, state_rng)
            near_thresh_frac = float(np.mean(np.abs(mu_t[s["is_comp"]]) < lsm.NEAR_THRESHOLD_MARGIN_PP))

            res = compare_state(s, tstep, eta_arr, d_t, r_t)
            res["scenario"] = scenario
            res["sample_index"] = j
            res["near_threshold_frac_competitive_races"] = near_thresh_frac
            all_results.append(res)
            print(f"  [{scenario}] sample {j}: t={tstep} ({res['days_remaining']}d left), "
                  f"near_thresh_frac={near_thresh_frac:.3f}, nonlinear={res['nonlinear_value']:.4f}, "
                  f"surrogate={res['surrogate_value']:.4f}, err={res['objective_error']:+.4f}, "
                  f"alloc_L1/F0={res['allocation_l1_frac_budget']:.4f} ({res['nonlinear_elapsed_s']:.1f}s)")

    abs_errs = np.array([r["abs_objective_error"] for r in all_results])
    alloc_l1s = np.array([r["allocation_l1_frac_budget"] for r in all_results])
    near_thresh = np.array([r["near_threshold_frac_competitive_races"] for r in all_results])
    deploy_agree_count = int(np.sum(
        [r["n_funded_nonlinear"] > 0 for r in all_results] == np.array(
            [r["n_funded_surrogate"] > 0 for r in all_results])))

    # Stratify by competitiveness tercile of the sampled state.
    terc = np.quantile(near_thresh, [1 / 3, 2 / 3])
    strat = {}
    for label, mask in [
        ("low_competitiveness", near_thresh <= terc[0]),
        ("mid_competitiveness", (near_thresh > terc[0]) & (near_thresh <= terc[1])),
        ("high_competitiveness", near_thresh > terc[1]),
    ]:
        if mask.sum() == 0:
            continue
        strat[label] = {
            "n": int(mask.sum()),
            "mean_abs_error": float(abs_errs[mask].mean()),
            "max_abs_error": float(abs_errs[mask].max()),
        }

    summary = {
        "n_samples": len(all_results),
        "n_per_scenario": N_PER_SCENARIO,
        "scenarios": SCENARIOS,
        "mean_abs_objective_error": float(abs_errs.mean()),
        "max_abs_objective_error": float(abs_errs.max()),
        "p50_abs_objective_error": float(np.percentile(abs_errs, 50)),
        "p90_abs_objective_error": float(np.percentile(abs_errs, 90)),
        "p99_abs_objective_error": float(np.percentile(abs_errs, 99)),
        "mean_allocation_l1_frac_budget": float(alloc_l1s.mean()),
        "max_allocation_l1_frac_budget": float(alloc_l1s.max()),
        "by_competitiveness_tercile": strat,
    }
    print("\n=== Summary across all samples ===")
    for k, v in summary.items():
        if k not in ("by_competitiveness_tercile",):
            print(f"  {k}: {v}")
    print(f"  by_competitiveness_tercile: {json.dumps(strat, indent=2)}")

    out_path = ROOT / "outputs/theta_surrogate_broad_validation.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": all_results}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
