#!/usr/bin/env python3
"""
Paper III revision, next-step item 6: the paper's only quantified source of
uncertainty around Theta(0) so far is Monte Carlo simulation noise (Section
8.8's five-seed SE of 0.0044), which holds every CALIBRATED parameter fixed
at its point estimate. This script addresses the larger, previously
unquantified source: uncertainty in the calibration itself. A full nested
bootstrap (refit eta/trickle/lambda on resampled historical panels, re-solve
Theta under each resample) is the ideal but is not attempted here; this is
the "at minimum" fallback the revision plan calls for -- structured
sensitivity ranges around the three calibrated parameters most exposed to
real historical instability (Table 5's I^2, Section 6.4's 16-43% swing when
the trickle panel was extended from 2 to 7 cycles, and the OU-fit lambda),
plus a lightweight joint/randomized sensitivity that reports how often
Theta_surrogate(0)'s sign flips.

Two distinct sources of uncertainty are kept separate deliberately:
  1. Cross-cycle SAMPLING variation in eta/resid, already captured by
     bootstrap_eta_resid_paths' per-path draws (used in every scenario
     already reported).
  2. STRUCTURAL/calibration uncertainty on top of that -- eta's untested
     extrapolation to the candidate-committee spending channel (Section
     10.1), the trickle rate's own demonstrated historical instability, and
     lambda's fitted-vs-plausible-alternative decay speed -- represented
     here as multiplicative scale factors applied on top of the existing
     bootstrap draw, not a replacement for it.

Scale ranges:
  - eta_scale in [0.7, 1.3]: roughly the swing between the 2-cycle and
    7-cycle eta re-estimates already documented (Section 6.1), and a
    conservative bound on the IE-to-candidate-committee extrapolation risk.
  - trickle_scale in [0.6, 1.4]: brackets the exact 16-43% swing Section 6.4
    reports when the trickle panel was extended from 2 to 7 cycles.
  - lambda_scale in [0.85, 1.15]: the term structure (Table 6) is stable to
    within a few percent from 30-270 days; this brackets a wider margin than
    that observed stability to account for the decay-rate fit itself, not
    just the term structure's raw stability.

Runs entirely on the validated concave-envelope surrogate (LP-speed,
~0.03s/call) -- this analysis is only tractable because that surrogate
exists; it would not have been practical against optimize_nonlinear() or
even the LP allocator at the K needed for stable point estimates.

Output: outputs/theta_parameter_sensitivity.json
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

import solve_bellman_lsm as lsm
import simulate_and_validate as sv

ROOT = Path(__file__).parent.parent
BASE_LAMBDA = sv.LAMBDA_DECAY
_ORIG_TRICKLE = lsm.load_trickle_rate_per_day

OAT_K = 500
OAT_SEED = 42
JOINT_K = 200
JOINT_N_DRAWS = 80
JOINT_MASTER_SEED = 20260803


def _scaled_trickle_fn(scale: float):
    def f(tiers_per_race):
        return _ORIG_TRICKLE(tiers_per_race) * scale
    return f


_CACHE = {}


def _build_cache():
    """Precompute the race universe and every per-cycle eta/resid fit ONCE.
    bootstrap_eta_resid_paths() re-reads and re-processes the raw multi-cycle
    IE panel from disk on every call; calling it fresh inside run_once (once
    per OAT config, once per joint draw -- 93 calls total) made a first
    version of this script re-load all 7 historical cycles' raw data 93
    times over. Cached once here and reused for every draw below."""
    if _CACHE:
        return _CACHE
    races = lsm.build_universe(cycle=2026)
    tiers = [r.cook_rating for r in races]
    per_cycle_fits = {c: lsm.fit_eta_and_resid(c) for c in lsm.BOOTSTRAP_CYCLES}
    eta_by_tier_cycle = {t: [] for t in lsm.TIERS}
    resid_by_tier_cycle = {t: [] for t in lsm.TIERS}
    for c in lsm.BOOTSTRAP_CYCLES:
        eta_c, resid_c = per_cycle_fits[c]
        for t in lsm.TIERS:
            if t in eta_c:
                eta_by_tier_cycle[t].append(eta_c[t])
                resid_by_tier_cycle[t].append(resid_c[t])
    _CACHE["tiers"] = tiers
    _CACHE["eta_by_tier_cycle"] = eta_by_tier_cycle
    _CACHE["resid_by_tier_cycle"] = resid_by_tier_cycle
    return _CACHE


def _bootstrap_draw_cached(k_paths: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    cache = _build_cache()
    tiers = cache["tiers"]
    eta_by_tier_cycle = cache["eta_by_tier_cycle"]
    resid_by_tier_cycle = cache["resid_by_tier_cycle"]
    n = len(tiers)
    eta_paths = np.zeros((k_paths, n))
    resid_paths = np.zeros((k_paths, n))
    summary = {}
    for t in lsm.TIERS:
        idx = [i for i, race_tier in enumerate(tiers) if race_tier == t]
        available_eta = np.array(eta_by_tier_cycle[t])
        available_resid = np.array(resid_by_tier_cycle[t])
        if not idx or len(available_eta) == 0:
            continue
        draw_idx = rng.integers(0, len(available_eta), size=k_paths)
        eta_draw = available_eta[draw_idx]
        resid_draw = available_resid[draw_idx]
        for i in idx:
            eta_paths[:, i] = eta_draw
            resid_paths[:, i] = resid_draw
        summary[t] = {"n_cycles_available": int(len(available_eta)),
                       "path_draw_mean": float(eta_draw.mean()), "path_draw_sd": float(eta_draw.std())}
    return eta_paths, resid_paths, summary


def run_once(eta_scale: float, trickle_scale: float, lambda_scale: float,
             k_paths: int, seed: int, label: str) -> float:
    sv.LAMBDA_DECAY = BASE_LAMBDA * lambda_scale
    lsm.load_trickle_rate_per_day = _scaled_trickle_fn(trickle_scale)
    lsm.K_PATHS = k_paths
    try:
        _build_cache()
        eta_rng = np.random.default_rng(seed)
        eta_arr, resid_arr, summary = _bootstrap_draw_cached(k_paths, eta_rng)
        eta_arr = eta_arr * eta_scale
        lsm.RNG = np.random.default_rng(seed)
        res = lsm.run_lsm(eta_arr, resid_arr, label, eta_summary=summary,
                           use_surrogate_allocator=True)
        return float(res["theta_by_period"][0]["mean_theta"])
    finally:
        sv.LAMBDA_DECAY = BASE_LAMBDA
        lsm.load_trickle_rate_per_day = _ORIG_TRICKLE


def run_oat():
    print("=== One-at-a-time sensitivity (K={}, seed={}) ===".format(OAT_K, OAT_SEED))
    grids = {
        "eta_scale": [0.7, 0.85, 1.0, 1.15, 1.3],
        "trickle_scale": [0.6, 0.8, 1.0, 1.2, 1.4],
        "lambda_scale": [0.85, 1.0, 1.15],
    }
    results = {}
    for param, values in grids.items():
        rows = []
        for v in values:
            kwargs = {"eta_scale": 1.0, "trickle_scale": 1.0, "lambda_scale": 1.0}
            kwargs[param] = v
            t0 = time.time()
            theta0 = run_once(k_paths=OAT_K, seed=OAT_SEED, label=f"oat_{param}_{v}", **kwargs)
            elapsed = time.time() - t0
            rows.append({"value": v, "theta0": theta0, "elapsed_s": elapsed})
            print(f"  {param}={v:.2f}: Theta(0)={theta0:+.4f}  ({elapsed:.1f}s)")
        results[param] = rows
    return results


def run_joint():
    print(f"\n=== Joint randomized sensitivity ({JOINT_N_DRAWS} draws, K={JOINT_K}) ===")
    master_rng = np.random.default_rng(JOINT_MASTER_SEED)
    draws = []
    for j in range(JOINT_N_DRAWS):
        eta_scale = float(master_rng.uniform(0.7, 1.3))
        trickle_scale = float(master_rng.uniform(0.6, 1.4))
        lambda_scale = float(master_rng.uniform(0.85, 1.15))
        seed = int(master_rng.integers(0, 2**31 - 1))
        t0 = time.time()
        theta0 = run_once(eta_scale, trickle_scale, lambda_scale, JOINT_K, seed, f"joint_{j}")
        elapsed = time.time() - t0
        draws.append({"eta_scale": eta_scale, "trickle_scale": trickle_scale,
                       "lambda_scale": lambda_scale, "seed": seed, "theta0": theta0})
        print(f"  draw {j}: eta_scale={eta_scale:.3f} trickle_scale={trickle_scale:.3f} "
              f"lambda_scale={lambda_scale:.3f} -> Theta(0)={theta0:+.4f}  ({elapsed:.1f}s)")
    thetas = np.array([d["theta0"] for d in draws])
    frac_negative = float(np.mean(thetas < 0))
    summary = {
        "n_draws": JOINT_N_DRAWS, "k_paths": JOINT_K,
        "mean_theta0": float(thetas.mean()), "sd_theta0": float(thetas.std()),
        "min_theta0": float(thetas.min()), "max_theta0": float(thetas.max()),
        "frac_negative_deploy_favored": frac_negative,
        "frac_positive_hold_favored": float(np.mean(thetas > 0)),
    }
    print(f"\nJoint sensitivity summary: mean={summary['mean_theta0']:+.4f} "
          f"sd={summary['sd_theta0']:.4f} range=[{summary['min_theta0']:+.4f}, {summary['max_theta0']:+.4f}] "
          f"frac_negative(deploy favored)={frac_negative:.3f}")
    return {"summary": summary, "draws": draws}


def main():
    oat_results = run_oat()
    joint_results = run_joint()
    out_path = ROOT / "outputs/theta_parameter_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump({
            "scenario": "eta_bootstrap_all_cycles", "allocator": "surrogate",
            "oat_k_paths": OAT_K, "oat_seed": OAT_SEED,
            "joint_k_paths": JOINT_K, "joint_n_draws": JOINT_N_DRAWS,
            "one_at_a_time": oat_results, "joint_randomized": joint_results,
        }, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
