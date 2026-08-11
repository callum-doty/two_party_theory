#!/usr/bin/env python3
"""
Longstaff-Schwartz backward induction for Theta(t) (Paper III Section 7.2),
run only after Section 7.1's simulator self-consistency gate passed.

Setup: 2026 live universe (434 races), "wait" branch simulated forward (no
DCCC discretionary deployment; candidate-committee floors D_i,t grow via a
calibrated non-discretionary spending trickle -- data_catalog.md Section
2.7 / scripts/estimate_candidate_spend_trickle.py, unblocking
docs/theta_followup_plan.md Section 0.1.1's previously-blocked fix -- and
R_i,t reacts to that trickle via eta on top of residual noise), K paths,
biweekly periods from today to Election Day 2026-11-03.

Per (path, period), two values are compared:
  - "Deploy now": close the discretionary reserve immediately, apply the
    resulting Delta_mu_i, then apply the closed-form "let remaining drift
    resolve" widening: Phi((mu_i,t + Delta_mu_i) / sqrt(sigma_i^2 + V_i(t))).
    Three interchangeable allocator branches exist for the deploy step
    (see deploy_value(), below): the original fast LP allocator (optimize(),
    ~11ms/call), the exact nonlinear allocator (optimize_nonlinear(), far
    slower -- 40s to over an hour per call, infeasible at Monte Carlo path
    counts), and a validated concave-envelope surrogate (~0.03s/call) that
    tracks the nonlinear allocator's value within 0.11-0.19 seats while
    staying LP-speed. paper/paper3_final.md's "Allocator-Robustness
    Finding" documents that the LP branch's own knapsack-style degeneracy
    manufactures artificial option value for "wait" and FLIPS THE SIGN of
    the headline Theta(0) result relative to the nonlinear/surrogate
    branches. main() below therefore uses the surrogate allocator by
    default (fixed 2026-08 after this was found to still be silently
    computing the superseded, wrong-sign LP calculation) -- the LP branch
    remains available via use_nonlinear_allocator=False,
    use_surrogate_allocator=False for side-by-side comparison scripts
    (e.g. theta_nonlinear_multiseed.py), not as this module's own default.
    sigma_i (Paper I's static residual) and V_i(t) (Paper III's remaining-
    drift variance, Section 7.1) are ADDITIVE, not substitutive: mu_i,T =
    mu_i,t + Delta_mu_i + xi, xi ~ N(0, V_i(t)), and
    E[Phi((mu+xi)/sigma)] = Phi(mu / sqrt(sigma^2 + V)) by the standard
    normal-CDF-convolution identity.
  - "Wait": regression-estimated continuation value, basis = Section
    7.2's four compressed features (E[Seats]_t, Var[Seats]_t, max MSG_t,
    near-threshold count), fit on paths' own realized V*_{t+1}.

Var[Seats]_t here is Sum_i p_i(1-p_i) -- an independence approximation,
not Paper I's full factor-covariance model -- stated explicitly as a
simplification of this first pass, not a silent omission.

Run three scenarios:
  - eta_fit_2022 / eta_fit_2024: eta fit on a single cycle, held identical
    across all K paths -- the original cycle-instability bracket (Section 5.5).
  - eta_bootstrap_all_cycles (added 2026-07-17, per docs/theta_followup_plan.md
    Section 6): each of the K simulated paths draws its OWN per-tier
    (eta, resid_std) pair from a randomly chosen historical cycle
    (2012-2024, whichever have >=10 obs for that tier), held fixed for
    that path's whole campaign. This propagates the confirmed real
    cycle-to-cycle eta variation (5 of 7 tiers, scripts/
    reconcile_eta_sigma_g_instability.py) into Theta directly, replacing
    two hand-picked brackets with an empirical distribution -- the
    "random effects, draw once per simulated election" approach preferred
    over a continuous stochastic process, since the data show cycle-to-
    cycle jumps, not within-cycle drift (this regression cannot even see
    the latter -- it fits one eta per cycle by construction).

Output: outputs/theta_schedule.json
"""

from __future__ import annotations
import dataclasses
import functools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.optimizer.allocator import (
    optimize, optimize_nonlinear, _precompute_race_arrays, _reactive_r, _apply_ceiling,
)
from backtest.optimizer.robust import optimize_nonlinear_robust
from backtest.estimation import eta_hierarchical
from backtest.types import SigmaModel, ModelOutputs

from estimate_eta_reaction import build_period_panel, build_delta_panel, TIERS
from simulate_and_validate import incremental_variances, remaining_variance, SIGMA_G_PER_SQRT_DAY
from concave_surrogate import surrogate_allocate

ROOT = Path(__file__).parent.parent
RNG = np.random.default_rng(20260716)

# Wait-branch spending trickle (docs/theta_followup_plan.md Section 0.1.1's
# blocked fix -- unblocked this session by data_catalog.md Section 2.7's
# dated candidate-committee periodic-reports panel). See
# scripts/estimate_candidate_spend_trickle.py for how this is fit.
_TRICKLE_PATH = ROOT / "data/processed/candidate_spend_trickle.json"

# Single source of truth: data/processed/live_2026_state.json, written by
# scripts/plot_2026_live_allocation.py. Previously TODAY/ELECTION_DAY/F0 were
# independent hardcoded literals here -- a stale TODAY already caused a real
# 98-vs-110-days-remaining mismatch against scripts/make_theta_paper_figures.py
# before this fix (Paper III audit, 2026-07-16).
with open(ROOT / "data/processed/live_2026_state.json") as _f:
    _live_state = json.load(_f)

PERIOD_DAYS = config.period_days()
TODAY = date.fromisoformat(_live_state["as_of"])
ELECTION_DAY = date.fromisoformat(_live_state["election_day"])
N_PERIODS = max(1, (ELECTION_DAY - TODAY).days // PERIOD_DAYS)
K_PATHS = 2000
COMPETITIVE = {"Toss-Up", "Lean D", "Lean R"}
NEAR_THRESHOLD_MARGIN_PP = 3.0   # points, matching Section 7.2's stated "e.g. 2 points" spec (widened slightly for path-count stability)
F0 = _live_state["f0"]          # deployable capital, single source of truth (Paper II Section 7.1's live figure)


def load_coef_and_sigma():
    with open(ROOT / "data/processed/margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                              "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(ROOT / "data/processed/sigma_model.json") as f:
        sigma_coef = json.load(f)
    return coef, SigmaModel(_coef=sigma_coef)


def load_trickle_rate_per_day(tiers_per_race: list[str]) -> np.ndarray:
    """Per-race $/day candidate-committee spending trickle rate (data_catalog.md
    Section 2.7 / scripts/estimate_candidate_spend_trickle.py), keyed by each
    race's own Cook tier. Falls back to the pooled rate for a tier with no
    fitted observations, and to 0.0 (the old, documented behavior -- D_i,t
    held perfectly fixed while waiting) if the trickle file has not been
    generated yet, so this script remains runnable before the historical
    dated panel exists for any cycle.

    Reads mean_rate_per_day, not median: checked directly against the real
    2022/2024 panel, median is exactly $0.00/day in every tier -- FEC's
    quarterly filing cadence against this project's biweekly period grid
    means most 14-day windows contain no new filing, so a median across
    mostly-zero deltas is zero regardless of real underlying growth (see
    estimate_candidate_spend_trickle.py's module docstring for the full
    reasoning). Uses trickle["preferred_estimator"] rather than hardcoding
    the key name, so a future re-calibration's own choice of estimator is
    respected here without a second edit."""
    if not _TRICKLE_PATH.exists():
        print(f"  [trickle] {_TRICKLE_PATH} not found -- wait-branch D_i,t will stay fixed "
              "(pre-fix behavior). Run scripts/estimate_candidate_spend_trickle.py once the "
              "dated candidate periodic-reports panel is fetched.")
        return np.zeros(len(tiers_per_race))
    with open(_TRICKLE_PATH) as f:
        trickle = json.load(f)
    by_tier = trickle["by_tier"]
    estimator = trickle.get("preferred_estimator", "mean_rate_per_day")
    pooled_rate = by_tier.get("_pooled", {}).get(estimator, 0.0)
    return np.array([
        by_tier.get(t, {}).get(estimator, pooled_rate) for t in tiers_per_race
    ])


def fit_eta_and_resid(fit_cycle: int) -> tuple[dict, dict]:
    panel = build_period_panel(fit_cycle)
    delta = build_delta_panel(panel)
    eta_by_tier, resid_std_by_tier = {}, {}
    for tier in TIERS:
        mask = delta["tier"] == tier
        if mask.sum() < 10:
            continue
        X = sm.add_constant(delta.loc[mask, "d_ie_delta_lag_dm"])
        y = delta.loc[mask, "r_ie_delta_dm"]
        fit = sm.OLS(y, X).fit()
        eta_by_tier[tier] = float(fit.params.get("d_ie_delta_lag_dm", 0.0))
        resid_std_by_tier[tier] = float(fit.resid.std())
    return eta_by_tier, resid_std_by_tier


BOOTSTRAP_CYCLES = [2012, 2014, 2016, 2018, 2020, 2022, 2024]


def tile_single_cycle(eta_by_tier: dict, resid_std_by_tier: dict, tiers_per_race: list[str],
                       k_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast one cycle's (eta, resid_std) per tier identically across
    every simulated path -- the original eta_fit_2022/eta_fit_2024 brackets,
    expressed in the same (K_PATHS, n) shape the bootstrap scenario uses,
    so run_lsm() doesn't need two different code paths."""
    n = len(tiers_per_race)
    eta_row = np.array([eta_by_tier.get(t, 0.0) for t in tiers_per_race])
    resid_row = np.array([resid_std_by_tier.get(t, 0.0) for t in tiers_per_race])
    return np.tile(eta_row, (k_paths, 1)), np.tile(resid_row, (k_paths, 1))


def bootstrap_eta_resid_paths(cycles: list[int], tiers_per_race: list[str], k_paths: int,
                               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Each of k_paths simulated elections draws its OWN per-tier
    (eta, resid_std) pair from one randomly chosen historical cycle, held
    fixed for that whole path -- not eta and resid_std from independently
    chosen cycles, which would break the real within-cycle relationship
    between how opponents reacted and how noisy that reaction was that
    year. This is the empirical-bootstrap option (assumption-light: no
    parametric shape imposed on a distribution whose real data include a
    sign flip -- Toss-Up went to -0.22 in 2016) recommended over a fitted
    Normal or a continuous stochastic process, per docs/theta_followup_plan.md
    Section 6."""
    per_cycle_fits = {c: fit_eta_and_resid(c) for c in cycles}

    eta_by_tier_cycle: dict[str, list[float]] = {t: [] for t in TIERS}
    resid_by_tier_cycle: dict[str, list[float]] = {t: [] for t in TIERS}
    for c in cycles:
        eta_c, resid_c = per_cycle_fits[c]
        for t in TIERS:
            if t in eta_c:
                eta_by_tier_cycle[t].append(eta_c[t])
                resid_by_tier_cycle[t].append(resid_c[t])

    n = len(tiers_per_race)
    eta_paths = np.zeros((k_paths, n))
    resid_paths = np.zeros((k_paths, n))
    summary = {}
    for t in TIERS:
        idx = [i for i, race_tier in enumerate(tiers_per_race) if race_tier == t]
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
        summary[t] = {
            "n_cycles_available": int(len(available_eta)),
            "historical_values": [float(v) for v in available_eta],
            "path_draw_mean": float(eta_draw.mean()), "path_draw_sd": float(eta_draw.std()),
        }
    return eta_paths, resid_paths, summary


def margin_gradient(coef, pvi, incumb_status, d_total, r_total, eta: float = 0.0) -> float:
    """d(mu_i)/d(D_i), Part I Section I.5's chain rule.

    eta=0 (default): holds R fixed, c*(1/D - 1/T) [algebraically c*R/(D*T)].
    eta>0: R reacts dollar-for-dollar at rate eta to new D spend (dT/dD=1+eta),
    matching allocator.py's _msg_vec eta-adjusted gradient (docs/theta_followup_plan.md
    Section 0.1.2) -- the entire deploy-branch increment is "new" spend from a
    d_total floor baseline, so unlike _reactive_r's party_obs threshold there is
    no "already-observed" spend to gate the reaction on here.
    """
    c = coef.beta1 + coef.beta2 * abs(pvi) + coef.beta3 * (1.0 if incumb_status == "Incumbent" else 0.0)
    if incumb_status == "Open" and coef.beta1_open is not None:
        c = coef.beta1_open + coef.beta2 * abs(pvi)
    d = max(d_total, 1.0)
    t = max(d_total + r_total, 1.0)
    return c * (1.0 / d - (1.0 + eta) / t)


def mu_struct_at(coef, pvi_arr, is_incumb_arr, gb_national, d_arr, r_arr, is_open_arr=None):
    """Recompute the structural mu (no epsilon) at an arbitrary (d, r) pair.

    Module-level twin of solve_bellman_lsm_continuous_phi.py's _mu_struct()
    (that file's own docstring calls the two "identical"; TestMuStructConsistency
    in tests/test_bellman_lsm.py checks this against the real code, not just
    the comment). Kept as an intentionally separate copy rather than an
    import, since solve_bellman_lsm_continuous_phi.py already imports this
    module as `lsm` -- importing back would be circular. One real, pre-existing
    numerical difference between the two copies: this version floors t_arr at
    1.0 (np.maximum(d_arr + r_arr, 1.0)), inherited unchanged from this
    module's own former closure, while the cphi copy does not -- immaterial at
    real 2026 dollar spend levels (d+r is never remotely close to $1), so
    preserved as-is here rather than silently "fixed" as part of an unrelated
    refactor.

    Extracted (2026-08) from run_lsm()'s former closure, where it was
    `_mu_struct_at(d, r)`, capturing coef/pvi_arr/is_incumb_arr/gb_national
    from the enclosing scope -- this module-level version takes them as
    explicit arguments so deploy_value() below can be unit-tested directly.
    """
    if is_open_arr is not None and coef.beta1_open is not None:
        beta1_eff = np.where(np.asarray(is_open_arr) > 0, coef.beta1_open, coef.beta1)
    else:
        beta1_eff = coef.beta1
    t_arr = np.maximum(d_arr + r_arr, 1.0)
    ratio = np.clip(d_arr / t_arr, 1e-6, 1 - 1e-6)
    log_ratio = np.log(ratio)
    c_arr = beta1_eff + coef.beta2 * np.abs(pvi_arr) + coef.beta3 * is_incumb_arr
    return (coef.alpha0 + coef.alpha1 * pvi_arr + coef.alpha2 * is_incumb_arr
            + coef.alpha3 * gb_national + c_arr * log_ratio)


def deploy_value(mu_t, r_t, widened_sigma, eta_arr_k, d_t, d_terminal, *,
                  races, coef, sigma_model, F0, n,
                  is_open_arr, pvi_arr, incumb_arr, is_incumb_arr, sigma_arr, gb_national,
                  use_nonlinear_allocator: bool = False, use_surrogate_allocator: bool = False,
                  use_robust_allocator: bool = False, eta_uncertainty_by_tier: dict | None = None):
    """Close the reserve now via one of four allocator branches, apply the
    resulting Delta_mu, then evaluate expected seats against widened_sigma
    (sigma_i, or sqrt(sigma_i^2+V_i(t)) if time remains). Shared by the
    terminal condition and every backward step so both use identical
    mechanics -- this is the fix for a bug this session found via a smoke
    test: the terminal value must ALSO deploy, or it is not a valid anchor
    for the recursion.

    grad (LP branch only) uses eta_arr_k (Section 0.1.2), the (n,) eta slice
    for THIS path k -- previously a single shared eta_arr, now indexed per
    path so a bootstrap-drawn per-path eta (Section 6) discounts the LP's
    linearized gradient exactly like a single-cycle bracket's shared value
    did, just varying path to path instead of being identical across all K
    paths.

    d_t is the CURRENT (trickled) candidate floor at this period/path
    (Section 0.1.1's fix), not the period-0 floor_arr unconditionally --
    deploying at period t adds discretionary spend on top of however much
    the candidate's own committee has already spent by t, consistent with
    mu_t (passed in) already reflecting that same trickled d_t structurally.

    d_terminal / trickle-drift correction (found and fixed the same session
    the trickle mechanism was added, before trusting any reported Theta):
    the widened_sigma convolution
    (E[Phi((mu+xi)/sigma)]=Phi(mu/sqrt(sigma^2+V)) for xi~N(0,V)) is only
    valid when the future movement being integrated over is MEAN-ZERO --
    true of idiosyncratic epsilon (by construction) and, pre-fix, true of D
    itself (D_i,t was perfectly fixed, so there was no future D movement to
    have a mean at all). Now that D grows via a real, deterministic
    (non-zero-mean) trickle, evaluating the deploy branch at mu_t (today's
    mu) plus only the DCCC's own delta_mu, widened by idiosyncratic sigma
    alone, silently omits the DETERMINISTIC mu appreciation the candidate's
    own future organic spending will produce between now and Election Day
    regardless of today's decision -- while the recursion's "wait"
    alternative automatically picks this up, because it is fit against real
    future mu_paths that already reflect the grown D. Leaving this
    uncorrected would have made "wait" look favored for a reason having
    nothing to do with genuine option value (it would just be capturing
    organic growth the deploy branch's shortcut failed to credit itself
    with). The fix: recompute the structural mu at the fully-trickled
    terminal floor d_terminal (known in advance -- trickle is deterministic)
    and an expected terminal R (r_t plus eta's deterministic reaction to the
    D_terminal-D_t gap; the mean-zero residual noise component is correctly
    left to widened_sigma, unchanged), and add the resulting shift on top of
    mu_t + delta_mu before convolving.

    Extracted (2026-08) from run_lsm()'s former closure -- see
    tests/test_bellman_lsm.py's TestDeployValueBranches, which previously
    could only exercise this indirectly through a full run_lsm() run
    (the test file's own scope note called this out as a known gap)."""
    r_terminal_expected = np.maximum(r_t + eta_arr_k * (d_terminal - d_t), 1.0)
    trickle_drift = (
        mu_struct_at(coef, pvi_arr, is_incumb_arr, gb_national, d_terminal, r_terminal_expected, is_open_arr)
        - mu_struct_at(coef, pvi_arr, is_incumb_arr, gb_national, d_t, r_t, is_open_arr)
    )

    if use_surrogate_allocator:
        # Item (5) of Section 8.9's investigation plan: a validated,
        # LP-speed (~0.025s/call vs. optimize_nonlinear()'s 40s-3,600s)
        # surrogate that still respects diminishing returns, unlike the
        # LP allocator. Validated (scripts/theta_concave_surrogate.py)
        # against optimize_nonlinear() at 4 representative states before
        # being used here: within 0.11-0.19 expected seats of the true
        # optimum out of ~235-240 (i.e. >99.9% of optimal value
        # captured), at roughly 2,000-2,700x the speed. Exploits
        # _reactive_r()'s separability (R_i depends only on race i's
        # own party spend) to solve the piecewise-linear-concave
        # relaxation EXACTLY via a greedy water-filling sort, not an
        # iterative solve -- this is what makes it fast.
        races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                        d_total=float(d_t[i]))
                   for i, r in enumerate(races)]
        party, arrays = surrogate_allocate(races_t, coef, sigma_model, F0, 0.15, eta_arr_k)
        d = np.maximum(arrays["floors"] + party, 1.0)
        r = _reactive_r(party, arrays)
        t_ = d + r
        ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
        log_ratio = np.log(ratio)
        log_total_pv = np.log(t_ / arrays["cvap"])
        mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
        mu_capped, _ = _apply_ceiling(mu_raw, arrays)
        deployed_mu = mu_capped + trickle_drift
    elif use_nonlinear_allocator:
        # Option B: the allocation AND the resulting mu both come from
        # the true, diminishing-returns-respecting objective -- reusing
        # the allocator's own internal ceiling math directly (rather
        # than re-deriving it) is what Option A's smoke test showed is
        # required; a naive margin_gradient()-based delta_mu, applied to
        # this allocator's (much larger, since it isn't knapsack-
        # constrained by a linear objective) chosen allocations, bypasses
        # the persuasion ceiling and inflates deploy value enormously.
        #
        # races carries each race's ORIGINAL, t=0 cand_d_total/r_total --
        # optimize_nonlinear() reads its per-race floor and R directly
        # from those fields (it takes no separate floor argument), so at
        # any period after t=0, d_t/r_t (this period's TRICKLED,
        # simulated state) must be baked into fresh RaceRecord copies
        # first, exactly the established pattern dynamic/ledger.py's
        # apply_to_races() already uses for the same reason -- passing
        # the stale, unmodified `races` here would optimize against the
        # wrong (t=0) state for every period but the first.
        # d_total is ALSO overridden (equal to cand_d_total=d_t), not just
        # r_total/cand_d_total: _precompute_race_arrays derives party_obs
        # (the "already observed" party-spend baseline _reactive_r's eta
        # threshold is measured against) from race.d_total. Leaving it at
        # its real, historical 2026 value -- unrelated to the simulated
        # d_t -- would give _reactive_r a stale, wrong threshold; setting
        # d_total=d_t makes party_obs=0, matching the LP branch's own
        # d_total_obs=d_t (zero party spend "already observed" relative
        # to this period's own trickled baseline).
        races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                        d_total=float(d_t[i]))
                   for i, r in enumerate(races)]
        res = optimize_nonlinear(races_t, coef, sigma_model, budget=F0, cov_matrix=np.eye(n) * 1e-6,
                                  gamma=0.0, cap_fraction=0.15, party_budget=F0, eta=eta_arr_k)
        arrays = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr_k)
        party = np.maximum(res.allocations - d_t, 0.0)
        d = np.maximum(arrays["floors"] + party, 1.0)
        r = _reactive_r(party, arrays)
        t_ = d + r
        ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
        log_ratio = np.log(ratio)
        log_total_pv = np.log(t_ / arrays["cvap"])
        mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
        mu_capped, _ = _apply_ceiling(mu_raw, arrays)
        deployed_mu = mu_capped + trickle_drift
    elif use_robust_allocator:
        # docs/theta_followup_plan.md Section 6's robust/max-min-over-eta
        # optimization: the ALLOCATION is chosen to hedge against each
        # race's own worst-case eta (per-tier p95, robust.py's monotonicity
        # reduction -- max_D min_eta collapses to a single optimize_nonlinear
        # call at eta_high, verified post-hoc, not just assumed), but the
        # resulting mu is then EVALUATED at eta_arr_k, this path's own
        # actually-realized eta -- the Bellman recursion needs "what
        # happens on this simulated path given the chosen action," not the
        # hypothetical worst case the allocation was chosen to hedge
        # against. Same races_t construction as the nonlinear branch, for
        # the same reason (this period's trickled state must be baked in).
        if eta_uncertainty_by_tier is None:
            raise ValueError("eta_uncertainty_by_tier is required when use_robust_allocator=True")
        races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                        d_total=float(d_t[i]))
                   for i, r in enumerate(races)]
        res = optimize_nonlinear_robust(races_t, coef, sigma_model, budget=F0,
                                         cov_matrix=np.eye(n) * 1e-6, gamma=0.0, cap_fraction=0.15,
                                         eta_uncertainty_by_tier=eta_uncertainty_by_tier, party_budget=F0)
        arrays = _precompute_race_arrays(races_t, coef, sigma_model, eta=eta_arr_k)
        party = np.maximum(res.allocations - d_t, 0.0)
        d = np.maximum(arrays["floors"] + party, 1.0)
        r = _reactive_r(party, arrays)
        t_ = d + r
        ratio = np.clip(d / t_, 1e-15, 1 - 1e-15)
        log_ratio = np.log(ratio)
        log_total_pv = np.log(t_ / arrays["cvap"])
        mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
        mu_capped, _ = _apply_ceiling(mu_raw, arrays)
        deployed_mu = mu_capped + trickle_drift
    else:
        p_win0 = norm.cdf(mu_t / sigma_arr)
        phi0 = norm.pdf(mu_t / sigma_arr)
        grad = np.array([margin_gradient(coef, pvi_arr[i], incumb_arr[i], d_t[i], r_t[i], eta_arr_k[i])
                          for i in range(n)])
        msg = phi0 / sigma_arr * grad
        outs = [ModelOutputs(district_id=races[i].district_id, ratio=d_t[i] / (d_t[i] + r_t[i]),
                              mu_hat=mu_t[i], sigma_i=sigma_arr[i], p_win=p_win0[i], msg_i=msg[i])
                for i in range(n)]
        res = optimize(outs, budget=F0, cov_matrix=np.eye(n) * 1e-6,
                        gamma=0.0, cap_fraction=0.15, floor_allocations=d_t, party_budget=F0,
                        d_total_obs=d_t)
        delta_s = np.maximum(res.allocations - d_t, 0.0)
        delta_mu = grad * delta_s
        deployed_mu = mu_t + delta_mu + trickle_drift

    return norm.cdf(deployed_mu / widened_sigma).sum()


def run_lsm(eta_arr_by_path: np.ndarray, resid_std_arr_by_path: np.ndarray, label: str,
            eta_summary: dict | None = None, enable_trickle: bool = True,
            enable_stochastic: bool = True, enable_opponent_reaction: bool = True,
            held_out_frac: float = 0.0, use_nonlinear_allocator: bool = False,
            use_surrogate_allocator: bool = False, use_robust_allocator: bool = False,
            eta_uncertainty_by_tier: dict | None = None,
            return_period0_action: bool = False) -> dict:
    """eta_arr_by_path / resid_std_arr_by_path: shape (K_PATHS, n) -- either
    a single cycle's fit tiled identically across every path (tile_single_cycle,
    the original eta_fit_2022/eta_fit_2024 brackets) or a genuine per-path
    bootstrap draw (bootstrap_eta_resid_paths). run_lsm() itself is agnostic
    to which; unifying the two here (rather than a separate code path per
    scenario) is what makes the bootstrap scenario a small addition instead
    of a duplicated ~150-line function.

    Mechanism-decomposition toggles (reviewer-requested isolation of Theta's
    drivers -- Paper III revision, 2026-07-28): enable_trickle controls
    candidate-committee organic spending growth (D_i,t fixed at the floor
    when False); enable_stochastic controls ALL random sources at once
    (idiosyncratic epsilon, the G_t national-environment walk, and R_i,t's
    residual reaction noise -- forced to exactly zero when False, giving a
    fully deterministic path); enable_opponent_reaction zeroes eta so R never
    reacts to D's growth (moot when enable_trickle is also False, since
    reaction = eta * delta_d = 0 regardless of eta once delta_d = 0). All
    default True, matching the full reported model (scenario E).

    held_out_frac (statistical-rigor addition, 2026-07-28): if > 0, a random
    held_out_frac share of the K_PATHS paths is excluded from every period's
    continuation-value regression fit (still receives a predicted wait_val
    from that fit, and still participates in the backward recursion), so the
    reported theta/frac_deploy_now can be computed on paths whose own
    realized future payoff never informed the regression that decided their
    stopping choice -- the standard fix for Longstaff-Schwartz's in-sample
    look-ahead bias. When 0 (default), behavior is unchanged: every path is
    used both to fit and to evaluate, exactly as before this addition.

    use_nonlinear_allocator (LP-vs-nonlinear reduced-scope comparison,
    "Option B" -- external review, 2026-07-29): if True, EVERY period's
    deploy branch (not just t=0, unlike the cheaper "Option A" one-time
    comparison in scripts/theta_lp_vs_nonlinear_deploy_branch.py) uses
    optimize_nonlinear() instead of the fast LP allocator, scored via the
    SAME ceiling-respecting mu computation Option A required after its own
    smoke test caught a naive linear-gradient shortcut inflating deploy
    value by ~46 seats out of 434 (bypassing the persuasion ceiling
    entirely). Default False reproduces the existing LP-based behavior
    exactly -- this flag changes nothing about any already-reported figure
    unless explicitly set. optimize_nonlinear() is far slower per call
    (tens of seconds to, occasionally, over an hour, vs. ~11ms for the LP),
    so this is only ever run at a drastically reduced K_PATHS."""
    coef, sigma_model = load_coef_and_sigma()
    races = build_universe(cycle=2026)
    n = len(races)
    outputs0 = compute_outputs_batch(races, coef, sigma_model)
    sigma_arr = np.array([o.sigma_i for o in outputs0])
    pvi_arr = np.array([r.pvi for r in races])
    incumb_arr = [r.incumb_status for r in races]
    floor_arr = np.array([r.cand_d_total for r in races])
    r0_arr = np.array([r.r_total for r in races])
    tiers = [r.cook_rating for r in races]
    is_comp = np.array([t in COMPETITIVE for t in tiers])
    gb_national = races[0].generic_ballot
    is_incumb_arr = np.array([1.0 if s == "Incumbent" else 0.0 for s in incumb_arr])
    # beta1_eff_arr: per-race spending elasticity, substituting beta1_open for
    # Open-seat races -- matching margin_gradient()'s already-correct branch and
    # win_prob.predict()'s static-pipeline behavior. mu_struct below previously
    # used coef.beta1 unconditionally, which meant the LEVEL of mu for an
    # Open-seat race used a different elasticity than the GRADIENT
    # margin_gradient() computed for that same race (found while writing
    # tests/test_bellman_lsm.py; fixed here).
    # is_open_arr is hoisted out of the branch below (unlike before the
    # deploy_value() extraction) so it's always defined for the module-level
    # mu_struct_at()/deploy_value() calls further down, not just when
    # coef.beta1_open is set -- mu_struct_at() itself no-ops on it when
    # coef.beta1_open is None, so this changes no behavior.
    is_open_arr = np.array([1.0 if s == "Open" else 0.0 for s in incumb_arr])
    if coef.beta1_open is not None:
        beta1_eff_arr = np.where(is_open_arr > 0, coef.beta1_open, coef.beta1)
    else:
        beta1_eff_arr = np.full(n, coef.beta1)
    resid_std_arr = resid_std_arr_by_path if enable_stochastic else np.zeros_like(resid_std_arr_by_path)
    eta_arr = eta_arr_by_path if enable_opponent_reaction else np.zeros_like(eta_arr_by_path)

    # --- Simulate the "wait" branch forward (Section 0.1.1's fix, unblocked this
    # session by data_catalog.md Section 2.7's dated candidate-periodic-reports
    # panel): D_i,t now grows via a real, calibrated non-discretionary spending
    # trickle (scripts/estimate_candidate_spend_trickle.py -- candidate-committee
    # disbursement growth that happens regardless of any DCCC deployment
    # decision), and R_i,t reacts to that trickle via eta_arr, on top of the
    # residual noise the pre-fix model already had. Deterministic trickle (same
    # $/day for every path, per race, at this pass's tier-median rate) --
    # a stochastic trickle is a natural extension, not attempted here, since
    # the calibration itself (estimate_candidate_spend_trickle.py) reports a
    # point rate per tier, not a distribution, unlike eta's bootstrap treatment.
    #
    # SCOPE BOUNDARY, stated explicitly rather than left implicit: eta_by_tier
    # (fit_eta_and_resid, above) was estimated from IE-to-IE reaction only
    # (Paper III Section 4.1's stated scope -- opponent IE spend reacting to
    # this side's IE spend). Applying that same eta here, to R_i,t's reaction
    # to a candidate-COMMITTEE spending trickle rather than an IE increment,
    # is a new, untested application of an existing estimate -- an assumption
    # of the same kind Section 5.5 already makes explicit for alpha3 (never
    # re-estimated for the estimand it's now applied to). This is not
    # re-validated by simulate_and_validate.py's Section 7.1 self-consistency
    # gate, which checks eta recovery against the same IE-to-IE mechanism it
    # was fit on (Check B) -- not this candidate-spend-to-IE mechanism, which
    # would require a separate regression (opponent IE reaction to candidate-
    # committee spend specifically) not yet run. Reported here rather than
    # silently assumed validated.
    trickle_per_day = load_trickle_rate_per_day(tiers) if enable_trickle else np.zeros(n)   # (n,) $/day
    trickle_per_period = trickle_per_day * PERIOD_DAYS           # (n,) $/period

    d_paths = np.zeros((K_PATHS, N_PERIODS + 1, n))
    d_paths[:, 0, :] = floor_arr[None, :]
    r_paths = np.zeros((K_PATHS, N_PERIODS + 1, n))
    r_paths[:, 0, :] = r0_arr
    for tstep in range(N_PERIODS):
        d_paths[:, tstep + 1, :] = d_paths[:, tstep, :] + trickle_per_period[None, :]
        delta_d = d_paths[:, tstep + 1, :] - d_paths[:, tstep, :]   # (K_PATHS, n)
        reaction = eta_arr * delta_d   # eta discounts/amplifies R's reaction to the trickle, per path/race
        r_paths[:, tstep + 1, :] = (
            r_paths[:, tstep, :] + reaction + RNG.normal(0, resid_std_arr, size=(K_PATHS, n))
        )
    r_paths = np.maximum(r_paths, 1.0)

    # --- Simulate G_t (Section 0.1.3): standalone zero-drift random walk, matching
    # simulate_and_validate.py's construction. NOT fed into mu_i's structural
    # formula -- alpha3 was estimated entirely from between-cycle variation
    # (paper3_draft.md Section 5.5's scope boundary) -- only tracked as a state
    # variable and added below as a fifth continuation-value regression feature,
    # so the LSM step can pick up a G_t-dependent effect empirically if one
    # exists, without applying alpha3 to an estimand it was never fit against.
    # scripts/estimate_gb_ou_drift.py fit an OU-with-drift model on the pooled
    # 4-cycle series and found the drift term statistically indistinguishable
    # from zero (p=0.37) and numerically negligible over the ~110 remaining days
    # to Election Day (implied E[delta_G] = -0.02 points, vs sigma_G~2 points at
    # that horizon) -- consistent with Section 5.3's finding that RW is a good
    # approximation at this horizon, so a zero-drift walk is used here.
    g_step_std = (SIGMA_G_PER_SQRT_DAY * np.sqrt(PERIOD_DAYS)) if enable_stochastic else 0.0
    g_paths = np.cumsum(RNG.normal(0, g_step_std, size=(K_PATHS, N_PERIODS)), axis=1)
    g_paths = np.concatenate([np.zeros((K_PATHS, 1)), g_paths], axis=1)   # G_0 = 0 (relative to today)

    eps_cum = np.zeros((K_PATHS, N_PERIODS + 1, n))
    for i in range(n):
        v = incremental_variances(sigma_arr[i], N_PERIODS) if enable_stochastic else np.zeros(N_PERIODS)
        incr = RNG.normal(0, np.sqrt(v), size=(K_PATHS, N_PERIODS))
        eps_cum[:, 1:, i] = np.cumsum(incr, axis=1)

    # mu_i,t = structural(trickled D_t, simulated R_t, static GB) + accumulated epsilon
    mu_paths = np.zeros((K_PATHS, N_PERIODS + 1, n))
    for tstep in range(N_PERIODS + 1):
        d_t = d_paths[:, tstep, :]
        t_t = d_t + r_paths[:, tstep, :]
        ratio = np.clip(d_t / t_t, 1e-6, 1 - 1e-6)
        log_ratio = np.log(ratio)
        c_arr = beta1_eff_arr[None, :] + coef.beta2 * np.abs(pvi_arr)[None, :] + coef.beta3 * is_incumb_arr[None, :]
        mu_struct = (coef.alpha0 + coef.alpha1 * pvi_arr[None, :] + coef.alpha2 * is_incumb_arr[None, :]
                     + coef.alpha3 * gb_national + c_arr * log_ratio)
        mu_paths[:, tstep, :] = mu_struct + eps_cum[:, tstep, :]

    # deploy_value_fn: this call's fixed context (races/coef/sigma_model/F0/n,
    # the per-race arrays, and this run's allocator-choice flags) bound via
    # functools.partial, so every per-path/per-period call site below only
    # has to pass the state that actually varies (mu_t, r_t, widened_sigma,
    # eta_arr_k, d_t, d_terminal) -- deploy_value() itself is now the
    # module-level function defined above run_lsm(), extracted from what
    # used to be this function's own closure (`_deploy_value`) so it can be
    # unit-tested directly (see tests/test_bellman_lsm.py's
    # TestDeployValueBranches).
    deploy_value_fn = functools.partial(
        deploy_value,
        races=races, coef=coef, sigma_model=sigma_model, F0=F0, n=n,
        is_open_arr=is_open_arr, pvi_arr=pvi_arr, incumb_arr=incumb_arr,
        is_incumb_arr=is_incumb_arr, sigma_arr=sigma_arr, gb_national=gb_national,
        use_nonlinear_allocator=use_nonlinear_allocator,
        use_surrogate_allocator=use_surrogate_allocator,
        use_robust_allocator=use_robust_allocator,
        eta_uncertainty_by_tier=eta_uncertainty_by_tier,
    )

    # --- Backward induction ---
    remaining_days = np.array([(N_PERIODS - t) * PERIOD_DAYS for t in range(N_PERIODS + 1)])

    # Terminal boundary, made fully consistent with the intermediate-period
    # fix (2026-07-28 audit, extended after external review caught the
    # remaining inconsistency): mu_paths[:, -1, :] already IS the fully
    # resolved simulated margin at T (eps_cum has accumulated its complete
    # budget by construction -- Appendix B.2's telescoping identity). There
    # is no separate, additional sigma_i-scale noise left to apply on top of
    # that -- doing so was the same double-count as the intermediate-period
    # bug, just at the one period where it happens to leave Theta(T)=0
    # unaffected (both branches are still evaluated identically there) while
    # still distorting the terminal ANCHOR value the whole backward
    # induction is regressed against. remaining_variance(sigma, 0) is
    # exactly 0 for every race regardless of the mechanism-decomposition
    # toggles, so this is now the single formula used at every period,
    # unconditionally -- no enable_stochastic branch needed here at all.
    terminal_sigma = np.sqrt(np.maximum(remaining_variance(sigma_arr, 0.0), 1e-6))
    allocator_label = (
        "surrogate" if use_surrogate_allocator
        else "nonlinear" if use_nonlinear_allocator
        else "robust" if use_robust_allocator
        else "lp"
    )
    if allocator_label == "lp":
        print(f"  [{label}] ** allocator=lp -- this reproduces the SUPERSEDED, wrong-sign "
              f"calculation documented in paper/paper3_final.md's Allocator-Robustness Finding. "
              f"Pass use_surrogate_allocator=True (or use_nonlinear_allocator=True) for the "
              f"corrected result. **")
    else:
        print(f"  [{label}] allocator={allocator_label}")
    print(f"  [{label}] computing terminal condition (forced deploy, {K_PATHS} paths)...")
    V_star = np.array([
        deploy_value_fn(mu_paths[k, -1, :], r_paths[k, -1, :], terminal_sigma, eta_arr[k],
                         d_paths[k, -1, :], d_paths[k, -1, :])   # at T, d_t IS d_terminal: drift=0
        for k in range(K_PATHS)   # V=0 at T: no widening
    ])

    # Fixed, independent RNG for the train/test split -- deliberately NOT the
    # module-level RNG driving path simulation, so requesting held_out_frac>0
    # doesn't perturb the simulated paths themselves relative to a
    # held_out_frac=0 run (only which rows feed each period's regression fit
    # changes).
    held_out_mask = np.zeros(K_PATHS, dtype=bool)
    if held_out_frac > 0:
        split_rng = np.random.default_rng(999)
        held_out_mask[split_rng.choice(K_PATHS, size=int(K_PATHS * held_out_frac), replace=False)] = True
    train_mask = ~held_out_mask

    theta_by_period = []
    for tstep in range(N_PERIODS - 1, -1, -1):
        v_remaining = (remaining_variance(sigma_arr, remaining_days[tstep])
                       if enable_stochastic else np.zeros_like(sigma_arr))   # vectorized over races
        # Fix (2026-07-28 audit): mu_paths[:, tstep, :] already embeds eps_cum(tstep),
        # the resolved-to-date share of the same sigma_i^2 idiosyncratic budget
        # (incremental_variances telescopes so that Var(eps_cum(t)) + v_remaining(t)
        # is constant in t -- see simulate_and_validate.py's incremental_variances).
        # Widening by sqrt(sigma_i^2 + v_remaining) on top of that double-counts:
        # once via the simulated eps_cum realization, once via the extra + sigma_i^2
        # term. The correct remaining uncertainty, given mu_t already reflects
        # everything resolved up to tstep, is v_remaining(t) alone. (The t=T terminal
        # condition below is untouched: it uses sigma_arr directly, matching Appendix
        # C.1's stated boundary Phi(mu_T/sigma_i), and Theta(T)=0 holds regardless
        # since both branches evaluate the identical mu_T through the same transform.)
        widened_sigma = np.sqrt(np.maximum(v_remaining, 1e-6))

        deploy_vals = np.array([
            deploy_value_fn(mu_paths[k, tstep, :], r_paths[k, tstep, :], widened_sigma,
                             eta_arr[k], d_paths[k, tstep, :], d_paths[k, -1, :])
            for k in range(K_PATHS)
        ])

        # Compressed basis, evaluated at period t (unwidened -- standard convention, matches Section 5.5's Validation A)
        p_win_t = norm.cdf(mu_paths[:, tstep, :] / sigma_arr[None, :])
        phi_t = norm.pdf(mu_paths[:, tstep, :] / sigma_arr[None, :])
        e_seats_t = p_win_t.sum(axis=1)
        var_seats_t = (p_win_t * (1 - p_win_t))[:, is_comp].sum(axis=1)   # independence approximation, stated explicitly
        max_msg_t = (phi_t / sigma_arr[None, :])[:, is_comp].max(axis=1)
        near_thresh_t = (np.abs(mu_paths[:, tstep, :][:, is_comp]) < NEAR_THRESHOLD_MARGIN_PP).sum(axis=1)
        g_t = g_paths[:, tstep]   # 5th feature (Section 0.1.3) -- G_t as a state descriptor only,
                                  # never fed into mu_i's structural formula (Section 5.5's scope boundary)

        # has_constant="add" (not the default "skip"): g_t is deterministically 0 for every
        # path at tstep=0 (G_0=0), which add_constant's default treats as an already-present
        # constant column and skips adding its own intercept -- silently shrinking X from 6
        # columns to 5 rather than raising, first caught via an IndexError on cont_fit.params[5].
        X = sm.add_constant(np.column_stack([e_seats_t, var_seats_t, max_msg_t, near_thresh_t, g_t]),
                             has_constant="add")
        # Fit only on train_mask rows when held_out_frac>0 (statistical-rigor
        # addition, above) -- predict() still applies to every row, so
        # held-out paths get a genuine out-of-sample continuation-value
        # estimate rather than one their own realized future contributed to.
        cont_fit = sm.OLS(V_star[train_mask], X[train_mask]).fit()
        wait_vals = cont_fit.predict(X)

        theta_t = wait_vals - deploy_vals
        deploy_now = deploy_vals >= wait_vals
        V_star = np.where(deploy_now, deploy_vals, wait_vals)
        if return_period0_action and tstep == 0:
            period0_action = deploy_now.copy()

        # rsquared = 1 - ssr/centered_tss is -inf, not the mathematically
        # sensible 0, when V_star has ~zero cross-path variance (centered_tss
        # underflows to exactly 0.0) -- a legitimate outcome, not a bug: the
        # allocator can be structurally deterministic enough (same top-MSG
        # race funded to cap regardless of path noise) that the continuation
        # value doesn't vary path-to-path at a given (t, small-universe) cell.
        _r2 = float(cont_fit.rsquared)
        basis_r2 = _r2 if np.isfinite(_r2) else 0.0

        entry = {
            "period": tstep, "days_remaining": int(remaining_days[tstep]),
            "mean_theta": float(np.mean(theta_t)), "frac_deploy_now": float(np.mean(deploy_now)),
            "basis_r2": basis_r2,
            "g_t_coef": float(cont_fit.params[5]), "g_t_pvalue": float(cont_fit.pvalues[5]),
        }
        if held_out_frac > 0:
            entry["mean_theta_held_out"] = float(np.mean(theta_t[held_out_mask]))
            entry["frac_deploy_now_held_out"] = float(np.mean(deploy_now[held_out_mask]))
        theta_by_period.append(entry)
        held_out_msg = (f", held-out Theta={entry['mean_theta_held_out']:+.4f}"
                         if held_out_frac > 0 else "")
        print(f"  [{label}] t={tstep} ({remaining_days[tstep]}d left): "
              f"mean Theta={np.mean(theta_t):+.4f} seats, frac(deploy now)={np.mean(deploy_now):.3f}, "
              f"basis R2={basis_r2:.3f}, g_t_coef={cont_fit.params[5]:+.5f} (p={cont_fit.pvalues[5]:.3f})"
              f"{held_out_msg}")

    theta_by_period = list(reversed(theta_by_period))
    out = {"label": label, "allocator": allocator_label, "eta_summary": eta_summary,
           "n_periods": N_PERIODS, "k_paths": K_PATHS, "theta_by_period": theta_by_period}
    if return_period0_action:
        out["period0_action_deploy_now"] = period0_action.tolist()
    return out


def main():
    # use_surrogate_allocator=True (fixed 2026-08): this entrypoint previously
    # called run_lsm() with neither allocator flag set, silently defaulting to
    # the LP allocator -- the SUPERSEDED, wrong-sign calculation per
    # paper/paper3_final.md's "Allocator-Robustness Finding" and
    # "Why the Corner Flipped Twice" sections. The validated fast surrogate
    # (concave_surrogate.py) is what scripts/theta_surrogate_headline.py used
    # for the paper's own "decisive re-solve" at full K=2000 -- this makes
    # the standard pipeline output (outputs/theta_schedule.json) match that
    # corrected result by default, instead of only a separately-named file
    # that most consumers of this script would never find.
    print(f"N_PERIODS={N_PERIODS} ({N_PERIODS*PERIOD_DAYS} days), K_PATHS={K_PATHS}, "
          f"allocator=surrogate\n")
    races = build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    results = {}
    for label, fit_cycle in [("eta_fit_2022", 2022), ("eta_fit_2024", 2024)]:
        print(f"=== {label} ===")
        eta_by_tier, resid_std_by_tier = fit_eta_and_resid(fit_cycle)
        print(f"  eta(tier): {eta_by_tier}")
        eta_arr_by_path, resid_std_arr_by_path = tile_single_cycle(
            eta_by_tier, resid_std_by_tier, tiers_per_race, K_PATHS)
        res = run_lsm(eta_arr_by_path, resid_std_arr_by_path, label,
                       eta_summary={"single_cycle_fit": eta_by_tier},
                       use_surrogate_allocator=True)
        results[label] = res

    print("=== eta_bootstrap_all_cycles ===")
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = bootstrap_eta_resid_paths(
        BOOTSTRAP_CYCLES, tiers_per_race, K_PATHS, RNG)
    for tier, s in boot_summary.items():
        print(f"  {tier}: {s['n_cycles_available']} historical cycles {s['historical_values']}, "
              f"path draws mean={s['path_draw_mean']:+.3f} sd={s['path_draw_sd']:.3f}")
    res = run_lsm(eta_arr_by_path, resid_std_arr_by_path, "eta_bootstrap_all_cycles",
                   eta_summary=boot_summary, use_surrogate_allocator=True)
    results["eta_bootstrap_all_cycles"] = res

    out_path = ROOT / "outputs/theta_schedule.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
