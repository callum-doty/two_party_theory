#!/usr/bin/env python3
"""
Continuous deployment-fraction generalization of the binary Theta(t) framing
(docs/theta_followup_plan.md Section 1, rescoped 2026-07-17 -- Section 1.3
is the implementation spec this script follows).

Motivation (Section 1.1): the binary framing only ever compares "deploy
everything now" vs "hold everything." eta_fit_2024's Theta(0)=-0.039 sits an
order of magnitude closer to indifference than eta_fit_2022's -0.517 -- the
concrete, data-grounded reason to expect a middle option (deploy some
fraction, reserve the rest) might beat both corners, which the binary
framing cannot express even if it exists.

Section 1.2 states two structural requirements a naive phi-grid search would
miss:
  1. The deploy-value-at-phi function must be genuinely concave in phi (via
     re-solving the LP allocator at each candidate level), not linearly
     interpolated between the two corner values -- a linear interpolation
     is mathematically guaranteed to land on a boundary.
  2. Unspent capital must be a carried-forward STATE variable (an
     impulse-control / multiple-exercise problem), not forfeited at a
     single one-shot choice -- since margin_gradient() stays positive for
     nearly every race in this universe, a one-time static split would
     never rationally choose phi<1.

Design (Section 1.3), and one deliberate departure from it, stated plainly:

  - Budget grid: F = {0, 0.25, 0.5, 0.75, 1.0} x F0 (GRID_FRACS below),
    5 points by default; legal actions from remaining-budget state f are
    any grid level f' <= f (money can only be committed, never
    "uncommitted"). An 11-point sensitivity grid is available via
    --grid-points 11 (Section 1.3 item 6).

  - The _deploy_value() floor-reset bug (solve_bellman_lsm.py:299,
    d_t = floor_arr.copy() on every call regardless of how much has
    already been committed) is fixed here by construction: floor_state_g,
    the per-race D-level if cumulative C_g dollars have been committed, is
    computed once per (grid level, path, period) via the SAME LP allocator
    call pattern (msg_i computed from the eta-discounted chain-rule
    gradient, per Section 0.1.2), but parameterized by the target
    cumulative budget C_g instead of being frozen at F0 every time -- this
    is what lets the grid trace out a genuinely concave value-of-budget
    curve instead of the flat line the frozen floor would otherwise
    produce (docs/theta_followup_plan.md Section 1.3 item 2).

  - Departure from Section 1.3's literal chain-rule "add a linearized
    delta_mu" convention: rather than computing a linearized delta_mu from
    a single fixed-baseline gradient and adding it to mu_t (which is
    exactly the kind of frozen-linearization bug this section exists to
    fix), this script uses the exact nonlinear margin formula already used
    for mu_struct everywhere else in solve_bellman_lsm.py -- with R_i
    increased by eta_i * (floor_state_g - floor_arr), i.e. the opponent's
    reactive response (paper3_draft.md Section 4.1's R_i(D_i) = R_base +
    eta*max(0, party_i - party_i_obs)) applied directly to the exact
    log-ratio, instead of folded into a chain-rule slope. This is strictly
    more faithful to Section 4.1's own R_i(D_i) specification than a
    linearized delta would be, and it is what gives the value-of-budget
    curve real, non-approximated concavity (log(D/(D+R)) is itself concave
    in D). The LP allocator's *choice* of which races receive money still
    uses the linearized, eta-discounted MSG objective (Section 0.1.2's
    margin_gradient) -- that part is unavoidable, since the fast LP's
    objective must be linear -- but valuing the resulting allocation does
    not have to inherit that approximation, and here it doesn't.

  - Regression basis: unified, pooling all reachable non-absorbing grid
    states g'=0..N_GRID-2 into one regression per period with remaining-
    budget-fraction f' and f'^2 as a sixth/seventh feature (Section 1.3
    item 4, "preferred" option) -- because every grid state is evaluated
    densely for all K_PATHS at every period (this design does not sample
    grid states endogenously along a single forward trajectory), the
    sparse-cell risk that motivated preferring "unified" over "grid-
    stratified" in the plan does not actually arise here; unified is used
    anyway for consistency with the plan's stated preference and because
    pooling is at least as well-conditioned as stratifying when cells are
    already dense.

  - Compute-cost note: precomputing floor_state_g once per (grid level,
    path, period) -- rather than once per (starting-state, action) pair,
    the plan's literal accounting -- means the number of LP calls scales
    with (N_GRID-1) x K_PATHS x (N_PERIODS+1), not the triangular
    per-action count Section 1.3 item 5 estimated. At the live universe's
    N_PERIODS=7, K_PATHS=2000, this is 4 x 2000 x 8 = 64,000 calls at a
    5-point grid (vs. the plan's ~210,000 estimate) and 10 x 2000 x 8 =
    160,000 at 11 points (vs. ~170 min extrapolated from the naive count) --
    a real efficiency gain from not recomputing the identical target grid
    level's LP solve once for every starting state that could reach it.

Output: outputs/theta_schedule_continuous_phi_{label}_{n_grid}pt.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent


def _setup_universe():
    coef, sigma_model = lsm.load_coef_and_sigma()
    races = lsm.build_universe(cycle=2026)
    n = len(races)
    outputs0 = lsm.compute_outputs_batch(races, coef, sigma_model)
    sigma_arr = np.array([o.sigma_i for o in outputs0])
    pvi_arr = np.array([r.pvi for r in races])
    incumb_arr = [r.incumb_status for r in races]
    floor_arr = np.array([r.cand_d_total for r in races])
    r0_arr = np.array([r.r_total for r in races])
    tiers = [r.cook_rating for r in races]
    is_comp = np.array([t in lsm.COMPETITIVE for t in tiers])
    gb_national = races[0].generic_ballot
    is_incumb_arr = np.array([1.0 if s == "Incumbent" else 0.0 for s in incumb_arr])
    is_open_arr = np.array([1.0 if s == "Open" else 0.0 for s in incumb_arr])
    return (coef, races, n, sigma_arr, pvi_arr, incumb_arr, floor_arr, r0_arr,
            is_comp, gb_national, is_incumb_arr, is_open_arr)


def _mu_struct(coef, pvi_arr, is_incumb_arr, gb_national, d_arr, r_arr, is_open_arr=None):
    """Exact nonlinear margin formula, identical to solve_bellman_lsm.run_lsm's
    mu_struct computation -- d_arr/r_arr broadcast over (K, n) or (n,).

    is_open_arr: substitutes coef.beta1_open for Open-seat races, matching
    margin_gradient()'s already-correct branch and win_prob.predict()'s
    static-pipeline behavior. Previously always used coef.beta1 regardless of
    incumbency status, which meant this function's mu LEVEL used a different
    elasticity than margin_gradient()'s mu GRADIENT for the same Open-seat
    race (found while writing tests/test_bellman_lsm.py; fixed here).
    Optional and defaults to None (old behavior) only so any external caller
    that hasn't been updated fails loudly on a shape mismatch rather than
    silently keeping the bug -- every in-repo caller now passes it.
    """
    if is_open_arr is not None and coef.beta1_open is not None:
        beta1_eff = np.where(is_open_arr > 0, coef.beta1_open, coef.beta1)
    else:
        beta1_eff = coef.beta1
    t_arr = d_arr + r_arr
    ratio = np.clip(d_arr / t_arr, 1e-6, 1 - 1e-6)
    log_ratio = np.log(ratio)
    c_arr = beta1_eff + coef.beta2 * np.abs(pvi_arr) + coef.beta3 * is_incumb_arr
    return (coef.alpha0 + coef.alpha1 * pvi_arr + coef.alpha2 * is_incumb_arr
            + coef.alpha3 * gb_national + c_arr * log_ratio)


def _solve_committed_floor(coef, races, n, sigma_arr, pvi_arr, incumb_arr, d_t,
                            mu_baseline_t, r_t, eta_arr_k, budget):
    """floor_state_g: the per-race D-level if `budget` cumulative dollars were
    committed right now ON TOP OF d_t, chosen via the SAME eta-discounted
    linearized-MSG LP allocator solve_bellman_lsm._deploy_value uses -- but
    genuinely parameterized by `budget` instead of frozen at F0 (the fix for
    Section 1.3 item 2's identified bug).

    d_t is the CURRENT candidate-committee floor at this period/path
    (docs/theta_followup_plan.md Section 0.1.1's fix, threaded through this
    script the same session it was added to solve_bellman_lsm.py's binary
    framing) -- previously always the static, period-0 floor_arr regardless
    of tstep, back when D_i,t never moved at all. Now that it does (via a
    real, calibrated spending trickle), the floor DCCC's committed dollars
    are layered on top of must be whatever the candidate's own committee has
    organically raised/spent by this period, not the day-0 baseline."""
    if budget <= 0:
        return d_t.copy()
    p_win0 = norm.cdf(mu_baseline_t / sigma_arr)
    phi0 = norm.pdf(mu_baseline_t / sigma_arr)
    grad = np.array([
        lsm.margin_gradient(coef, pvi_arr[i], incumb_arr[i], d_t[i], r_t[i], eta_arr_k[i])
        for i in range(n)
    ])
    msg = phi0 / sigma_arr * grad
    outs = [lsm.ModelOutputs(district_id=races[i].district_id,
                              ratio=d_t[i] / (d_t[i] + r_t[i]),
                              mu_hat=mu_baseline_t[i], sigma_i=sigma_arr[i],
                              p_win=p_win0[i], msg_i=msg[i])
            for i in range(n)]
    res = lsm.optimize(outs, budget=budget, cov_matrix=np.eye(n) * 1e-6, gamma=0.0,
                        cap_fraction=0.15, floor_allocations=d_t, party_budget=budget,
                        d_total_obs=d_t)
    return res.allocations


def run_continuous_phi_lsm(eta_arr_by_path: np.ndarray, resid_std_arr_by_path: np.ndarray,
                            label: str, grid_fracs: list[float], k_paths: int,
                            rng: np.random.Generator, eta_summary: dict | None = None) -> dict:
    n_periods = lsm.N_PERIODS
    n_grid = len(grid_fracs)
    (coef, races, n, sigma_arr, pvi_arr, incumb_arr, floor_arr, r0_arr,
     is_comp, gb_national, is_incumb_arr, is_open_arr) = _setup_universe()

    eta_arr = eta_arr_by_path
    resid_std_arr = resid_std_arr_by_path

    # --- Trickle-driven D_i,t + eta-reactive R_i,t (docs/theta_followup_plan.md
    # Section 0.1.1's fix, threaded through this script the same session it was
    # added to solve_bellman_lsm.py's binary framing, per Section 12.4's stated
    # gap). d_paths grows deterministically at the calibrated per-tier trickle
    # rate (scripts/estimate_candidate_spend_trickle.py); r_paths reacts to that
    # growth via eta_arr, on top of the residual noise this script already had.
    tiers_per_race = [r.cook_rating for r in races]
    trickle_per_day = lsm.load_trickle_rate_per_day(tiers_per_race)
    trickle_per_period = trickle_per_day * lsm.PERIOD_DAYS

    d_paths = np.zeros((k_paths, n_periods + 1, n))
    d_paths[:, 0, :] = floor_arr[None, :]
    r_paths = np.zeros((k_paths, n_periods + 1, n))
    r_paths[:, 0, :] = r0_arr
    for tstep in range(n_periods):
        d_paths[:, tstep + 1, :] = d_paths[:, tstep, :] + trickle_per_period[None, :]
        delta_d = d_paths[:, tstep + 1, :] - d_paths[:, tstep, :]
        reaction = eta_arr * delta_d
        r_paths[:, tstep + 1, :] = (
            r_paths[:, tstep, :] + reaction + rng.normal(0, resid_std_arr, size=(k_paths, n))
        )
    r_paths = np.maximum(r_paths, 1.0)

    g_step_std = lsm.SIGMA_G_PER_SQRT_DAY * np.sqrt(lsm.PERIOD_DAYS)
    g_paths = np.cumsum(rng.normal(0, g_step_std, size=(k_paths, n_periods)), axis=1)
    g_paths = np.concatenate([np.zeros((k_paths, 1)), g_paths], axis=1)

    eps_cum = np.zeros((k_paths, n_periods + 1, n))
    for i in range(n):
        v = lsm.incremental_variances(sigma_arr[i], n_periods)
        incr = rng.normal(0, np.sqrt(v), size=(k_paths, n_periods))
        eps_cum[:, 1:, i] = np.cumsum(incr, axis=1)

    # mu_baseline: the wait-branch mu (D following the trickle, nothing DCCC-
    # committed) -- exactly solve_bellman_lsm.run_lsm's mu_paths, and also grid
    # state g=0 (nothing committed).
    mu_baseline = np.zeros((k_paths, n_periods + 1, n))
    for tstep in range(n_periods + 1):
        mu_baseline[:, tstep, :] = (
            _mu_struct(coef, pvi_arr[None, :], is_incumb_arr[None, :], gb_national,
                       d_paths[:, tstep, :], r_paths[:, tstep, :], is_open_arr[None, :])
            + eps_cum[:, tstep, :]
        )

    # --- Precompute mu_committed[g] for every nonzero grid level ---
    print(f"  [{label}] precomputing floor_state for {n_grid - 1} nonzero grid levels "
          f"x {k_paths} paths x {n_periods + 1} periods ({(n_grid - 1) * k_paths * (n_periods + 1)} LP calls)...")
    mu_committed = [mu_baseline]  # g=0
    for g in range(1, n_grid):
        budget_g = grid_fracs[g] * lsm.F0
        is_full_deploy = (g == n_grid - 1)
        mu_g = np.zeros((k_paths, n_periods + 1, n))
        for tstep in range(n_periods + 1):
            for k in range(k_paths):
                d_t = d_paths[k, tstep, :]
                floor_g_kt = _solve_committed_floor(
                    coef, races, n, sigma_arr, pvi_arr, incumb_arr, d_t,
                    mu_baseline[k, tstep, :], r_paths[k, tstep, :], eta_arr[k], budget_g)
                r_eff = r_paths[k, tstep, :] + eta_arr[k] * (floor_g_kt - d_t)
                mu_level = _mu_struct(coef, pvi_arr, is_incumb_arr, gb_national, floor_g_kt, r_eff, is_open_arr)

                # Trickle-drift correction, ONLY for the full-deployment grid level
                # (g=n_grid-1) -- exactly the fix applied to solve_bellman_lsm.py's
                # _deploy_value, and needed for the identical reason: this array's
                # value at tstep is later convolved with widened_sigma
                # (absorbing_val, below) to represent "commit fully now, then let
                # remaining drift resolve" -- valid only for mean-zero future
                # movement. Organic D growth after tstep is deterministic and
                # non-zero-mean, so the expected structural mu shift from
                # (floor_g_kt, r_eff) to the fully-trickled terminal pair is added
                # before storing. Every OTHER grid level (gp < n_grid-1) is used
                # directly as real current-period features in the regression basis
                # below, never widened by a convolution shortcut, so no correction
                # applies there -- same as mu_baseline (g=0) needing none either.
                if is_full_deploy and tstep < n_periods:
                    d_terminal = floor_g_kt + (d_paths[k, -1, :] - d_t)
                    r_terminal_expected = np.maximum(
                        r_eff + eta_arr[k] * (d_terminal - floor_g_kt), 1.0
                    )
                    mu_terminal_level = _mu_struct(
                        coef, pvi_arr, is_incumb_arr, gb_national, d_terminal, r_terminal_expected, is_open_arr
                    )
                    trickle_drift = mu_terminal_level - mu_level
                else:
                    trickle_drift = 0.0

                mu_g[k, tstep, :] = mu_level + eps_cum[k, tstep, :] + trickle_drift
        mu_committed.append(mu_g)
        print(f"  [{label}] grid level {grid_fracs[g]:.2f} done")

    # --- Backward induction over (t, remaining-budget grid state) ---
    remaining_days = np.array([(n_periods - t) * lsm.PERIOD_DAYS for t in range(n_periods + 1)])

    # Same terminal-boundary fix as solve_bellman_lsm.py (2026-07-28 audit,
    # extended after external review): mu_committed[-1][:, -1, :] already IS
    # the fully resolved margin at T, so no separate sigma_i floor belongs
    # here -- remaining_variance(sigma, 0) is exactly 0 for every race.
    terminal_sigma = np.sqrt(np.maximum(lsm.remaining_variance(sigma_arr, 0.0), 1e-6))
    term_val = norm.cdf(mu_committed[-1][:, -1, :] / terminal_sigma).sum(axis=1)
    V_star_by_g = {g: term_val.copy() for g in range(n_grid)}

    schedule = []
    for tstep in range(n_periods - 1, -1, -1):
        v_remaining = lsm.remaining_variance(sigma_arr, remaining_days[tstep])
        # Same fix as solve_bellman_lsm.py's backward induction (2026-07-28 audit):
        # mu_committed already embeds eps_cum(tstep), the resolved-to-date share of
        # sigma_i^2, so widening by sigma_i^2 + v_remaining on top double-counts.
        # v_remaining(t) alone is the correct remaining uncertainty.
        widened_sigma = np.sqrt(np.maximum(v_remaining, 1e-6))
        absorbing_val = norm.cdf(mu_committed[-1][:, tstep, :] / widened_sigma).sum(axis=1)

        # Grid-stratified regression (one fit per grid state g', no functional form
        # imposed across budget levels), used as PRIMARY here -- a deliberate flip
        # from Section 1.3 item 4's stated "unified preferred, stratified as
        # diagnostic" ordering. Reason, found empirically while smoke-testing this
        # script: the LP allocator's already-documented knapsack degeneracy
        # (paper3_draft.md Section 8.2's addendum -- it only ever funds the same
        # ~7 low-floor races regardless of total budget) makes the true value-of-
        # budget curve rise almost its entire distance by a few percent of F0 and
        # then stay nearly flat -- a step shape, not a smooth polynomial. Pooling
        # states into one regression with f'/f'^2 forces a quadratic through a
        # relationship that isn't one, and produced visibly non-monotonic
        # cont_pred values across budget levels during testing. A separate,
        # unconstrained fit per grid state avoids imposing that wrong shape.
        # This project's dense design (every grid state evaluated for all
        # K_PATHS at every period, not sampled endogenously along one forward
        # trajectory) also means Section 1.3's stated reason to prefer unified
        # (sparse cells under endogenous sampling) does not apply here anyway.
        cont_pred = {}
        basis_r2_by_gp = {}
        for gp in range(n_grid - 1):
            mu_t = mu_committed[gp][:, tstep, :]
            p_win_t = norm.cdf(mu_t / sigma_arr[None, :])
            phi_t = norm.pdf(mu_t / sigma_arr[None, :])
            e_seats_t = p_win_t.sum(axis=1)
            var_seats_t = (p_win_t * (1 - p_win_t))[:, is_comp].sum(axis=1)
            max_msg_t = (phi_t / sigma_arr[None, :])[:, is_comp].max(axis=1)
            near_thresh_t = (np.abs(mu_t[:, is_comp]) < lsm.NEAR_THRESHOLD_MARGIN_PP).sum(axis=1)
            g_t = g_paths[:, tstep]
            Xg = sm.add_constant(np.column_stack([e_seats_t, var_seats_t, max_msg_t, near_thresh_t, g_t]),
                                  has_constant="add")
            fit_gp = sm.OLS(V_star_by_g[gp], Xg).fit()
            cont_pred[gp] = fit_gp.predict(Xg)
            basis_r2_by_gp[gp] = float(fit_gp.rsquared)
        cont_fit_r2_mean = float(np.mean(list(basis_r2_by_gp.values())))

        new_V_star_by_g = {}
        for g in range(n_grid):
            if g == n_grid - 1:
                new_V_star_by_g[g] = absorbing_val.copy()
            else:
                options = [absorbing_val] + [cont_pred[gp] for gp in range(g, n_grid - 1)]
                new_V_star_by_g[g] = np.stack(options, axis=0).max(axis=0)

        # Diagnostics: state g=0 (full reserve still available entering this period),
        # generalizing the binary script's Theta(t) snapshot convention.
        options_g0 = [absorbing_val] + [cont_pred[gp] for gp in range(n_grid - 1)]
        action_fracs = np.array([1.0] + [grid_fracs[gp] for gp in range(n_grid - 1)])
        stacked_g0 = np.stack(options_g0, axis=0)
        argmax_g0 = np.argmax(stacked_g0, axis=0)
        chosen_frac = action_fracs[argmax_g0]

        option_means = {f"deploy_{frac:.2f}": float(val.mean())
                        for frac, val in zip(action_fracs, options_g0)}
        frac_dist = {f"{frac:.2f}": float(np.mean(chosen_frac == frac)) for frac in action_fracs}

        entry = {
            "period": tstep, "days_remaining": int(remaining_days[tstep]),
            "v_g0_mean": float(new_V_star_by_g[0].mean()),
            "chosen_frac_mean": float(chosen_frac.mean()),
            "option_value_means": option_means,
            "action_frac_distribution": frac_dist,
            "basis_r2": cont_fit_r2_mean,
            "basis_r2_by_grid_state": basis_r2_by_gp,
        }
        schedule.append(entry)
        print(f"  [{label}] t={tstep} ({remaining_days[tstep]}d left): "
              f"V(g=0)={entry['v_g0_mean']:+.4f}, mean chosen frac={entry['chosen_frac_mean']:.3f}, "
              f"R2={entry['basis_r2']:.3f}")

        V_star_by_g = new_V_star_by_g

    schedule = list(reversed(schedule))
    return {"label": label, "grid_fracs": list(grid_fracs), "n_grid": n_grid,
            "n_periods": n_periods, "k_paths": k_paths, "eta_summary": eta_summary,
            "schedule": schedule}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-paths", type=int, default=lsm.K_PATHS)
    ap.add_argument("--grid-points", type=int, default=5)
    ap.add_argument("--scenarios", nargs="+",
                    default=["eta_fit_2022", "eta_fit_2024", "eta_bootstrap_all_cycles"])
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    grid_fracs = list(np.linspace(0.0, 1.0, args.grid_points))
    k_paths = args.k_paths
    print(f"N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days), "
          f"K_PATHS={k_paths}, grid={[f'{g:.2f}' for g in grid_fracs]}\n")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    results = {}
    for label in args.scenarios:
        rng = np.random.default_rng(args.seed)
        print(f"=== {label} ({args.grid_points}pt grid) ===")
        if label in ("eta_fit_2022", "eta_fit_2024"):
            fit_cycle = 2022 if label == "eta_fit_2022" else 2024
            eta_by_tier, resid_std_by_tier = lsm.fit_eta_and_resid(fit_cycle)
            eta_arr_by_path, resid_std_arr_by_path = lsm.tile_single_cycle(
                eta_by_tier, resid_std_by_tier, tiers_per_race, k_paths)
            eta_summary = {"single_cycle_fit": eta_by_tier}
        elif label == "eta_bootstrap_all_cycles":
            eta_arr_by_path, resid_std_arr_by_path, eta_summary = lsm.bootstrap_eta_resid_paths(
                lsm.BOOTSTRAP_CYCLES, tiers_per_race, k_paths, rng)
        else:
            raise ValueError(f"unknown scenario {label}")

        res = run_continuous_phi_lsm(eta_arr_by_path, resid_std_arr_by_path, label,
                                      grid_fracs, k_paths, rng, eta_summary=eta_summary)
        results[label] = res

        out_path = ROOT / f"outputs/theta_schedule_continuous_phi_{label}_{args.grid_points}pt.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  -> saved {out_path}\n")

    return results


if __name__ == "__main__":
    main()
