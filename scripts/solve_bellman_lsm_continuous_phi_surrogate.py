#!/usr/bin/env python3
"""
Surrogate-allocator re-run of the continuous deployment-fraction analysis
(paper3 next-step item 4). Table 13's LP-based continuous-phi result is
superseded by the same allocator-robustness finding that overturned the
binary Theta(0) result (Section 8.9). This script re-solves the continuous
framing under the validated concave-envelope surrogate.

THIS IS NOT A THIN ALLOCATOR SWAP, and an earlier version of this script
that tried to be one was wrong -- caught before trusting its output, by
exactly the mechanism-checking discipline this paper's central
methodological argument insists on. That first version monkeypatched only
solve_bellman_lsm_continuous_phi.py's `_solve_committed_floor` (the LP call
that determines the cumulative D-level after committing a grid-level
budget) and left everything downstream -- in particular the mu formula
used to SCORE the resulting allocation -- untouched. That downstream
formula, `_mu_struct`, deliberately omits the persuasion ceiling and the
alpha4*log(total_spend/cvap) term (see that script's own header comment,
"Departure from Section 1.3's literal convention"). Under the LP allocator
this was tolerated: the LP concentrates nearly all spending into ~7 races,
so the ceiling rarely binds and the omission barely moves the answer. The
surrogate spreads spending across 100-250 races instead, and the omission
stopped being harmless: a smoke test comparing the two allocators' RAW,
single-state objective values (scripts/theta_concave_surrogate.py's own
methodology, reused here) showed that scoring the surrogate's allocation
through the full ceiling+alpha4 formula gives a MONOTONICALLY INCREASING
value in the deployed budget (as any sane allocator must produce), while
scoring the identical allocations through the simplified `_mu_struct` used
to inflate values further still -- and, critically, when threaded through
this script's own multi-period regression/backward-induction machinery,
produced deploy_1.00 substantially BELOW deploy_0.00 (full hold), directly
contradicting the binary surrogate result (Table 13i: deploy favored, small
negative Theta) for the identical three calibration scenarios. That
contradiction was the tell that something was wrong, not a genuine finding
-- solve_bellman_lsm.py's own binary run_lsm() uses the FULL ceiling-
respecting formula (via `_apply_ceiling`, `arrays["alpha4"]`) for its deploy
branch specifically because an active DCCC spending decision needs it; only
the WAIT/organic-baseline trajectory legitimately uses the simpler formula
(no ceiling correction needed there since, at zero DCCC increment, party=0
reproduces mu_floor exactly by construction). The original (LP-based)
continuous-phi script applied the simple formula to every grid level,
including deployed ones -- a latent inconsistency with run_lsm's own
convention that the LP's degenerate concentration pattern happened to mask,
and that the surrogate's very different (spread-out) allocation pattern
exposed.

The fix, applied here: grid level g=0 (nothing committed) keeps the simple
formula (matches run_lsm's wait-branch convention exactly, and is a no-op
correction anyway since no DCCC increment exists to trigger the ceiling).
Every other grid level (any committed DCCC spend) is scored via the SAME
ceiling+alpha4-respecting formula run_lsm's surrogate deploy branch already
uses and that Table 13i was validated against -- computed directly from the
`arrays` dict scripts/concave_surrogate.py's `surrogate_allocate` already
returns, with no extra allocator calls needed. The deterministic trickle-
drift correction (full-deploy grid level only, matching solve_bellman_lsm.py
and the original continuous-phi script) is left on the simple formula,
exactly matching run_lsm's own precedent (`_mu_struct_at` there is also the
simple, non-ceiling helper).

This is therefore a fork of solve_bellman_lsm_continuous_phi.py's
run_continuous_phi_lsm(), not a monkeypatch: the simulation setup, the
regression-basis construction, and the backward induction over (t,
remaining-budget grid state) are unchanged; only the grid-level mu
computation block differs, in the way described above.

Output: outputs/theta_schedule_continuous_phi_surrogate_{scenario}_{n}pt.json
"""

from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

import solve_bellman_lsm as lsm
import solve_bellman_lsm_continuous_phi as cphi
from concave_surrogate import surrogate_allocate
from backtest.optimizer.allocator import _reactive_r, _apply_ceiling

ROOT = Path(__file__).parent.parent
_SIGMA_MODEL = None


def run_continuous_phi_lsm_surrogate(eta_arr_by_path: np.ndarray, resid_std_arr_by_path: np.ndarray,
                                      label: str, grid_fracs: list[float], k_paths: int,
                                      rng: np.random.Generator, eta_summary: dict | None = None) -> dict:
    n_periods = lsm.N_PERIODS
    n_grid = len(grid_fracs)
    (coef, races, n, sigma_arr, pvi_arr, incumb_arr, floor_arr, r0_arr,
     is_comp, gb_national, is_incumb_arr, is_open_arr) = cphi._setup_universe()

    eta_arr = eta_arr_by_path
    resid_std_arr = resid_std_arr_by_path

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

    # mu_baseline (g=0, nothing committed): the simple structural formula,
    # matching run_lsm's own wait-branch convention exactly -- no ceiling
    # correction needed since party=0 reproduces mu_floor by construction.
    mu_baseline = np.zeros((k_paths, n_periods + 1, n))
    for tstep in range(n_periods + 1):
        mu_baseline[:, tstep, :] = (
            cphi._mu_struct(coef, pvi_arr[None, :], is_incumb_arr[None, :], gb_national,
                             d_paths[:, tstep, :], r_paths[:, tstep, :], is_open_arr[None, :])
            + eps_cum[:, tstep, :]
        )

    def _mu_struct_simple(d, r):
        return cphi._mu_struct(coef, pvi_arr, is_incumb_arr, gb_national, d, r, is_open_arr)

    print(f"  [{label}] precomputing surrogate allocations for {n_grid - 1} nonzero grid levels "
          f"x {k_paths} paths x {n_periods + 1} periods ({(n_grid - 1) * k_paths * (n_periods + 1)} surrogate calls)...")
    mu_committed = [mu_baseline]  # g=0
    for g in range(1, n_grid):
        budget_g = grid_fracs[g] * lsm.F0
        is_full_deploy = (g == n_grid - 1)
        mu_g = np.zeros((k_paths, n_periods + 1, n))
        for tstep in range(n_periods + 1):
            for k in range(k_paths):
                d_t = d_paths[k, tstep, :]
                r_t = r_paths[k, tstep, :]
                eta_k = eta_arr[k]

                races_t = [dataclasses.replace(r, cand_d_total=float(d_t[i]), r_total=float(r_t[i]),
                                                d_total=float(d_t[i])) for i, r in enumerate(races)]
                party, arrays = surrogate_allocate(races_t, coef, _SIGMA_MODEL, budget_g, 0.15, eta_k)
                floor_g_kt = d_t + party
                d_full = np.maximum(arrays["floors"] + party, 1.0)
                r_full = _reactive_r(party, arrays)
                t_full = d_full + r_full
                log_ratio = np.log(np.clip(d_full / t_full, 1e-15, 1 - 1e-15))
                log_total_pv = np.log(t_full / arrays["cvap"])
                mu_raw = arrays["mu_const"] + arrays["c_spend"] * log_ratio + arrays["alpha4"] * log_total_pv
                mu_level, _ = _apply_ceiling(mu_raw, arrays)

                r_eff = r_t + eta_k * (floor_g_kt - d_t)

                # Trickle-drift baseline, fixed relative to the original (inherited)
                # continuous-phi script: that script computed this delta relative to
                # floor_g_kt/r_eff (the POST-deploy state), so the log-ratio term's
                # concavity made the same organic growth look smaller the more DCCC
                # money had already been committed -- a spurious budget-size-dependent
                # penalty on larger deployments having nothing to do with organic
                # growth itself. run_lsm's own _deploy_value (validated, Table 13i)
                # computes this delta relative to the ORIGINAL pre-deploy (d_t, r_t)
                # baseline and the natural organic-only trickle to T (d_paths[k,-1,:],
                # with no DCCC money in it at all) -- matched exactly here.
                if is_full_deploy and tstep < n_periods:
                    d_terminal_organic = d_paths[k, -1, :]
                    r_terminal_expected = np.maximum(r_t + eta_k * (d_terminal_organic - d_t), 1.0)
                    trickle_drift = _mu_struct_simple(d_terminal_organic, r_terminal_expected) - _mu_struct_simple(d_t, r_t)
                else:
                    trickle_drift = 0.0

                mu_g[k, tstep, :] = mu_level + eps_cum[k, tstep, :] + trickle_drift
        mu_committed.append(mu_g)
        print(f"  [{label}] grid level {grid_fracs[g]:.2f} done")

    # --- Backward induction over (t, remaining-budget grid state) --- (unchanged from original)
    remaining_days = np.array([(n_periods - t) * lsm.PERIOD_DAYS for t in range(n_periods + 1)])
    terminal_sigma = np.sqrt(np.maximum(lsm.remaining_variance(sigma_arr, 0.0), 1e-6))
    term_val = norm.cdf(mu_committed[-1][:, -1, :] / terminal_sigma).sum(axis=1)
    V_star_by_g = {g: term_val.copy() for g in range(n_grid)}

    schedule = []
    for tstep in range(n_periods - 1, -1, -1):
        v_remaining = lsm.remaining_variance(sigma_arr, remaining_days[tstep])
        widened_sigma = np.sqrt(np.maximum(v_remaining, 1e-6))
        absorbing_val = norm.cdf(mu_committed[-1][:, tstep, :] / widened_sigma).sum(axis=1)

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
            "schedule": schedule, "allocator": "surrogate_full_ceiling_formula"}


def main():
    global _SIGMA_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-paths", type=int, default=lsm.K_PATHS)
    ap.add_argument("--grid-points", type=int, default=5)
    ap.add_argument("--scenarios", nargs="+",
                    default=["eta_fit_2022", "eta_fit_2024", "eta_bootstrap_all_cycles"])
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    _coef, _SIGMA_MODEL = lsm.load_coef_and_sigma()

    grid_fracs = list(np.linspace(0.0, 1.0, args.grid_points))
    k_paths = args.k_paths
    print(f"[surrogate-v2, full ceiling formula] N_PERIODS={lsm.N_PERIODS} "
          f"({lsm.N_PERIODS * lsm.PERIOD_DAYS} days), K_PATHS={k_paths}, "
          f"grid={[f'{g:.2f}' for g in grid_fracs]}\n")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    results = {}
    for label in args.scenarios:
        rng = np.random.default_rng(args.seed)
        print(f"=== {label} ({args.grid_points}pt grid, surrogate allocator, full formula) ===")
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

        res = run_continuous_phi_lsm_surrogate(eta_arr_by_path, resid_std_arr_by_path, label,
                                                grid_fracs, k_paths, rng, eta_summary=eta_summary)
        results[label] = res

        out_path = ROOT / f"outputs/theta_schedule_continuous_phi_surrogate_{label}_{args.grid_points}pt.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  -> saved {out_path}\n")

    return results


if __name__ == "__main__":
    main()
