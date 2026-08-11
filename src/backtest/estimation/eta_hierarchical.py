"""
Hierarchical (partial-pooling) tier x cycle model for the opponent-reaction
coefficient eta -- docs/theta_followup_plan.md Section 6 explicitly named
this the deferred "right long-run answer" to eta's confirmed cycle-to-cycle
instability (5 of 7 tiers show statistically significant cycle-to-cycle
variation), calling the existing DerSimonian-Laird per-tier random-effects
model (scripts/build_eta_uncertainty_distribution.py) "a single-tier special
case" of the crossed model this module implements.

DESIGN CHANGE FROM THE ORIGINAL PLAN, stated explicitly: the plan called for
a PyMC model fit directly on the observation-level IE-spend delta panel
(eta as a random SLOPE varying by tier and cycle). PyMC could not be
installed in this environment -- its numba/llvmlite dependency needs a
matching-version LLVM toolchain (not just cmake) that isn't available here,
and the user chose (over attempting a multi-GB LLVM install with a real
risk of a version-mismatch failure anyway) to use statsmodels' MixedLM
instead, which is already a project dependency.

statsmodels.MixedLM does not cleanly support a nested/crossed RANDOM-SLOPE
structure the way PyMC would have. The design used here instead is a
standard two-stage ("estimate then meta-analyze") approach, a direct
generalization of the codebase's own existing DerSimonian-Laird per-tier
model to a genuine two-way (tier AND cycle) structure:

  Stage 1: fit eta_hat(tier, cycle) by simple OLS per (tier, cycle) cell
           (fit_per_cell_eta(), the same regression
           scripts/solve_bellman_lsm.py's fit_eta_and_resid() already runs,
           re-implemented here to also capture each cell's own standard
           error, which fit_eta_and_resid() does not return).
  Stage 2: partial-pool those per-cell point estimates via a crossed
           tier + cycle random-intercept model (fit_hierarchical_eta(),
           statsmodels MixedLM with groups=tier and a cycle variance
           component) -- eta_hat(tier,cycle) = mu_global + tier_effect(tier)
           + cycle_effect(cycle) + noise.

This is NOT the same model as a full observation-level Bayesian fit (it
does not propagate each cell's differing sample size as a Bayesian
likelihood would -- Stage 2 pools the point ESTIMATES, unweighted, treating
each surviving cell equally regardless of how many transactions it was
estimated from), a real simplification stated here rather than glossed
over. It still delivers the substantive goal: partial pooling across BOTH
tier and cycle (not just tier, as DerSimonian-Laird did), replacing the
per-cycle-per-tier point estimates bootstrap_eta_resid_paths() currently
draws from.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


def fit_per_cell_eta(cycles: list[int], build_period_panel, build_delta_panel, tiers: list[str],
                      min_obs: int = 10) -> pd.DataFrame:
    """Stage 1: one eta(tier, cycle) OLS fit per cell, WITH its standard
    error (fit_eta_and_resid() in scripts/solve_bellman_lsm.py runs the
    identical regression but only returns the point estimate and residual
    std, not the coefficient's own SE -- needed here for Stage 2's
    weighting). build_period_panel/build_delta_panel are passed in (from
    scripts/estimate_eta_reaction.py) rather than imported at module level,
    since scripts/ is not normally on src/'s import path -- callers
    (scripts/estimate_eta_hierarchical.py) pass the real functions in.

    Returns: cycle, tier, eta_hat, eta_se, resid_std, n_obs -- one row per
    (cycle, tier) cell with at least min_obs observations."""
    rows = []
    for cycle in cycles:
        panel = build_period_panel(cycle)
        delta = build_delta_panel(panel)
        for tier in tiers:
            mask = delta["tier"] == tier
            if mask.sum() < min_obs:
                continue
            X = sm.add_constant(delta.loc[mask, "d_ie_delta_lag_dm"])
            y = delta.loc[mask, "r_ie_delta_dm"]
            fit = sm.OLS(y, X).fit()
            rows.append(dict(
                cycle=cycle, tier=tier,
                eta_hat=float(fit.params.get("d_ie_delta_lag_dm", 0.0)),
                eta_se=float(fit.bse.get("d_ie_delta_lag_dm", float("nan"))),
                resid_std=float(fit.resid.std()),
                n_obs=int(mask.sum()),
            ))
    return pd.DataFrame(rows)


def fit_hierarchical_eta(per_cell: pd.DataFrame) -> dict:
    """Stage 2: crossed tier + cycle random-intercept model on the Stage-1
    point estimates, via statsmodels MixedLM.

    Both tier and cycle are fit as independent variance components under a
    single dummy top-level group (vc_formula={"tier": ..., "cycle": ...},
    groups=a constant) -- the standard statsmodels recipe for a genuinely
    CROSSED (not nested) structure, treating tier and cycle symmetrically.
    An earlier attempt using groups="tier" with cycle as a nested variance
    component converged to a degenerate fit (tier variance collapsed to
    exactly 0 with a nonsensical near-zero intercept SE) -- this
    symmetric-vc formulation converges cleanly (checked directly against
    real 2012-2024 data) and is used here instead.

    per_cell must have columns tier, cycle, eta_hat (from fit_per_cell_eta).
    cycle is treated as a categorical grouping variable (not numeric), i.e.
    this is NOT the OU-drift kind of model estimate_gb_ou_drift.py fits --
    it estimates how much of the cross-cell spread is a cycle-level draw vs.
    tier-level vs. pure noise, not a trend over calendar time.

    Returns dict with mu_global, tier_effects (dict), cycle_effects (dict),
    tier_var, cycle_var, resid_var, and the fitted MixedLMResults object
    itself (as "model_result") for diagnostics."""
    df = per_cell.copy()
    df["tier"] = df["tier"].astype("category")
    df["cycle"] = df["cycle"].astype("category")
    df["_dummy_group"] = 1

    model = smf.mixedlm(
        "eta_hat ~ 1", data=df, groups="_dummy_group",
        vc_formula={"tier": "0 + C(tier)", "cycle": "0 + C(cycle)"},
    )
    result = model.fit(reml=True)

    mu_global = float(result.params.get("Intercept", float("nan")))

    # Single dummy group -> one Series of BLUPs, prefixed "tier[C(tier)[X]]"
    # / "cycle[C(cycle)[Y]]" by statsmodels' vc_formula naming convention.
    re_series = result.random_effects[1]
    tier_effects = {
        tier: float(re_series.get(f"tier[C(tier)[{tier}]]", 0.0))
        for tier in df["tier"].cat.categories
    }
    cycle_effects = {
        cycle: float(re_series.get(f"cycle[C(cycle)[{cycle}]]", 0.0))
        for cycle in df["cycle"].cat.categories
    }

    vc_names = list(model.exog_vc.names) if hasattr(model, "exog_vc") else ["tier", "cycle"]
    vcomp = {name: float(v) for name, v in zip(vc_names, result.vcomp)}
    tier_var = vcomp.get("tier", 0.0)
    cycle_var = vcomp.get("cycle", 0.0)
    resid_var = float(result.scale)

    return dict(
        mu_global=mu_global, tier_effects=tier_effects, cycle_effects=cycle_effects,
        tier_var=tier_var, cycle_var=cycle_var, resid_var=resid_var,
        model_result=result,
    )


def posterior_predictive_eta_draws(
    fit: dict, tiers_per_race: list[str], k_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    """Drop-in replacement for scripts/solve_bellman_lsm.py's
    bootstrap_eta_resid_paths()'s eta output: draws k_paths samples of
    eta[tier] for a NEW, unseen cycle -- same (k_paths, n) shape, same
    broadcast-by-tier-to-races convention.

    For an unseen cycle, the appropriate predictive draw is
    mu_global + tier_effect[tier] + N(0, cycle_var) + N(0, resid_var/n_eff)
    -- i.e. propagate BOTH the cycle-level uncertainty (a genuinely new
    cycle's own draw, not one of the observed ones) and residual estimation
    noise, matching bootstrap_eta_resid_paths()'s own "new path, new
    cycle-like draw" semantics. resid_var is used directly (not divided by
    a fitted n_eff) as a conservative, slightly-too-wide approximation --
    Stage 1's unweighted-pooling simplification (module docstring) means
    there's no single clean n_eff to divide by here either."""
    n = len(tiers_per_race)
    eta_paths = np.zeros((k_paths, n))

    unique_tiers = sorted(set(tiers_per_race))
    for tier in unique_tiers:
        idx = [i for i, t in enumerate(tiers_per_race) if t == tier]
        if not idx:
            continue
        tier_effect = fit["tier_effects"].get(tier, 0.0)
        cycle_draw = rng.normal(0.0, np.sqrt(max(fit["cycle_var"], 0.0)), size=k_paths)
        resid_draw = rng.normal(0.0, np.sqrt(max(fit["resid_var"], 0.0)), size=k_paths)
        draws = fit["mu_global"] + tier_effect + cycle_draw + resid_draw
        for i in idx:
            eta_paths[:, i] = draws

    return eta_paths
