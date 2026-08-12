"""
Shared two-player payoff model, p_i(x_D_i, x_R_i) = P(Democrat wins race i)
(project_spec.md Section 4):

    U_D(D, R) = sum_i p_i(x_D_i, x_R_i)
    U_R(D, R) = N - U_D(D, R)

ONE probability model for both sides, evaluated as a SIGNED, symmetric
saturation around a FIXED, party-neutral baseline -- not the D-anchored,
moving-floor construction this project used through 2026-08-10 (see git
history for that version). That construction re-derived its persuasion
ceiling at whatever R total was being evaluated -- fine as long as D was the
only side ever varying (every caller before this project), but handing it
to BR_R as a literal optimization objective exposed a real gap: the ceiling
capped D's own excursion ABOVE its floor and did nothing to bound how far
the floor itself fell as R's total grew. BR_R under that formula found the
gap immediately: it dumped ~$3M into races like HI-02 (R candidate spend on
record: $10), dragging a 99%-D seat to a modeled 33% -- an unsupported
extrapolation, not a real strategic opportunity (docs/methodology.md has
the full incident writeup).

baseline_arrays/p_win_shared/grad_shared fix this with a FIXED baseline:
mu_0_i = mu_raw(F^D_i, F^R_i), the race's margin at BOTH sides'
uncontrolled floor (candidate + state party + outside --
estimation.control_provenance.apply_control_floor's F^D/F^R, already
threaded through this codebase as RaceRecord.cand_d_total and the
separately-returned cand_r_total array). Either side's induced deviation
from that fixed point, Delta_mu_raw = mu_raw(F^D+x_D, F^R+x_R) - mu_0, is
saturated with a SIGNED, symmetric tanh: Delta_mu_cap = C*tanh(Delta_mu_raw/C),
so -C < Delta_mu_cap < C regardless of which side is spending or how much --
neither player's best response can extrapolate the fitted log-ratio response
surface beyond the region the ceiling regularizes. C_i keeps the exact
functional form of the original one-sided ceiling
(c_max * 4*p_0*(1-p_0), p_0 = Phi(mu_0/sigma)), since that shape is already
party-symmetric (p(1-p) = (1-p)p); only its ANCHOR changes, from a moving
D-side floor to the fixed, two-sided baseline. tanh (not the original
one-sided exp form) is used because it is smooth and antisymmetric by
construction (g(-z) = -g(z)) -- there is exactly one p_i(x_D, x_R), used as
both the search objective and the reported score for both sides; there is
no separate D-anchored formula or R-mirrored formula left to keep
consistent with each other.

c_max is still the DCCC-calibrated value from config.yaml, applied
identically to R's side -- an explicit, stated assumption (not an
independently-validated NRCC figure), the same status the old mirrored
ceiling's c_max carried. The open D/R elasticity- and scale-symmetry
question (project_spec.md Section 19) is unresolved either way; this fix
only removes the unbounded-extrapolation failure mode, it does not validate
that c_max=10 is the right number for R.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import norm

from backtest import config
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import _precompute_race_arrays


def baseline_arrays(races, coef, sigma_model, floor_r: np.ndarray) -> dict:
    """Fixed per-race baseline for the two-player game: mu_0 = mu_raw(F^D,
    F^R) at both sides' UNCONTROLLED floors, the persuasion-ceiling width
    C_i anchored there, and the static per-race quantities (mu_const,
    c_spend, sigma, cvap, alpha4) neither depends on. Computed ONCE per race
    universe; reused by every p_win_shared/grad_shared call against
    different (x_D, x_R) candidate allocations -- mu_0/C never move as the
    optimizer searches, unlike the old D-anchored formula's moving floor."""
    floor_r = np.asarray(floor_r, dtype=float)
    races_at_floor = [dataclasses.replace(r, r_total=float(floor_r[i])) for i, r in enumerate(races)]
    base = _precompute_race_arrays(races_at_floor, coef, sigma_model, eta=0.0)
    mu_0 = base["mu_floor"]  # predict_floor_margin(cand_d_total=F^D, r_total=F^R) == mu_raw(F^D, F^R)
    c_max = config.persuasion_ceiling_c_max()
    C = ceiling_mod.ceiling(mu_0, base["sigma"], c_max, maturity_factor=1.0)
    return dict(
        mu_const=base["mu_const"], c_spend=base["c_spend"], sigma=base["sigma"],
        cvap=base["cvap"], alpha4=base["alpha4"],
        floor_d=base["floors"], floor_r=floor_r, mu_0=mu_0, C=C,
    )


def _mu_raw(total_d: np.ndarray, total_r: np.ndarray, arrays: dict) -> np.ndarray:
    d = np.maximum(total_d, 1.0)
    r = np.maximum(total_r, 1.0)
    t = d + r
    ratio = np.clip(d / t, 1e-15, 1 - 1e-15)
    return arrays["mu_const"] + arrays["c_spend"] * np.log(ratio) + arrays["alpha4"] * np.log(t / arrays["cvap"])


def p_win_shared(party_d: np.ndarray, party_r: np.ndarray, arrays: dict) -> np.ndarray:
    """p_i(x_D, x_R): the two-player game's canonical payoff. Both sides'
    controllable dollars (party_d = x_D, party_r = x_R) enter through the
    SAME formula, saturated symmetrically around the fixed baseline mu_0 --
    see module docstring. This is the ONLY p_i this project's BR_D/BR_R
    should optimize AND report against; there is no separate D-anchored or
    R-mirrored variant left to keep consistent with it."""
    total_d = arrays["floor_d"] + np.maximum(party_d, 0.0)
    total_r = arrays["floor_r"] + np.maximum(party_r, 0.0)
    mu_raw = _mu_raw(total_d, total_r, arrays)
    delta_raw = mu_raw - arrays["mu_0"]
    C = arrays["C"]
    delta_cap = C * np.tanh(delta_raw / C)
    mu = arrays["mu_0"] + delta_cap
    return norm.cdf(mu / arrays["sigma"])


def grad_shared(party_d: np.ndarray, party_r: np.ndarray, arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """(dp_i/dx_D_i, dp_i/dx_R_i), analytic, from p_win_shared's own
    formula -- both derived the same way (chain rule through the same
    tanh saturation), unlike gradients.py's msg_d/msg_r, which differentiate
    two formulas that used to disagree away from x=0."""
    total_d = arrays["floor_d"] + np.maximum(party_d, 0.0)
    total_r = arrays["floor_r"] + np.maximum(party_r, 0.0)
    mu_raw = _mu_raw(total_d, total_r, arrays)
    C = arrays["C"]
    delta_raw = mu_raw - arrays["mu_0"]
    tanh_term = np.tanh(delta_raw / C)
    sech2 = 1.0 - tanh_term ** 2
    mu = arrays["mu_0"] + C * tanh_term
    sigma = arrays["sigma"]
    phi = norm.pdf(mu / sigma)

    # Same floor_dollars=1.0 clamp _mu_raw applies internally -- d(mu_raw)/dx
    # must be taken w.r.t. the SAME clamped d/r/t _mu_raw actually used, or
    # a race sitting exactly at the clamp (floor=$0, party=$0) divides by
    # zero here even though mu_raw itself was computed safely.
    d = np.maximum(total_d, 1.0)
    r = np.maximum(total_r, 1.0)
    t = d + r
    d_muraw_d_xd = arrays["c_spend"] * (1.0 / d - 1.0 / t) + arrays["alpha4"] / t
    d_muraw_d_xr = (arrays["alpha4"] - arrays["c_spend"]) / t

    common = (phi / sigma) * sech2
    return common * d_muraw_d_xd, common * d_muraw_d_xr
