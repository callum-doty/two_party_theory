"""
Race-level marginal values (project_spec.md Section 6):

    MSG_i^D = d p_i / d D_i
    MSG_i^R = - d p_i / d R_i

Both derived from the SAME shared p_i(D, R) model (payoff.py), in a common
unit of expected seats per dollar (multiply by 1e6 for "per $1M", matching
the spec's suggested reporting unit). MSG^D reuses
backtest.optimizer.allocator._msg_vec unmodified (it's exactly d p / d D at
eta=0). MSG^R has no existing counterpart in the old codebase -- Paper III's
nash.py only ever differentiates R's own MIRRORED, uncalibrated ceiling
formula (_r_mu_and_grad), which answers a different question ("R's own
belief about d P_R_win / d party_R") than what this game needs ("how R's
total spending moves the SAME calibrated p_i used for D's own gradient").
d_p_d_R below is the missing derivative, worked out from the identical
d/dr/ratio/total_pv construction _msg_vec already uses for d/dD, differing
only in which variable (D fixed vs. R fixed) is held constant:

    t = D_i + R_i
    d(log(D_i/t))/dR_i   = -1/t
    d(log(t/cvap))/dR_i  =  1/t
    => d(mu_raw)/dR_i = (alpha4 - c_spend) / t

which is the same expression nash.py's _r_mu_and_grad independently derives
(confirms the two are consistent up to which ceiling anchors them) -- see
that module's docstring for the derivation in its own words.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from backtest.optimizer.allocator import _apply_ceiling, _msg_vec


def msg_d(party_d: np.ndarray, arrays: dict) -> np.ndarray:
    """MSG_i^D = d p_i / d D_i, at fixed R (arrays already bakes in R's
    total via payoff.race_arrays_at)."""
    return _msg_vec(np.maximum(party_d, 0.0), arrays)


def d_p_d_R(party_d: np.ndarray, total_r: np.ndarray, arrays: dict) -> np.ndarray:
    """d p_i / d R_i, at fixed D, using D's own calibrated ceiling (see
    module docstring for why this -- not R's mirrored ceiling -- is used)."""
    d = np.maximum(arrays["floors"] + party_d, 1.0)
    r = np.maximum(total_r, 1.0)
    t = d + r
    mu_raw = (
        arrays["mu_const"]
        + arrays["c_spend"] * np.log(np.clip(d / t, 1e-15, 1 - 1e-15))
        + arrays["alpha4"] * np.log(t / arrays["cvap"])
    )
    mu_capped, grad_factor = _apply_ceiling(mu_raw, arrays)
    sigma = arrays["sigma"]
    phi = norm.pdf(mu_capped / sigma)

    d_mu_raw_d_r = (arrays["alpha4"] - arrays["c_spend"]) / t
    d_mu_capped_d_r = grad_factor * d_mu_raw_d_r
    return (phi / sigma) * d_mu_capped_d_r


def msg_r(party_d: np.ndarray, total_r: np.ndarray, arrays: dict) -> np.ndarray:
    """MSG_i^R = - d p_i / d R_i (spec Section 6): positive when more R
    dollars raise R's own win chance."""
    return -d_p_d_R(party_d, total_r, arrays)
