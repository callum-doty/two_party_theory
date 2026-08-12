"""
Race-level marginal values (project_spec.md Section 6):

    MSG_i^D = d p_i / d x_D_i
    MSG_i^R = - d p_i / d x_R_i

Both derived from payoff.grad_shared -- the single symmetric, fixed-baseline
formula (payoff.py's module docstring) used for BR_D/BR_R and every reported
utility in this package. There is no longer a separate D-anchored derivation
and R-mirrored derivation to keep consistent with each other: msg_d and
msg_r are the same chain rule through the same p_i(x_D, x_R), differing only
in which partial derivative is returned.
"""

from __future__ import annotations

import numpy as np

from . import payoff


def msg_d(party_d: np.ndarray, party_r: np.ndarray, arrays: dict) -> np.ndarray:
    """MSG_i^D = d p_i / d x_D_i, at fixed R."""
    dp_dxd, _ = payoff.grad_shared(party_d, party_r, arrays)
    return dp_dxd


def msg_r(party_d: np.ndarray, party_r: np.ndarray, arrays: dict) -> np.ndarray:
    """MSG_i^R = - d p_i / d x_R_i (spec Section 6): positive when more R
    dollars raise R's own win chance."""
    _, dp_dxr = payoff.grad_shared(party_d, party_r, arrays)
    return -dp_dxr
