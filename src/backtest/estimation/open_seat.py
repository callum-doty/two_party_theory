"""
Open-seat spending elasticity calibration (§8.3).

Problem: β_RC is estimated from repeat-challenger pairs (incumbent races). Open seats
have no incumbent anchor, higher variance, and different fundraising dynamics. Using
β_RC directly for open seats likely misestimates the spending elasticity.

Procedure (following spec §8.3):
  1. β_panel^OS  — estimated from the panel interaction term log(ratio) × is_open
  2. τ           — prior uncertainty; set by covariate overlap between the RC subsample
                   (incumbents vs. challengers) and the open-seat population. Larger
                   distance → larger τ → more shrinkage toward β_RC.
  3. κ           — shrinkage weight (precision-weighted posterior on β_panel^OS):
                   κ = precision_data / (precision_prior + precision_data)
                     = (1/SE_panel_OS²) / (1/τ² + 1/SE_panel_OS²) = τ² / (τ² + SE_panel_OS²)
                   (Corrected 2026-07-23: an earlier version of this docstring stated
                   κ = 1/(1+τ²/SE²), the algebraic complement of what's actually
                   computed below — that would be the weight on β_RC, not on
                   β_panel^OS. docs/full_derivation.md §I.4 already matches the code.)
  4. β_OS^calib  — posterior: κ × β_panel^OS + (1 − κ) × β_RC
  5. β_OS^lb     — conservative lower bound: β_OS^calib − 1.64 × posterior_SE

The calibrated β_OS^calib replaces β_RC for open seats in _precompute_race_arrays.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenSeatCalibration:
    """Output of the κ calibration procedure."""
    beta_rc:         float    # repeat-challenger prior
    beta_panel_os:   float    # raw panel estimate (β_RC + β₄_panel)
    beta4_se:        float    # SE of β₄ from panel regression
    tau:             float    # prior uncertainty (covariate-overlap-based)
    kappa:           float    # shrinkage weight on panel estimate
    beta_os_calib:   float    # posterior point estimate (β_OS^calib)
    posterior_se:    float    # posterior standard error
    beta_os_lb:      float    # lower bound = β_OS^calib − 1.64 × posterior_SE


def _covariate_overlap_tau(beta_rc_se: float, n_open_seats: int) -> float:
    """
    Set τ by covariate distance between RC pairs and open seats.

    Repeat-challenger pairs are incumbent-vs-same-challenger across cycles,
    drawn from a population of competitive races where the same candidate
    ran twice. Open seats are structurally different: no incumbent signal,
    typically higher-quality challengers on both sides, different fundraising.

    Conservatively assume open-seat dynamics deviate from RC dynamics by
    about 2× the RC estimation uncertainty — reflecting that the RC design
    is a noisy proxy for open-seat elasticity, not a precise one. Additional
    penalty: fewer open seats in the panel (reduce effective n).

    τ = 2.0 × β_RC_SE × sqrt(max(50, n_open_seats) / n_open_seats)
    """
    n_penalty = math.sqrt(max(50, n_open_seats) / max(n_open_seats, 1))
    return 2.0 * beta_rc_se * n_penalty


def calibrate_open_seat(
    beta_rc: float,
    beta_rc_se: float,
    beta_panel_os: float,
    beta4_se: float,
    n_open_seats: int,
) -> OpenSeatCalibration:
    """
    Apply Bayesian shrinkage to produce β_OS^calib.

    Parameters
    ----------
    beta_rc       : repeat-challenger point estimate (prior mean)
    beta_rc_se    : SE of β_RC (determines prior uncertainty τ)
    beta_panel_os : β_RC + β₄_panel (panel open-seat elasticity)
    beta4_se      : SE of the β₄ interaction term from panel regression
    n_open_seats  : number of open-seat observations in the historical panel
    """
    tau = _covariate_overlap_tau(beta_rc_se, n_open_seats)

    # Conjugate Gaussian update:
    #   prior     ~ N(β_RC, τ²)
    #   likelihood ~ N(β_panel_OS, SE_panel_OS²)
    #   posterior ~ N(β_OS^calib, σ_post²)
    se_panel_os = beta4_se   # SE of the incremental β₄ also bounds total panel SE
    if se_panel_os <= 0:
        se_panel_os = tau    # degenerate: no panel info, full shrinkage

    precision_prior = 1.0 / (tau ** 2)
    precision_data  = 1.0 / (se_panel_os ** 2)
    precision_post  = precision_prior + precision_data

    beta_os_calib = (precision_prior * beta_rc + precision_data * beta_panel_os) / precision_post
    posterior_se  = math.sqrt(1.0 / precision_post)
    kappa         = precision_data / precision_post   # weight on panel estimate

    # Lower bound at ~90th-percentile conservative (1.64 SE below posterior mean)
    beta_os_lb = beta_os_calib - 1.64 * posterior_se

    result = OpenSeatCalibration(
        beta_rc=beta_rc,
        beta_panel_os=beta_panel_os,
        beta4_se=beta4_se,
        tau=tau,
        kappa=kappa,
        beta_os_calib=beta_os_calib,
        posterior_se=posterior_se,
        beta_os_lb=beta_os_lb,
    )

    logger.info(
        f"Open-seat κ calibration:\n"
        f"  β_RC (prior)    = {beta_rc:.4f} ± {beta_rc_se:.4f}  (τ = {tau:.4f})\n"
        f"  β_panel_OS      = {beta_panel_os:.4f} ± {se_panel_os:.4f}\n"
        f"  κ (weight)      = {kappa:.3f}  (higher → trust panel more)\n"
        f"  β_OS^calib      = {beta_os_calib:.4f} ± {posterior_se:.4f}\n"
        f"  β_OS^lb (90%)   = {beta_os_lb:.4f}\n"
        f"  open-seat obs n = {n_open_seats}"
    )

    return result
