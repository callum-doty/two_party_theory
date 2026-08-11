"""
Spending response surface — the margin model.

Margin_i = α₀ + α₁·PVI_i + α₂·incumb_i + α₃·GB
         + β₁·log(ratio_i) + β₂·log(ratio_i)·|PVI_i|
         + β₃·log(ratio_i)·incumb_i + εᵢ

β₁ = β_RC (fixed from 2012–2022 estimation, not re-estimated on 2024 data)
β₂, β₃ are interaction modifiers from the historical panel.
α₀–α₃ are control surface coefficients from the historical panel.

This module provides:
  - estimate_from_panel(): fit α and β on the 2012–2022 panel
  - predict(): apply the model to 2024 districts at a given β_RC draw
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarginModelCoefficients:
    """All estimated coefficients for the spending response surface."""
    alpha0: float         # intercept
    alpha1: float         # PVI
    alpha2: float         # incumbency (Incumbent dummy)
    alpha3: float         # generic ballot
    alpha4: float = 0.0   # log((D+R)/CVAP) — total spending intensity per voter
    alpha5: float = 0.0   # indiv_share — D candidate individual-contribution fraction
    beta1:  float = 0.0   # β_RC — log(ratio) base spending elasticity
    beta2:  float = 0.0   # log(ratio) × |PVI|
    beta3:  float = 0.0   # log(ratio) × Incumbent
    # Open-seat calibration (§8.3): Bayesian-shrunk spending elasticity for open seats.
    # When set, replaces beta1 for open seats in optimizer and MSG computation.
    # None = use beta1 for all race types (pre-calibration behavior).
    beta1_open: float | None = None


def estimate_from_panel(
    panel_results: pd.DataFrame,
    panel_spend: pd.DataFrame,
    panel_incumb: pd.DataFrame,
    panel_pvi: pd.DataFrame,
    generic_ballot_by_cycle: dict[int, float],
    beta_rc_estimate: float,
    cvap_df: pd.DataFrame | None = None,
    panel_indiv_df: pd.DataFrame | None = None,
) -> tuple[MarginModelCoefficients, float]:
    """
    Fit the margin model on the 2012–2022 panel.

    β₁ (β_RC) is constrained to its repeat-challenger estimate.
    α and β₂, β₃ are estimated by OLS with β₁ pre-imposed via offset.

    Parameters
    ----------
    beta_rc_estimate : pre-estimated β_RC from estimation.beta_rc

    Returns
    -------
    (MarginModelCoefficients, r_squared_competitive)
    """
    df = (
        panel_results
        .merge(panel_spend, on=["district_id", "cycle"])
        .merge(panel_incumb, on=["district_id", "cycle"])
        .merge(panel_pvi, on=["district_id", "cycle"])
    )

    df["gb"] = df["cycle"].map(generic_ballot_by_cycle)
    df = df[(df["d_total"] + df["r_total"]) > 0]
    df["ratio"] = df["d_total"] / (df["d_total"] + df["r_total"])
    df = df[df["ratio"] > 0]
    df["log_ratio"] = np.log(df["ratio"])
    df["abs_pvi"] = df["pvi"].abs()
    df["is_incumb"] = (df["incumb_status"] == "Incumbent").astype(float)

    # Spending intensity: log((D+R)/CVAP) — total dollars per eligible voter.
    # Uses 2022 ACS5 CVAP as an approximation for all cycles; districts that
    # don't match (pre-2022 redistricting) receive median-CVAP imputation.
    if cvap_df is not None:
        df = df.merge(cvap_df[["district_id", "cvap"]], on="district_id", how="left")
        median_cvap = df["cvap"].median()
        df["cvap"] = df["cvap"].fillna(median_cvap).clip(lower=1)
    else:
        df["cvap"] = 500_000   # national median fallback
    df["log_total_per_voter"] = np.log((df["d_total"] + df["r_total"]) / df["cvap"])

    # Impose β_RC via offset: y* = margin − β_RC·log(ratio)
    df["y_offset"] = df["margin_pp"] - beta_rc_estimate * df["log_ratio"]

    df["is_open"] = (df["incumb_status"] == "Open").astype(float)
    df["log_ratio_x_abs_pvi"] = df["log_ratio"] * df["abs_pvi"]
    df["log_ratio_x_incumb"] = df["log_ratio"] * df["is_incumb"]
    df["log_ratio_x_open"] = df["log_ratio"] * df["is_open"]
    # Merge indiv_share for D candidate quality control
    if panel_indiv_df is not None:
        df = df.merge(panel_indiv_df[["district_id", "cycle", "indiv_share"]],
                      on=["district_id", "cycle"], how="left")
        df["indiv_share"] = df["indiv_share"].fillna(0.0)
    else:
        df["indiv_share"] = 0.0

    # Note: log_total_per_voter excluded — endogeneity; see note in code.
    feature_cols = ["pvi", "is_incumb", "gb",
                    "log_ratio_x_abs_pvi", "log_ratio_x_incumb",
                    "log_ratio_x_open", "indiv_share"]

    X = sm.add_constant(df[feature_cols])
    y = df["y_offset"]
    fit = sm.OLS(y, X).fit(cov_type="HC3")

    # beta4_panel: additional spending elasticity for open seats above beta_RC.
    # beta_panel_OS = beta_RC + beta4_panel (raw panel estimate for open seats).
    beta4_panel = float(fit.params.get("log_ratio_x_open", 0.0))
    beta4_se = float(fit.bse.get("log_ratio_x_open", float("inf")))
    beta_panel_os = beta_rc_estimate + beta4_panel
    logger.info(
        f"Open-seat panel spending elasticity: β_RC={beta_rc_estimate:.3f}, "
        f"β₄={beta4_panel:.3f} (SE={beta4_se:.3f}), β_panel_OS={beta_panel_os:.3f}"
    )

    coef = MarginModelCoefficients(
        alpha0=float(fit.params["const"]),
        alpha1=float(fit.params["pvi"]),
        alpha2=float(fit.params["is_incumb"]),
        alpha3=float(fit.params["gb"]),
        alpha4=0.0,   # constrained to zero; see note above
        alpha5=0.0,   # constrained to zero: indiv_share is a race-salience proxy,
                      # not candidate quality — competitive races attract grassroots
                      # donors, penalising them suppresses the DCCC baseline without
                      # changing optimizer allocations (see §5.4 of FINDINGS.md)
        beta1=beta_rc_estimate,
        beta2=float(fit.params["log_ratio_x_abs_pvi"]),
        beta3=float(fit.params["log_ratio_x_incumb"]),
        beta1_open=None,   # filled by calibrate_open_seat() in run_estimation.py
    )

    # R² on competitive subset
    df["fitted"] = predict_batch(df, coef)
    competitive_mask = df["abs_pvi"] <= 10
    ss_res = ((df.loc[competitive_mask, "margin_pp"] - df.loc[competitive_mask, "fitted"]) ** 2).sum()
    ss_tot = ((df.loc[competitive_mask, "margin_pp"] - df.loc[competitive_mask, "margin_pp"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    logger.info(f"Margin model R² (competitive subset): {r2:.3f}")
    for name, val in coef.__dict__.items():
        logger.info(f"  {name}: {val}")

    return coef, r2, {"beta_panel_os": beta_panel_os, "beta4_se": beta4_se}


def predict(
    pvi: float,
    incumb_status: str,
    generic_ballot: float,
    ratio: float,
    coef: MarginModelCoefficients,
    beta1_override: float | None = None,
    total_spend: float = 0.0,
    cvap: int = 0,
    indiv_share: float = 0.0,
) -> float:
    """
    Compute fitted expected margin for a single district.

    Parameters
    ----------
    beta1_override  : if provided, substitute this value for coef.beta1
                      (used during β_RC uncertainty draws)
    total_spend     : D + R total spending ($); used for spending intensity term
    cvap            : citizen voting age population; used for per-voter normalization
    """
    # beta1_open (Open-seat calibration) takes priority over beta1_override:
    # beta1_override carries β_RC bootstrap-draw uncertainty (uncertainty.py),
    # a different, separately-calibrated quantity from beta1_open -- letting
    # it clobber beta1_open would recenter every Open race's elasticity on
    # β_RC (~5.46) instead of its own calibrated value (~6.98) in 100% of
    # draws, not add noise around the right point.
    if coef.beta1_open is not None and incumb_status == "Open":
        b1 = coef.beta1_open
    elif beta1_override is not None:
        b1 = beta1_override
    else:
        b1 = coef.beta1
    is_incumb = 1.0 if incumb_status == "Incumbent" else 0.0
    log_ratio = np.log(ratio)
    abs_pvi = abs(pvi)

    log_total_pv = np.log(max(total_spend, 1.0) / max(cvap, 1)) if cvap > 0 else 0.0

    return (
        coef.alpha0
        + coef.alpha1 * pvi
        + coef.alpha2 * is_incumb
        + coef.alpha3 * generic_ballot
        + coef.alpha4 * log_total_pv
        + coef.alpha5 * indiv_share
        + b1 * log_ratio
        + coef.beta2 * log_ratio * abs_pvi
        + coef.beta3 * log_ratio * is_incumb
    )


def predict_floor_margin(
    pvi: float,
    incumb_status: str,
    generic_ballot: float,
    cand_d_total: float,
    r_total: float,
    coef: MarginModelCoefficients,
    beta1_override: float | None = None,
    cvap: int = 0,
    indiv_share: float = 0.0,
    floor_dollars: float = 1.0,
) -> float:
    """
    Expected margin at a race's own candidate-only spending floor: D =
    cand_d_total (no party money added), R = r_total.

    Canonical floor-ratio regularization (see FINDINGS.md's persuasion-
    ceiling section): D and R are each floored at `floor_dollars` before
    forming the ratio, rather than clipping the resulting ratio itself.
    log(D/(D+R)) is meant to encode *relative* spending strength — ratio
    clipping collapses every sufficiently lopsided race to the same value
    regardless of the opponent's actual spend (a $0-vs-$800K race and a
    $0-vs-$8M race both hit the same clipped ratio), destroying exactly the
    information the term exists to carry. Flooring the dollar amounts first
    preserves it: the floor only matters when D or R is near zero, and has
    no detectable effect once both are well above floor_dollars, same as any
    additive smoothing constant. floor_dollars represents the smallest
    economically meaningful positive expenditure in the model's spending
    units — rescale it if those units ever change.

    This is the single source of truth for the floor-margin computation;
    model/win_prob.py and optimizer/allocator.py both call this rather than
    each re-deriving their own floor-ratio convention (see FINDINGS.md —
    two independently-written versions previously disagreed on lopsided
    near-zero-floor races, though the disagreement happened to be inert in
    the ranges the ceiling was actually saturating at).
    """
    d0 = max(cand_d_total, floor_dollars)
    r0 = max(r_total, floor_dollars)
    t0 = d0 + r0
    ratio0 = np.clip(d0 / t0, 1e-15, 1 - 1e-15)
    return predict(
        pvi=pvi,
        incumb_status=incumb_status,
        generic_ballot=generic_ballot,
        ratio=ratio0,
        coef=coef,
        beta1_override=beta1_override,
        total_spend=t0,
        cvap=cvap,
        indiv_share=indiv_share,
    )


def predict_batch(df: pd.DataFrame, coef: MarginModelCoefficients) -> pd.Series:
    """Vectorised version of predict() for a DataFrame with precomputed columns.

    Requires df to have: pvi, is_incumb, gb, log_ratio, log_total_per_voter
    Optional: is_open (0/1) — substitutes coef.beta1_open for open-seat rows,
    matching predict()'s single-row beta1/beta1_open switch (§8.3). Without
    this column (or before beta1_open is calibrated), every row falls back to
    coef.beta1, same as predict() would for a non-open-seat race.
    """
    log_tpv = df["log_total_per_voter"] if "log_total_per_voter" in df.columns else 0.0
    indiv = df["indiv_share"] if "indiv_share" in df.columns else 0.0
    if coef.beta1_open is not None and "is_open" in df.columns:
        b1 = np.where(df["is_open"].astype(bool), coef.beta1_open, coef.beta1)
    else:
        b1 = coef.beta1
    return (
        coef.alpha0
        + coef.alpha1 * df["pvi"]
        + coef.alpha2 * df["is_incumb"]
        + coef.alpha3 * df["gb"]
        + coef.alpha4 * log_tpv
        + coef.alpha5 * indiv
        + b1 * df["log_ratio"]
        + coef.beta2 * df["log_ratio"] * df["pvi"].abs()
        + coef.beta3 * df["log_ratio"] * df["is_incumb"]
    )
