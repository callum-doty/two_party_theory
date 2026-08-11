"""
Heteroskedastic σᵢ estimation from 2012–2022 margin residuals.

σᵢ is the standard deviation of the margin prediction error, conditional on:
  - |PVI| (absolute partisan lean)
  - Incumbency status (three categories)
  - National environment (generic ballot)

Fitting a log-linear model:
  log(|residual_i|) = a0 + a1·|PVI_i| + a2·is_open_i + a3·is_challenger_i + νᵢ

The fitted model is carried forward to 2024 district characteristics.
No 2024 outcome data enters σᵢ estimation.

Expected ordering: σ^open > σ^challenger > σ^incumbent at matched |PVI|.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..types import SigmaModel

logger = logging.getLogger(__name__)


def estimate_sigma(residuals: pd.DataFrame) -> SigmaModel:
    """
    Fit the heteroskedastic σᵢ model on historical margin residuals.

    Parameters
    ----------
    residuals : DataFrame with columns
        district_id, cycle, abs_pvi, incumb_status, margin_residual
        where margin_residual = actual_margin − fitted_margin (from the panel regression)

    Returns
    -------
    SigmaModel ready for predict(abs_pvi, incumb_status).
    """
    df = residuals.copy()
    df = df[df["margin_residual"].notna() & (df["margin_residual"].abs() > 0.01)]

    df["log_abs_resid"] = np.log(df["margin_residual"].abs())
    df["is_open"] = (df["incumb_status"] == "Open").astype(float)
    df["is_challenger"] = (df["incumb_status"] == "Challenger").astype(float)
    # |GB| captures wave-year amplification of forecast error — high polarization
    # cycles (2018, 2020) produce larger residuals independent of district type.
    df["abs_gb"] = df["gb"].abs() if "gb" in df.columns else 0.0

    feature_cols = ["abs_pvi", "is_open", "is_challenger"]
    if "gb" in df.columns:
        feature_cols.append("abs_gb")

    X = sm.add_constant(df[feature_cols])
    y = df["log_abs_resid"]
    fit = sm.OLS(y, X).fit()

    # Duan (1983) smearing estimator: exp(fitted) recovers the median of the
    # log-normal |residual|, not its mean — undercorrected retransformation
    # would systematically understate σᵢ. Rescale by the mean of exp(residual)
    # from this fit (no normality assumption required).
    smearing_factor = float(np.mean(np.exp(fit.resid)))

    coef = {
        "intercept":       float(fit.params["const"]),
        "abs_pvi":         float(fit.params["abs_pvi"]),
        "is_open":         float(fit.params["is_open"]),
        "is_challenger":   float(fit.params["is_challenger"]),
        "abs_gb":          float(fit.params.get("abs_gb", 0.0)),
        "smearing_factor": smearing_factor,
    }

    logger.info(
        f"σ model: intercept={coef['intercept']:.3f}, abs_pvi={coef['abs_pvi']:.3f}, "
        f"is_open={coef['is_open']:.3f}, is_challenger={coef['is_challenger']:.3f}, "
        f"abs_gb={coef['abs_gb']:.4f}, smearing_factor={coef['smearing_factor']:.4f}"
    )

    _check_sigma_ordering(SigmaModel(_coef=coef))
    return SigmaModel(_coef=coef)


def _check_sigma_ordering(model: SigmaModel) -> None:
    """Validate σ^open > σ^challenger > σ^incumbent at each |PVI| level."""
    pvi_bins = np.arange(0, 25, 5)
    violations = 0
    for pvi in pvi_bins:
        s_open = model.predict(float(pvi), "Open")
        s_chall = model.predict(float(pvi), "Challenger")
        s_incumb = model.predict(float(pvi), "Incumbent")
        if not (s_open > s_chall > s_incumb):
            violations += 1
            logger.warning(
                f"|PVI|={pvi}: ordering violated — open={s_open:.2f}, "
                f"chall={s_chall:.2f}, incumb={s_incumb:.2f}"
            )
    n_bins = len(pvi_bins)
    frac_ok = 1 - violations / n_bins
    logger.info(f"σ ordering holds in {frac_ok:.0%} of PVI bins ({n_bins - violations}/{n_bins})")


def compute_residuals_from_panel(
    panel_results: pd.DataFrame,
    panel_spend: pd.DataFrame,
    panel_incumb: pd.DataFrame,
    panel_pvi: pd.DataFrame,
    alpha_coef: dict,
    beta_coef: dict,
    generic_ballot_by_cycle: dict[int, float],
    cvap_df: pd.DataFrame | None = None,
    panel_indiv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute margin residuals for all historical panel observations.

    Uses the fitted margin model (α and β coefficients from panel estimation)
    to produce residuals: actual_margin − fitted_margin.

    Parameters
    ----------
    panel_results : [district_id, cycle, margin_pp]
    panel_spend   : [district_id, cycle, d_total, r_total]
    panel_incumb  : [district_id, cycle, incumb_status]
    panel_pvi     : [district_id, cycle, pvi]
    alpha_coef    : {"intercept", "pvi", "incumb", "gb", "alpha4", "alpha5"} —
                    control surface coefficients. "alpha4"/"alpha5" default to
                    0.0 if omitted. "alpha5" (indiv_share) requires
                    panel_indiv_df to have an effect — see below.
    beta_coef     : {"b1", "b2", "b3", "b1_open"} — spending response coefficients.
                    "b1_open" (optional) is substituted for "b1" on Open-seat rows,
                    matching model.margin.predict()'s beta1/beta1_open switch —
                    otherwise Open-seat residuals are computed against the wrong
                    μ̂ and bias the σᵢ "is_open" coefficient.
    generic_ballot_by_cycle : {cycle: GB_value}
    panel_indiv_df : [district_id, cycle, indiv_share], optional. Mirrors
                    model.margin.estimate_from_panel()'s own panel_indiv_df
                    parameter — omitted rows (or a fully-omitted argument)
                    fall back to indiv_share=0.0, matching that function's
                    fallback. Needed so alpha_coef["alpha5"] (if non-zero)
                    is applied against the same indiv_share values the
                    margin model itself was fit on, rather than silently
                    computing residuals against a μ̂ that omits the α₅ term
                    entirely (found in the 2026-07-24 codebase audit —
                    alpha5 is currently hardcoded to 0.0 in
                    model.margin.estimate_from_panel(), so this was dormant,
                    not yet a live bias).

    Returns
    -------
    DataFrame with columns: district_id, cycle, abs_pvi, incumb_status, margin_residual
    """
    df = (
        panel_results
        .merge(panel_spend, on=["district_id", "cycle"])
        .merge(panel_incumb, on=["district_id", "cycle"])
        .merge(panel_pvi, on=["district_id", "cycle"])
    )

    df["gb"] = df["cycle"].map(generic_ballot_by_cycle)
    df = df[df["d_total"] + df["r_total"] > 0]
    df["ratio"] = df["d_total"] / (df["d_total"] + df["r_total"])
    df = df[df["ratio"] > 0]
    df["log_ratio"] = np.log(df["ratio"])
    df["abs_pvi"] = df["pvi"].abs()
    df["is_incumb"] = (df["incumb_status"] == "Incumbent").astype(float)
    df["is_open"] = (df["incumb_status"] == "Open").astype(float)

    if cvap_df is not None:
        df = df.merge(cvap_df[["district_id", "cvap"]], on="district_id", how="left")
        df["cvap"] = df["cvap"].fillna(df["cvap"].median()).clip(lower=1)
    else:
        df["cvap"] = 500_000
    df["log_total_per_voter"] = np.log((df["d_total"] + df["r_total"]) / df["cvap"])

    if panel_indiv_df is not None:
        df = df.merge(panel_indiv_df[["district_id", "cycle", "indiv_share"]],
                      on=["district_id", "cycle"], how="left")
        df["indiv_share"] = df["indiv_share"].fillna(0.0)
    else:
        df["indiv_share"] = 0.0

    # Fitted margin from the full model
    a = alpha_coef
    b = beta_coef
    b1_open = b.get("b1_open")
    b1_eff = np.where(df["is_open"] == 1, b1_open, b["b1"]) if b1_open is not None else b["b1"]
    df["mu_hat"] = (
        a["intercept"]
        + a["pvi"] * df["pvi"]
        + a["incumb"] * df["is_incumb"]
        + a["gb"] * df["gb"]
        + a.get("alpha4", 0.0) * df["log_total_per_voter"]
        + a.get("alpha5", 0.0) * df["indiv_share"]
        + b1_eff * df["log_ratio"]
        + b["b2"] * df["log_ratio"] * df["abs_pvi"]
        + b["b3"] * df["log_ratio"] * df["is_incumb"]
    )

    df["margin_residual"] = df["margin_pp"] - df["mu_hat"]

    return df[["district_id", "cycle", "abs_pvi", "incumb_status", "margin_residual", "gb"]]
