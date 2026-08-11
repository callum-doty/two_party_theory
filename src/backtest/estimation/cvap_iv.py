"""
CVAP spending-intensity fixed-effects estimation (FINDINGS.md Section 10.7,
Gap 1) -- alpha4 (log((D+R)/CVAP), total spending per eligible voter) is
currently hardcoded to 0.0 in MarginModelCoefficients because naive OLS on
the historical panel found it endogenous: high-spending races are
structurally more competitive (DCCC over-invests where wins are needed
most), so OLS on the CROSS-SECTIONAL relationship picks up selection bias,
not a causal effect of spending intensity. A within-district fixed-effects
regression, using genuine time variation in a district's own CVAP across
Census vintages, is one way to net out time-invariant district confounders
that a pure cross-section cannot.

SCOPE BOUNDARY, stated explicitly rather than glossed over (this module does
NOT deliver a genuine instrumental-variable estimate):

The original plan considered two identification strategies: (1) a
multi-vintage CVAP panel exploited via within-district fixed effects, and
(2) redistricting-driven mechanical CVAP jumps as a quasi-experimental
instrument. Strategy (2) requires a real geographic (GIS) crosswalk between
pre- and post-2022-redistricting district boundaries -- scripts/
fetch_cvap_panel.py verified EMPIRICALLY that Census re-tabulates CVAP under
whatever congressional map is CURRENT at publication time, not the map in
effect during the ACS collection window, so a district_id's boundary
identity is genuinely different before vs. after the 2020-census
redistricting (confirmed directly: TX 36->38 districts, MT 1->2, CA/IL/MI/
NY/OH/PA/WV each -1, exactly matching the real 2020 apportionment, at the
2021->2022 vintage boundary). Building that crosswalk (matching pre- and
post-boundary polygons by geographic overlap) is genuine GIS work this
module does not attempt -- strategy (2) is NOT implemented here. Only
strategy (1) (FE only, no instrument) is.

This means what follows is a FIXED-EFFECTS estimate, not an
instrumental-variable estimate -- it nets out time-invariant district
confounders (partisan lean, urbanicity, historical competitiveness level)
but does NOT address any TIME-VARYING confounder that moves together with
both a district's CVAP and its competitiveness within the same district
over time (e.g., a district trending more competitive AND gaining
population in the same years). Reported as a data point, not as a validated
causal estimate -- see estimate_alpha4_fe()'s docstring for the honest
caveats on what this can and cannot rule out.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def nearest_vintage_for_cycle(cycle: int, available_vintages: list[int],
                               is_post_redistricting: dict[int, bool]) -> int | None:
    """Contemporaneous vintage mapping (the convention this project uses,
    chosen over a lagged mapping): a cycle uses the CVAP vintage whose
    5-year window END YEAR is closest to that cycle's election year --
    matching the timing convention log_total_per_voter already uses
    elsewhere in this codebase (both measured "as of" the cycle).

    Boundary-consistency constraint, enforced here rather than left to
    "nearest year" alone: a pre-2022-redistricting cycle must never be
    matched to a post-redistricting vintage (or vice versa), since the
    district_id would not refer to the same underlying geography. Returns
    None only if NO vintage sharing the cycle's redistricting era is present
    in available_vintages at all (e.g. if the caller only fetched
    post-redistricting vintages, no pre-2022 cycle could be matched to
    anything). In practice, with the real vintage set this project fetches
    (2016 onward -- Census's CD-level CVAP tabulation was verified to not
    exist before the 2016 vintage: 2014 and earlier only publish MCD.csv,
    not CD.csv), even a cycle as early as 2012 still maps to the nearest
    available same-era vintage (2016) rather than returning None -- the gap
    is simply wider for those cycles, not absent."""
    cycle_is_post = cycle >= 2022   # the 2022 election was the first run under new maps
    candidates = [v for v in available_vintages if is_post_redistricting.get(v) == cycle_is_post]
    if not candidates:
        return None
    return min(candidates, key=lambda v: abs(v - cycle))


def build_fe_estimation_panel(
    historical_panel: pd.DataFrame,
    cvap_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the historical margin-model panel (district_id, cycle,
    margin_pp, d_total, r_total, pvi, incumb_status, gb -- the same
    intermediate frame src.backtest.model.margin.estimate_from_panel()
    builds internally) with the multi-vintage CVAP panel
    (scripts/fetch_cvap_panel.py's district_id, cvap, vintage_end_year,
    is_post_redistricting), via nearest_vintage_for_cycle()'s
    contemporaneous, boundary-consistent mapping.

    Cycles with NO same-era CVAP vintage available at all (not just a
    distant one) are dropped, not silently imputed -- logged explicitly so
    the resulting panel's cycle coverage is visible, not assumed. With the
    real fetched vintage set (2016 onward), this only drops cycles if the
    caller fetched an incomplete vintage set; 2012/2014 still map to the
    nearest available pre-redistricting vintage (2016) rather than being
    dropped -- see nearest_vintage_for_cycle()'s docstring."""
    available = sorted(cvap_panel["vintage_end_year"].unique())
    is_post = cvap_panel.drop_duplicates("vintage_end_year").set_index("vintage_end_year")["is_post_redistricting"].to_dict()

    cycles = sorted(historical_panel["cycle"].unique())
    cycle_to_vintage = {c: nearest_vintage_for_cycle(c, available, is_post) for c in cycles}
    dropped = [c for c, v in cycle_to_vintage.items() if v is None]
    if dropped:
        logger.warning(f"build_fe_estimation_panel: no same-era CVAP vintage available for "
                        f"cycle(s) {dropped} -- dropped from the FE panel, not imputed.")

    df = historical_panel[historical_panel["cycle"].isin(
        [c for c, v in cycle_to_vintage.items() if v is not None]
    )].copy()
    df["cvap_vintage_end_year"] = df["cycle"].map(cycle_to_vintage)

    merged = df.merge(
        cvap_panel[["district_id", "vintage_end_year", "cvap"]],
        left_on=["district_id", "cvap_vintage_end_year"],
        right_on=["district_id", "vintage_end_year"],
        how="inner",
    ).drop(columns=["vintage_end_year"])

    n_before = len(df)
    n_after = len(merged)
    if n_after < n_before:
        logger.warning(f"build_fe_estimation_panel: {n_before - n_after} of {n_before} "
                        f"panel rows had no matching district_id in their cycle's CVAP "
                        f"vintage and were dropped (real, not imputed).")

    merged["total_spend"] = merged["d_total"] + merged["r_total"]
    merged = merged[(merged["total_spend"] > 0) & (merged["cvap"] > 0)]
    merged["log_total_per_voter"] = np.log(merged["total_spend"] / merged["cvap"])
    merged["ratio"] = merged["d_total"] / merged["total_spend"]
    merged = merged[merged["ratio"] > 0]
    merged["log_ratio"] = np.log(merged["ratio"])

    return merged


def estimate_alpha4_fe(panel: pd.DataFrame, min_periods_per_district: int = 2) -> dict:
    """Within-district fixed-effects regression of margin_pp on
    log_total_per_voter, controlling for log_ratio (the existing beta1
    spending-elasticity channel) and generic ballot (gb), via
    linearmodels.PanelOLS with entity (district_id) effects.

    Restricted to districts observed in at least min_periods_per_district
    cycles (default 2) -- a district appearing only once contributes no
    within-district variation and would be silently dropped by PanelOLS's
    own demeaning anyway; filtering explicitly here makes the effective
    sample size visible in the returned diagnostics rather than only
    showing up as a smaller-than-expected nobs.

    HONEST CAVEATS, not proof this estimate is valid:
    - This nets out TIME-INVARIANT district confounders (e.g. a district's
      general partisan lean or urban/rural character) but NOT time-varying
      ones that move with both CVAP and competitiveness in the same years
      (e.g. a district simultaneously gaining population and becoming more
      competitive for unrelated reasons).
    - The available CVAP vintages are overlapping 5-year ACS windows
      (2016-2021 vintages share 4 of their 5 underlying years with their
      immediate neighbor), so within-district variation across "different"
      vintages is itself partly mechanical smoothing, not independent
      measurement -- flagged in scripts/fetch_cvap_panel.py's own docstring
      as an expected source of weak power, confirmed empirically if this
      function's standard errors turn out large relative to the point
      estimate.
    - This is NOT an instrumental-variable estimate (see this module's
      top-level docstring) -- no exclusion restriction is invoked, so this
      does not, on its own, resolve the original OLS endogeneity concern
      that motivated Gap 1 in the first place. It answers a narrower
      question (does the within-district relationship survive controlling
      for time-invariant confounders) than the original ambition (a fully
      identified causal estimate).

    Returns dict with alpha4_fe, se, pvalue, n_obs, n_districts,
    n_periods_range, r_squared_within.
    """
    from linearmodels.panel import PanelOLS

    counts = panel.groupby("district_id")["cycle"].nunique()
    keep_districts = counts[counts >= min_periods_per_district].index
    df = panel[panel["district_id"].isin(keep_districts)].copy()
    if df.empty or df["district_id"].nunique() < 2:
        return {
            "status": "insufficient_data",
            "n_obs": int(len(df)),
            "n_districts": int(df["district_id"].nunique()) if len(df) else 0,
            "reason": f"fewer than {min_periods_per_district} districts have "
                      f">={min_periods_per_district} observed cycles with matching CVAP data",
        }

    df = df.set_index(["district_id", "cycle"])
    y = df["margin_pp"]
    X = df[["log_total_per_voter", "log_ratio", "gb"]]
    X = X.assign(const=1.0)

    mod = PanelOLS(y, X, entity_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    alpha4_fe = float(res.params.get("log_total_per_voter", float("nan")))
    se = float(res.std_errors.get("log_total_per_voter", float("nan")))
    pvalue = float(res.pvalues.get("log_total_per_voter", float("nan")))

    return {
        "status": "ok",
        "alpha4_fe": alpha4_fe,
        "se": se,
        "pvalue": pvalue,
        "n_obs": int(res.nobs),
        "n_districts": int(df.index.get_level_values("district_id").nunique()),
        "cycles_used": sorted(df.index.get_level_values("cycle").unique().tolist()),
        "r_squared_within": float(res.rsquared_within),
        "full_summary": str(res),
    }
