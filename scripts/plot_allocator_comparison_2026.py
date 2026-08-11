#!/usr/bin/env python3
"""
Allocator spending comparison for 2026 races -- the live, in-progress analog
of plot_allocator_comparison.py's 2024 (completed-cycle) chart, and the same
"inefficient allocation" story Paper I told, applied to a decision that
hasn't been made yet.

Compares, for each race, how much money each strategy directs to it:

  1. Actual spend to date -- real, dated candidate-committee filings
     (data_catalog.md Section 2.7, this project's own dated periodic-reports
     panel) plus real coordinated/IE spend (RealizedSpendCommitmentSource),
     summed through today. PARTIAL, not final -- the cycle is still in
     progress and this number will keep growing regardless of what the
     model recommends.
  2. Trend forecast -- actual-to-date PLUS the candidate-committee component
     projected forward to Election Day at the calibrated per-tier trickle
     rate (scripts/estimate_candidate_spend_trickle.py). The coordinated/IE
     component is held flat at today's value, NOT extrapolated the same
     way -- this project's own research (docs/theta_followup_plan.md
     Section 10) found real outside-group spending is heavily back-loaded
     (95%+ in the final two months, not a smooth daily rate), so a linear
     trend is known to be a poor model for that component specifically.
     This forecast is very likely an UNDERESTIMATE of true final spend for
     exactly that reason, stated on the chart, not smoothed over.
  3. Model optimizer -- Paper II's nonlinear optimizer's recommended TOTAL
     party spend by Election Day (already-committed floor + recommended
     additional), reusing plot_2026_live_allocation.py's
     run_live_allocation() unmodified -- not a second, independent model run.
  4. Cook-implied -- proportional to Cook win probability, scaled to the
     actual-spend-to-date total for the chosen scope (same normalization
     convention as the 2024 chart).
  5. Uniform -- equal share across races in the chosen scope, same scaling.

Scope (--scope):
  competitive (default) -- Toss-Up / Lean D / Lean R only, same restriction
    the 2024 chart and this project's optimizer universe both use. Marginal
    dollars barely move Safe/Likely seats either way (Paper I's own MSG
    finding), so this is where the comparison has real signal.
  all -- every race in the 2026 universe (~434). Per-race x-axis labels are
    dropped (illegible at this count); a handful of races are still
    annotated. Included on request to show the same pattern isn't an
    artifact of pre-selecting only contested races.

Important differences from the 2024 chart, stated explicitly rather than
silently reused:
  - No outcome rings (D won / R won) -- the election hasn't happened. Real
    outcomes for a future event don't exist to plot; fabricating a "predicted
    winner" dressed up as fact would misrepresent what is actually known.
    Replaced with continuous shading by the model's CURRENT win-probability
    estimate (compute_outputs_batch's p_win, Paper I's unmodified static
    pipeline) -- explicitly labeled as a live estimate, not a result.
  - "Actual spend to date" is a partial-cycle number, not a finished-cycle
    total like the 2024 chart's d_total. This is not an apples-to-apples
    "how did the DCCC's final allocation compare" comparison -- it answers a
    different, live-decision-relevant question: which races are currently
    ahead of, or behind, the model's target pace.

Usage:
    python scripts/plot_allocator_comparison_2026.py                     # competitive, default
    python scripts/plot_allocator_comparison_2026.py --scope all         # all races
    python scripts/plot_allocator_comparison_2026.py --scope competitive --scope all   # both

Output: outputs/allocator_comparison_2026_live_{scope}.png, .csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.data import fec
from backtest.dynamic.ledger import RealizedSpendCommitmentSource
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.types import SigmaModel
from plot_2026_live_allocation import run_live_allocation

# Annotate a handful of races in the "all races" scope so it's not a wall of
# unlabeled dots -- the biggest outlier plus a few from each end of the sort.
ALL_SCOPE_ANNOTATE_TOP_N = 4


def load_coef_and_sigma():
    with open(ROOT / "data/processed/margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                              "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(ROOT / "data/processed/sigma_model.json") as f:
        sigma_coef = json.load(f)
    return coef, SigmaModel(_coef=sigma_coef)


def load_trickle_rate_per_day(tiers: list[str]) -> np.ndarray:
    """Per-race $/day candidate-committee trickle rate (data_catalog.md
    Section 2.7 / scripts/estimate_candidate_spend_trickle.py). Mirrors
    solve_bellman_lsm.py's load_trickle_rate_per_day (duplicated rather than
    imported: that module has heavy, slow-to-import Bellman/LSM machinery
    this plotting script has no other reason to load)."""
    path = ROOT / "data/processed/candidate_spend_trickle.json"
    if not path.exists():
        return np.zeros(len(tiers))
    with open(path) as f:
        trickle = json.load(f)
    by_tier = trickle["by_tier"]
    estimator = trickle.get("preferred_estimator", "mean_rate_per_day")
    pooled = by_tier.get("_pooled", {}).get(estimator, 0.0)
    return np.array([by_tier.get(t, {}).get(estimator, pooled) for t in tiers])


def build_frame(scope: str, as_of, races_all, model_by_district: dict) -> pd.DataFrame:
    if scope == "competitive":
        comp_ratings = config.competitive_ratings()
        races = [r for r in races_all if r.cook_rating in comp_ratings]
    elif scope == "all":
        races = races_all
    else:
        raise ValueError(f"unknown scope {scope!r}")

    district_ids = [r.district_id for r in races]
    tiers = [r.cook_rating for r in races]
    n = len(races)
    cook_map = config.cook_win_probs()

    # --- Actual spend to date: candidate-committee filings + coordinated/IE. ---
    cand = fec.cumulative_candidate_spend_as_of(2026, as_of)
    cand_by_key = {(row.district_id, row.party): row.disb_cum for row in cand.itertuples()}
    committed_d = RealizedSpendCommitmentSource(cycle=2026, party="D").committed_capital(0, as_of, races)
    committed_r = RealizedSpendCommitmentSource(cycle=2026, party="R").committed_capital(0, as_of, races)

    cand_d_only = np.array([cand_by_key.get((did, "D"), 0.0) for did in district_ids])
    actual_d = cand_d_only + np.array([committed_d.get(did, 0.0) for did in district_ids])
    actual_r = np.array([cand_by_key.get((did, "R"), 0.0) for did in district_ids]) + \
        np.array([committed_r.get(did, 0.0) for did in district_ids])
    actual_m = actual_d / 1e6

    # --- Trend forecast: candidate-committee component projected to Election
    # Day at the calibrated trickle rate; coordinated/IE held flat (see
    # module docstring for why extrapolating that component isn't done). ---
    election_day = config.election_day(2026)
    days_remaining = max(0, (election_day - as_of).days)
    trickle_per_day = load_trickle_rate_per_day(tiers)
    cand_d_forecast = cand_d_only + trickle_per_day * days_remaining
    forecast_d = cand_d_forecast + np.array([committed_d.get(did, 0.0) for did in district_ids])
    forecast_m = forecast_d / 1e6

    # --- Model optimizer: recommended TOTAL D party spend by Election Day. ---
    model_m = np.array([model_by_district.get(did, 0.0) for did in district_ids]) / 1e6

    # --- Cook-implied and Uniform, scaled to the actual-spend-to-date total
    # for this scope (distribution pattern, not overall cycle size). ---
    cook_probs = np.array([cook_map[r.cook_rating] for r in races])
    cook_share = cook_probs / cook_probs.sum()
    uniform_share = np.full(n, 1.0 / n)
    actual_total = actual_m.sum()
    cook_m = cook_share * actual_total
    uniform_m = uniform_share * actual_total

    # --- Win probability: a live estimate, NOT a realized outcome. Reflects
    # actual-to-date spend, not the static universe's cycle-cumulative
    # totals, so p_win describes the same spend state being plotted. ---
    coef, sigma_model = load_coef_and_sigma()
    races_asof = [replace(r, d_total=actual_d[i], r_total=actual_r[i]) for i, r in enumerate(races)]
    outputs = compute_outputs_batch(races_asof, coef, sigma_model)
    p_win = np.array([o.p_win for o in outputs])

    df = pd.DataFrame({
        "district_id": district_ids,
        "cook_rating": tiers,
        "actual_m": actual_m,
        "forecast_m": forecast_m,
        "model_m": model_m,
        "cook_m": cook_m,
        "uniform_m": uniform_m,
        "p_win": p_win,
    })
    df["delta_m"] = df["model_m"] - df["actual_m"]
    df = df.sort_values("delta_m", ascending=True).reset_index(drop=True)
    df["rank"] = df.index
    return df, days_remaining


def plot(df: pd.DataFrame, scope: str, as_of, days_remaining: int) -> None:
    n = len(df)
    label_races = (scope == "competitive")

    C_MODEL, C_COOK, C_UNIFORM, C_FORECAST = "#2a9d4f", "#e07b39", "#888888", "#8e44ad"

    fig, ax = plt.subplots(figsize=(17 if label_races else 15, 6))
    x = df["rank"].values

    cut_boundary = (df["delta_m"] < -0.05).sum()
    add_boundary = (df["delta_m"] < 0.05).sum()
    ymax = df[["actual_m", "model_m", "forecast_m"]].max().max()
    ax.axvspan(-1, cut_boundary - 0.5, alpha=0.06, color="#c0392b", zorder=0)
    ax.axvspan(cut_boundary - 0.5, add_boundary - 0.5, alpha=0.04, color="#888888", zorder=0)
    ax.axvspan(add_boundary - 0.5, n, alpha=0.06, color="#2a9d4f", zorder=0)
    for label, x_pos in [
        ("← ahead of target", cut_boundary / 2),
        ("on pace", (cut_boundary + add_boundary) / 2),
        ("behind target →", (add_boundary + n) / 2 - (0 if label_races else n * 0.03)),
    ]:
        ax.text(x_pos - 0.5, ymax * 1.04, label, fontsize=8, color="#555555",
                ha="center", va="bottom", style="italic")

    for _, row in df.iterrows():
        lo, hi = sorted([row["actual_m"], row["model_m"]])
        color = C_MODEL if row["model_m"] > row["actual_m"] else "#1a6faf"
        ax.plot([row["rank"], row["rank"]], [lo, hi], color=color, lw=0.6, alpha=0.35, zorder=1)

    ax.scatter(x, df["cook_m"], color=C_COOK, s=(28 if label_races else 12), zorder=3,
               marker="D", alpha=0.85, label="Cook-implied")
    ax.scatter(x, df["uniform_m"], color=C_UNIFORM, s=(22 if label_races else 10), zorder=2,
               marker="s", alpha=0.7, label=f"Uniform (${df['uniform_m'].iloc[0]:.2f}M each)")
    ax.scatter(x, df["forecast_m"], color=C_FORECAST, s=(30 if label_races else 14), zorder=3,
               marker="P", alpha=0.8,
               label=f"Trend forecast (candidate pace to Election Day,\nIE/coordinated held flat — likely an underestimate)")
    ax.scatter(x, df["model_m"], color=C_MODEL, s=(38 if label_races else 16), zorder=4,
               marker="^", alpha=0.9, label="Model recommended (total by Election Day)")

    cmap = mcolors.LinearSegmentedColormap.from_list("dr", ["#c0392b", "#dddddd", "#1a6faf"])
    sc = ax.scatter(x, df["actual_m"], c=df["p_win"], cmap=cmap, vmin=0, vmax=1,
                     s=(45 if label_races else 18), zorder=5, marker="o",
                     edgecolors="black", linewidths=0.4, label="Actual spend to date")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Model's current P(D win) — estimate, not an outcome", fontsize=8)

    if label_races:
        ax.set_xticks(x)
        ax.set_xticklabels(df["district_id"], rotation=90, fontsize=6.5)
    else:
        ax.set_xticks([])
        # Annotate the most extreme races at each end plus the single
        # biggest actual-spend outlier, so the "wall of dots" still has a
        # few concrete anchors.
        to_annotate = set(df.head(ALL_SCOPE_ANNOTATE_TOP_N)["district_id"]) \
            | set(df.tail(ALL_SCOPE_ANNOTATE_TOP_N)["district_id"]) \
            | set(df.nlargest(1, "actual_m")["district_id"])
        for _, row in df.iterrows():
            if row["district_id"] in to_annotate:
                y_val = max(row["actual_m"], row["model_m"], row["forecast_m"]) + ymax * 0.02
                ax.text(row["rank"], y_val, row["district_id"], fontsize=7, ha="center",
                        va="bottom", color="#333333")

    scope_label = f"{n} competitive races (Toss-Up / Lean D / Lean R)" if label_races else f"all {n} races in the 2026 universe"
    ax.set_xlabel(r"Race (sorted by $\Delta$ = Model target $-$ Actual-to-date)", fontsize=10)
    ax.set_ylabel("Spending ($M)", fontsize=11)
    ax.set_title(
        f"2026 Race Spending: Model Target vs. Actual-to-Date ({scope_label})\n"
        f"As of {as_of.isoformat()} ({days_remaining}d to Election Day) — PROSPECTIVE, cycle in progress: "
        "actual spend is partial, not final; no races have been decided.",
        fontsize=10.5,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.2, max(df["model_m"].max(), df["actual_m"].max(), df["forecast_m"].max()) * 1.15)

    fig.tight_layout()
    out_png = ROOT / "outputs" / f"allocator_comparison_2026_live_{scope}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["competitive", "all"], action="append",
                    help="Repeatable. Defaults to both if omitted.")
    args = ap.parse_args()
    scopes = args.scope or ["competitive", "all"]

    as_of = datetime.now(timezone.utc).date()
    races_all = build_universe(cycle=2026)

    # Model run is shared across scopes -- one nonlinear-optimizer call, not
    # re-run per scope.
    model_df, _meta = run_live_allocation()
    model_by_district = dict(zip(model_df["district_id"], model_df["recommended_total_party"]))

    for scope in scopes:
        df, days_remaining = build_frame(scope, as_of, races_all, model_by_district)
        plot(df, scope, as_of, days_remaining)

        out_csv = ROOT / "outputs" / f"allocator_comparison_2026_live_{scope}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved -> {out_csv}")

        print(f"\nTotals, {scope} scope, {len(df)} races ($M):")
        print(f"  Actual to date:  {df['actual_m'].sum():.1f}")
        print(f"  Trend forecast:  {df['forecast_m'].sum():.1f}")
        print(f"  Model target:    {df['model_m'].sum():.1f}")
        print(f"  Cook-implied:    {df['cook_m'].sum():.1f}")
        print(f"  Uniform:         {df['uniform_m'].sum():.1f}\n")


if __name__ == "__main__":
    main()
