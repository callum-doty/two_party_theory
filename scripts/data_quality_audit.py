#!/usr/bin/env python3
"""
Data-quality audit of the race universe: missing values, silent fills,
univariate outliers, distribution shape, and a bivariate anomaly check
that a univariate check alone cannot catch.

This project's data pipeline (src/backtest/data/universe.py) had NO
systematic data-hygiene layer before this script: missing values are
handled inconsistently (some dropped-with-a-log, some silently
zero/empty-string-filled, see §1 below), there is no outlier detection
anywhere, imputation exists only for CVAP (median-fill, two places
downstream of universe.py), and the existing "6 validation gates"
(src/backtest/validation/gates.py) all check MODEL OUTPUT validity
(R^2, Brier score, sign checks), not input distribution shape. Confirmed
by direct audit 2026-08-10; see docs/data_quality_audit.md for the
write-up this script's output feeds.

Sections
--------
1. Missing values / silent fills: cvap==0, cook_rating=="", outcome is
   None are all silent-fill or silent-gap artifacts detectable directly
   from the final RaceRecord (a real district can never have cvap==0,
   so any cvap==0 is definitionally a merge-fill artifact, not real data).
2. Univariate outliers: z-score AND IQR flags on pvi, log1p(d_total),
   log1p(r_total), cvap (excluding the item-1 zero-fills, which would
   otherwise dominate/distort the check), indiv_share, spend_ratio.
3. Distribution histograms for the same fields.
4. Bivariate anomaly check: fits a simple PVI-only logistic baseline
   (statsmodels Logit on historical outcome ~ pvi) and flags races where
   the FULL margin model's predicted win probability disagrees sharply
   with that naive baseline. This is the check that catches GA-07-type
   cases (§4.2 of race_level_exploitability's write-up): GA-07's PVI and
   spend ratio are each unremarkable on their own (confirmed by a 2026-
   08-10 audit), so no univariate check in §2 flags it -- the anomaly
   only exists in the INTERACTION of a favorable PVI with a near-
   uncontested spend ratio, which only a bivariate check can surface.

Usage:
    python scripts/data_quality_audit.py --cycle 2024
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.optimizer.allocator import _precompute_race_arrays, _p_win_vec

import solve_bellman_lsm as lsm  # noqa: E402 -- reuse its real-coefficient loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("data_quality_audit")


class _CapturedLogs(logging.Handler):
    """Captures universe.py's own drop/fill log messages so this script can
    quote them in its own report without duplicating universe.py's logic."""
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[str] = []

    def emit(self, record):
        if record.name == "backtest.data.universe":
            self.records.append(record.getMessage())


def to_df(races) -> pd.DataFrame:
    return pd.DataFrame([{
        "district_id": r.district_id, "cook_rating": r.cook_rating, "pvi": r.pvi,
        "incumb_status": r.incumb_status, "d_total": r.d_total, "r_total": r.r_total,
        "cvap": r.cvap, "cand_d_total": r.cand_d_total, "indiv_share": r.indiv_share,
        "outcome": r.outcome, "redistricting_flagged": r.redistricting_flagged,
    } for r in races])


def section_missing_and_fills(df: pd.DataFrame, captured: list[str]) -> dict:
    zero_cvap = df[df["cvap"] == 0]
    empty_rating = df[df["cook_rating"] == ""]
    missing_outcome = df[df["outcome"].isna()]
    zero_cand_d = df[df["cand_d_total"] == 0.0]
    zero_indiv_share = df[df["indiv_share"] == 0.0]

    logger.info(f"universe.py's own drop log during build: {len(captured)} messages captured")
    for m in captured:
        logger.info(f"  > {m}")

    logger.info(f"cvap==0 (impossible for a real district -> silent merge-fill artifact): "
                f"{len(zero_cvap)} races: {zero_cvap['district_id'].tolist()}")
    logger.info(f"cook_rating=='' (silent fill, no Cook rating merged): "
                f"{len(empty_rating)} races: {empty_rating['district_id'].tolist()}")
    logger.info(f"outcome missing (no MIT/election result merged): "
                f"{len(missing_outcome)} races: {missing_outcome['district_id'].tolist()}")
    logger.info(f"cand_d_total==0.0 (ambiguous: could be real near-zero candidate spend, or a "
                f"silent fill -- not distinguishable without the raw pre-merge source): "
                f"{len(zero_cand_d)} races")
    logger.info(f"indiv_share==0.0 (same ambiguity as cand_d_total): {len(zero_indiv_share)} races")

    return {
        "universe_build_log_messages": captured,
        "cvap_zero_silent_fill": zero_cvap["district_id"].tolist(),
        "cook_rating_empty_silent_fill": empty_rating["district_id"].tolist(),
        "outcome_missing": missing_outcome["district_id"].tolist(),
        "cand_d_total_zero_ambiguous": zero_cand_d["district_id"].tolist(),
        "indiv_share_zero_ambiguous": zero_indiv_share["district_id"].tolist(),
    }


def section_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    d["log_d_total"] = np.log1p(d["d_total"])
    d["log_r_total"] = np.log1p(d["r_total"])
    d["spend_ratio"] = d["d_total"] / (d["d_total"] + d["r_total"]).clip(lower=1.0)
    d.loc[d["cvap"] == 0, "cvap"] = np.nan  # exclude known silent-fill artifacts from the check

    fields = ["pvi", "log_d_total", "log_r_total", "cvap", "indiv_share", "spend_ratio"]
    summary = {}
    for f in fields:
        x = d[f].dropna()
        mu, sigma = x.mean(), x.std()
        z = (d[f] - mu) / sigma
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        d[f"{f}_zscore"] = z
        d[f"{f}_iqr_outlier"] = (d[f] < lo) | (d[f] > hi)
        d[f"{f}_z_outlier"] = z.abs() > 3.0
        n_z = int(d[f"{f}_z_outlier"].sum())
        n_iqr = int(d[f"{f}_iqr_outlier"].sum())
        summary[f] = {"mean": float(mu), "std": float(sigma), "n_zscore_gt3": n_z, "n_iqr_flagged": n_iqr,
                       "zscore_flagged_districts": d.loc[d[f"{f}_z_outlier"].fillna(False), "district_id"].tolist()}
        logger.info(f"{f}: mean={mu:.3g} sd={sigma:.3g} | |z|>3: {n_z} races | outside 1.5*IQR: {n_iqr} races")

    return d, summary


def make_distribution_plots(d: pd.DataFrame, cycle: int, out_dir: Path) -> None:
    fields = [("pvi", "PVI (D-positive)"), ("log_d_total", "log(1+D total $)"),
              ("log_r_total", "log(1+R total $)"), ("cvap", "CVAP (silent-fill zeros excluded)"),
              ("indiv_share", "D individual-contribution share"), ("spend_ratio", "D/(D+R) spend ratio")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (field, label) in zip(axes.flat, fields):
        x = d[field].dropna()
        ax.hist(x, bins=30, color="#4292c6", edgecolor="white", linewidth=0.4)
        ax.axvline(x.mean(), color="#de2d26", ls="--", lw=1, label=f"mean={x.mean():.2g}")
        ax.axvline(x.median(), color="#333333", ls=":", lw=1, label=f"median={x.median():.2g}")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Raw input distributions — {cycle} race universe (n={len(d)})")
    fig.tight_layout()
    path = out_dir / f"distributions_{cycle}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {path}")


def _flag_by_disagreement(d: pd.DataFrame, baseline_col: str, z_thresh: float = 2.5) -> pd.DataFrame:
    disagreement = d["model_p_win"] - d[baseline_col]
    mu, sigma = disagreement.mean(), disagreement.std()
    z = (disagreement - mu) / sigma
    out = d.copy()
    out[f"disagreement_{baseline_col}"] = disagreement
    out[f"z_{baseline_col}"] = z
    return out


def section_bivariate_anomaly(races, df: pd.DataFrame, coef, sigma_model, out_dir: Path, cycle: int) -> pd.DataFrame:
    """PVI-only logistic baseline vs. the full margin model's p_win. Flags
    races where they sharply disagree -- catches interaction anomalies
    (e.g. GA-07) that no univariate check in section_outliers() can see,
    since GA-07's PVI and spend_ratio are each unremarkable alone.

    Robustness checks added after an initial pass flagged 13 races
    disproportionately incumbent-held (69% vs. 44% base rate, checked
    2026-08-10) -- a confound the PVI-only baseline can't see, since PVI
    is a fixed geographic/historical quantity blind to a given cycle's
    incumbent's own fundraising strength:
      1. A statistical-significance note: at z_thresh, what fraction of
         433 races would be expected to cross that threshold BY CHANCE
         under a null of normally-distributed disagreement, vs. what was
         actually found.
      2. A richer baseline (PVI + incumbency dummies) run alongside the
         PVI-only one. If most PVI-only-flagged races stop being flagged
         once incumbency is controlled for, the "model overweights
         spending" reading is wrong (or at least incomplete) -- the
         PVI-only baseline was just missing a real predictor, not the
         model's spending term overreacting.
    """
    from scipy.stats import norm as scipy_norm

    has_outcome = df["outcome"].isin(["D", "R"])
    y = (df.loc[has_outcome, "outcome"] == "D").astype(int)

    X_pvi = sm.add_constant(df.loc[has_outcome, "pvi"])
    fit_pvi = sm.Logit(y, X_pvi).fit(disp=0)
    logger.info(f"PVI-only baseline logit: {fit_pvi.params.to_dict()}, "
                f"pseudo-R2={fit_pvi.prsquared:.3f} (n={has_outcome.sum()} races with a known outcome)")

    incumb_dummies = pd.get_dummies(df["incumb_status"], prefix="incumb", drop_first=True)
    X_rich_full = sm.add_constant(pd.concat([df["pvi"], incumb_dummies], axis=1)).astype(float)
    X_rich = X_rich_full.loc[has_outcome]
    fit_rich = sm.Logit(y, X_rich).fit(disp=0)
    logger.info(f"PVI+incumbency baseline logit: {fit_rich.params.to_dict()}, "
                f"pseudo-R2={fit_rich.prsquared:.3f}")

    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=0.0)
    party_d_obs = np.maximum(df["d_total"].to_numpy() - df["cand_d_total"].to_numpy(), 0.0)
    model_p_win = _p_win_vec(party_d_obs, arrays)

    d = df.copy()
    d["model_p_win"] = model_p_win
    d["pvi_baseline_p_win"] = fit_pvi.predict(sm.add_constant(df["pvi"])).to_numpy()
    d["pvi_incumb_baseline_p_win"] = fit_rich.predict(X_rich_full).to_numpy()

    d = _flag_by_disagreement(d, "pvi_baseline_p_win")
    d = _flag_by_disagreement(d, "pvi_incumb_baseline_p_win")

    z_thresh = 2.5
    expected_false_positives = len(d) * 2 * (1.0 - scipy_norm.cdf(z_thresh))
    flagged_pvi = d[d["z_pvi_baseline_p_win"].abs() > z_thresh].sort_values("z_pvi_baseline_p_win")
    flagged_rich = d[d["z_pvi_incumb_baseline_p_win"].abs() > z_thresh].sort_values("z_pvi_incumb_baseline_p_win")
    still_flagged = set(flagged_pvi["district_id"]) & set(flagged_rich["district_id"])

    logger.info(f"At |z|>{z_thresh}, under a normal null ~{expected_false_positives:.1f} of "
                f"{len(d)} races would be flagged by CHANCE ALONE.")
    logger.info(f"PVI-only baseline: {len(flagged_pvi)} races flagged (vs. ~{expected_false_positives:.1f} "
                f"expected by chance -- {len(flagged_pvi) / max(expected_false_positives, 0.01):.1f}x).")
    logger.info(f"PVI+incumbency baseline: {len(flagged_rich)} races flagged.")
    logger.info(f"Races flagged by BOTH baselines (survive the incumbency-confound check): "
                f"{len(still_flagged)}/{len(flagged_pvi)}: {sorted(still_flagged)}")
    for _, row in flagged_pvi.iterrows():
        survives = row["district_id"] in still_flagged
        logger.info(f"  {row['district_id']} ({row['cook_rating']}, PVI {row['pvi']:+.1f}, {row['incumb_status']}): "
                     f"model={row['model_p_win']:.3f} vs PVI-baseline={row['pvi_baseline_p_win']:.3f} "
                     f"(z={row['z_pvi_baseline_p_win']:.2f}) | survives PVI+incumbency baseline: {survives}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (baseline_col, z_col, title) in zip(axes, [
        ("pvi_baseline_p_win", "z_pvi_baseline_p_win", "PVI-only baseline"),
        ("pvi_incumb_baseline_p_win", "z_pvi_incumb_baseline_p_win", "PVI + incumbency baseline"),
    ]):
        disagreement = d["model_p_win"] - d[baseline_col]
        flagged = d[d[z_col].abs() > z_thresh]
        ax.scatter(d["pvi"], disagreement, s=18, alpha=0.6, color="#4292c6")
        ax.scatter(flagged["pvi"], d.loc[flagged.index, "model_p_win"] - flagged[baseline_col],
                    s=40, color="#de2d26", label=f"|z| > {z_thresh}", zorder=5)
        for _, row in flagged.iterrows():
            ax.annotate(row["district_id"], (row["pvi"], row["model_p_win"] - row[baseline_col]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("PVI (D-positive)")
        ax.set_ylabel("model p_win − baseline p_win")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle(f"Bivariate anomaly check, with a robustness baseline — {cycle}")
    fig.tight_layout()
    path = out_dir / f"pvi_vs_model_disagreement_{cycle}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {path}")

    d["disagreement"] = d["model_p_win"] - d["pvi_baseline_p_win"]  # backward-compat column name
    d["disagreement_z"] = d["z_pvi_baseline_p_win"]
    d.attrs["expected_false_positives"] = expected_false_positives
    d.attrs["still_flagged_both_baselines"] = sorted(still_flagged)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-quality audit of the race universe")
    parser.add_argument("--cycle", type=int, default=2024)
    args = parser.parse_args()

    out_dir = config.outputs_path() / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = _CapturedLogs()
    logging.getLogger("backtest.data.universe").addHandler(capture)
    races = build_universe(cycle=args.cycle)
    logging.getLogger("backtest.data.universe").removeHandler(capture)
    df = to_df(races)
    logger.info(f"Auditing {len(df)} races, {args.cycle} cycle")

    logger.info("=" * 20 + " 1. Missing values / silent fills " + "=" * 20)
    missing_summary = section_missing_and_fills(df, capture.records)

    logger.info("=" * 20 + " 2. Univariate outliers " + "=" * 20)
    d, outlier_summary = section_outliers(df)

    logger.info("=" * 20 + " 3. Distribution plots " + "=" * 20)
    make_distribution_plots(d, args.cycle, out_dir)

    logger.info("=" * 20 + " 4. Bivariate anomaly check " + "=" * 20)
    coef, sigma_model = lsm.load_coef_and_sigma()
    d = section_bivariate_anomaly(races, d, coef, sigma_model, out_dir, args.cycle)

    csv_path = out_dir / f"data_quality_flagged_races_{args.cycle}.csv"
    d.to_csv(csv_path, index=False)
    logger.info(f"Saved full per-race audit table -> {csv_path}")

    summary = {
        "cycle": args.cycle, "n_races": len(df),
        "missing_and_fills": missing_summary,
        "univariate_outliers": outlier_summary,
        "n_bivariate_flagged_pvi_only": int((d["disagreement_z"].abs() > 2.5).sum()),
        "bivariate_flagged_districts_pvi_only": d.loc[d["disagreement_z"].abs() > 2.5, "district_id"].tolist(),
        "expected_false_positives_by_chance": d.attrs.get("expected_false_positives"),
        "n_survives_pvi_plus_incumbency_baseline": len(d.attrs.get("still_flagged_both_baselines", [])),
        "districts_surviving_pvi_plus_incumbency_baseline": d.attrs.get("still_flagged_both_baselines"),
    }
    json_path = out_dir / f"data_quality_summary_{args.cycle}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary -> {json_path}")


if __name__ == "__main__":
    main()
