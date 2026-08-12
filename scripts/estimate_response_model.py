#!/usr/bin/env python3
"""
Response-model estimation entry point (project_spec.md Section 4) and the
D/R symmetry test (Section 19).

The margin/spending-response model itself is REUSED unchanged (spec
Section 4: "the existing research already represents race state using
expected margin, uncertainty, ..."); re-estimating it is scripts/
run_estimation.py's job (carried over unmodified from the old project). This
script does two things specific to the new project:

  1. Points at that existing estimation entry point rather than duplicating it.
  2. Runs the symmetry test spec Section 19 requires before assuming a
     mirrored D-side elasticity is a valid stand-in for an R-side one:
     "Do not simply mirror the Democratic response curve onto Republicans
     without testing it."

SYMMETRY TEST (implemented 2026-08-12): `game/payoff.py`'s shared formula
uses ONE coefficient, `coef.beta1` (== `beta_rc.estimate`, `src/backtest/
model/margin.py` line 139), for the log-spending-share-ratio term
`c_spend * log(x_D/(x_D+x_R))` -- it is not "D's beta_D mirrored onto a
separate beta_R," it is literally the SAME number applied to both sides'
dollars through the same functional form. The open question was never
"does the formula treat D and R differently" (it doesn't, by construction)
but "was that ONE coefficient estimated on a sample that could actually
reveal an asymmetry if one existed." `beta_rc.estimate_beta_rc` fits
ΔMargin = β·Δlog(D-spend-share) + ε on REPEAT-CHALLENGER pairs, and until
now `identify_repeat_pairs` only ever selected the sample where the
DEMOCRAT is the repeat challenger (`incumb_status == "Challenger"`, R holds
the seat) -- so beta1 has only ever been tested on "D attacking an R-held
seat," never "R attacking a D-held seat."

`identify_repeat_pairs(..., challenger_party="R")` (added alongside this
function) selects the mirror-image sample (`incumb_status == "Incumbent"`,
D holds the seat, R is the repeat challenger) using the SAME margin_pp/
D-spending-share units as the original -- so a beta fit on that sample is
directly comparable to `beta_rc`, and the two together test EXACTLY
"does the same relative-spending-to-margin relationship hold regardless of
which party is defending vs. challenging," which is what letting one
`c_spend` govern both x_D's and x_R's effect on the same log-ratio term
actually assumes. `test_d_r_symmetry` below fits both samples and runs a
pooled interaction-term test (equivalent to a Chow test for equal slopes)
for a single p-value on "beta_D_sample == beta_R_sample".

Usage:
    python scripts/estimate_response_model.py --cycle 2024   # points to run_estimation.py
    python scripts/estimate_response_model.py --run-symmetry-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from backtest import config  # noqa: E402
from backtest.data import fec, elections, incumbency  # noqa: E402
from backtest.estimation import beta_rc as beta_rc_module  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("estimate_response_model")


def _load_generic_ballot() -> dict[int, float]:
    gb_path = config.raw_path("generic_ballot") / "generic_ballot_by_cycle.csv"
    if gb_path.exists():
        df = pd.read_csv(gb_path)
        return dict(zip(df["cycle"].astype(int), df["generic_ballot"].astype(float)))
    return {2012: 1.2, 2014: -5.8, 2016: 1.3, 2018: 8.6, 2020: 7.0, 2022: -1.0}


def _fit_beta_or_warn(pairs: pd.DataFrame, min_pairs: int, label: str):
    """estimate_beta_rc, but tolerates n < min_pairs (with a loud warning)
    instead of raising -- needed for common-support re-tests, where
    trimming to an overlap band can legitimately push a sample below the
    threshold that's fine for the FULL, untrimmed fit."""
    if len(pairs) >= min_pairs:
        return beta_rc_module.estimate_beta_rc(pairs)
    logger.warning(f"{label}: only {len(pairs)} pairs (< min {min_pairs}) -- "
                    f"fitting anyway, treat SE as unreliable.")
    from backtest.types import BetaRC
    X = sm.add_constant(pairs["delta_log_ratio"])
    model = sm.OLS(pairs["delta_margin"], X).fit(cov_type="HC3")
    return BetaRC(estimate=float(model.params["delta_log_ratio"]),
                   se=float(model.bse["delta_log_ratio"]), n_pairs=len(pairs))


def _pooled_interaction_test(pairs_d: pd.DataFrame, pairs_r: pd.DataFrame, min_pairs: int, label: str) -> dict:
    """One symmetry test: fit beta on each sample, then a pooled OLS with a
    challenger-party interaction term whose t-test is the formal
    beta_D == beta_R test (Chow test for equal slopes, HC3 robust SEs).
    Shared by test_d_r_symmetry (full samples) and
    common_support_symmetry_test (ratio-band-trimmed samples)."""
    beta_d = _fit_beta_or_warn(pairs_d, min_pairs, f"{label} D-challenger")
    beta_r = _fit_beta_or_warn(pairs_r, min_pairs, f"{label} R-challenger")

    pd_tagged = pairs_d.assign(is_r_challenger=0)
    pr_tagged = pairs_r.assign(is_r_challenger=1)
    pooled = pd.concat([pd_tagged, pr_tagged], ignore_index=True)
    pooled["interaction"] = pooled["delta_log_ratio"] * pooled["is_r_challenger"]
    X = sm.add_constant(pooled[["delta_log_ratio", "is_r_challenger", "interaction"]])
    pooled_model = sm.OLS(pooled["delta_margin"], X).fit(cov_type="HC3")

    b3 = float(pooled_model.params["interaction"])
    se3 = float(pooled_model.bse["interaction"])
    t3 = float(pooled_model.tvalues["interaction"])
    p3 = float(pooled_model.pvalues["interaction"])

    ci_d = (beta_d.estimate - 1.96 * beta_d.se, beta_d.estimate + 1.96 * beta_d.se)
    ci_r = (beta_r.estimate - 1.96 * beta_r.se, beta_r.estimate + 1.96 * beta_r.se)
    ci_overlap = not (ci_d[1] < ci_r[0] or ci_r[1] < ci_d[0])
    reject_symmetry = p3 < 0.05

    logger.info(f"[{label}] beta_D (n={beta_d.n_pairs}) = {beta_d.estimate:.4f} "
                f"(SE={beta_d.se:.4f}, 95% CI=[{ci_d[0]:.4f}, {ci_d[1]:.4f}])")
    logger.info(f"[{label}] beta_R (n={beta_r.n_pairs}) = {beta_r.estimate:.4f} "
                f"(SE={beta_r.se:.4f}, 95% CI=[{ci_r[0]:.4f}, {ci_r[1]:.4f}])")
    logger.info(f"[{label}] interaction test: diff={b3:.4f}, SE={se3:.4f}, t={t3:.3f}, p={p3:.4f} "
                f"-> {'REJECT' if reject_symmetry else 'CANNOT REJECT'} symmetry at alpha=0.05")

    return {
        "beta_d": {"estimate": beta_d.estimate, "se": beta_d.se, "n_pairs": beta_d.n_pairs, "ci_95": list(ci_d)},
        "beta_r": {"estimate": beta_r.estimate, "se": beta_r.se, "n_pairs": beta_r.n_pairs, "ci_95": list(ci_r)},
        "ci_overlap": ci_overlap,
        "interaction_test": {"diff": b3, "se": se3, "t": t3, "p_value": p3},
        "reject_symmetry_at_0.05": reject_symmetry,
        "min_repeat_pairs_threshold": min_pairs,
        "d_sample_below_min_pairs": len(pairs_d) < min_pairs,
        "r_sample_below_min_pairs": len(pairs_r) < min_pairs,
    }


def common_support_symmetry_test(bands: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.1, 0.9), (0.2, 0.8))) -> dict:
    """Re-runs the D/R symmetry test restricted to races where BOTH samples
    have actual data -- the full-sample test (`test_d_r_symmetry`) found
    beta_D=5.47 vs. beta_R=24.17, but the two samples sit at nearly
    disjoint points on the D-spending-share axis (D-challenger pairs:
    mean D-share ~0.21-0.28; R-challenger pairs: mean D-share ~0.89-0.91)
    -- so part or all of that gap could be the payoff's log-ratio
    functional form being measured at very different, non-overlapping
    points on what this project's own ceiling/saturation work established
    is a genuinely CONCAVE response curve, not a clean party-vs-party
    comparison. `ratio_mid = (ratio_prev + ratio_curr) / 2` summarizes
    "where on the spending-share axis does this pair sit," and each band
    below trims both samples to a shared window on that axis before
    refitting -- (0.0, 1.0) is the untrimmed baseline for reference,
    (0.1, 0.9) is the widest band where BOTH trimmed samples still clear
    `min_repeat_pairs` (project_spec.md's own precision bar), (0.2, 0.8) is
    a tighter robustness check where the R sample no longer clears it (kept
    anyway, flagged, since the point estimate and its direction are still
    informative even when the SE is not trustworthy)."""
    cycles = config.panel_cycles()
    gb = _load_generic_ballot()
    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)
    panel_spend = pd.concat([fec.build_total_spend(c) for c in cycles], ignore_index=True)
    panel_incumb = pd.concat([incumbency.load_incumbency(c) for c in cycles], ignore_index=True)

    pairs_d = beta_rc_module.identify_repeat_pairs(
        panel_results, panel_spend, panel_incumb, gb, challenger_party="D")
    pairs_r = beta_rc_module.identify_repeat_pairs(
        panel_results, panel_spend, panel_incumb, gb, challenger_party="R")
    pairs_d = pairs_d.assign(ratio_mid=(pairs_d["ratio_prev"] + pairs_d["ratio_curr"]) / 2)
    pairs_r = pairs_r.assign(ratio_mid=(pairs_r["ratio_prev"] + pairs_r["ratio_curr"]) / 2)

    logger.info(f"D-challenger ratio_mid: mean={pairs_d.ratio_mid.mean():.3f}, "
                f"median={pairs_d.ratio_mid.median():.3f} (n={len(pairs_d)})")
    logger.info(f"R-challenger ratio_mid: mean={pairs_r.ratio_mid.mean():.3f}, "
                f"median={pairs_r.ratio_mid.median():.3f} (n={len(pairs_r)})")

    min_pairs = config.min_repeat_pairs()
    band_results = {}
    for lo, hi in bands:
        label = f"[{lo:.2f},{hi:.2f}]" if (lo, hi) != (0.0, 1.0) else "untrimmed"
        d_band = pairs_d[(pairs_d.ratio_mid >= lo) & (pairs_d.ratio_mid <= hi)]
        r_band = pairs_r[(pairs_r.ratio_mid >= lo) & (pairs_r.ratio_mid <= hi)]
        logger.info(f"Band {label}: D n={len(d_band)}, R n={len(r_band)}")
        band_results[label] = dict(
            band=[lo, hi], n_d=len(d_band), n_r=len(r_band),
            **_pooled_interaction_test(d_band, r_band, min_pairs, label),
        )

    out = {
        "d_challenger_ratio_mid": {"mean": float(pairs_d.ratio_mid.mean()),
                                    "median": float(pairs_d.ratio_mid.median()), "n": len(pairs_d)},
        "r_challenger_ratio_mid": {"mean": float(pairs_r.ratio_mid.mean()),
                                    "median": float(pairs_r.ratio_mid.median()), "n": len(pairs_r)},
        "bands": band_results,
    }
    out_path = REPO_ROOT / "results" / "d_r_symmetry_common_support.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")
    return out


def test_d_r_symmetry() -> dict:
    """beta_D_sample == beta_R_sample? (spec Section 19). Fits
    `beta_rc.estimate_beta_rc` on both the D-repeat-challenger sample
    (`beta_rc.json`'s original estimate) and the R-repeat-challenger
    mirror-image sample, then runs a pooled OLS with a challenger-party
    interaction term -- the interaction coefficient's t-test IS the
    symmetry test (a Chow test for equal slopes across the two subsamples,
    same HC3 robust covariance convention `estimate_beta_rc` already uses).

    See `common_support_symmetry_test` for the follow-up this result
    motivated: the two samples sit at nearly disjoint points on the
    D-spending-share axis, so part of the gap found here could be the
    log-ratio functional form's concavity rather than a clean party
    difference -- read the two together, not this one in isolation."""
    cycles = config.panel_cycles()
    gb = _load_generic_ballot()

    logger.info(f"Loading panel results/spend/incumbency for cycles {cycles}…")
    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)
    panel_spend = pd.concat([fec.build_total_spend(c) for c in cycles], ignore_index=True)
    panel_incumb = pd.concat([incumbency.load_incumbency(c) for c in cycles], ignore_index=True)

    logger.info("Identifying D-repeat-challenger pairs (original beta_rc sample)…")
    pairs_d = beta_rc_module.identify_repeat_pairs(
        panel_results, panel_spend, panel_incumb, gb, challenger_party="D")
    logger.info("Identifying R-repeat-challenger pairs (mirror-image sample, "
                "D holds the seat, R is the repeat challenger)…")
    pairs_r = beta_rc_module.identify_repeat_pairs(
        panel_results, panel_spend, panel_incumb, gb, challenger_party="R")

    min_pairs = config.min_repeat_pairs()
    logger.info(f"D-challenger pairs: {len(pairs_d)}; R-challenger pairs: {len(pairs_r)} "
                f"(min required for a standalone fit: {min_pairs})")

    result = _pooled_interaction_test(pairs_d, pairs_r, min_pairs, "full sample")
    out_path = REPO_ROOT / "results" / "d_r_symmetry_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-model estimation (reused) + D/R symmetry test")
    parser.add_argument("--run-symmetry-test", action="store_true")
    parser.add_argument("--common-support-test", action="store_true",
                         help="Re-test symmetry restricted to overlapping D-spending-share bands.")
    args = parser.parse_args()

    logger.info("The spending-response model is reused unchanged from the old project. "
                "Run scripts/run_estimation.py to (re-)fit beta_D/sigma/etc.")
    if args.run_symmetry_test:
        test_d_r_symmetry()
    if args.common_support_test:
        common_support_symmetry_test()


if __name__ == "__main__":
    main()
