#!/usr/bin/env python3
"""
Run historical panel estimation.

Estimates and persists:
  1. β_RC — repeat-challenger spending coefficient
  2. Full margin model coefficients (α, β₂, β₃)
  3. σᵢ heteroskedastic model

Outputs go to data/processed/ (default) or a suffixed sub-directory
when --panel-end-cycle is used for cross-cycle validation.

Usage:
    python scripts/run_estimation.py                        # standard 2012–2022 panel
    python scripts/run_estimation.py --panel-end-cycle 2020 # OOS 2022 validation
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import skew

from backtest import config
from backtest.data import fec, elections, incumbency, census
from backtest.estimation import beta_rc as beta_rc_module
from backtest.estimation import sigma as sigma_module
from backtest.estimation import open_seat as open_seat_module
from backtest.model import margin as margin_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_estimation")


def _load_generic_ballot() -> dict[int, float]:
    """
    Load historical generic ballot values from data/raw/generic_ballot/generic_ballot_by_cycle.csv.
    Falls back to hardcoded values if the file is missing.
    """
    gb_path = config.raw_path("generic_ballot") / "generic_ballot_by_cycle.csv"
    if gb_path.exists():
        df = pd.read_csv(gb_path)
        result = dict(zip(df["cycle"].astype(int), df["generic_ballot"].astype(float)))
        logger.info(f"Loaded generic ballot values from {gb_path.name}: {result}")
        return result
    logger.warning(
        f"Generic ballot file not found at {gb_path}. "
        "Using hardcoded fallback values. Create the file to manage GB values properly."
    )
    return {
        2012: 1.2,
        2014: -5.8,
        2016: 1.3,
        2018: 8.6,
        2020: 7.0,
        2022: -1.0,
    }


GENERIC_BALLOT_BY_CYCLE: dict[int, float] = _load_generic_ballot()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run panel estimation")
    parser.add_argument(
        "--panel-end-cycle", type=int, default=None,
        help="Exclude cycles after this year (e.g. 2020 for OOS 2022 validation). "
             "Outputs written to data/processed_oos_{year}/",
    )
    args = parser.parse_args()

    all_cycles = config.panel_cycles()
    if args.panel_end_cycle is not None:
        cycles = [c for c in all_cycles if c <= args.panel_end_cycle]
        out_dir = config.processed_path().parent / f"processed_oos_{args.panel_end_cycle}"
        logger.info(f"OOS mode: panel cycles {cycles}, writing to {out_dir.name}/")
    else:
        cycles = all_cycles
        out_dir = config.processed_path()

    processed = out_dir
    processed.mkdir(parents=True, exist_ok=True)

    # ── Load panel data ───────────────────────────────────────────────────────
    logger.info(f"Loading panel results for cycles {cycles}…")
    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)

    logger.info("Loading panel spend…")
    spend_frames = [fec.build_total_spend(c) for c in cycles]
    panel_spend = pd.concat(spend_frames, ignore_index=True)

    logger.info("Loading panel individual-contribution shares…")
    cand_frames = [fec.load_candidate_disbursements(c) for c in cycles]
    panel_cand = pd.concat(cand_frames, ignore_index=True)
    panel_indiv_df = (
        panel_cand[panel_cand["party"] == "D"][["district_id", "cycle", "indiv_share"]]
        .copy()
    )

    logger.info("Loading panel incumbency…")
    incumb_frames = [incumbency.load_incumbency(c) for c in cycles]
    panel_incumb = pd.concat(incumb_frames, ignore_index=True)

    from backtest.data.cook import load_pvi
    logger.info("Loading panel PVI…")
    pvi_frames = [load_pvi(c).assign(cycle=c) for c in cycles]
    panel_pvi = pd.concat(pvi_frames, ignore_index=True)

    # ── Step 1: β_RC ─────────────────────────────────────────────────────────
    logger.info("Identifying repeat-challenger pairs…")
    pairs = beta_rc_module.identify_repeat_pairs(
        panel_results, panel_spend, panel_incumb,
        generic_ballot_by_cycle=GENERIC_BALLOT_BY_CYCLE,
    )
    logger.info(f"Found {len(pairs)} pairs across cycles")

    logger.info("Estimating β_RC…")
    beta_rc = beta_rc_module.estimate_beta_rc(pairs)

    beta_path = processed / "beta_rc.json"
    with open(beta_path, "w") as f:
        json.dump({"estimate": beta_rc.estimate, "se": beta_rc.se, "n_pairs": beta_rc.n_pairs}, f, indent=2)
    logger.info(f"β_RC saved to {beta_path}")

    # ── Step 1b: Non-parametric bootstrap of β_RC ────────────────────────────
    # The uncertainty propagation elsewhere in this pipeline (K=1000 draws in
    # propagate_beta_rc_uncertainty) samples from the parametric N(β̂, SE²)
    # posterior implied by OLS asymptotics. That assumes the 118-pair sampling
    # distribution is well-approximated by a normal — untested, and suspect
    # given the identifying sample's skew toward Safe R pairs (FINDINGS.md
    # §10.1). This resamples the pairs directly to characterize the actual
    # finite-sample distribution, including any skew the normal approximation
    # would miss.
    n_boot = config.uncertainty_cfg()["beta_rc_bootstrap_draws"]
    logger.info(f"Bootstrapping β_RC ({n_boot} resamples of {len(pairs)} repeat-challenger pairs)…")
    boot_draws = beta_rc_module.bootstrap_beta_rc(pairs, n_boot=n_boot, rng=np.random.default_rng(42))

    boot_ci_low, boot_ci_high = float(np.percentile(boot_draws, 2.5)), float(np.percentile(boot_draws, 97.5))
    param_ci_low, param_ci_high = beta_rc.estimate - 1.96 * beta_rc.se, beta_rc.estimate + 1.96 * beta_rc.se
    boot_mean, boot_std, boot_skew = float(boot_draws.mean()), float(boot_draws.std(ddof=1)), float(skew(boot_draws))

    logger.info(
        f"β_RC bootstrap: mean={boot_mean:.4f}, std={boot_std:.4f}, skew={boot_skew:.4f}, "
        f"95% CI=[{boot_ci_low:.4f}, {boot_ci_high:.4f}] vs. "
        f"parametric N(β̂,SE²) 95% CI=[{param_ci_low:.4f}, {param_ci_high:.4f}]"
    )

    boot_path = processed / "beta_rc_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump({
            "n_boot": n_boot,
            "n_pairs": beta_rc.n_pairs,
            "bootstrap_mean": boot_mean,
            "bootstrap_std": boot_std,
            "bootstrap_skew": boot_skew,
            "bootstrap_ci_95": [boot_ci_low, boot_ci_high],
            "parametric_estimate": beta_rc.estimate,
            "parametric_se": beta_rc.se,
            "parametric_ci_95": [param_ci_low, param_ci_high],
        }, f, indent=2)
    logger.info(f"β_RC bootstrap distribution saved to {boot_path}")

    # scripts/plot_beta_rc_bootstrap.py always reads the unsuffixed filename and
    # always compares it against the *primary* config.processed_path() estimate
    # -- it has no OOS mode. An OOS run (--panel-end-cycle) must not overwrite
    # that shared file with 2012-2020-panel draws, or the primary chart would
    # silently start plotting OOS data under a primary-implied title. Give OOS
    # runs their own suffixed file instead (verified 2026-07-23: this file had
    # in fact been silently overwritten with 2022-OOS draws by a prior run).
    boot_draws_filename = (
        "beta_rc_bootstrap_distribution.csv" if args.panel_end_cycle is None
        else f"beta_rc_bootstrap_distribution_oos_{args.panel_end_cycle}.csv"
    )
    boot_draws_path = config.outputs_path() / boot_draws_filename
    boot_draws_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"beta_rc_draw": boot_draws}).to_csv(boot_draws_path, index=False)
    logger.info(f"β_RC raw bootstrap draws saved to {boot_draws_path} (for plot_beta_rc_bootstrap.py)")

    # ── Step 2: Full margin model ─────────────────────────────────────────────
    logger.info("Loading CVAP data…")
    cvap_df = census.load_cvap()

    logger.info("Fitting margin model on panel…")
    coef, r2, os_stats = margin_module.estimate_from_panel(
        panel_results=panel_results,
        panel_spend=panel_spend,
        panel_incumb=panel_incumb,
        panel_pvi=panel_pvi,
        generic_ballot_by_cycle=GENERIC_BALLOT_BY_CYCLE,
        beta_rc_estimate=beta_rc.estimate,
        cvap_df=cvap_df,
        panel_indiv_df=panel_indiv_df,
    )

    # ── Step 2b: Open-seat κ calibration (§8.3) ──────────────────────────────
    logger.info("Running open-seat Bayesian shrinkage (§8.3)…")
    # Count open-seat observations in the panel for the sample-size penalty
    n_open = int(panel_incumb[panel_incumb["incumb_status"] == "Open"].shape[0])
    os_cal = open_seat_module.calibrate_open_seat(
        beta_rc=beta_rc.estimate,
        beta_rc_se=beta_rc.se,
        beta_panel_os=os_stats["beta_panel_os"],
        beta4_se=os_stats["beta4_se"],
        n_open_seats=n_open,
    )
    # Wire calibrated open-seat β into the coefficient set
    coef.beta1_open = os_cal.beta_os_calib

    os_path = processed / "open_seat_calibration.json"
    with open(os_path, "w") as f:
        json.dump({
            "beta_rc":       os_cal.beta_rc,
            "beta_panel_os": os_cal.beta_panel_os,
            "beta4_se":      os_cal.beta4_se,
            "tau":           os_cal.tau,
            "kappa":         os_cal.kappa,
            "beta_os_calib": os_cal.beta_os_calib,
            "posterior_se":  os_cal.posterior_se,
            "beta_os_lb":    os_cal.beta_os_lb,
        }, f, indent=2)
    logger.info(f"Open-seat calibration saved to {os_path}")

    coef_path = processed / "margin_model_coef.json"
    with open(coef_path, "w") as f:
        json.dump({
            "alpha0": coef.alpha0, "alpha1": coef.alpha1,
            "alpha2": coef.alpha2, "alpha3": coef.alpha3,
            "alpha4": coef.alpha4, "alpha5": coef.alpha5,
            "beta1":  coef.beta1,  "beta2":  coef.beta2,
            "beta3":  coef.beta3,
            "beta1_open": coef.beta1_open,
            "r2_competitive": r2,
        }, f, indent=2)
    logger.info(f"Margin model coefficients saved to {coef_path}")

    # ── Step 3: σᵢ model ─────────────────────────────────────────────────────
    logger.info("Computing margin residuals for σᵢ estimation…")
    alpha_coef = {"intercept": coef.alpha0, "pvi": coef.alpha1,
                  "incumb": coef.alpha2, "gb": coef.alpha3, "alpha4": coef.alpha4,
                  "alpha5": coef.alpha5}
    beta_coef = {"b1": coef.beta1, "b2": coef.beta2, "b3": coef.beta3,
                 "b1_open": coef.beta1_open}

    residuals = sigma_module.compute_residuals_from_panel(
        panel_results=panel_results,
        panel_spend=panel_spend,
        panel_incumb=panel_incumb,
        panel_pvi=panel_pvi,
        alpha_coef=alpha_coef,
        beta_coef=beta_coef,
        generic_ballot_by_cycle=GENERIC_BALLOT_BY_CYCLE,
        cvap_df=cvap_df,
        panel_indiv_df=panel_indiv_df,
    )

    logger.info("Estimating σᵢ model…")
    sigma_model = sigma_module.estimate_sigma(residuals)

    sigma_path = processed / "sigma_model.json"
    with open(sigma_path, "w") as f:
        json.dump(sigma_model._coef, f, indent=2)
    logger.info(f"σᵢ model saved to {sigma_path}")

    logger.info("Estimation complete. Run scripts/run_backtest.py to execute the backtest.")


if __name__ == "__main__":
    main()
