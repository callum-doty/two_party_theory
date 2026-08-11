#!/usr/bin/env python3
"""
Estimate alpha4 (CVAP spending-intensity) via within-district fixed effects
(FINDINGS.md Section 10.7, Gap 1) -- see src/backtest/estimation/cvap_iv.py's
module docstring for the full scope boundary (this is FE-only, not a genuine
instrumental-variable estimate; the redistricting-jump instrument requires a
GIS crosswalk this pass does not build).

Assembles the same historical panel scripts/run_estimation.py uses (loading
directly from the same source functions, not duplicating logic), merges in
the multi-vintage CVAP panel (scripts/fetch_cvap_panel.py -- run that first),
and reports the FE result honestly regardless of outcome. Does NOT write
alpha4 into data/processed/margin_model_coef.json automatically -- per the
project's standing practice for anything this consequential (the original
naive-OLS alpha4 attempt was itself rejected after a manual OOS Brier
check), any adoption of this result requires a human to re-run that same
check first.

Usage:
    python scripts/fetch_cvap_panel.py          # run first if not already done
    python scripts/estimate_alpha4_iv.py
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data import fec, elections, incumbency
from backtest.data.cook import load_pvi
from backtest.estimation import cvap_iv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("estimate_alpha4_iv")


def _load_generic_ballot_by_cycle() -> dict[int, float]:
    path = config.raw_path("generic_ballot") / "generic_ballot_by_cycle.csv"
    df = pd.read_csv(path)
    return dict(zip(df["cycle"], df["generic_ballot"]))


def build_historical_panel(cycles: list[int]) -> pd.DataFrame:
    """Same assembly as scripts/run_estimation.py's main() -- district_id,
    cycle, margin_pp, d_total, r_total, pvi, incumb_status, gb -- kept as a
    thin, independent re-derivation (not importing run_estimation.py's
    main() directly, which has side effects / writes files) rather than a
    shared helper, since this is the only other caller so far."""
    panel_results = pd.concat([elections.load_results(c) for c in cycles], ignore_index=True)
    panel_spend = pd.concat([fec.build_total_spend(c) for c in cycles], ignore_index=True)
    panel_incumb = pd.concat([incumbency.load_incumbency(c) for c in cycles], ignore_index=True)
    panel_pvi = pd.concat([load_pvi(c).assign(cycle=c) for c in cycles], ignore_index=True)
    gb_by_cycle = _load_generic_ballot_by_cycle()

    df = (
        panel_results
        .merge(panel_spend, on=["district_id", "cycle"])
        .merge(panel_incumb, on=["district_id", "cycle"])
        .merge(panel_pvi, on=["district_id", "cycle"])
    )
    df["gb"] = df["cycle"].map(gb_by_cycle)
    return df


def main() -> None:
    out_dir = config.outputs_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    cvap_panel_path = config.raw_path("census") / "cvap_panel_all_vintages.csv"
    if not cvap_panel_path.exists():
        raise FileNotFoundError(
            f"{cvap_panel_path} not found -- run scripts/fetch_cvap_panel.py first."
        )
    cvap_panel = pd.read_csv(cvap_panel_path, dtype={"district_id": str})

    cycles = config.panel_cycles()
    logger.info(f"Building historical panel for cycles {cycles}…")
    historical_panel = build_historical_panel(cycles)
    logger.info(f"Historical panel: {len(historical_panel)} district-cycle rows")

    fe_panel = cvap_iv.build_fe_estimation_panel(historical_panel, cvap_panel)
    logger.info(f"FE estimation panel: {len(fe_panel)} rows, "
                f"{fe_panel['district_id'].nunique()} districts, "
                f"cycles {sorted(fe_panel['cycle'].unique().tolist())}")

    result = cvap_iv.estimate_alpha4_fe(fe_panel)

    if result["status"] != "ok":
        logger.warning(f"FE estimation did not run: {result}")
    else:
        logger.info(f"alpha4_fe = {result['alpha4_fe']:.4f} (SE={result['se']:.4f}, "
                     f"p={result['pvalue']:.4f}), n_obs={result['n_obs']}, "
                     f"n_districts={result['n_districts']}, "
                     f"R2_within={result['r_squared_within']:.4f}")
        print(result["full_summary"])

    weak_or_insignificant = (
        result["status"] != "ok"
        or not (result["pvalue"] == result["pvalue"])  # NaN check
        or result["pvalue"] > 0.10
    )

    out = {
        "status": result["status"],
        "method": "within_district_fixed_effects",
        "note": (
            "This is a fixed-effects estimate, NOT an instrumental-variable "
            "estimate -- the redistricting-jump instrument requires a GIS "
            "crosswalk not built in this pass. See "
            "src/backtest/estimation/cvap_iv.py's module docstring for the "
            "full scope boundary."
        ),
        "result": {k: v for k, v in result.items() if k != "full_summary"},
        "recommendation": (
            "alpha4 remains 0.0 in production (data/processed/margin_model_coef.json "
            "unchanged) -- this result is reported for the record, not auto-adopted. "
            "Any adoption requires a manual OOS Brier re-check, per this project's "
            "standing practice for consequential coefficient changes."
            if weak_or_insignificant else
            "alpha4_fe reached conventional significance; still requires a manual "
            "OOS Brier re-check before being written into "
            "data/processed/margin_model_coef.json -- not done automatically."
        ),
    }
    out_path = out_dir / "alpha4_iv_attempt.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Saved → {out_path}")
    logger.info(out["recommendation"])


if __name__ == "__main__":
    main()
