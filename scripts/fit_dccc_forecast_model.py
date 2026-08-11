#!/usr/bin/env python3
"""
Fit and validate the DCCC-forecast model: a two-part (Cragg) hurdle model
predicting each race's eventual DCCC party-dollar share from information
available 91 days before Election Day -- Stage 1 (logit) predicts whether
DCCC funds the race at all; Stage 2 (OLS on log-share, funded subset only)
predicts how much, conditional on funding. Mirrors this project's own
"selection vs. intensity" language (Paper I Table 5) directly, not by
coincidence -- it's the same distinction, applied prospectively instead of
retrospectively.

Two validations, per the reviewer's held-out-cycle discipline:
  1. Leave-one-cycle-out CV across all 7 cycles (2012-2024) -- for each
     held-out cycle, train on the other 6, predict, and report both
     statistical fit (AUC, Brier, R^2) and the metric that actually
     matters: E[Seats | forecasted-DCCC] vs. the real E[Seats | actual
     DCCC], using the identical nonlinear evaluation used throughout this
     document.
  2. The reviewer's specific request: train on 2012-2022, test on 2024.

The final model (refit on all 7 cycles) is saved for
apply_dccc_forecast_2026.py to use on the live cycle.

Usage:
    python scripts/fit_dccc_forecast_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.optimizer.allocator import nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts  # type: ignore

FEATURES = ["pvi", "abs_pvi", "is_challenger", "is_open", "cook_ordinal", "cand_ratio_t", "r_ratio_t", "generic_ballot"]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_challenger"] = (df["incumb_status"] == "Challenger").astype(float)
    df["is_open"] = (df["incumb_status"] == "Open").astype(float)
    return df


def fit_hurdle(train_df: pd.DataFrame):
    X = sm.add_constant(train_df[FEATURES])
    selection_model = sm.Logit(train_df["funded"], X).fit(disp=0)

    funded = train_df[train_df["funded"] == 1].copy()
    funded["log_share"] = np.log(funded["party_share_final"].clip(lower=1e-6))
    X_funded = sm.add_constant(funded[FEATURES])
    intensity_model = sm.OLS(funded["log_share"], X_funded).fit()

    return selection_model, intensity_model


def predict_shares(selection_model, intensity_model, df: pd.DataFrame) -> np.ndarray:
    X = sm.add_constant(df[FEATURES], has_constant="add")
    p_funded = selection_model.predict(X)
    log_share = intensity_model.predict(X)
    raw = p_funded * np.exp(log_share)
    return raw


def evaluate_cycle(test_df: pd.DataFrame, predicted_share: np.ndarray, cycle: int, processed_dir) -> dict:
    """Rescale predicted shares to the cycle's real party budget, then score
    both statistical fit and the downstream seats metric against the real,
    known outcome for this (held-out) cycle."""
    _, coef, sigma_model = load_processed_artifacts(processed_dir)
    races = build_universe(cycle=cycle)
    races_by_id = {r.district_id: r for r in races}
    races_ordered = [races_by_id[d] for d in test_df["district_id"]]
    party_budget = sum(r.d_total - r.cand_d_total for r in races)

    predicted_dollars = predicted_share / predicted_share.sum() * party_budget
    actual_dollars = test_df["party_share_final"].values * party_budget

    forecast_seats = nonlinear_expected_seats_at_party_dollars(races_ordered, coef, sigma_model, predicted_dollars)
    actual_seats = nonlinear_expected_seats_at_party_dollars(races_ordered, coef, sigma_model, actual_dollars)

    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(test_df["funded"], predicted_share) if test_df["funded"].nunique() > 1 else float("nan")
    brier = brier_score_loss(test_df["funded"], np.clip(predicted_share / predicted_share.max(), 0, 1))

    funded_mask = test_df["funded"] == 1
    if funded_mask.sum() > 1:
        actual_log = np.log(test_df.loc[funded_mask, "party_share_final"].clip(lower=1e-6))
        pred_log = np.log(np.clip(predicted_share[funded_mask.values], 1e-9, None))
        r2 = 1 - np.sum((actual_log - pred_log) ** 2) / np.sum((actual_log - actual_log.mean()) ** 2)
    else:
        r2 = float("nan")

    return {
        "cycle": cycle, "auc": auc, "brier": brier, "intensity_r2": r2,
        "forecast_seats": forecast_seats, "actual_seats": actual_seats,
        "forecast_error_seats": forecast_seats - actual_seats,
    }


def main() -> None:
    df = _prep(pd.read_csv(config.processed_path() / "dccc_forecast_training_data.csv"))
    cycles = sorted(df["cycle"].unique())
    print(f"Loaded {len(df)} rows across cycles: {cycles}\n")

    def processed_dir_for(cycle: int):
        # Reuse Paper I's own estimation artifacts: 2024 pipeline for cycle==2024,
        # OOS-2020-trained pipeline for cycle==2022, and the live 2024-fit pipeline
        # as the best available stand-in for other historical cycles (this repo's
        # estimation artifacts are only fit for these two configurations).
        if cycle == 2022:
            return Path("data/processed_oos_2020")
        return config.processed_path()

    # ── 1. Leave-one-cycle-out CV ────────────────────────────────────────────
    print("[1/2] Leave-one-cycle-out cross-validation:")
    loco_results = []
    for held_out in cycles:
        train_df = df[df["cycle"] != held_out]
        test_df = df[df["cycle"] == held_out]
        sel, inten = fit_hurdle(train_df)
        pred = predict_shares(sel, inten, test_df)
        res = evaluate_cycle(test_df, pred, held_out, processed_dir_for(held_out))
        loco_results.append(res)
        print(f"  held out {held_out}: AUC={res['auc']:.3f}  Brier={res['brier']:.4f}  "
              f"intensity R2={res['intensity_r2']:.3f}  "
              f"forecast E[Seats]={res['forecast_seats']:.2f} vs actual={res['actual_seats']:.2f} "
              f"(error {res['forecast_error_seats']:+.2f})")

    loco_df = pd.DataFrame(loco_results)
    print(f"\n  Mean |forecast error|: {loco_df['forecast_error_seats'].abs().mean():.3f} seats")
    print(f"  Mean AUC: {loco_df['auc'].mean():.3f}   Mean intensity R2: {loco_df['intensity_r2'].mean():.3f}")

    # ── 2. Reviewer's specific test: train 2012-2022, test 2024 ────────────
    print(f"\n[2/2] Train on 2012-2022 (6 cycles), test on 2024:")
    train_df = df[df["cycle"] < 2024]
    test_df = df[df["cycle"] == 2024]
    sel, inten = fit_hurdle(train_df)
    pred = predict_shares(sel, inten, test_df)
    res_2024 = evaluate_cycle(test_df, pred, 2024, config.processed_path())
    print(f"  AUC={res_2024['auc']:.3f}  Brier={res_2024['brier']:.4f}  "
          f"intensity R2={res_2024['intensity_r2']:.3f}")
    print(f"  Forecasted DCCC E[Seats]={res_2024['forecast_seats']:.3f} vs. "
          f"real DCCC E[Seats]={res_2024['actual_seats']:.3f} "
          f"(forecast error {res_2024['forecast_error_seats']:+.3f} seats)")
    print(f"  For reference, Paper I's headline: real DCCC=215.115, model-optimal=217.940 (+2.83)")

    out_dir = config.outputs_path()
    loco_df.to_csv(out_dir / "dccc_forecast_loco_cv.csv", index=False)
    pd.DataFrame([res_2024]).to_csv(out_dir / "dccc_forecast_2012_2022_train_2024_test.csv", index=False)
    print(f"\nSaved: {out_dir / 'dccc_forecast_loco_cv.csv'}")
    print(f"Saved: {out_dir / 'dccc_forecast_2012_2022_train_2024_test.csv'}")

    # ── Final model: refit on all 7 cycles for application to live 2026 ────
    sel_final, inten_final = fit_hurdle(df)
    model_out = {
        "features": FEATURES,
        "selection_params": {"const": float(sel_final.params["const"]),
                              **{f: float(sel_final.params[f]) for f in FEATURES}},
        "intensity_params": {"const": float(inten_final.params["const"]),
                              **{f: float(inten_final.params[f]) for f in FEATURES}},
        "trained_on_cycles": [int(c) for c in cycles],
    }
    model_path = config.processed_path() / "dccc_forecast_model.json"
    with open(model_path, "w") as f:
        json.dump(model_out, f, indent=2)
    print(f"Saved: {model_path}")

    print("\nSelection-stage (funded?) coefficients (all 7 cycles):")
    print(sel_final.summary().tables[1])
    print("\nIntensity-stage (log share | funded) coefficients (all 7 cycles):")
    print(inten_final.summary().tables[1])


if __name__ == "__main__":
    main()
