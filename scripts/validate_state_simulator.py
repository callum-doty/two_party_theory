#!/usr/bin/env python3
"""
Assemble the state-transition simulator (Paper III Section 2) and validate
it per Section 5.4's standard: rank correlation between a pre-election
snapshot's mu_i and eventual realized margin, using 2022 and 2024 as
held-out cycles, plus direct out-of-sample checks of the two calibrated
components (eta, sigma_G) fit earlier this session.

IMPORTANT boundary respected here (dynamic/simulate.py's own documented
constraint): alpha3 (the margin model's GB coefficient) was estimated
entirely from BETWEEN-cycle variation (one GB value per cycle). Applying
it to within-cycle G_t movement would use it against an estimand it was
never fit against. This script therefore does NOT fold the calibrated
Delta-G shock into mu_i -- it validates sigma_G(Delta t) separately,
against realized historical G_t movement, on its own terms.

Three validations:
  A. Primary (Section 5.4's literal test): reconstruct each cycle's race
     state at a September snapshot (real, dated IE spend only, matching
     dynamic/simulate.py's existing point-in-time convention), compute
     mu_i via Paper I's unmodified pipeline, and Spearman-correlate
     against realized November margin -- does the model's Sept view
     rank-order eventual outcomes correctly?
  B. sigma_G(Delta t) check: does the calibrated term structure match
     REALIZED |Delta G| over the actual Sept -> Election Day window in
     2022 and 2024 (using the real historical 538 series)?
  C. eta(tier) stability check: refit eta on each cycle separately
     (leave-one-cycle-out) and compare against the pooled estimate --
     is the tiered pattern a stable feature or an artifact of pooling?

Output: outputs/simulator_validation_summary.json
"""

from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data import elections
from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.types import SigmaModel
from backtest.dynamic.simulate import (
    _static_floor_totals, _reconstruct_races_at, _has_dated_candidate_panel,
    _candidate_fallback_totals,
)

from estimate_eta_reaction import build_period_panel, build_delta_panel, fit_tiered_eta, TIERS
from estimate_gb_volatility import fit_lambda_from_term_structure

ROOT = Path(__file__).parent.parent
COMPETITIVE = {"Toss-Up", "Lean D", "Lean R"}


class SimulatorValidationError(Exception):
    """Raised when a validation check falls outside its pass criterion."""


# Pass criteria, calibrated against observed values (2026-07-23):
# A: rho was 0.407/0.556 (both p<0.01) -- positive AND significant is the
#    docstring's own stated test ("does the model's Sept view rank-order
#    eventual outcomes correctly"), not just a nonzero correlation.
# B: z was -1.44/-1.31 -- within 2 SD is the standard tolerance; both fields
#    already existed, just never enforced.
# C: per-tier eta across the two leave-one-cycle-out fits should be a stable
#    (same-signed) pattern, not independent noise -- checked via Spearman rho
#    between the two cycles' per-tier eta values across all 7 tiers.
VALIDATION_A_MIN_RHO = 0.0
VALIDATION_A_MAX_P = 0.05

CYCLE_CONFIG = {
    2022: {"processed_dir": ROOT / "data/processed_oos_2020", "election_day": date(2022, 11, 8)},
    2024: {"processed_dir": ROOT / "data/processed",          "election_day": date(2024, 11, 5)},
}
SEPTEMBER_1 = {2022: date(2022, 9, 1), 2024: date(2024, 9, 1)}


def load_coef_and_sigma(processed_dir: Path) -> tuple[MarginModelCoefficients, SigmaModel]:
    with open(processed_dir / "margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                              "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(processed_dir / "sigma_model.json") as f:
        sigma_coef = json.load(f)
    return coef, SigmaModel(_coef=sigma_coef)


# ─── Validation A: rank correlation, September snapshot vs. realized November ──

def validate_september_forecast(cycle: int) -> dict:
    cfg = CYCLE_CONFIG[cycle]
    coef, sigma_model = load_coef_and_sigma(cfg["processed_dir"])

    base_races = build_universe(cycle=cycle)
    static_totals = _static_floor_totals(cycle)
    use_dated_candidate_spend = _has_dated_candidate_panel(cycle)
    candidate_fallback_totals = None if use_dated_candidate_spend else _candidate_fallback_totals(cycle)
    sept_races = _reconstruct_races_at(
        period_index=0, period_date=SEPTEMBER_1[cycle], cycle=cycle,
        base_races=base_races, static_totals=static_totals,
        use_dated_candidate_spend=use_dated_candidate_spend,
        candidate_fallback_totals=candidate_fallback_totals,
    )
    outputs = compute_outputs_batch(sept_races, coef, sigma_model)

    mu_by_district = {o.district_id: o.mu_hat for o in outputs}
    tier_by_district = {r.district_id: r.cook_rating for r in sept_races}

    results = elections.load_results(cycle)
    realized = dict(zip(results["district_id"], results["margin_pp"]))

    rows = []
    for did, mu in mu_by_district.items():
        if did in realized and tier_by_district.get(did) in COMPETITIVE:
            rows.append({"district_id": did, "mu_sept": mu, "realized_margin": realized[did],
                         "tier": tier_by_district[did]})
    df = pd.DataFrame(rows)

    rho, p = stats.spearmanr(df["mu_sept"], df["realized_margin"])
    return {
        "cycle": cycle, "n_competitive": len(df),
        "spearman_rho": float(rho), "p_value": float(p),
        "sept_snapshot_date": str(SEPTEMBER_1[cycle]),
    }


# ─── Validation B: sigma_G(Delta t) vs. realized |Delta G| ─────────────────────

def realized_delta_g(cycle: int) -> float:
    """Actual |G(Sept 1) - G(Election Day)| from the real historical 538 series."""
    df = pd.read_csv(ROOT / "data/raw/generic_ballot/generic_ballot_historical_538.csv")
    df = df[df["cycle"] == cycle].copy()
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="candidate", values="pct_estimate").sort_index()
    gt = (piv["Democrats"] - piv["Republicans"]).asfreq("D").interpolate()

    sept = pd.Timestamp(SEPTEMBER_1[cycle])
    elec = pd.Timestamp(CYCLE_CONFIG[cycle]["election_day"])
    g_sept = gt.asof(sept)
    g_elec = gt.asof(elec)
    return float(g_elec - g_sept), (elec - sept).days


def validate_sigma_g(cycle: int, sigma_per_sqrt_day: float) -> dict:
    delta_g, delta_t_days = realized_delta_g(cycle)
    predicted_sd = sigma_per_sqrt_day * np.sqrt(delta_t_days)
    z_score = delta_g / predicted_sd
    return {
        "cycle": cycle, "delta_t_days": delta_t_days,
        "realized_delta_g": delta_g, "predicted_sd": predicted_sd,
        "z_score": z_score,
        "within_1_sd": bool(abs(z_score) <= 1.0),
        "within_2_sd": bool(abs(z_score) <= 2.0),
    }


# ─── Validation C: eta(tier) leave-one-cycle-out stability ─────────────────────

def validate_eta_stability() -> pd.DataFrame:
    rows = []
    for holdout_cycle in (2022, 2024):
        fit_cycle = 2024 if holdout_cycle == 2022 else 2022
        panel = build_period_panel(fit_cycle)
        delta = build_delta_panel(panel)
        result, _ = fit_tiered_eta(delta)
        result["fit_on_cycle"] = fit_cycle
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def check_lambda_consistency() -> dict:
    """Re-fit lambda fresh from outputs/gb_volatility_term_structure_historical_only.csv
    (same function scripts/estimate_gb_volatility.py used to write
    data/processed/gb_dynamics.json) and check it agrees with the stored,
    single-source-of-truth value -- a real self-consistency check, replacing
    an earlier version of this function that re-fit the same number and then
    discarded it without comparing to anything (Paper III audit, 2026-07-16)."""
    ts = pd.read_csv(ROOT / "outputs/gb_volatility_term_structure_historical_only.csv")
    lam_refit, a_refit, tau_refit = fit_lambda_from_term_structure(ts)
    with open(ROOT / "data/processed/gb_dynamics.json") as f:
        stored = json.load(f)
    rel_diff = abs(lam_refit - stored["lambda_decay"]) / stored["lambda_decay"]
    return {
        "lambda_refit": lam_refit, "tau_refit_days": tau_refit,
        "lambda_stored": stored["lambda_decay"], "tau_stored_days": stored["tau_days"],
        "relative_diff": rel_diff, "consistent": bool(rel_diff < 1e-6),
    }


def main():
    failures = []

    print("=== Validation A: Spearman(mu_Sept, realized_margin), competitive set ===")
    a_results = [validate_september_forecast(c) for c in (2022, 2024)]
    for r in a_results:
        r["passed"] = r["spearman_rho"] > VALIDATION_A_MIN_RHO and r["p_value"] < VALIDATION_A_MAX_P
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}] {r['cycle']}: n={r['n_competitive']}, rho={r['spearman_rho']:.3f} (p={r['p_value']:.4f})")
        if not r["passed"]:
            failures.append(f"{r['cycle']} Validation A: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f} "
                             f"does not meet rho>{VALIDATION_A_MIN_RHO} and p<{VALIDATION_A_MAX_P}")

    print("\n=== Validation B: sigma_G(Delta t) vs. realized |Delta G|, Sept->Election Day ===")
    with open(ROOT / "data/processed/gb_dynamics.json") as f:
        gb_dynamics = json.load(f)
    sigma_per_sqrt_day = gb_dynamics["sigma_g_per_sqrt_day"]   # single source of truth (Section 5.3)
    b_results = [validate_sigma_g(c, sigma_per_sqrt_day) for c in (2022, 2024)]
    for r in b_results:
        tag = "PASS" if r["within_2_sd"] else "FAIL"
        print(f"  [{tag}] {r['cycle']}: Delta_t={r['delta_t_days']}d, realized_dG={r['realized_delta_g']:+.2f}, "
              f"predicted_sd={r['predicted_sd']:.2f}, z={r['z_score']:+.2f}, "
              f"within 1sd={r['within_1_sd']}, within 2sd={r['within_2_sd']}")
        if not r["within_2_sd"]:
            failures.append(f"{r['cycle']} Validation B: z={r['z_score']:+.2f} exceeds 2 SD")

    print("\n=== Validation C: eta(tier) leave-one-cycle-out ===")
    c_results = validate_eta_stability()
    pivot = c_results.pivot(index="tier", columns="fit_on_cycle", values="eta")
    print(pivot.to_string())
    stability_rho, stability_p = stats.spearmanr(pivot[2022], pivot[2024])
    c_passed = bool(stability_rho > 0)
    tag = "PASS" if c_passed else "FAIL"
    print(f"  [{tag}] cross-cycle per-tier eta stability: Spearman rho={stability_rho:.3f} (p={stability_p:.4f})")
    if not c_passed:
        failures.append(f"Validation C: cross-cycle per-tier eta Spearman rho={stability_rho:.3f} is not positive "
                         "-- the tiered eta pattern does not replicate across held-out cycles")

    print("\n=== lambda consistency check: re-fit vs. data/processed/gb_dynamics.json ===")
    lam_check = check_lambda_consistency()
    tag = "PASS" if lam_check["consistent"] else "FAIL"
    print(f"  [{tag}] refit lambda = {lam_check['lambda_refit']:.5f} (tau = {lam_check['tau_refit_days']:.1f}d)  "
          f"vs. stored = {lam_check['lambda_stored']:.5f} (tau = {lam_check['tau_stored_days']:.1f}d)  "
          f"-- consistent: {lam_check['consistent']} (rel. diff {lam_check['relative_diff']:.2e})")
    if not lam_check["consistent"]:
        failures.append(f"lambda consistency check: relative diff {lam_check['relative_diff']:.2e} exceeds 1e-6")

    summary = {
        "validation_a_september_forecast": a_results,
        "validation_b_sigma_g_check": b_results,
        "validation_c_eta_stability": c_results.to_dict(orient="records"),
        "validation_c_stability_rho": stability_rho,
        "validation_c_stability_p": stability_p,
        "lambda_consistency_check": lam_check,
    }
    out_path = ROOT / "outputs/simulator_validation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")

    if failures:
        raise SimulatorValidationError(
            "Simulator validation check(s) failed:\n  " + "\n  ".join(failures)
        )
    print("\nAll simulator validation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except SimulatorValidationError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
