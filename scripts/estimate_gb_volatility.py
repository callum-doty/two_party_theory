#!/usr/bin/env python3
"""
Estimate the national-environment (generic ballot) volatility term
structure sigma_G(delta_t) -- Paper III Section 5.

Pools four historical cycles (2018, 2020, 2022, 2024; 538's own daily
trend estimate, recovered via the Wayback Machine -- see
docs/data_catalog.md Section 1.8b) with the live 2026 series (this
project's own 21-day trailing average of raw polls) to get a genuinely
cross-cycle realized-volatility estimate, rather than the single-path,
one-cycle exercise this project could previously only support.

Output: outputs/gb_volatility_term_structure.csv
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).parent.parent
HORIZONS_DAYS = [30, 60, 90, 180, 270, 365, 450]

# Reference horizon for the single representative sigma_G(per-sqrt-day) figure
# quoted throughout Paper III (docs/paper3_draft.md Section 5.3's "working
# number for the calibration") -- chosen to match Paper II Section 7.1's live
# run horizon (~4 months out from a typical decision point), not hand-picked.
REFERENCE_HORIZON_DAYS = 120.0


def load_historical_series() -> dict[int, pd.Series]:
    """Return {cycle: G_t series indexed by date}, G_t = dem% - rep%."""
    df = pd.read_csv(ROOT / "data/raw/generic_ballot/generic_ballot_historical_538.csv")
    df["date"] = pd.to_datetime(df["date"])
    series = {}
    for cycle, g in df.groupby("cycle"):
        piv = g.pivot_table(index="date", columns="candidate", values="pct_estimate")
        piv = piv.sort_index()
        gt = (piv["Democrats"] - piv["Republicans"]).asfreq("D").interpolate()
        series[int(cycle)] = gt
    return series


def load_live_2026_series() -> pd.Series:
    """21-day trailing average of the raw 2026 polls (same construction as
    Paper III Section 5.3's original single-cycle exercise)."""
    df = pd.read_csv(ROOT / "data/live/generic_ballot_polls.csv")
    df["mid_date"] = pd.to_datetime(df["start_date"]) + \
        (pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])) / 2
    df = df.sort_values("mid_date").reset_index(drop=True)
    daily = df.set_index("mid_date")["gb"].resample("D").mean().interpolate()
    return daily.rolling("21D").mean().dropna()


def realized_vol_by_horizon(series: pd.Series, horizons: list[int]) -> dict[int, tuple[float, int]]:
    """Return {horizon_days: (std(delta_G), n_pairs)} for one series."""
    out = {}
    for h in horizons:
        shifted = series.shift(-h)
        delta = (shifted - series).dropna()
        if len(delta) < 5:
            continue
        out[h] = (float(delta.std()), int(len(delta)))
    return out


def pooled_vol_by_horizon(all_series: dict[str, pd.Series], horizons: list[int]) -> pd.DataFrame:
    """Pool all delta_G observations across every series (cycle) at each
    horizon, then take one std -- this is the actual cross-cycle pooling,
    not just an average of each cycle's own std."""
    rows = []
    for h in horizons:
        pooled_deltas = []
        n_cycles_covering = 0
        for name, s in all_series.items():
            shifted = s.shift(-h)
            delta = (shifted - s).dropna()
            if len(delta) >= 5:
                pooled_deltas.append(delta.values)
                n_cycles_covering += 1
        if not pooled_deltas:
            continue
        pooled = np.concatenate(pooled_deltas)
        rows.append({
            "horizon_days": h,
            "horizon_months": round(h / 30.44, 1),
            "n_cycles_covering": n_cycles_covering,
            "n_pooled_obs": len(pooled),
            "pooled_std_dG": float(np.std(pooled, ddof=1)),
            "pooled_std_over_sqrt_days": float(np.std(pooled, ddof=1) / np.sqrt(h)),
        })
    return pd.DataFrame(rows)


def representative_sigma_g_per_sqrt_day(term_structure: pd.DataFrame,
                                         horizon_days: float = REFERENCE_HORIZON_DAYS) -> float:
    """Interpolate the pooled std(dG)/sqrt(days) term structure at a single
    reference horizon, rather than hand-picking one of the table's rows --
    this is what makes the "0.186" figure quoted in Paper III reproducible
    from data instead of a literal someone typed once and never revisited."""
    return float(np.interp(horizon_days, term_structure["horizon_days"],
                            term_structure["pooled_std_over_sqrt_days"]))


def fit_lambda_from_term_structure(term_structure: pd.DataFrame) -> tuple[float, float, float]:
    """Fit Var(dG)(t) = A(1 - exp(-t/tau)) to a pooled term structure,
    return (lambda=1/tau, A, tau). Used for Section 6.2's epsilon-decay
    proxy rate -- moved here (previously duplicated/discarded in
    scripts/validate_state_simulator.py) so there is exactly one function
    that fits this number, called once here and re-checked (not re-derived
    independently) by validate_state_simulator.py."""
    t = term_structure["horizon_days"].values.astype(float)
    var = term_structure["pooled_std_dG"].values ** 2

    def model(t, A, tau):
        return A * (1 - np.exp(-t / tau))

    popt, _ = curve_fit(model, t, var, p0=[20.0, 300.0], maxfev=5000)
    a_fit, tau_fit = popt
    return 1.0 / tau_fit, a_fit, tau_fit


def main():
    hist = load_historical_series()
    live_2026 = load_live_2026_series()

    all_series = {f"{c}_538": s for c, s in hist.items()}
    all_series["2026_live"] = live_2026

    print("=== Per-series length ===")
    for name, s in all_series.items():
        print(f"  {name}: {len(s)} days, {s.index.min().date()} -> {s.index.max().date()}")

    print("\n=== Per-cycle realized volatility (not pooled) ===")
    per_cycle_rows = []
    for name, s in all_series.items():
        vol = realized_vol_by_horizon(s, HORIZONS_DAYS)
        for h, (sd, n) in vol.items():
            per_cycle_rows.append({"series": name, "horizon_days": h, "std_dG": sd, "n_pairs": n})
    per_cycle_df = pd.DataFrame(per_cycle_rows)
    print(per_cycle_df.pivot(index="horizon_days", columns="series", values="std_dG").to_string())

    print("\n=== Pooled across all 5 series (4 historical cycles + 2026 live) ===")
    pooled = pooled_vol_by_horizon(all_series, HORIZONS_DAYS)
    print(pooled.to_string(index=False))

    print("\n=== Pooled across the 4 HISTORICAL cycles only (excludes 2026, methodologically cleaner) ===")
    hist_only = {f"{c}_538": s for c, s in hist.items()}
    pooled_hist = pooled_vol_by_horizon(hist_only, HORIZONS_DAYS)
    print(pooled_hist.to_string(index=False))

    out_dir = ROOT / "outputs"
    per_cycle_df.to_csv(out_dir / "gb_volatility_by_cycle.csv", index=False)
    pooled.to_csv(out_dir / "gb_volatility_term_structure.csv", index=False)
    pooled_hist.to_csv(out_dir / "gb_volatility_term_structure_historical_only.csv", index=False)
    print(f"\nSaved -> {out_dir / 'gb_volatility_term_structure.csv'}")
    print(f"Saved -> {out_dir / 'gb_volatility_term_structure_historical_only.csv'}")

    # --- Single source of truth for sigma_G(dt) and lambda (Paper III audit, 2026-07-16):
    # write the two calibrated constants every downstream script/paper cites into one
    # JSON, instead of each re-typing its own copy of a number computed here. Uses the
    # historical-only (4-cycle) term structure, per Section 5.3's stated preference for
    # methodological cleanliness (538's own aggregation vs. this project's raw-poll
    # smoothing are not identically constructed for the live 2026 series).
    sigma_g = representative_sigma_g_per_sqrt_day(pooled_hist)
    lam, a_fit, tau_fit = fit_lambda_from_term_structure(pooled_hist)
    gb_dynamics = {
        "sigma_g_per_sqrt_day": sigma_g,
        "reference_horizon_days": REFERENCE_HORIZON_DAYS,
        "lambda_decay": lam,
        "tau_days": tau_fit,
        "lambda_fit_A": a_fit,
        "source": "scripts/estimate_gb_volatility.py, historical-only (4-cycle) pooled term structure",
    }
    with open(ROOT / "data/processed/gb_dynamics.json", "w") as f:
        json.dump(gb_dynamics, f, indent=2)
    print(f"\n=== Single source of truth written: data/processed/gb_dynamics.json ===")
    print(f"  sigma_g_per_sqrt_day={sigma_g:.4f} (at {REFERENCE_HORIZON_DAYS:.0f}d), "
          f"lambda_decay={lam:.5f} (tau={tau_fit:.1f}d)")


if __name__ == "__main__":
    main()
