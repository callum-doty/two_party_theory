#!/usr/bin/env python3
"""
Test a specific hypothesis about the persuasion ceiling's "not full
resolution" of the Likely-R/Safe-R concentration in the 2026 live gain
decomposition (gain_decomposition_2026_by_race.csv): that the ceiling
successfully suppresses truly deterministic (Safe-tier, Phi0 near 0 or 1)
races, but under-suppresses an intermediate zone -- races with moderate
persuadability (Phi0 not extreme) that ALSO have a near-zero real floor,
where the raw MSG blowup as D->0 and the ceiling's own persuadability-
scaled headroom both stay large at the same time.

Reuses check_deep_pvi_ceiling.py's exact saturation/robustness methodology
(model/ceiling.py's own functions -- no reimplementation), applied to the
races actually driving the Likely R / Safe R contribution
(gain_decomposition_2026_by_race.csv, already computed).

Usage:
    python scripts/check_2026_likely_r_ceiling_balance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import predict, predict_floor_margin
from backtest.model import ceiling as ceiling_mod
from run_backtest import load_processed_artifacts  # type: ignore


def main() -> None:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races_by_id = {r.district_id: r for r in build_universe(cycle=2026)}
    c_max = config.persuasion_ceiling_c_max()

    df = pd.read_csv(config.outputs_path() / "gain_decomposition_2026_by_race.csv")

    def diagnose(subset_df: pd.DataFrame, label: str, n: int) -> pd.DataFrame:
        rows = []
        for _, row in subset_df.head(n).iterrows():
            race = races_by_id[row["district_id"]]
            mu_floor = predict_floor_margin(
                pvi=race.pvi, incumb_status=race.incumb_status,
                generic_ballot=race.generic_ballot, cand_d_total=race.cand_d_total,
                r_total=race.r_total, coef=coef, cvap=race.cvap, indiv_share=race.indiv_share,
            )
            sigma_i = sigma_model.predict(abs(race.pvi), race.incumb_status, race.generic_ballot)
            phi0 = float(norm.cdf(mu_floor / sigma_i))
            persuadability = float(4.0 * phi0 * (1.0 - phi0))
            C_i = float(ceiling_mod.ceiling(mu_floor, sigma_i, c_max))

            d_recommended = float(row["model_party_dollars"]) + race.cand_d_total
            ratio = np.clip(d_recommended / (d_recommended + race.r_total), 1e-6, 1 - 1e-6)
            mu_raw = predict(
                pvi=race.pvi, incumb_status=race.incumb_status, generic_ballot=race.generic_ballot,
                ratio=ratio, coef=coef, total_spend=d_recommended + race.r_total,
                cvap=race.cvap, indiv_share=race.indiv_share,
            )
            mu_capped, grad_factor = ceiling_mod.apply(mu_raw, mu_floor, sigma_i, c_max)
            saturation = 1.0 - float(grad_factor)

            rows.append({
                "district_id": race.district_id, "pvi": round(race.pvi, 2),
                "cand_floor": round(race.cand_d_total, 0),
                "phi0_at_floor": round(phi0, 4),
                "persuadability_4phi0(1-phi0)": round(persuadability, 4),
                "C_i_ceiling_pp": round(C_i, 3),
                "mu_floor_pp": round(mu_floor, 2),
                "mu_raw_uncapped_pp": round(mu_raw, 2),
                "mu_capped_pp": round(mu_capped, 2),
                "saturation_pct": round(saturation * 100, 1),
                "model_party_$": round(row["model_party_dollars"], 0),
                "delta_seats": round(row["delta_seats"], 4),
            })
        out = pd.DataFrame(rows)
        print(f"\n{label} (n={len(out)}):")
        print(out.to_string(index=False))
        return out

    likely_r = df[df.cook_rating == "Likely R"].sort_values("delta_seats", ascending=False)
    safe_r = df[df.cook_rating == "Safe R"].sort_values("delta_seats", ascending=False)

    likely_r_diag = diagnose(likely_r, "Top Likely R contributors", 10)
    safe_r_diag = diagnose(safe_r, "Top Safe R contributors", 6)

    print(f"\nMean Phi0 at floor -- Likely R top-10: {likely_r_diag['phi0_at_floor'].mean():.3f}   "
          f"Safe R top-6: {safe_r_diag['phi0_at_floor'].mean():.3f}")
    print(f"Mean persuadability -- Likely R top-10: {likely_r_diag['persuadability_4phi0(1-phi0)'].mean():.3f}   "
          f"Safe R top-6: {safe_r_diag['persuadability_4phi0(1-phi0)'].mean():.3f}")
    print(f"Mean saturation -- Likely R top-10: {likely_r_diag['saturation_pct'].mean():.1f}%   "
          f"Safe R top-6: {safe_r_diag['saturation_pct'].mean():.1f}%")

    all_races = pd.DataFrame([{"district_id": d, "pvi": races_by_id[d].pvi,
                                "cand_floor": races_by_id[d].cand_d_total}
                               for d in df["district_id"]])
    out_path = config.outputs_path() / "likely_r_ceiling_balance_2026.csv"
    pd.concat([likely_r_diag.assign(tier="Likely R"), safe_r_diag.assign(tier="Safe R")]) \
        .to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
