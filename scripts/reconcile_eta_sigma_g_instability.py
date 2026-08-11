#!/usr/bin/env python3
"""
Reconciles two calibration-reliability caveats already on record in Paper III
Section 5.5, both flagged but never resolved (raised again while reviewing
outputs/eta_reaction_by_tier_fig.png and outputs/gb_asymmetry_check_fig.png):

Part 1 -- eta(tier) leave-one-cycle-out instability (Validation C). Toss-Up
swings 0.73 (2022) to 0.34 (2024); Safe D and Likely R flip sign entirely.
Question: is this real cycle-to-cycle heterogeneity, or just estimation
noise from small per-tier, per-cycle samples? A formal check -- does each
tier's 2022 CI overlap its 2024 CI, accounting for each fit's own SE -- can
tell these apart, the same way the interaction test did for beta_RC's
tier-composition question (Section 10.1 addendum, 2026-07-17).

Part 2 -- sigma_G's realized late-cycle bias (Validation B) only checked
2022 and 2024 (the two cycles used elsewhere as held-out validation years),
both landing at z~-1.3 to -1.4 in the SAME direction (toward Republicans).
This session's OU-drift fit (scripts/estimate_gb_ou_drift.py) found no
significant LEVEL mean-reversion pooling all lags across all 4 historical
cycles (p=0.37) -- a different question than whether the specific LATE-CYCLE
(Sept-to-Election-Day) window shows a directional pattern the OU model,
fit on all lags uniformly, wouldn't isolate. Extending Validation B's exact
check to all 4 historical cycles (not just 2) tests this directly.

Part 3 (added 2026-07-17, extending Part 1 back to 2012) -- the 2-cycle
CI-overlap check in Part 1 is honestly underpowered (only one comparison per
tier). `load_ie_transactions_dated()` already has usable, dated IE data for
2012/2014/2016/2018/2020 -- and per Paper III Section 4.2's own data-quality
table, blank exp_date rates are actually LOWER in these older cycles
(0-17.5%) than in 2022/2024 (28-33%), so this is not a data-quality downgrade,
it is better raw data than the two cycles currently used. This fits eta
separately on each of the 7 cycles per tier (a forest-plot-style table) and
runs one joint test per tier -- does allowing eta to vary across all 7
cycles significantly improve the fit over a single pooled eta? -- rather
than 21 unadjusted pairwise comparisons.

Output: outputs/eta_instability_reconciliation.csv,
        outputs/sigma_g_late_cycle_reconciliation.csv,
        outputs/eta_seven_cycle_extension.csv, printed summary.
"""

from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from estimate_eta_reaction import build_period_panel, build_delta_panel, fit_tiered_eta, TIERS
from estimate_gb_volatility import load_historical_series

COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
ELECTION_DAYS = {2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
SIGMA_PER_SQRT_DAY = 0.1838   # data/processed/gb_dynamics.json's historical-only figure
ALL_ETA_CYCLES = [2012, 2014, 2016, 2018, 2020, 2022, 2024]


# ═══ Part 1: eta(tier) instability -- CI overlap, not just point-estimate comparison ═══

def eta_instability_check() -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        per_cycle = {}
        for cycle in (2022, 2024):
            panel = build_period_panel(cycle)
            delta = build_delta_panel(panel)
            fit, _ = fit_tiered_eta(delta)
            match = fit[fit["tier"] == tier]
            if match.empty:
                continue
            per_cycle[cycle] = match.iloc[0]

        if len(per_cycle) < 2:
            continue
        e22, e24 = per_cycle[2022], per_cycle[2024]
        lo22, hi22 = e22["eta"] - 1.96 * e22["se"], e22["eta"] + 1.96 * e22["se"]
        lo24, hi24 = e24["eta"] - 1.96 * e24["se"], e24["eta"] + 1.96 * e24["se"]
        overlap = not (hi22 < lo24 or hi24 < lo22)

        rows.append({
            "tier": tier,
            "n_2022": int(e22["n_obs"]), "eta_2022": e22["eta"], "se_2022": e22["se"],
            "n_2024": int(e24["n_obs"]), "eta_2024": e24["eta"], "se_2024": e24["se"],
            "ci_2022": f"[{lo22:.2f}, {hi22:.2f}]", "ci_2024": f"[{lo24:.2f}, {hi24:.2f}]",
            "cis_overlap": overlap,
            "point_estimate_ratio": abs(e22["eta"]) / max(abs(e24["eta"]), 1e-9),
        })
    return pd.DataFrame(rows)


# ═══ Part 2: sigma_G late-cycle realized move, all 4 historical cycles ═══

def sigma_g_late_cycle_check() -> pd.DataFrame:
    hist = load_historical_series()
    sept_1 = {c: date(c, 9, 1) for c in ELECTION_DAYS}

    rows = []
    for cycle, gt in hist.items():
        elec = pd.Timestamp(ELECTION_DAYS[cycle])
        sept = pd.Timestamp(sept_1[cycle])
        g_sept = gt.asof(sept)
        g_elec = gt.asof(elec)
        delta_t_days = (elec - sept).days
        realized_delta = float(g_elec - g_sept)
        predicted_sd = SIGMA_PER_SQRT_DAY * np.sqrt(delta_t_days)
        z = realized_delta / predicted_sd
        rows.append({
            "cycle": cycle, "delta_t_days": delta_t_days,
            "realized_delta_g": realized_delta, "predicted_sd": predicted_sd,
            "z_score": z, "direction": "toward R" if realized_delta < 0 else "toward D",
        })
    return pd.DataFrame(rows).sort_values("cycle")


# ═══ Part 3: eta(tier) across all 7 cycles (2012-2024) -- forest table + joint test ═══

def build_seven_cycle_panel() -> pd.DataFrame:
    frames = []
    for cycle in ALL_ETA_CYCLES:
        panel = build_period_panel(cycle)
        delta = build_delta_panel(panel)
        delta = delta.copy()
        delta["cycle"] = cycle
        frames.append(delta)
    return pd.concat(frames, ignore_index=True)


def eta_per_cycle_forest(all_panel: pd.DataFrame) -> pd.DataFrame:
    """Single-cycle eta fit for every (tier, cycle) with >=10 observations --
    the 7-cycle generalization of Part 1's 2-point comparison."""
    rows = []
    for tier in TIERS:
        for cycle in ALL_ETA_CYCLES:
            sub = all_panel[(all_panel["tier"] == tier) & (all_panel["cycle"] == cycle)]
            if len(sub) < 10:
                continue
            X = sm.add_constant(sub["d_ie_delta_lag_dm"])
            y = sub["r_ie_delta_dm"]
            f = sm.OLS(y, X).fit(cov_type="HC3")
            rows.append({
                "tier": tier, "cycle": cycle, "n_obs": len(sub),
                "eta": float(f.params["d_ie_delta_lag_dm"]),
                "se": float(f.bse["d_ie_delta_lag_dm"]),
            })
    return pd.DataFrame(rows)


def eta_joint_cycle_test(all_panel: pd.DataFrame) -> pd.DataFrame:
    """For each tier: does letting eta vary freely across all 7 cycles
    (interaction terms) significantly beat a single pooled eta for that
    tier? One joint Wald test per tier, using the same HC3-robust covariance
    as every other eta regression in this project, instead of 21 unadjusted
    pairwise CI comparisons."""
    rows = []
    for tier in TIERS:
        sub = all_panel[all_panel["tier"] == tier].copy()
        cycles_present = sorted(sub["cycle"].unique())
        if len(cycles_present) < 2 or len(sub) < 20:
            continue
        base_cycle = cycles_present[0]
        interaction_cols = []
        for c in cycles_present[1:]:
            col = f"dlr_x_{c}"
            sub[col] = sub["d_ie_delta_lag_dm"] * (sub["cycle"] == c).astype(float)
            interaction_cols.append(col)

        X = sm.add_constant(sub[["d_ie_delta_lag_dm"] + interaction_cols])
        y = sub["r_ie_delta_dm"]
        f = sm.OLS(y, X).fit(cov_type="HC3")

        hyp = ", ".join(f"{c} = 0" for c in interaction_cols)
        test = f.f_test(hyp)

        rows.append({
            "tier": tier, "n_cycles": len(cycles_present), "n_obs_total": len(sub),
            "pooled_eta_all_cycles": float(f.params["d_ie_delta_lag_dm"]),
            "joint_f_stat": float(test.fvalue), "joint_p_value": float(test.pvalue),
            "eta_varies_by_cycle": bool(test.pvalue < 0.05),
        })
    return pd.DataFrame(rows)


def main():
    print("=== Part 1: eta(tier) leave-one-cycle-out -- CI overlap, not just point estimates ===\n")
    eta_check = eta_instability_check()
    pd.set_option("display.width", 160)
    print(eta_check[["tier", "n_2022", "eta_2022", "n_2024", "eta_2024", "cis_overlap"]]
          .to_string(index=False))

    n_overlap = eta_check["cis_overlap"].sum()
    print(f"\n{n_overlap}/{len(eta_check)} tiers have OVERLAPPING 95% CIs across cycles despite the "
          f"large point-estimate swings.")
    print("-> Where CIs overlap: cannot distinguish real cycle-to-cycle change from estimation noise")
    print("   at this sample size -- the 'instability' is honestly reported as unresolved, not as a")
    print("   confirmed unstable parameter.")
    print("-> Where CIs do NOT overlap: a case for genuine cycle-to-cycle heterogeneity, independent")
    print("   of sample-size limitations.")
    for _, r in eta_check.iterrows():
        if not r["cis_overlap"]:
            print(f"     {r['tier']}: {r['ci_2022']} (2022) vs {r['ci_2024']} (2024) -- NO overlap")

    eta_check.to_csv(ROOT / "outputs/eta_instability_reconciliation.csv", index=False)
    print(f"\nSaved -> outputs/eta_instability_reconciliation.csv")

    print("\n=== Part 2: sigma_G late-cycle realized move, all 4 historical cycles (not just 2022/2024) ===\n")
    sigma_check = sigma_g_late_cycle_check()
    print(sigma_check.to_string(index=False))

    same_direction = (sigma_check["realized_delta_g"] < 0).sum()
    print(f"\n{same_direction}/4 cycles moved toward Republicans in the Sept-1-to-Election-Day window.")
    mean_z = sigma_check["z_score"].mean()
    print(f"Mean z-score across all 4 cycles: {mean_z:+.2f} (2022/2024-only mean was ~-1.37)")

    sigma_check.to_csv(ROOT / "outputs/sigma_g_late_cycle_reconciliation.csv", index=False)
    print(f"\nSaved -> outputs/sigma_g_late_cycle_reconciliation.csv")

    print("\n=== Part 3: eta(tier) across all 7 cycles (2012-2024), not just 2022 vs. 2024 ===\n")
    all_panel = build_seven_cycle_panel()
    print(f"Combined panel: {len(all_panel)} delta-panel rows across {ALL_ETA_CYCLES}\n")

    forest = eta_per_cycle_forest(all_panel)
    print("Per-cycle single-cycle eta fits (n>=10 required):")
    pivot = forest.pivot(index="tier", columns="cycle", values="eta").reindex(COOK_ORDER)
    print(pivot.to_string(float_format=lambda x: f"{x:+.2f}" if pd.notna(x) else "  . "))
    forest.to_csv(ROOT / "outputs/eta_seven_cycle_extension.csv", index=False)

    print("\nJoint test per tier: does eta vary significantly across all 7 cycles?")
    joint = eta_joint_cycle_test(all_panel)
    print(joint[["tier", "n_cycles", "n_obs_total", "pooled_eta_all_cycles",
                  "joint_f_stat", "joint_p_value", "eta_varies_by_cycle"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    n_real = int(joint["eta_varies_by_cycle"].sum())
    print(f"\n{n_real}/{len(joint)} tiers show STATISTICALLY SIGNIFICANT cycle-to-cycle variation "
          f"(joint test, all 7 cycles pooled) -- a properly powered answer, not the 2-cycle\n"
          f"pairwise check's necessarily weaker one.")

    joint.to_csv(ROOT / "outputs/eta_seven_cycle_joint_test.csv", index=False)
    print(f"Saved -> outputs/eta_seven_cycle_extension.csv, outputs/eta_seven_cycle_joint_test.csv")


if __name__ == "__main__":
    main()
