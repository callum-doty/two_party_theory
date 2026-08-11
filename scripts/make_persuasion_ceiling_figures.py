#!/usr/bin/env python3
"""
Four figures for the persuasion-ceiling fix (FINDINGS.md Section 7.3b,
src/backtest/model/ceiling.py), matching the blog-post narrative walkthrough
of why the ceiling was built and what it changed:

  1. persuasion_ceiling_curve_fig.png       -- C(Phi_0) = c_max * 4*Phi_0*(1-Phi_0),
     the ceiling equation itself.
  2. persuasion_ceiling_response_surface_fig.png -- raw (uncapped) vs. capped
     win probability as party $ grows, on four real 2024 exemplar races
     (Toss-Up, Lean D, Safe D, Safe R).
  3. persuasion_ceiling_cmax_sweep_fig.png   -- party-budget share by tier
     level, across an 8-point c_max robustness sweep {3,5,7,10,15,20,30}
     (the range behind config.yaml's persuasion_ceiling.c_max choice).
  4. persuasion_ceiling_tier_allocation_fig.png -- average model-recommended
     vs. DCCC-observed party $ per race, by tier level, at the shipped
     c_max=10.0.

Everything here is recomputed live from the fitted 2024 pipeline
(data/processed/margin_model_coef.json, sigma_model.json, config.yaml) --
no cached/precomputed numbers. The c_max sweep re-solves optimize_nonlinear()
from scratch at each of 8 c_max values (~60-90s each on the 433-race live
universe), so this script takes several minutes end to end.

Output: outputs/persuasion_ceiling_*_fig.png (4 files)
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients, predict
from backtest.model.win_prob import compute_outputs_batch
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import optimize_nonlinear
from backtest.types import SigmaModel

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

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

ACCENT = "#2a78d6"       # capped / model-recommended
BASELINE = "#9a988f"     # raw (uncapped) / DCCC observed
SERIES_COMP = "#2a78d6"
SERIES_LIKELY = "#c9702e"
SERIES_SAFE = "#1b9e6f"


def load_pipeline():
    with open(ROOT / "data/processed/margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(ROOT / "data/processed/sigma_model.json") as f:
        sigma_coef = json.load(f)
    sigma_model = SigmaModel(_coef=sigma_coef)
    races = build_universe(cycle=2024)
    return races, coef, sigma_model


# ═══ 1. The persuasion ceiling curve ═════════════════════════════════════════

def fig1_ceiling_curve(c_max: float):
    phi0 = np.linspace(0.001, 0.999, 400)
    persuadability = 4.0 * phi0 * (1.0 - phi0)
    ceiling = c_max * persuadability

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(phi0, ceiling, color=ACCENT, alpha=0.12, linewidth=0)
    ax.plot(phi0, ceiling, color=ACCENT, linewidth=2.2)
    ax.scatter([0.5], [c_max], color=ACCENT, s=36, zorder=5, edgecolor="white", linewidth=1.2)
    ax.annotate(f"Φ₀ = 0.5 → C = {c_max:.1f} (peak)", xy=(0.5, c_max), xytext=(0.56, c_max * 0.92),
                fontsize=9.5, color="#333")
    ax.axvline(0.5, color="#c9c8bd", linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, c_max * 1.1)
    ax.set_xlabel("win probability at candidate-only floor (Φ₀)")
    ax.set_ylabel("persuasion ceiling C (margin points)")
    ax.set_title(f"The persuasion ceiling: C(Φ₀) = c_max · 4Φ₀(1−Φ₀), c_max = {c_max:.1f}", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "persuasion_ceiling_curve_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote persuasion_ceiling_curve_fig.png")


# ═══ 2. Response surface: raw vs. capped, on 4 real exemplar races ══════════

def pick_exemplar(races, tier):
    cands = [r for r in races if r.cook_rating == tier and r.cand_d_total > 1000 and r.r_total > 1000]
    cands.sort(key=lambda r: abs(r.pvi))
    return cands[len(cands) // 2]


def fig2_response_surface(races, coef, sigma_model, c_max: float):
    outputs0 = compute_outputs_batch(races, coef, sigma_model)
    sigma_by_id = {o.district_id: o.sigma_i for o in outputs0}

    tiers = ["Toss-Up", "Lean D", "Safe D", "Safe R"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))

    for ax, tier in zip(axes.flat, tiers):
        r = pick_exemplar(races, tier)
        floor_d = r.cand_d_total
        floor_total = floor_d + r.r_total
        floor_ratio = np.clip(floor_d / floor_total, 1e-6, 1 - 1e-6)
        mu_floor = predict(pvi=r.pvi, incumb_status=r.incumb_status, generic_ballot=r.generic_ballot,
                            ratio=floor_ratio, coef=coef, total_spend=floor_total, cvap=r.cvap,
                            indiv_share=r.indiv_share)
        sigma_i = sigma_by_id[r.district_id]
        phi0 = float(norm.cdf(mu_floor / sigma_i))

        party_grid = np.linspace(0, max(floor_d * 8, 2_000_000), 250)
        d_grid = floor_d + party_grid
        total_grid = d_grid + r.r_total
        ratio_grid = np.clip(d_grid / total_grid, 1e-6, 1 - 1e-6)
        mu_raw = np.array([
            predict(pvi=r.pvi, incumb_status=r.incumb_status, generic_ballot=r.generic_ballot,
                    ratio=rat, coef=coef, total_spend=tot, cvap=r.cvap, indiv_share=r.indiv_share)
            for rat, tot in zip(ratio_grid, total_grid)
        ])
        mu_capped, _ = ceiling_mod.apply(mu_raw, mu_floor, sigma_i, c_max)
        p_raw = norm.cdf(mu_raw / sigma_i)
        p_capped = norm.cdf(mu_capped / sigma_i)

        ax.plot(party_grid / 1e6, p_raw, color=BASELINE, linewidth=2, linestyle=(0, (4, 3)),
                label="raw (uncapped)")
        ax.plot(party_grid / 1e6, p_capped, color=ACCENT, linewidth=2, label="capped")
        ax.scatter([party_grid[-1] / 1e6], [p_raw[-1]], color=BASELINE, s=26, zorder=5,
                   edgecolor="white", linewidth=1)
        ax.scatter([party_grid[-1] / 1e6], [p_capped[-1]], color=ACCENT, s=26, zorder=5,
                   edgecolor="white", linewidth=1)
        ax.set_title(f"{tier} — {r.district_id}  (Φ₀={phi0:.2f})", fontsize=10.5)
        ax.set_xlabel("additional party $ (millions)", fontsize=9.5)
        ax.set_ylabel("win probability", fontsize=9.5)
        ax.tick_params(labelsize=9)

    axes.flat[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Response surface, before and after the ceiling — 4 real 2024 races", fontsize=12.5, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "persuasion_ceiling_response_surface_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote persuasion_ceiling_response_surface_fig.png")


# ═══ 3. c_max robustness sweep ═══════════════════════════════════════════════

SAFE_TIERS = {"Safe D", "Safe R"}
COMPETITIVE_TIERS = {"Toss-Up", "Lean D", "Lean R"}
LIKELY_TIERS = {"Likely D", "Likely R"}
SWEEP_C_MAX = [3, 5, 7, 10, 15, 20, 30]


SWEEP_CACHE = OUT / ".persuasion_ceiling_sweep_cache.npz"


def run_cmax_sweep(races, coef, sigma_model, c_max_default: float, use_cache: bool = True):
    """Re-solves optimize_nonlinear() at each SWEEP_C_MAX value (~60-90s each,
    ~7-9 minutes total on the 433-race live universe). Cached to
    SWEEP_CACHE (gitignored, dev-ergonomics only) since this is the
    expensive step and fig3/fig4 are pure post-processing of its output --
    delete the cache file (or pass use_cache=False) to force a fresh solve,
    e.g. after config.yaml or the fitted coefficients change."""
    if use_cache and SWEEP_CACHE.exists():
        cached = np.load(SWEEP_CACHE)
        if list(cached["c_max"]) == SWEEP_C_MAX:
            print("    (using cached sweep -- delete "
                  f"{SWEEP_CACHE.relative_to(ROOT)} to force a fresh solve)")
            return [
                {"c_max": float(cached["c_max"][i]), "expected_seats": float(cached["expected_seats"][i]),
                 "safe": float(cached["safe"][i]), "competitive": float(cached["competitive"][i]),
                 "likely": float(cached["likely"][i]), "allocations": cached["allocations"][i]}
                for i in range(len(SWEEP_C_MAX))
            ]

    budget = sum(r.d_total for r in races)
    party_budget = sum(r.d_total - r.cand_d_total for r in races)
    cov = np.eye(len(races)) * 1e-9

    rows = []
    for cm in SWEEP_C_MAX:
        t0 = time.time()
        config._cfg["persuasion_ceiling"]["c_max"] = cm
        res = optimize_nonlinear(races, coef, sigma_model, budget=budget, cov_matrix=cov,
                                  gamma=0.0, cap_fraction=0.15, party_budget=party_budget)
        party_alloc = res.allocations - np.array([r.cand_d_total for r in races])
        shares = {}
        for tier_set, name in [(SAFE_TIERS, "safe"), (COMPETITIVE_TIERS, "competitive"), (LIKELY_TIERS, "likely")]:
            idx = np.array([r.cook_rating in tier_set for r in races])
            shares[name] = float(party_alloc[idx].sum() / party_alloc.sum())
        rows.append({"c_max": cm, "expected_seats": res.expected_seats, **shares,
                     "allocations": res.allocations})
        print(f"    c_max={cm:>4}  seats={res.expected_seats:.3f}  safe={shares['safe']:.1%}  "
              f"competitive={shares['competitive']:.1%}  likely={shares['likely']:.1%}  ({time.time()-t0:.0f}s)")
    config._cfg["persuasion_ceiling"]["c_max"] = c_max_default

    np.savez(
        SWEEP_CACHE,
        c_max=np.array([r["c_max"] for r in rows], dtype=float),
        expected_seats=np.array([r["expected_seats"] for r in rows]),
        safe=np.array([r["safe"] for r in rows]),
        competitive=np.array([r["competitive"] for r in rows]),
        likely=np.array([r["likely"] for r in rows]),
        allocations=np.array([r["allocations"] for r in rows]),
    )
    return rows


def fig3_cmax_sweep(sweep_rows, c_max_default: float):
    xs = [r["c_max"] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for key, color, label in [("competitive", SERIES_COMP, "Competitive (Toss-Up/Lean)"),
                               ("likely", SERIES_LIKELY, "Likely"),
                               ("safe", SERIES_SAFE, "Safe")]:
        ys = [r[key] for r in sweep_rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, label=label)

    y_max = max(0.72, max(r["competitive"] for r in sweep_rows) * 1.12)
    ax.axhline(0.45, color="#7a7972", linewidth=0.9, linestyle=(0, (2, 3)))
    ax.text(xs[-1], 0.45 + y_max * 0.015, "pre-fix (uncapped), documented: 45%", ha="right", va="bottom",
            fontsize=8.5, color="#555")
    ax.axvline(c_max_default, color="#c9c8bd", linewidth=0.9, linestyle=(0, (2, 3)))
    ax.text(c_max_default, 0.99, "chosen", ha="center", va="top", fontsize=8.5,
            color="#555", transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("c_max (log scale)")
    ax.set_ylabel("party budget share")
    ax.set_title("Party budget share by tier, across the c_max robustness sweep", fontsize=11.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "persuasion_ceiling_cmax_sweep_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote persuasion_ceiling_cmax_sweep_fig.png")


# ═══ 4. Emergent tier prioritization at the shipped c_max ═══════════════════

def fig4_tier_allocation(races, sweep_rows, c_max_default: float):
    row = next(r for r in sweep_rows if r["c_max"] == c_max_default)
    model_alloc = row["allocations"]
    model_party = model_alloc - np.array([rc.cand_d_total for rc in races])
    dccc_party = np.maximum(np.array([rc.d_total for rc in races]) - np.array([rc.cand_d_total for rc in races]), 0.0)
    tiers = np.array([rc.cook_rating for rc in races])

    levels = [("Toss-Up", ["Toss-Up"]), ("Lean", ["Lean D", "Lean R"]),
              ("Likely", ["Likely D", "Likely R"]), ("Safe", ["Safe D", "Safe R"])]
    labels, model_per_race, dccc_per_race, ns = [], [], [], []
    for name, tier_list in levels:
        idx = np.isin(tiers, tier_list)
        n = int(idx.sum())
        labels.append(name); ns.append(n)
        model_per_race.append(model_party[idx].sum() / n)
        dccc_per_race.append(dccc_party[idx].sum() / n)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    h = 0.32
    ax.barh(y + h/2, model_per_race, height=h, color=ACCENT, label="Model-recommended")
    ax.barh(y - h/2, dccc_per_race, height=h, color=BASELINE, label="DCCC observed")

    for yi, v in zip(y + h/2, model_per_race):
        ax.text(v * 1.08, yi, f"${v/1e6:.1f}M" if v >= 1e5 else f"${v/1e3:.0f}K",
                va="center", fontsize=9, color=ACCENT)
    for yi, v in zip(y - h/2, dccc_per_race):
        ax.text(v * 1.08, yi, f"${v/1e6:.1f}M" if v >= 1e5 else f"${v/1e3:.0f}K",
                va="center", fontsize=9, color="#6b6a63")

    ax.set_xscale("log")
    ax.set_xlim(1e4, 1.2e7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)])
    ax.set_xlabel("average party $ per race (log scale)")
    ax.set_title(f"Average party $ per race, by tier — model vs. DCCC (c_max = {c_max_default:.1f})", fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "persuasion_ceiling_tier_allocation_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote persuasion_ceiling_tier_allocation_fig.png")


def main():
    print("Loading fitted 2024 pipeline...")
    races, coef, sigma_model = load_pipeline()
    c_max_default = config.persuasion_ceiling_c_max()
    print(f"  {len(races)} races, c_max_default={c_max_default}")

    print("\n[1/4] Persuasion ceiling curve")
    fig1_ceiling_curve(c_max_default)

    print("\n[2/4] Response surface (4 real exemplar races)")
    fig2_response_surface(races, coef, sigma_model, c_max_default)

    print(f"\n[3/4] c_max robustness sweep ({len(SWEEP_C_MAX)} points, ~60-90s each)")
    sweep_rows = run_cmax_sweep(races, coef, sigma_model, c_max_default)
    fig3_cmax_sweep(sweep_rows, c_max_default)

    print("\n[4/4] Tier allocation at shipped c_max")
    fig4_tier_allocation(races, sweep_rows, c_max_default)

    print(f"\nDone. 4 figures written to {OUT}/")


if __name__ == "__main__":
    main()
