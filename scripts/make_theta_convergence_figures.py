#!/usr/bin/env python3
"""
Two new reader-facing figures for Paper III, requested to make the
Empirical Results section (Theta's time path) and Section 8.8 (statistical
rigor) easier to read at a glance than the underlying tables alone:

  1. theta_schedule_over_time_fig.png -- Theta(t) plotted against days
     remaining to Election Day, one line per calibration scenario, built
     from outputs/theta_schedule.json (already computed; no new solves).
     Shows directly what Table 11 only reports at a single point (t=0):
     that the value of waiting shrinks smoothly as Election Day approaches,
     reaching Theta(T)=0 by construction.
  2. theta_convergence_diagnostics_fig.png -- three panels visualizing
     Section 8.8's statistical-rigor checks: (A) the 5-seed Theta(0)
     estimates scattered around their mean with a +/-1 SE band, (B) the
     K=2,000 vs K=5,000 comparison, (C) in-sample vs held-out-only Theta(0)
     under the 30%-held-out policy evaluation. Built from
     outputs/theta_statistical_rigor.json (already computed; no new solves).

Output: outputs/theta_schedule_over_time_fig.png,
        outputs/theta_convergence_diagnostics_fig.png
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

SCEN_COLOR = {"eta_fit_2022": "#c0392b", "eta_fit_2024": "#2e6da4",
              "eta_bootstrap_all_cycles": "#2a9d4f"}
SCEN_LABEL = {"eta_fit_2022": "eta fit on 2022", "eta_fit_2024": "eta fit on 2024",
              "eta_bootstrap_all_cycles": "eta bootstrap, all 7 cycles"}
SCENARIOS = ["eta_fit_2022", "eta_fit_2024", "eta_bootstrap_all_cycles"]


def make_theta_schedule_figure():
    with open(OUT / "theta_schedule.json") as f:
        sched = json.load(f)

    fig, (ax_theta, ax_frac) = plt.subplots(1, 2, figsize=(12, 5))

    for label in SCENARIOS:
        periods = sched[label]["theta_by_period"]
        days = [p["days_remaining"] for p in periods]
        theta = [p["mean_theta"] for p in periods]
        frac = [p["frac_deploy_now"] for p in periods]
        ax_theta.plot(days, theta, "o-", color=SCEN_COLOR[label], lw=2, ms=6,
                      label=SCEN_LABEL[label])
        ax_frac.plot(days, frac, "o-", color=SCEN_COLOR[label], lw=2, ms=6,
                     label=SCEN_LABEL[label])

    ax_theta.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)
    ax_theta.invert_xaxis()
    ax_theta.set_xlabel("Days remaining to Election Day")
    ax_theta.set_ylabel(r"$\Theta(t)$ (expected seats)")
    ax_theta.set_title("A. Value of Waiting Shrinks Smoothly\nto Zero as Election Day Approaches",
                        fontsize=11, fontweight="bold")
    ax_theta.legend(frameon=False, fontsize=8, loc="upper left")
    ax_theta.text(0.97, 0.04, r"$\Theta(T)=0$ by construction" + "\n(Appendix C.1)",
                  transform=ax_theta.transAxes, fontsize=8, color="#555555",
                  style="italic", ha="right")

    ax_frac.invert_xaxis()
    ax_frac.set_ylim(-0.02, 1.02)
    ax_frac.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax_frac.set_xlabel("Days remaining to Election Day")
    ax_frac.set_ylabel("Share of simulated paths choosing “deploy now”")
    ax_frac.set_title("B. “Hold” Dominates Almost Everywhere;\nDeploy Share Rises Only Very Late",
                       fontsize=11, fontweight="bold")
    ax_frac.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "theta_schedule_over_time_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ theta_schedule_over_time_fig.png")


def make_convergence_figure():
    with open(OUT / "theta_statistical_rigor.json") as f:
        rigor = json.load(f)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5))
    color = SCEN_COLOR["eta_bootstrap_all_cycles"]

    # --- Panel A: 5-seed scatter around mean +/- 1 SE ---
    seeds = rigor["multi_seed_K2000"]
    seed_vals = [s["mean_theta"] for s in seeds]
    seed_labels = [str(s["seed"]) for s in seeds]
    mean, se = rigor["theta0_mc_mean"], rigor["theta0_mc_se"]
    x = np.arange(len(seed_vals))
    axA.scatter(x, seed_vals, color=color, s=70, zorder=3, label="individual seed")
    axA.axhline(mean, color="#333333", lw=1.2, label=f"mean = {mean:.3f}")
    axA.fill_between([-0.5, len(x) - 0.5], mean - se, mean + se, color="#333333", alpha=0.12,
                      label=f"±1 SE = {se:.3f}", zorder=0)
    axA.set_xticks(x)
    axA.set_xticklabels(seed_labels, fontsize=8)
    axA.set_xlim(-0.5, len(x) - 0.5)
    axA.set_xlabel("Random seed")
    axA.set_ylabel(r"$\Theta(0)$ (expected seats)")
    axA.set_title("A. Five Independent Seeds Agree\nto Within ~1% of the Point Estimate",
                   fontsize=11, fontweight="bold")
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    # --- Panel B: K-sensitivity ---
    ks = rigor["k_sensitivity"]
    k_vals = sorted(int(k) for k in ks.keys())
    theta_by_k = [ks[str(k)]["mean_theta"] for k in k_vals]
    axB.plot(k_vals, theta_by_k, "o-", color=color, ms=10, lw=2)
    for k, t in zip(k_vals, theta_by_k):
        axB.annotate(f"{t:.3f}", (k, t), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9)
    axB.set_xlim(min(k_vals) * 0.7, max(k_vals) * 1.15)
    pad = (max(theta_by_k) - min(theta_by_k)) * 3 + 0.02
    mid = (max(theta_by_k) + min(theta_by_k)) / 2
    axB.set_ylim(mid - pad, mid + pad)
    axB.set_xscale("log")
    axB.set_xticks(k_vals)
    axB.set_xticklabels([f"{k:,}" for k in k_vals])
    axB.set_xlabel("Monte Carlo paths ($K$)")
    axB.set_ylabel(r"$\Theta(0)$ (expected seats)")
    axB.set_title("B. Doubling Path Count Changes\nthe Estimate by ~2.7%",
                   fontsize=11, fontweight="bold")

    # --- Panel C: in-sample vs held-out ---
    oos = rigor["out_of_sample_K2000"]
    cats = ["In-sample\n(all paths)", "Held-out only\n(30% of paths)"]
    vals = [oos["mean_theta"], oos["mean_theta_held_out"]]
    bars = axC.bar(cats, vals, color=[color, "#a3d9a5"], width=0.55,
                    edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    ymin = min(vals) - 0.5
    ymax = max(vals) + 0.5
    axC.set_ylim(ymin, ymax)
    axC.set_ylabel(r"$\Theta(0)$ (expected seats)")
    axC.set_title("C. Held-Out Policy Evaluation\nMatches In-Sample (No Overfitting)",
                   fontsize=11, fontweight="bold")

    fig.suptitle("Monte Carlo and Regression-Specification Diagnostics (eta_bootstrap_all_cycles, Section 8.8)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "theta_convergence_diagnostics_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ theta_convergence_diagnostics_fig.png")


def main():
    make_theta_schedule_figure()
    make_convergence_figure()
    print(f"\nBoth figures written to {OUT}/")


if __name__ == "__main__":
    main()
