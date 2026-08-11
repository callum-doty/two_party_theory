#!/usr/bin/env python3
"""
Figure for the continuous-deployment-fraction result (docs/theta_followup_plan.md
Section 8; docs/paper3_draft.md Section 8.4): does allowing a genuinely
continuous, sequential reserve fraction reveal an interior optimum the
binary deploy/hold framing couldn't express? Three panels answer the three
parts of that question:

  A. Why the value-of-budget curve is a step, not a smooth concave curve
     (the LP allocator's knapsack degeneracy saturates the same ~7 races
     almost immediately) -- shown at t=0 for all three eta scenarios (5pt
     grid) plus the eta_fit_2024 11pt sensitivity check overlaid, to show
     resolution doesn't change the shape.
  B. What the optimal action schedule actually looks like over time for
     the representative eta_fit_2024 case -- a stacked composition of which
     grid action wins, by days remaining. Corners dominate throughout;
     mid-cycle periods split between the two corners rather than
     converging on an interior fraction.
  C. That the corner holds with a comfortable margin in every scenario,
     not a narrow one grid resolution could plausibly flip -- the gap
     between the full-deploy option and the next-best alternative at t=0.

Output: outputs/continuous_phi_result_fig.png
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _load(label, n_grid):
    with open(OUT / f"theta_schedule_continuous_phi_{label}_{n_grid}pt.json") as f:
        return json.load(f)


def _t0_curve(d):
    e0 = d["schedule"][0]
    fracs = sorted(float(k.removeprefix("deploy_")) for k in e0["option_value_means"])
    vals = [e0["option_value_means"][f"deploy_{f:.2f}"] for f in fracs]
    return np.array(fracs), np.array(vals)


def main():
    d5 = {label: _load(label, 5) for label in SCENARIOS}
    d11 = _load("eta_fit_2024", 11)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    axA, axB, axC = axes

    # ── Panel A: value-of-budget curve at t=0, step-shaped, corner wins ──
    for label in SCENARIOS:
        fracs, vals = _t0_curve(d5[label])
        axA.plot(fracs, vals, "o-", color=SCEN_COLOR[label], lw=2, ms=7,
                  label=SCEN_LABEL[label], zorder=3)
        win_i = np.argmax(vals)
        axA.plot(fracs[win_i], vals[win_i], "*", color=SCEN_COLOR[label], ms=20,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    fracs11, vals11 = _t0_curve(d11)
    axA.plot(fracs11, vals11, "--", color="#2e6da4", lw=1.2, alpha=0.6, zorder=2,
              label="eta_fit_2024, 11pt grid check")

    axA.set_xlabel("Fraction of $F_0$ committed today")
    axA.set_ylabel("E[Seats] at $t=0$")
    axA.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axA.set_title("A. Value-of-Budget Is a Step, Not a Smooth\nCurve — Same ~7 Races Saturate Fast",
                   fontsize=11, fontweight="bold")
    axA.legend(frameon=False, fontsize=8, loc="lower right")
    axA.text(0.03, 0.06,
             "★ = winning action\n(always a corner)",
             transform=axA.transAxes, fontsize=8, color="#555555", style="italic")

    # ── Panel B: action-schedule composition over time, eta_fit_2024 ──
    sched = d5["eta_fit_2024"]["schedule"]
    days = [p["days_remaining"] for p in sched]
    frac_keys = sorted(sched[0]["action_frac_distribution"].keys(), key=float)
    cmap = plt.get_cmap("RdYlBu")
    seg_colors = {k: cmap(i / (len(frac_keys) - 1)) for i, k in enumerate(frac_keys)}
    bottom = np.zeros(len(sched))
    for k in frac_keys:
        vals = np.array([p["action_frac_distribution"][k] for p in sched])
        axB.bar(days, vals, bottom=bottom, width=10, color=seg_colors[k],
                 edgecolor="white", linewidth=0.5,
                 label=f"{float(k):.0%} committed" if float(k) in (0.0, 1.0) else None)
        bottom += vals
    axB.invert_xaxis()
    axB.set_ylim(0, 1.02)
    axB.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axB.set_xlabel("Days remaining to Election Day")
    axB.set_ylabel("Share of simulated paths choosing each action")
    axB.set_title("B. eta_fit_2024: Corners Dominate\nThroughout — Mid-Cycle Splits Between\nThem, Never Converges Interior",
                   fontsize=11, fontweight="bold")
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    # ── Panel C: margin between the winning corner and next-best option at t=0 ──
    # Data-driven, not hardcoded to "full deploy wins" -- that assumption held
    # pre-trickle-fix but is now false (hold wins in every scenario, §12.6).
    # win_fracs lets the label/annotation reflect whichever fraction actually
    # wins in each scenario, rather than silently mislabeling a "hold wins" bar
    # as "Full deploy (winner)".
    x = np.arange(len(SCENARIOS))
    win_vals, next_vals, win_fracs = [], [], []
    for label in SCENARIOS:
        fracs, vals = _t0_curve(d5[label])
        order = np.argsort(vals)[::-1]
        win_vals.append(vals[order[0]])
        next_vals.append(vals[order[1]])
        win_fracs.append(fracs[order[0]])
    win_vals, next_vals, win_fracs = np.array(win_vals), np.array(next_vals), np.array(win_fracs)
    margin = win_vals - next_vals

    axC.bar(x - 0.16, win_vals, width=0.32, color=[SCEN_COLOR[l] for l in SCENARIOS],
             label="Winning action")
    axC.bar(x + 0.16, next_vals, width=0.32, color=[SCEN_COLOR[l] for l in SCENARIOS],
             alpha=0.35, label="Next-best option")
    ymin = min(next_vals.min(), win_vals.min()) - 0.15
    ymax = max(next_vals.max(), win_vals.max()) + 0.15
    axC.set_ylim(ymin, ymax)
    for i, m in enumerate(margin):
        win_label = "hold" if win_fracs[i] == 0.0 else ("deploy" if win_fracs[i] == 1.0 else f"{win_fracs[i]:.0%}")
        axC.text(i, win_vals[i] + 0.02, f"+{m:.2f}\n({win_label})", ha="center", fontsize=8.5, fontweight="bold")
    axC.set_xticks(x)
    axC.set_xticklabels([SCEN_LABEL[l].replace("eta ", "").replace(", ", "\n") for l in SCENARIOS],
                          fontsize=8.5)
    axC.set_ylabel("E[Seats] at $t=0$")
    axC.set_title("C. The Corner Wins by a Comfortable\nMargin in Every Scenario — Not a\nResolution-Dependent Knife-Edge",
                   fontsize=11, fontweight="bold")
    axC.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Does a Continuous Deployment Fraction Beat the Corners? No. (Paper III §8.4/§8.6)",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.text(0.5, -0.06,
             "A genuinely concave, sequential, carried-forward-budget generalization of the binary deploy/hold decision (theta_followup_plan.md §1.3)\n"
             "still lands on a corner, not an interior fraction, in all three eta scenarios, confirmed at an 11-point grid. Updated 2026-07-20 (§12.6):\n"
             "with a real, dated candidate-spending trickle now driving D_i,t, the corner that wins has flipped from full deployment to HOLDING the\n"
             "reserve entirely (100% hold in two of three scenarios, 92% in the third) -- the mechanism behind the step shape (panel A) is unrelated\n"
             "to the trickle fix and persists regardless of which corner wins.",
             ha="center", fontsize=9, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUT / "continuous_phi_result_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ continuous_phi_result_fig.png")


if __name__ == "__main__":
    main()
