#!/usr/bin/env python3
"""
Six figures supporting Paper III (docs/paper3_draft.md) and the follow-up
plan (docs/theta_followup_plan.md), one per question a reader asked:

  1. gb_volatility_term_structure.png  -- how much does the national
     environment (generic ballot) move as a function of horizon? (Section 5)
  2. eta_reaction_by_tier.png          -- how much does the opponent react
     to a dollar of new spending, by competitiveness tier? (Section 4)
  3. epsilon_uncertainty_decay.png     -- how fast does idiosyncratic
     race-level uncertainty resolve as Election Day approaches? (Section 6.2)
  4. theta_binary_decision_motivation.png -- what does the CURRENT
     binary deploy/hold model actually produce, as motivation for testing
     whether a continuous reserve fraction would reveal something the
     binary framing cannot (theta_followup_plan.md Section 1). The
     continuous-phi generalization this figure motivates has since been
     implemented and run (Section 1.3/8) -- see
     scripts/make_continuous_phi_figure.py for that result; this figure
     is kept as-is, showing only the binary run's own motivating output.
  5. gb_asymmetry_check.png            -- does the historical record show
     a real asymmetry from a similarly-favorable starting point, or is a
     symmetric zero-drift process defensible? (theta_followup_plan.md
     Section 2, OU-with-drift fit)
  6. msg_low_leverage_check.png        -- does the objective function's
     marginal-seat-gain (MSG) calculation overvalue near-zero-spend "safe"
     races because of the log-ratio functional form's 1/D singularity,
     independent of whether those races are competitive?

All figures are built from data already computed and checked into
outputs/ this session, plus one fresh computation for (6) that reuses
Paper I's unmodified compute_outputs_batch() on the live 2026 universe --
no new modeling assumptions are introduced.

Output: outputs/*.png (6 files)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
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

COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
COOK_COLOR = {
    "Safe D":   "#1a3a5c",
    "Likely D": "#2e6da4",
    "Lean D":   "#5b9bd5",
    "Toss-Up":  "#e6a817",
    "Lean R":   "#e87c7c",
    "Likely R": "#c0392b",
    "Safe R":   "#7b0000",
}


# ═══ 1. GB volatility term structure (Section 5.3) ══════════════════════════

def fig1_gb_volatility():
    pooled = pd.read_csv(OUT / "gb_volatility_term_structure.csv")
    hist_only = pd.read_csv(OUT / "gb_volatility_term_structure_historical_only.csv")
    with open(ROOT / "data/processed/live_2026_state.json") as f:
        live_days_remaining = json.load(f)["days_remaining"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(pooled["horizon_days"], pooled["pooled_std_dG"], "o-",
             color="#2e6da4", lw=2, ms=6, label="Pooled: 4 historical + live 2026")
    ax1.plot(hist_only["horizon_days"], hist_only["pooled_std_dG"], "s--",
             color="#c0392b", lw=1.8, ms=6, label="4 historical cycles only")
    ref_days = np.linspace(20, 460, 100)
    ax1.plot(ref_days, 0.19 * np.sqrt(ref_days), color="#999999", lw=1.2, ls=":",
              label=r"Pure random walk ($0.19\sqrt{\Delta t}$)", zorder=1)
    ax1.set_xlabel("Horizon (days)")
    ax1.set_ylabel(r"Realized std($\Delta G$), points")
    ax1.set_title("Generic-Ballot Move Grows With Horizon", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2.plot(pooled["horizon_days"], pooled["pooled_std_over_sqrt_days"], "o-",
             color="#2e6da4", lw=2, ms=6, label="Pooled (5 series)")
    ax2.plot(hist_only["horizon_days"], hist_only["pooled_std_over_sqrt_days"], "s--",
             color="#c0392b", lw=1.8, ms=6, label="Historical only (4 cycles)")
    ax2.axhline(0.19, color="#999999", lw=1, ls=":", zorder=1)
    ax2.axvline(live_days_remaining, color="#e6a817", lw=1.4, ls="--", alpha=0.8)
    ax2.text(live_days_remaining + 5, 0.21, f"~{live_days_remaining}d: live 2026\ndecision horizon",
              fontsize=8, color="#8a6d0a")
    ax2.set_xlabel("Horizon (days)")
    ax2.set_ylabel(r"std($\Delta G$) / $\sqrt{\Delta t}$")
    ax2.set_title("Flat 30-270d → Random Walk Is a Good Fit\n(Decline only past ~1 year)",
                   fontweight="bold", fontsize=11.5)
    ax2.set_ylim(0.10, 0.22)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle("How Much Does the National Environment Move? (Paper III §5.3)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "gb_volatility_term_structure_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ gb_volatility_term_structure_fig.png")


# ═══ 2. Opponent reaction eta by tier (Section 4.4, §5.5 Validation C) ══════

def fig2_eta_by_tier():
    pooled = pd.read_csv(OUT / "eta_reaction_estimates.csv").set_index("tier").reindex(COOK_ORDER)
    with open(OUT / "simulator_validation_summary.json") as f:
        val = json.load(f)
    loco = pd.DataFrame(val["validation_c_eta_stability"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.arange(len(COOK_ORDER))
    colors = [COOK_COLOR[t] for t in COOK_ORDER]
    ci = 1.96 * pooled["se"].values
    bars = ax1.bar(x, pooled["eta"].values, yerr=ci, color=colors, capsize=4,
                    edgecolor="white", linewidth=0.6, error_kw=dict(lw=1.2, ecolor="#333333"))
    ax1.axhline(0, color="#333333", lw=0.8)
    for i, t in enumerate(COOK_ORDER):
        p = pooled.loc[t, "p_value"]
        marker = "*" if p < 0.05 else ("†" if p < 0.10 else "")
        ax1.text(i, pooled.loc[t, "eta"] + np.sign(pooled.loc[t, "eta"] + 1e-9) * (ci[i] + 0.04),
                  marker, ha="center", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(COOK_ORDER, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel(r"$\hat\eta$ (pooled, 2022+2024)")
    ax1.set_title("Opponent Reaction Is Strongest in\nContested Races (95% CI, * p<.05)",
                   fontweight="bold", fontsize=11.5)

    reliable = ["Lean D", "Toss-Up", "Lean R"]
    for i, t in enumerate(COOK_ORDER):
        sub = loco[loco["tier"] == t]
        y2022 = sub[sub["fit_on_cycle"] == 2022]["eta"].values
        y2024 = sub[sub["fit_on_cycle"] == 2024]["eta"].values
        alpha = 1.0 if t in reliable else 0.4
        if len(y2022):
            ax2.plot([i - 0.08], y2022, "o", color="#333333", ms=7, alpha=alpha)
        if len(y2024):
            ax2.plot([i + 0.08], y2024, "^", color="#e6a817", ms=7, alpha=alpha)
        if len(y2022) and len(y2024):
            ax2.plot([i - 0.08, i + 0.08], [y2022[0], y2024[0]], "-", color="#999999",
                      lw=1, alpha=alpha, zorder=1)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(COOK_ORDER, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\hat\eta$, fit on one cycle only")
    ax2.set_title("Leave-One-Cycle-Out: Sign Is Stable\nfor Contested Tiers, Magnitude Is Not",
                   fontweight="bold", fontsize=11.5)
    legend_els = [
        plt.Line2D([0], [0], marker="o", color="#333333", lw=0, ms=7, label="Fit on 2022"),
        plt.Line2D([0], [0], marker="^", color="#e6a817", lw=0, ms=7, label="Fit on 2024"),
    ]
    ax2.legend(handles=legend_els, frameon=False, fontsize=9, loc="upper left")
    ax2.text(0.98, 0.03, "Faded = flagged unreliable\n(§4.4 / §5.5)", transform=ax2.transAxes,
              ha="right", va="bottom", fontsize=8, color="#777777")

    fig.suptitle(r"How Much Do Opponents React to a Dollar of New Spending? ($\hat\eta$, Paper III §4)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "eta_reaction_by_tier_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ eta_reaction_by_tier_fig.png")


# ═══ 3. Epsilon uncertainty decay (Section 6.2 / 7.1) ═══════════════════════

def fig3_epsilon_decay():
    with open(ROOT / "data/processed/gb_dynamics.json") as f:
        gb_dynamics = json.load(f)
    LAMBDA = gb_dynamics["lambda_decay"]   # single source of truth (Paper III audit, 2026-07-16)
    TAU = gb_dynamics["tau_days"]

    days_remaining = np.linspace(0, 450, 500)
    frac_remaining = np.sqrt(1 - np.exp(-LAMBDA * days_remaining))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(days_remaining, frac_remaining, color="#2e6da4", lw=2.4)
    ax.fill_between(days_remaining, 0, frac_remaining, color="#2e6da4", alpha=0.08)

    ax.axvline(TAU, color="#999999", lw=1.2, ls=":")
    ax.text(TAU + 6, 0.15, rf"$\tau$={TAU:.0f}d", fontsize=9, color="#666666")

    with open(ROOT / "data/processed/live_2026_state.json") as f:
        live_state = json.load(f)
    today_days = live_state["days_remaining"]   # single source of truth (Paper III audit, 2026-07-16)
    y_today = np.sqrt(1 - np.exp(-LAMBDA * today_days))
    ax.axvline(today_days, color="#e6a817", lw=1.6, ls="--")
    ax.plot([today_days], [y_today], "o", color="#e6a817", ms=8, zorder=5)
    ax.annotate(f"Today (~{today_days}d out)\n{y_today:.0%} of static $\\sigma_i$ still \"live\"",
                xy=(today_days, y_today), xytext=(today_days + 30, y_today - 0.22),
                fontsize=9, color="#8a6d0a",
                arrowprops=dict(arrowstyle="->", color="#8a6d0a", lw=1))

    ax.set_xlabel("Days remaining to Election Day  (T − t)")
    ax.set_ylabel(r"$\sigma_{i,t} \,/\, \sigma_i^{\mathrm{static}}$"
                  "\n(fraction of idiosyncratic uncertainty still unresolved)")
    ax.set_title("How Fast Does Race-Level Uncertainty Resolve?\n"
                 r"$\sigma_{i,t}=\sigma_i^{\mathrm{static}}\sqrt{1-e^{-\lambda(T-t)}}$"
                 rf", $\lambda$={LAMBDA:.5f} borrowed from $\sigma_G(\Delta t)$'s decay (Paper III §6.2)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 450)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(OUT / "epsilon_uncertainty_decay_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ epsilon_uncertainty_decay_fig.png")


# ═══ 4. Current binary deploy/hold model output (motivation for §1) ═════════

def fig4_binary_motivation():
    with open(OUT / "theta_schedule.json") as f:
        sched = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"eta_fit_2022": "#c0392b", "eta_fit_2024": "#2e6da4", "eta_bootstrap_all_cycles": "#2a9d4f"}
    labels = {"eta_fit_2022": "eta fit on 2022", "eta_fit_2024": "eta fit on 2024",
              "eta_bootstrap_all_cycles": "eta bootstrap, all 7 cycles"}

    for label, res in sched.items():
        periods = res["theta_by_period"]
        days = [p["days_remaining"] for p in periods]
        theta = [p["mean_theta"] for p in periods]
        frac = [p["frac_deploy_now"] for p in periods]
        axes[0].plot(days, theta, "o-", color=colors[label], lw=2, ms=6, label=labels[label])
        axes[1].plot(days, frac, "o-", color=colors[label], lw=2, ms=6, label=labels[label])

    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Days remaining to Election Day")
    axes[0].set_ylabel(r"Mean $\Theta(t)$ (expected seats: wait − deploy)")
    axes[0].set_title(r"$\Theta>0$ = patience has value; $\Theta<0$ = deploy now wins",
                       fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].invert_xaxis()
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_xlabel("Days remaining to Election Day")
    axes[1].set_ylabel("Fraction of simulated paths\nwhere deploy-now wins")
    axes[1].set_title("The Binary Model's Own Output Is Not a Clean\n0%/100% Step Function Mid-Cycle",
                       fontsize=10.5)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Is the All-or-Nothing Framing Hiding a Partial Reserve? (§1, motivation only)",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.text(0.5, -0.10,
             "This is the BINARY model's own path-level split (deploy-everything vs. hold-everything), shown only as motivation -- it is\n"
             "already far from 0%/100% in the 28-84 day window, which is exactly the situation where a continuous reserve fraction could\n"
             "reveal a real small-but-nonzero optimum the binary comparison cannot express. Updated 2026-07-20 (theta_followup_plan.md §12):\n"
             "with a real, dated candidate-spending trickle now driving D_i,t (previously held perfectly fixed), Theta is positive throughout\n"
             "the horizon shown and every scenario recommends HOLDING the reserve, not deploying it -- the first flipped corner in this\n"
             "research line. The continuous-phi generalization this motivates was re-run under the same fix (§12.6) -- see\n"
             "make_continuous_phi_figure.py: the answer is that the corner holds there too, now favoring hold, for the same reason.",
             ha="center", fontsize=8.5, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])
    fig.savefig(OUT / "theta_binary_decision_motivation_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ theta_binary_decision_motivation_fig.png")


# ═══ 5. Symmetric vs. asymmetric environment check (§2, OU-with-drift) ══════

def fig5_gb_asymmetry():
    with open(OUT / "gb_ou_drift_fit.json") as f:
        ou = json.load(f)

    hist_raw = pd.read_csv(ROOT / "data/raw/generic_ballot/generic_ballot_historical_538.csv")
    hist_raw["date"] = pd.to_datetime(hist_raw["date"])

    cycle_colors = {2018: "#2e6da4", 2020: "#5b9bd5", 2022: "#e6a817", 2024: "#c0392b"}
    election_days = {2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for cycle, color in cycle_colors.items():
        g = hist_raw[hist_raw["cycle"] == cycle]
        piv = g.pivot_table(index="date", columns="candidate", values="pct_estimate").sort_index()
        gt = (piv["Democrats"] - piv["Republicans"]).asfreq("D").interpolate()
        elec = pd.Timestamp(election_days[cycle])
        days_to_elec = (elec - gt.index).days
        ax.plot(days_to_elec, gt.values, color=color, lw=1.8, alpha=0.85, label=f"{cycle}")

    for row in ou["historical_sanity_check"]:
        if not row["found_match"]:
            continue
        ax.plot([row["days_to_election"]], [row["g_at_match"]], "o",
                 color=cycle_colors[row["cycle"]], ms=10, mec="black", mew=1.2, zorder=5)

    ax.axhline(ou["today_gb"], color="#333333", lw=1.4, ls="--")
    ax.text(560, ou["today_gb"] + 0.25, f"Today: D+{ou['today_gb']}", fontsize=9, color="#333333")
    ax.axhline(ou["ou_drift_fit"]["g_bar"], color="#888888", lw=1.2, ls=":")
    ax.text(560, ou["ou_drift_fit"]["g_bar"] - 0.6,
            rf"Fitted $\bar G$={ou['ou_drift_fit']['g_bar']:.2f} (OU drift, p={ou['ou_drift_fit']['p_value_b']:.2f}, n.s.)",
            fontsize=8.5, color="#666666")

    ax.invert_xaxis()
    ax.set_xlabel("Days before Election Day")
    ax.set_ylabel("Generic ballot, D − R (points)")
    ax.set_title("Does a Similarly-Favorable Starting Point Predict Direction?\n"
                 "Circled: each cycle's day nearest today's D+5.02 level", fontweight="bold")
    ax.legend(title="Cycle", frameon=False, fontsize=9, loc="lower left")
    fig.text(0.5, -0.02,
             "2018 & 2020 moved further favorable to Democrats from a similar level; 2022 swung sharply the other way.\n"
             "Direction is genuinely mixed across the only 4 available cycles -- consistent with the OU fit's statistically insignificant drift (p=0.37).",
             ha="center", fontsize=8.5, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "gb_asymmetry_check_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ gb_asymmetry_check_fig.png")


# ═══ 6. Does the objective overvalue low-leverage (safe) races? ═════════════

def fig6_msg_low_leverage():
    with open(ROOT / "data/processed/margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                              "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(ROOT / "data/processed/sigma_model.json") as f:
        sigma_coef = json.load(f)
    sigma_model = SigmaModel(_coef=sigma_coef)

    races = build_universe(cycle=2026)
    outputs = compute_outputs_batch(races, coef, sigma_model)
    tier_by_district = {r.district_id: r.cook_rating for r in races}

    rows = [{"district_id": o.district_id, "p_win": o.p_win,
             "msg_per_1m": o.msg_i * 1e6, "tier": tier_by_district[o.district_id]}
            for o in outputs]
    df = pd.DataFrame(rows)

    alloc = pd.read_csv(OUT / "allocation_2026_live.csv")
    df = df.merge(alloc[["district_id", "recommended_total_party"]], on="district_id", how="left")
    total_party = df["recommended_total_party"].sum()
    df["share_of_party_budget"] = df["recommended_total_party"] / total_party

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for tier in COOK_ORDER:
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        ax1.scatter(sub["p_win"], sub["msg_per_1m"].clip(lower=1e-6), c=COOK_COLOR[tier],
                    s=32, alpha=0.75, edgecolors="none", label=tier, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$P_{\mathrm{win}}$  (0 = safe R, 1 = safe D)")
    ax1.set_ylabel("Marginal Seat Gain per $1M\n(log scale)")
    ax1.set_title("MSG Spikes at the Extremes\n(1/D singularity, not real competitiveness)",
                   fontweight="bold", fontsize=11)
    ax1.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center")

    tier_share = df.groupby("tier")["share_of_party_budget"].sum().reindex(COOK_ORDER).fillna(0)
    safe_share = tier_share[["Safe R", "Likely R"]].sum()
    bars = ax2.bar(range(len(COOK_ORDER)), tier_share.values,
                    color=[COOK_COLOR[t] for t in COOK_ORDER], edgecolor="white", linewidth=0.6)
    ax2.set_xticks(range(len(COOK_ORDER)))
    ax2.set_xticklabels(COOK_ORDER, rotation=30, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_ylabel("Share of recommended party budget")
    ax2.set_title(f"...And {safe_share:.1%} of the Live 2026 Budget\nGoes to Safe R + Likely R",
                   fontweight="bold", fontsize=11)
    ax2.text(0.97, 0.95, f"Safe R + Likely R = {safe_share:.1%}\n"
             "(Paper II §7.1's live allocation)", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    fig.suptitle("Does the Objective Overvalue Spending in Low-Leverage Races?",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.03,
             r"MSG's log-ratio term ($\propto R_i/(D_i\cdot T_i)$) blows up as $D_i\to0$, which happens precisely in safe seats with little"
             "\ncandidate/party spend -- a functional-form artifact, not evidence those seats are strategically pivotal (allocator.py's own docstring flags this).",
             ha="center", fontsize=8.5, color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "msg_low_leverage_check_fig.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ msg_low_leverage_check_fig.png")


def main():
    fig1_gb_volatility()
    fig2_eta_by_tier()
    fig3_epsilon_decay()
    fig4_binary_motivation()
    fig5_gb_asymmetry()
    fig6_msg_low_leverage()
    print(f"\nAll 6 figures written to {OUT}/")


if __name__ == "__main__":
    main()
