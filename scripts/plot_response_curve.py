#!/usr/bin/env python3
"""
Plot spending response curves for the paper.

Illustrates why the nonlinear optimizer finds gains that the linear MSG
approximation misses. Two panels:

  Left  — High-spending incumbent race (PA-07 analog, D=$27M, R=$4M):
           the curve is flat here; removing $5M barely changes expected seats.

  Right — Underfunded open-seat race (TX-15 analog, D=$4M, R=$10M):
           the curve is steep here; adding $5M meaningfully raises win probability.

The linear approximation (tangent line at observed spending) diverges from
the true curve in both cases, explaining the +11.9 nonlinear vs +3.3 linear
seat-gain discrepancy documented in the backtest.

Output: outputs/response_curve.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtest.types import SigmaModel


# ── Load model ────────────────────────────────────────────────────────────────

with open(ROOT / "data/processed/margin_model_coef.json") as f:
    coef = json.load(f)
with open(ROOT / "data/processed/sigma_model.json") as f:
    sigma_coef = json.load(f)

sigma_model = SigmaModel(_coef=sigma_coef)


# ── Response curve function ───────────────────────────────────────────────────

def response_curve(
    d_range_m: np.ndarray,
    r_fixed_m: float,
    pvi: float,
    incumb_status: str,
    gb: float,
    indiv_share: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """
    Compute P_win and MSG over a range of D spending values.

    Returns (p_win array, sigma, c_spend) for annotation.
    """
    is_incumb = 1.0 if incumb_status == "Incumbent" else 0.0
    b1 = coef["beta1_open"] if incumb_status == "Open" else coef["beta1"]

    mu_const = (
        coef["alpha0"]
        + coef["alpha1"] * pvi
        + coef["alpha2"] * is_incumb
        + coef["alpha3"] * gb
        + coef["alpha5"] * indiv_share
    )
    c_spend = b1 + coef["beta2"] * abs(pvi) + coef["beta3"] * is_incumb
    sigma = sigma_model.predict(abs(pvi), incumb_status, gb)

    ratio = d_range_m / (d_range_m + r_fixed_m)
    mu = mu_const + c_spend * np.log(np.clip(ratio, 1e-10, 1 - 1e-10))
    return norm.cdf(mu / sigma), sigma, c_spend, mu_const


def msg_at(
    d_obs_m: float,
    r_fixed_m: float,
    sigma: float,
    c_spend: float,
    mu_const: float,
) -> float:
    """MSG (seats per $1M) at observed spending."""
    ratio = d_obs_m / (d_obs_m + r_fixed_m)
    mu = mu_const + c_spend * np.log(ratio)
    phi = norm.pdf(mu / sigma)
    # ∂P_win/∂D ($M) = φ(μ/σ)/σ · c_spend · R/(D·(D+R))
    return (phi / sigma) * c_spend * r_fixed_m / (d_obs_m * (d_obs_m + r_fixed_m))


# ── Race parameters ───────────────────────────────────────────────────────────

# Race 1: High-spending incumbent (PA-07 analog)
#   PVI = -2 (slight R lean), D incumbent, generic ballot = -1 (2024)
#   Republicans raised only $4M; DCCC poured in $27M
R1, D1_OBS = 4.0, 27.0
P1_PARAMS = dict(pvi=-2.0, incumb_status="Incumbent", gb=-1.0)

# Race 2: Underfunded open seat (TX-15/VA-10 analog)
#   PVI = 0 (exactly competitive), open seat, generic ballot = -1 (2024)
#   DCCC allocated only $4M against $10M Republican spending
R2, D2_OBS = 10.0, 4.0
P2_PARAMS = dict(pvi=0.0, incumb_status="Open", gb=-1.0)

d1 = np.linspace(0.5, 32, 800)
d2 = np.linspace(0.5, 22, 800)

p1, sigma1, c1, mu1 = response_curve(d1, R1, **P1_PARAMS)
p2, sigma2, c2, mu2 = response_curve(d2, R2, **P2_PARAMS)

msg1 = msg_at(D1_OBS, R1, sigma1, c1, mu1)
msg2 = msg_at(D2_OBS, R2, sigma2, c2, mu2)

# P_win at observed spending
p1_obs = response_curve(np.array([D1_OBS]), R1, **P1_PARAMS)[0][0]
p2_obs = response_curve(np.array([D2_OBS]), R2, **P2_PARAMS)[0][0]

# Linear tangent lines
tangent1 = np.clip(p1_obs + msg1 * (d1 - D1_OBS), 0, 1)
tangent2 = np.clip(p2_obs + msg2 * (d2 - D2_OBS), 0, 1)

# Reallocation: remove $5M from Race 1, add $5M to Race 2
DELTA = 5.0
p1_new = response_curve(np.array([D1_OBS - DELTA]), R1, **P1_PARAMS)[0][0]
p2_new = response_curve(np.array([D2_OBS + DELTA]), R2, **P2_PARAMS)[0][0]
nonlinear_gain = (p2_new - p2_obs) - (p1_obs - p1_new)
linear_gain    = msg2 * DELTA - msg1 * DELTA


# ── Plot ──────────────────────────────────────────────────────────────────────

BLUE   = "#1a6faf"
GREEN  = "#2a9d4f"
ORANGE = "#e07b39"
GRAY   = "#888888"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Spending Response Curves: Why Nonlinear Optimization Outperforms Linear MSG Approximation",
    fontsize=12, fontweight="bold", y=1.01,
)

# ─── Panel 1: high-spending incumbent ────────────────────────────────────────
ax1.plot(d1, p1, color=BLUE, lw=2.5, label="Win probability P(win)")
ax1.plot(d1, tangent1, "--", color=ORANGE, lw=1.8,
         label=f"Linear approx. (MSG = {msg1:.4f} seats/$1M)")

ax1.axvline(D1_OBS, color=GRAY, ls=":", lw=1.2, alpha=0.8)
ax1.scatter([D1_OBS], [p1_obs], color=BLUE, s=70, zorder=5)
ax1.scatter([D1_OBS - DELTA], [p1_new], color=ORANGE, s=60, zorder=5, marker="v")

# Arrow: reallocation removes $5M
ax1.annotate(
    "",
    xy=(D1_OBS - DELTA, p1_new + 0.01),
    xytext=(D1_OBS, p1_obs),
    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5),
)

ax1.text(
    D1_OBS + 0.3, p1_obs + 0.005,
    f"Observed: D=${D1_OBS:.0f}M\nP(win) = {p1_obs:.1%}",
    fontsize=9, color=BLUE, va="bottom",
)
ax1.text(
    D1_OBS - DELTA - 0.5, p1_new - 0.04,
    f"Remove ${DELTA:.0f}M\nP(win) → {p1_new:.1%}\n(cost: {(p1_obs-p1_new)*100:.1f} pp)",
    fontsize=8.5, color=ORANGE, ha="right", va="top",
)

# Flat-region shading
ax1.axvspan(20, 32, alpha=0.07, color=BLUE, label="Flat region — low MSG")
ax1.text(24, 0.12, "Flat region\nlow MSG", fontsize=8, color=BLUE, ha="center", alpha=0.8)

ax1.set_xlabel("Democratic Spending ($M)", fontsize=11)
ax1.set_ylabel("Win Probability", fontsize=11)
ax1.set_title(
    "Race 1: High-spending incumbent\n"
    f"(PVI={P1_PARAMS['pvi']:+.0f}, Incumbent, R=${R1:.0f}M fixed)",
    fontsize=10,
)
ax1.set_ylim(-0.02, 1.08)
ax1.set_xlim(0, 32)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.legend(fontsize=8.5, loc="lower right")
ax1.grid(axis="y", alpha=0.3)

# ─── Panel 2: underfunded open seat ──────────────────────────────────────────
ax2.plot(d2, p2, color=GREEN, lw=2.5, label="Win probability P(win)")
ax2.plot(d2, tangent2, "--", color=ORANGE, lw=1.8,
         label=f"Linear approx. (MSG = {msg2:.4f} seats/$1M)")

ax2.axvline(D2_OBS, color=GRAY, ls=":", lw=1.2, alpha=0.8)
ax2.scatter([D2_OBS], [p2_obs], color=GREEN, s=70, zorder=5)
ax2.scatter([D2_OBS + DELTA], [p2_new], color=ORANGE, s=60, zorder=5, marker="^")

# Arrow: reallocation adds $5M
ax2.annotate(
    "",
    xy=(D2_OBS + DELTA, p2_new - 0.01),
    xytext=(D2_OBS, p2_obs),
    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5),
)

ax2.text(
    D2_OBS - 0.3, p2_obs - 0.015,
    f"Observed: D=${D2_OBS:.0f}M\nP(win) = {p2_obs:.1%}",
    fontsize=9, color=GREEN, ha="right", va="top",
)
ax2.text(
    D2_OBS + DELTA + 0.3, p2_new + 0.01,
    f"Add ${DELTA:.0f}M\nP(win) → {p2_new:.1%}\n(gain: {(p2_new-p2_obs)*100:.1f} pp)",
    fontsize=8.5, color=ORANGE, ha="left", va="bottom",
)

# Steep-region shading
ax2.axvspan(0.5, 8, alpha=0.07, color=GREEN, label="Steep region — high MSG")
ax2.text(4, 0.62, "Steep region\nhigh MSG", fontsize=8, color=GREEN, ha="center", alpha=0.8)

ax2.set_xlabel("Democratic Spending ($M)", fontsize=11)
ax2.set_title(
    "Race 2: Underfunded open seat\n"
    f"(PVI={P2_PARAMS['pvi']:+.0f}, Open seat, R=${R2:.0f}M fixed)",
    fontsize=10,
)
ax2.set_ylim(-0.02, 1.08)
ax2.set_xlim(0, 22)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(axis="y", alpha=0.3)

# ─── Bottom annotation ────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    f"Reallocation of ${DELTA:.0f}M from Race 1 → Race 2: "
    f"nonlinear gain = {nonlinear_gain*100:+.1f} pp  |  "
    f"linear approximation = {linear_gain*100:+.1f} pp  |  "
    f"linear overestimates gain by {(linear_gain - nonlinear_gain)*100:.1f} pp",
    ha="center", fontsize=9.5, style="italic", color="#444444",
)

plt.tight_layout()
out = ROOT / "outputs" / "response_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
print(f"\nKey numbers:")
print(f"  Race 1 observed: D=${D1_OBS}M, P(win)={p1_obs:.1%}, MSG={msg1:.5f} seats/$1M")
print(f"  Race 2 observed: D=${D2_OBS}M, P(win)={p2_obs:.1%}, MSG={msg2:.5f} seats/$1M")
print(f"  Remove ${DELTA}M from Race 1: P(win) {p1_obs:.1%} → {p1_new:.1%} (cost {(p1_obs-p1_new)*100:.2f} pp)")
print(f"  Add ${DELTA}M to Race 2:      P(win) {p2_obs:.1%} → {p2_new:.1%} (gain {(p2_new-p2_obs)*100:.2f} pp)")
print(f"  Nonlinear net gain: {nonlinear_gain*100:+.2f} pp")
print(f"  Linear approx net: {linear_gain*100:+.2f} pp")
