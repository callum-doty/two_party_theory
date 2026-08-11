#!/usr/bin/env python3
"""
Single-race response curve: true (nonlinear) vs. linear MSG approximation.

Race: VA-10 (Virginia's 10th Congressional District, 2024)
  Cook rating:    Lean D
  Incumbent:      Open seat (no incumbent)
  PVI:            +4.9 (slight D lean)
  D spending:     $2.95M observed  →  $14.71M model-recommended
  R spending:     $9.52M (held fixed)
  Generic ballot: −1.2 pp (2024 cycle)
  indiv_share:    0.800 (share of D receipts from individual donors)
  Outcome:        D won

Despite a favorable PVI, Republicans outspent Democrats 3:1 -- see the
printed "Key numbers" at the bottom of this script's output for the current
model win probability at observed spending (a coefficient/sigma-model
update will shift the exact figure; do not hardcode it here). The optimizer
identified VA-10 as sharply underfunded and recommended a large increase.
The linear MSG approximation overestimates the gain from additional spending
because it cannot capture the S-curve's diminishing returns.

Output: outputs/single_race_response.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ── VA-10 real parameters ──────────────────────────────────────────────────────
# PVI/incumbency/GB/indiv_share are static district facts (don't move when the
# model is re-estimated) and stay hardcoded. D_OBS/D_OPT are *model outputs*
# (observed spend, optimizer recommendation) — read live from the current
# race_table_baseline.csv / aggregate_summary_baseline.csv instead of being
# hardcoded, so they can't silently go stale after a coefficient update.

PVI        = 4.930169768920273   # Cook PVI, D-favoring
INCUMB     = "Open"              # open seat
GB         = -1.2                # 2024 generic ballot
INDIV_SHR  = 0.7997268729439285  # share of D receipts from individuals
R_FIXED    = 9.517135            # $M, Republican total spending

_table = pd.read_csv(ROOT / "outputs" / "race_table_baseline.csv")
_row = _table[_table["district_id"] == "VA-10"].iloc[0]
_agg = pd.read_csv(ROOT / "outputs" / "aggregate_summary_baseline.csv").iloc[0]

D_OBS = float(_row["d_total"]) / 1e6                                    # $M, observed Democratic spending
D_OPT = float(_row["recommended_share"]) * float(_agg["total_budget"]) / 1e6   # $M, model optimizer recommendation


# ── Model helpers ─────────────────────────────────────────────────────────────

is_incumb = 0.0                  # open seat
b1 = coef["beta1_open"]          # open-seat spending coefficient

mu_const = (
    coef["alpha0"]
    + coef["alpha1"] * PVI
    + coef["alpha2"] * is_incumb
    + coef["alpha3"] * GB
    + coef["alpha5"] * INDIV_SHR
)
c_spend = b1 + coef["beta2"] * abs(PVI) + coef["beta3"] * is_incumb
sigma   = sigma_model.predict(abs(PVI), INCUMB, GB)


def p_win(d: np.ndarray) -> np.ndarray:
    ratio = d / (d + R_FIXED)
    mu    = mu_const + c_spend * np.log(np.clip(ratio, 1e-10, 1 - 1e-10))
    return norm.cdf(mu / sigma)


def msg_exact(d: float) -> float:
    """Exact MSG (seats / $1M) using analytical gradient."""
    ratio = d / (d + R_FIXED)
    mu    = mu_const + c_spend * np.log(ratio)
    phi   = norm.pdf(mu / sigma)
    return (phi / sigma) * c_spend * R_FIXED / (d * (d + R_FIXED))


# ── Compute curves ────────────────────────────────────────────────────────────

d_range = np.linspace(0.3, 28.0, 1400)
p_true  = p_win(d_range)

p_obs   = p_win(np.array([D_OBS]))[0]
msg_obs = msg_exact(D_OBS)

tangent = np.clip(p_obs + msg_obs * (d_range - D_OBS), 0.0, 1.0)

# Nonlinear and linear estimates at optimizer recommendation
p_true_opt   = p_win(np.array([D_OPT]))[0]
p_linear_opt = float(np.clip(p_obs + msg_obs * (D_OPT - D_OBS), 0.0, 1.0))
gap_opt       = p_linear_opt - p_true_opt  # positive = linear overestimates

# Structural ceiling: as D → ∞, log(D/(D+R)) → 0, so mu → mu_const
p_asymptote = norm.cdf(mu_const / sigma)


# ── Sanity-check against race table (read live, never hardcoded) ─────────────
print("Sanity check vs. race_table_baseline.csv:")
print(f"  p_win at D_OBS:  computed={p_obs:.4f}  table={float(_row['p_win']):.4f}")
print(f"  sigma:           computed={sigma:.3f}   table={float(_row['sigma_i']):.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────

BLUE   = "#1a6faf"
ORANGE = "#e07b39"
GRAY   = "#888888"
GREEN  = "#2a9d4f"

fig, ax = plt.subplots(figsize=(9, 5.5))

# True curve
ax.plot(d_range, p_true, color=BLUE, lw=2.5,
        label=r"True win probability  $P(\mathrm{win}) = \Phi(\mu/\sigma)$")

# Linear approximation
ax.plot(d_range, tangent, "--", color=ORANGE, lw=2.0,
        label=f"Linear approximation  (MSG = {msg_obs:.4f} seats/\\$1M at observed spending)")

# Divergence shading
ax.fill_between(d_range, p_true, tangent,
                where=(d_range > D_OBS),
                alpha=0.13, color=ORANGE,
                label="Linear overestimates gain (right of observed)")
ax.fill_between(d_range, p_true, tangent,
                where=(d_range < D_OBS),
                alpha=0.10, color=GREEN,
                label="Linear underestimates cost (left of observed)")

# Observed point
ax.axvline(D_OBS, color=GRAY, ls=":", lw=1.2, alpha=0.7)
ax.scatter([D_OBS], [p_obs], color=BLUE, s=90, zorder=6)
ax.annotate(
    f"Observed\nD = \\${D_OBS:.2f}M\n$P$ = {p_obs:.1%}",
    xy=(D_OBS, p_obs),
    xytext=(D_OBS + 0.8, p_obs - 0.07),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0),
)

# Optimizer recommendation
ax.axvline(D_OPT, color=GRAY, ls=":", lw=1.2, alpha=0.5)
ax.scatter([D_OPT], [p_true_opt],   color=BLUE,   s=75, zorder=6, marker="^",
           label=f"At optimizer recommendation (\\${D_OPT:.1f}M): true = {p_true_opt:.1%}")
ax.scatter([D_OPT], [p_linear_opt], color=ORANGE, s=75, zorder=6, marker="v",
           label=f"Linear prediction at \\${D_OPT:.1f}M: {p_linear_opt:.1%}")

# Gap arrow and annotation
ax.annotate("",
            xy=(D_OPT + 0.25, p_true_opt),
            xytext=(D_OPT + 0.25, p_linear_opt),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
ax.text(D_OPT + 1.0, (p_true_opt + p_linear_opt) / 2,
        f"{gap_opt*100:.1f} pp\noverestimate",
        fontsize=8.5, color=ORANGE, va="center")

# Asymptotic ceiling
ax.axhline(p_asymptote, color=BLUE, ls=(0, (4, 6)), lw=1.0, alpha=0.4)
ax.text(27.0, p_asymptote + 0.01,
        f"Ceiling ≈ {p_asymptote:.0%}",
        fontsize=8, color=BLUE, alpha=0.55, ha="right")

# Outcome note
ax.text(0.98, 0.04,
        "Outcome: Democrat won  ✓",
        transform=ax.transAxes, fontsize=9, color=GREEN,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN, alpha=0.8))

ax.set_xlabel("Democratic Spending ($M)", fontsize=12)
ax.set_ylabel("Win Probability", fontsize=12)
ax.set_title(
    "VA-10 (2024): Spending Response Curve — True vs. Linear Approximation\n"
    f"Open seat · PVI +{PVI:.1f} · R = \\${R_FIXED:.1f}M fixed · Generic ballot {GB:+.1f} pp",
    fontsize=11,
)
ax.set_xlim(0, 28)
ax.set_ylim(-0.02, 0.85)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
out = ROOT / "outputs" / "single_race_response.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
print(f"\nKey numbers:")
print(f"  Observed:     D=${D_OBS:.3f}M  P(win)={p_obs:.1%}  MSG={msg_obs:.5f} seats/$1M")
print(f"  At opt rec:   D=${D_OPT}M  true P(win)={p_true_opt:.1%}  linear={p_linear_opt:.1%}  gap={gap_opt*100:.1f} pp")
print(f"  Ceiling (D→∞): P(win) → {p_asymptote:.1%}")
