#!/usr/bin/env python3
"""
Plot the beta_RC bootstrap distribution against the parametric normal it
replaces (docs/paper1_draft.md Section 5.2, docs/full_derivation.md Part III
Section III.21). The paper's uncertainty propagation elsewhere (Monte Carlo
draws in comparison/uncertainty.py) assumes beta_RC's sampling distribution
is N(beta_hat, SE^2) -- symmetric by construction. The non-parametric
bootstrap (resampling the 118 repeat-challenger pairs directly, not assuming
a distribution) instead shows a mild right skew (~+0.2), which a symmetric
normal cannot represent by definition. This figure makes that visible.

Reads outputs/beta_rc_bootstrap_distribution.csv (raw draws) and
data/processed/beta_rc.json / beta_rc_bootstrap.json (summary stats), both
written by scripts/run_estimation.py. Run that script first if either file
is missing.

Output: outputs/beta_rc_bootstrap_distribution.png
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config

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

_COLOR_BOOT = "#1f4e9c"
_COLOR_NORMAL = "#d62728"


def main() -> None:
    processed = config.processed_path()
    outputs = config.outputs_path()

    draws_path = outputs / "beta_rc_bootstrap_distribution.csv"
    if not draws_path.exists():
        sys.exit(
            f"{draws_path} not found. Run `python scripts/run_estimation.py` "
            "first -- it writes the raw bootstrap draws this figure needs."
        )
    draws = pd.read_csv(draws_path)["beta_rc_draw"].to_numpy()

    with open(processed / "beta_rc.json") as f:
        beta_rc = json.load(f)
    beta_hat, se, n_pairs = beta_rc["estimate"], beta_rc["se"], beta_rc["n_pairs"]

    summary_path = processed / "beta_rc_bootstrap.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        skew = summary["bootstrap_skew"]
        boot_ci = summary["bootstrap_ci_95"]
    else:
        skew = float(scipy_stats.skew(draws))
        boot_ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    param_ci = [beta_hat - 1.96 * se, beta_hat + 1.96 * se]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    n_bins = max(20, min(40, int(np.sqrt(len(draws)))))
    ax.hist(
        draws, bins=n_bins, density=True, color=_COLOR_BOOT, alpha=0.55,
        edgecolor="white", linewidth=0.4,
        label=f"Bootstrap draws (n={len(draws):,} resamples of {n_pairs} pairs)",
    )

    x = np.linspace(draws.min() - 1.0, draws.max() + 1.0, 400)
    ax.plot(
        x, scipy_stats.norm.pdf(x, loc=beta_hat, scale=se),
        color=_COLOR_NORMAL, linewidth=2.2,
        label=rf"Parametric $N({beta_hat:.2f},\ {se:.2f}^2)$ — symmetric by construction",
    )

    ax.axvline(beta_hat, color="black", linewidth=1.0, alpha=0.6, zorder=1)

    ax.axvspan(boot_ci[0], boot_ci[1], color=_COLOR_BOOT, alpha=0.06, zorder=0)
    ax.axvline(boot_ci[0], color=_COLOR_BOOT, linestyle="--", linewidth=1.3, alpha=0.85)
    ax.axvline(
        boot_ci[1], color=_COLOR_BOOT, linestyle="--", linewidth=1.3, alpha=0.85,
        label=f"Bootstrap 95% CI [{boot_ci[0]:.2f}, {boot_ci[1]:.2f}]",
    )
    ax.axvline(param_ci[0], color=_COLOR_NORMAL, linestyle=":", linewidth=1.3, alpha=0.85)
    ax.axvline(
        param_ci[1], color=_COLOR_NORMAL, linestyle=":", linewidth=1.3, alpha=0.85,
        label=f"Parametric 95% CI [{param_ci[0]:.2f}, {param_ci[1]:.2f}]",
    )

    ax.annotate(
        f"skew = {skew:+.3f}",
        xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
        fontsize=12, fontweight="bold", color=_COLOR_BOOT,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
    )

    ax.set_xlabel(r"$\hat\beta_{RC}$ (repeat-challenger spending coefficient)")
    ax.set_ylabel("Density")
    ax.set_title(
        r"Non-parametric bootstrap of $\hat\beta_{RC}$ vs. the parametric normal"
        f"\n({n_pairs} repeat-challenger pairs, resampled with replacement)"
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    fig.tight_layout()

    save_path = outputs / "beta_rc_bootstrap_distribution.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved -> {save_path}")
    print(
        f"skew={skew:+.4f}  bootstrap 95% CI={boot_ci}  "
        f"parametric 95% CI={[round(v, 4) for v in param_ci]}"
    )


if __name__ == "__main__":
    main()
