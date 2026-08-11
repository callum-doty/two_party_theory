#!/usr/bin/env python3
"""
Race-level (Z_D, Z_R) strategic-surplus map for one cycle
(project_spec.md Sections 7-8) -- headline output #1: the scatter of every
district in (Z_D, Z_R) space, plus the quadrant taxonomy (Democratic
opportunity / Republican opportunity / under-contested / over-capitalized /
locally equilibrated). Descriptive until verified by
compute_persistent_value.py's explicit best-response calculations (spec
Section 8's own caveat).

Usage:
    python scripts/compute_race_surplus.py --cycle 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.dpi": 150,
})

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import exploitability  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_race_surplus")

COOK_COLORS = {
    "Safe D": "#08306b", "Likely D": "#2171b5", "Lean D": "#6baed6",
    "Toss-Up": "#969696",
    "Lean R": "#fc9272", "Likely R": "#de2d26", "Safe R": "#a50f15",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Race-level (Z_D, Z_R) strategic surplus map")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]

    surplus = exploitability.race_level_surplus(
        races, coef, sigma_model, state["cand_r_total"], state["budget_d"], state["budget_r"],
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
    )
    logger.info(f"lambda_D = {surplus['lambda_D']:.6e} seats/$, "
                f"lambda_R = {surplus['lambda_R']:.6e} seats/$")

    rows = []
    for i, r in enumerate(races):
        sd, sr = float(surplus["S_D"][i]), float(surplus["S_R"][i])
        rows.append({
            "district_id": r.district_id, "cook_rating": r.cook_rating, "pvi": r.pvi,
            "p_win_obs": float(surplus["p_win_obs"][i]),
            "MSG_D": float(surplus["g_D_obs"][i]), "MSG_R": float(surplus["g_R_obs"][i]),
            "S_D": sd, "S_R": sr,
            "Z_D": float(surplus["Z_D"][i]), "Z_R": float(surplus["Z_R"][i]),
            "quadrant": exploitability.quadrant(sd, sr),
        })
    df = pd.DataFrame(rows)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"race_surplus_{args.cycle}.csv"
    df.sort_values("S_D", ascending=False).to_csv(csv_path, index=False)
    logger.info(f"Saved -> {csv_path}")
    logger.info(f"Quadrant counts: {df['quadrant'].value_counts().to_dict()}")

    make_scatter(df, args.cycle, REPO_ROOT / "figures" / "static")


def make_scatter(df: pd.DataFrame, cycle: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def slog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    x, y = slog(df["Z_D"].to_numpy()), slog(df["Z_R"].to_numpy())
    colors = df["cook_rating"].map(COOK_COLORS).fillna("#969696")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(x, y, c=colors, s=28, alpha=0.8, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$Z_D = MSG_i^D / \lambda_D - 1$  (signed-log)")
    ax.set_ylabel(r"$Z_R = MSG_i^R / \lambda_R - 1$  (signed-log)")
    ax.set_title(f"Race-level two-player strategic-surplus map — {cycle}")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=k)
               for k, c in COOK_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8, title="Cook rating")

    fig.tight_layout()
    fig_path = out_dir / f"race_surplus_scatter_{cycle}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
