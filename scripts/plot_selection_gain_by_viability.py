#!/usr/bin/env python3
"""
Visualize the selection-gain-by-viability decomposition
(decompose_selection_gain_by_viability.py) for both cycles side by side.

Horizontal grouped bars, one group per scenario (A-E), 2024 and 2022 shown
together so the pattern's replication across cycles is visible directly.
Scenario A (headline) is visually separated from B-E (the robustness cuts)
with a divider, and B (hard-exclude flagged races) is called out as the
scenario closest to the reviewer's own "near zero" reading.

Reads outputs/selection_gain_by_viability{,_2022}.csv, written by
decompose_selection_gain_by_viability.py. Run that first (both cycles) if
either is missing.

Usage:
    python scripts/plot_selection_gain_by_viability.py

Output: outputs/selection_gain_by_viability.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

C_2024 = "#1a6faf"
C_2022 = "#2a9d4f"
C_CAVEAT = "#c0392b"

SHORT_LABELS = {
    "A. All races": "A. All races\n(headline)",
    "B. Flagged (publicly viable) excluded": "B. Flagged races\nhard-excluded",
    "C. Unexplained residual only, newly-fundable": "C. Unexplained\nresidual only",
    "D. Flagged races soft-penalized": "D. Flagged races\nsoft-penalized",
    "E. Material unexplained-only (5-6 races)": "E. Material\nunexplained-only",
}


def main() -> None:
    outputs = config.outputs_path()
    df24 = pd.read_csv(outputs / "selection_gain_by_viability.csv")
    df22 = pd.read_csv(outputs / "selection_gain_by_viability_2022.csv")

    scenarios = df24["scenario"].tolist()
    labels = [SHORT_LABELS.get(s, s) for s in scenarios]
    y = np.arange(len(scenarios))[::-1]  # A at top

    fig, ax = plt.subplots(figsize=(10, 6.5))
    h = 0.32
    ax.barh(y + h / 2 + 0.02, df24["modeled_gain"], height=h, color=C_2024, label="2024 (primary)", zorder=3)
    ax.barh(y - h / 2 - 0.02, df22["modeled_gain"], height=h, color=C_2022, label="2022 (OOS)", zorder=3)

    for yi, v24, v22 in zip(y, df24["modeled_gain"], df22["modeled_gain"]):
        ax.text(v24 + 0.05, yi + h / 2 + 0.02, f"+{v24:.2f}", va="center", fontsize=9, color=C_2024)
        ax.text(v22 + 0.05, yi - h / 2 - 0.02, f"+{v22:.2f}", va="center", fontsize=9, color=C_2022)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axhline(y[0] - 0.5, color="#999999", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axvline(0, color="#444444", linewidth=0.8)

    # Callout on scenario B, the reviewer's own "near zero vs. near 2-3" test.
    b_idx = scenarios.index("B. Flagged (publicly viable) excluded")
    ax.axhspan(y[b_idx] - 0.5, y[b_idx] + 0.5, color=C_CAVEAT, alpha=0.07, zorder=0)

    ax.set_xlabel("Modeled seat gain vs. DCCC observed")
    ax.set_title(
        "Selection gain by viability-flag treatment\n"
        "Scenario B (highlighted): 78-85% of the headline gain disappears when flagged races are hard-excluded",
        fontsize=11,
    )
    ax.grid(axis="x", color="#e5e4df", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    ax.set_xlim(0, max(df24["modeled_gain"].max(), df22["modeled_gain"].max()) * 1.25)

    fig.tight_layout()
    out_path = outputs / "selection_gain_by_viability.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
