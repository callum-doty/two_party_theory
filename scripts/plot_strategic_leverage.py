#!/usr/bin/env python3
"""
Four-panel summary figure for the strategic-leverage / response-displacement
findings (docs/methodology.md's "Strategic leverage and response
displacement" section, 2026-08-13). Consumes results/strategic_leverage_
{cycle}.json only -- no recomputation, per this project's Manim/figure
design rule (data -> estimation -> optimization -> frozen results -> chart).

  A (top-left):     Leverage curve (seats sacrificed per $1M) vs delta,
                     2024, top-3 D/R candidates.
  B (top-right):     Same, 2022.
  C (bottom-left):   V_uni -> PSV dumbbell per race -- how much of the
                     unilateral opportunity survives the opponent's best
                     response (retention), same 12 races shown in A/B.
  D (bottom-right):  Response-displacement heatmap, 2022 D-side pressure --
                     four different target races almost all draw from the
                     same 8-race Republican "financing pool."

Output: figures/static/strategic_leverage_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "figures" / "static"

D_COLOR = "#2e6da4"   # matches this project's existing Cook-rating palette ("Likely D")
R_COLOR = "#c0392b"   # matches "Likely R"
GRAY = "#9a9a9a"
INK = "#333333"


def load(cycle: int) -> dict:
    return json.load(open(RESULTS / f"strategic_leverage_{cycle}.json"))


def _curve_points(primary: list[dict], curve: list[dict], district_id: str,
                   large: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the $1M primary point, the small-delta curve, and (if given)
    the large-delta extension for one race, sorted by delta. Returns
    (delta_deployed_$M, leverage, capped_bool) -- large-delta rows use
    delta_deployed rather than the nominal request, since capping can make
    them differ once delta approaches a race's per-race spending cap."""
    rows = [r for r in primary if r["district_id"] == district_id]
    rows += [r for r in curve if r["district_id"] == district_id]
    if large:
        rows += [r for r in large if r["district_id"] == district_id]
    rows = sorted(rows, key=lambda r: r["delta"])
    deployed = np.array([r.get("delta_deployed", r["delta"]) for r in rows]) / 1e6
    leverage = np.array([r["leverage_seats_per_million"] for r in rows])
    capped = np.array([r.get("capped", False) for r in rows])
    return deployed, leverage, capped


def panel_curves(ax, data: dict, cycle: int, large: dict | None = None) -> None:
    top_d = sorted({r["district_id"] for r in data["leverage_D_curve"]})
    top_r = sorted({r["district_id"] for r in data["leverage_R_curve"]})
    large_d = large["leverage_D_large"] if large else None
    large_r = large["leverage_R_large"] if large else None

    labels: list[tuple[str, float, float, str]] = []  # (district_id, x, y, color)
    for did in top_d:
        x, y, capped = _curve_points(data["leverage_D_primary"], data["leverage_D_curve"], did, large_d)
        ax.plot(x, y, color=D_COLOR, marker="o", markersize=4, linewidth=1.6, alpha=0.85)
        if capped.any():
            ax.scatter(x[capped], y[capped], marker="x", s=45, color=D_COLOR, zorder=5)
        labels.append((did, x[-1], y[-1], D_COLOR))
    for did in top_r:
        x, y, capped = _curve_points(data["leverage_R_primary"], data["leverage_R_curve"], did, large_r)
        ax.plot(x, y, color=R_COLOR, marker="o", markersize=4, linewidth=1.6,
                 linestyle="--", alpha=0.85)
        if capped.any():
            ax.scatter(x[capped], y[capped], marker="x", s=45, color=R_COLOR, zorder=5)
        labels.append((did, x[-1], y[-1], R_COLOR))

    # Nudge apart labels whose endpoints land within ~3% of the y-range of
    # each other -- otherwise two nearly-equal-leverage races render as
    # illegible overlapping text (found in the first render, 2024's NV-01
    # vs. FL-22).
    ys = [l[2] for l in labels]
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    tol = 0.03 * y_span
    labels.sort(key=lambda l: l[2])
    POINTS_PER_COLLISION = 9.0
    offsets_pts = [0.0] * len(labels)
    for i in range(1, len(labels)):
        if labels[i][2] - labels[i - 1][2] < tol:
            offsets_pts[i] = offsets_pts[i - 1] + POINTS_PER_COLLISION
    for (did, x, y, color), off_pts in zip(labels, offsets_pts):
        ax.annotate(did, (x, y), xytext=(4, off_pts), textcoords="offset points",
                    fontsize=7.5, color=color, va="center")

    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.set_xlabel("Commitment size, dollars actually deployed ($M)")
    ax.set_ylabel("Leverage (expected seats sacrificed / $1M)")
    subtitle = "pool exhaustion at scale (x = capped at the race's spending cap)" if large else "leverage declines as commitment grows"
    ax.set_title(f"{cycle}: {subtitle}", fontsize=10.5, color=INK, loc="left")
    ax.set_xlim(0, 13 if large else 2.5)


def panel_dumbbell(ax, data_by_cycle: dict[int, dict]) -> None:
    rows = []
    for cycle, data in data_by_cycle.items():
        for side, primary_key, color in (("D", "leverage_D_primary", D_COLOR), ("R", "leverage_R_primary", R_COLOR)):
            curve_key = f"leverage_{side}_curve"
            top_ids = sorted({r["district_id"] for r in data[curve_key]})
            for r in data[primary_key]:
                if r["district_id"] in top_ids:
                    rows.append(dict(label=f"{r['district_id']} '{str(cycle)[2:]}", side=side, color=color,
                                      v_uni=r["V_uni"], psv=r["PSV"]))
    rows = sorted(rows, key=lambda r: (r["side"], r["psv"]))
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        ax.plot([r["v_uni"], r["psv"]], [y, y], color=GRAY, linewidth=1.2, zorder=1)
        ax.scatter([r["v_uni"]], [y], facecolor="white", edgecolor=r["color"], linewidth=1.4, s=32, zorder=2)
        ax.scatter([r["psv"]], [y], facecolor=r["color"], edgecolor=r["color"], s=32, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("Expected seats, $1M commitment")
    ax.set_title("Unilateral value → value after opponent's best response", fontsize=10.5, color=INK, loc="left")
    open_dot = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                           markeredgecolor=INK, markersize=6, label="V_uni (opponent fixed)")
    filled_dot = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
                             markeredgecolor=INK, markersize=6, label="PSV (opponent best-responds)")
    ax.legend(handles=[open_dot, filled_dot], loc="lower right", fontsize=7.5, frameon=False)


def panel_heatmap(ax, data: dict, cycle: int) -> None:
    """Signed displacement: negative (red) = R cuts from this race to fund
    its response, positive (blue) = R reinforces this race instead. Both
    directions draw on the SAME small set of races across different D-side
    targets -- some targets fund their response by cutting this pool, others
    by reinforcing it (and financing that reinforcement with cuts too
    diffuse, across too many races, to show up in any single race's
    top-8 movers list) -- either way, this pool is where the action is."""
    movers = [r for r in data["leverage_D_primary"] if r["r_top_cuts"] or r["r_top_adds"]]
    targets = sorted({r["district_id"] for r in movers})
    pool: list[str] = []
    for r in movers:
        for c in r["r_top_cuts"] + r["r_top_adds"]:
            if c["district_id"] not in pool:
                pool.append(c["district_id"])
    pool = sorted(pool)
    mat = np.full((len(targets), len(pool)), np.nan)
    for i, did in enumerate(targets):
        row = next(r for r in movers if r["district_id"] == did)
        for c in row["r_top_cuts"] + row["r_top_adds"]:
            j = pool.index(c["district_id"])
            mat[i, j] = c["delta"]

    vmax = np.nanmax(np.abs(mat))
    masked = np.ma.masked_invalid(mat)
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked / 1e3, cmap=cmap, aspect="auto", vmin=-vmax / 1e3, vmax=vmax / 1e3)
    ax.set_xticks(range(len(pool)))
    ax.set_xticklabels(pool, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8.5)
    ax.set_title(f"{cycle}: the same handful of Republican races absorb every D-side response\n"
                 f"(row = D's $1M target; column = where R's re-optimized budget moves; red=cut, blue=reinforced)",
                 fontsize=9.5, color=INK, loc="left")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$ moved, signed (thousands)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _load_large(cycle: int) -> dict | None:
    path = RESULTS / f"strategic_leverage_large_delta_{cycle}.json"
    return json.load(open(path)) if path.exists() else None


def main() -> None:
    data_2024 = load(2024)
    data_2022 = load(2022)
    large_2024 = _load_large(2024)
    large_2022 = _load_large(2022)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    panel_curves(axes[0, 0], data_2024, 2024, large_2024)
    panel_curves(axes[0, 1], data_2022, 2022, large_2022)
    panel_dumbbell(axes[1, 0], {2024: data_2024, 2022: data_2022})
    panel_heatmap(axes[1, 1], data_2022, 2022)

    d_line = plt.Line2D([0], [0], color=D_COLOR, linewidth=1.6, marker="o", markersize=4, label="Democratic commitment (R responds)")
    r_line = plt.Line2D([0], [0], color=R_COLOR, linewidth=1.6, linestyle="--", marker="o", markersize=4, label="Republican commitment (D responds)")
    x_marker = plt.Line2D([0], [0], color=INK, marker="x", linestyle="none", markersize=6, label="capped at race's spending limit")
    fig.legend(handles=[d_line, r_line, x_marker], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               fontsize=9.5, frameon=False)

    fig.suptitle("Strategic leverage and response displacement, 2022 & 2024",
                  fontsize=13.5, color=INK, y=1.035, fontweight="bold")
    fig.text(0.5, 0.005,
              "leverage = PSV / (delta deployed / \\$1M); PSV = expected-seat value retained after the opponent's exact best response (persistent_value.py's isolated-baseline convention). "
              "Top row extends to \\$12M, capped by each race's own spending limit. Exact SLSQP throughout.",
              ha="center", fontsize=7.5, color=GRAY)

    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "strategic_leverage_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
