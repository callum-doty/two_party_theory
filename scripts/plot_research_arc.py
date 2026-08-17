#!/usr/bin/env python3
"""
Research-arc overview figure for the final paper (2026-08-17): a single
frontispiece figure that lets a reader reconstruct the whole paper's logic
in about fifteen seconds -- static exploitability, competed away by
reciprocal optimization into a mixed-equilibrium distribution, reopened by
irreversible capital into a timing channel, and then narrowed down to one
surviving existence-proof case by four successive stress tests (Sections
15-18).

All numbers are transcribed from already-reported results elsewhere in the
paper (Sections 2-5, 15-19) -- no new computation. The stress-test values
and district lists were cross-checked directly against
results/theta_final_week_sensitivity.json, results/theta_unified_union.json
and results/theta_unified_union_excl_redistricting.json (not just the prose)
after an earlier draft of this figure mis-attributed a district (a
"TN-09" label that does not belong to the redistricting-flagged NC cluster
Section 17 actually describes) and understated the mixed-equilibrium panel
(the 0.06-0.68 residual-regret band is the best-response ORBIT's diagnostic,
not the double-oracle equilibrium itself, which converges exactly -- Table 2).

Output: figures/static/research_arc_overview.png
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "figures" / "static"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "figure.dpi": 150,
})

BLUE = "#3288bd"       # static exploitability
DARK = "#4d4d4d"        # mixed equilibrium
MIDGRAY = "#6b6b6b"     # most opportunity eliminated
ORANGE = "#e66101"      # timing channel appears
RED = "#d73027"         # eliminated
GREEN = "#1a9850"       # survives
INK = "#222222"
GRAY = "#8a8a8a"

STAGE_X = 0.30
STAGE_W = 0.50


def stage_box(ax, y, text, color, textcolor="white", fontsize=10.5, fontweight="bold"):
    ax.text(STAGE_X, y, text, transform=ax.transAxes, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=textcolor, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.55", fc=color, ec="none"), zorder=3)


def spine_arrow(ax, y0, y1, label=None, label_fontsize=8.3, label_color="#555555"):
    ax.annotate("", xy=(STAGE_X, y1 + 0.026), xytext=(STAGE_X, y0 - 0.026),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#666666", linewidth=1.6, shrinkA=0, shrinkB=0),
                 zorder=2)
    if label:
        ax.text(STAGE_X, (y0 + y1) / 2 - 0.003, label, transform=ax.transAxes,
                 ha="center", va="center", fontsize=label_fontsize, color=label_color,
                 style="italic", linespacing=1.3)


def eliminated_row(ax, y, tick_w, header, value, item, ok=False):
    """One stress-test row: a dotted tick from the spine to a right-hand
    block with a bold header, a monospace before/after value, and a
    red (or green, if it survives) verdict line."""
    ax.plot([STAGE_X, STAGE_X + tick_w], [y + 0.019, y + 0.019], color=GRAY, linewidth=0.9,
             linestyle=(0, (2, 2)), transform=ax.transAxes, zorder=1)
    text_x = STAGE_X + tick_w + 0.015
    ax.text(text_x, y + 0.032, header, transform=ax.transAxes, ha="left", va="bottom",
             fontsize=8.5, fontweight="bold", color=INK)
    ax.text(text_x, y + 0.006, value, transform=ax.transAxes, ha="left", va="bottom",
             fontsize=8.6, color="#444444", family="monospace")
    mark, color = ("✓", GREEN) if ok else ("✗", RED)
    ax.text(text_x, y - 0.014, f"{mark}  {item}", transform=ax.transAxes, ha="left", va="top",
             fontsize=8.1, color=color, linespacing=1.35)


def concept_bridge(ax, y):
    """STATIC -> PERSISTENT -> DYNAMIC framing, bridging the equilibrium
    result and the timing channel (this project's central conceptual
    distinction -- see Sections 12, 19-21)."""
    labels = [
        ("STATIC", "Can I gain if the\nopponent stays fixed?", BLUE),
        ("PERSISTENT", "Does it survive their\nbest response?", MIDGRAY),
        ("DYNAMIC", "Is waiting better than\ndeploying elsewhere now?", ORANGE),
    ]
    xs = [STAGE_X - 0.185, STAGE_X, STAGE_X + 0.185]
    for x, (name, desc, color) in zip(xs, labels):
        ax.text(x, y + 0.014, name, transform=ax.transAxes, ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color=color)
        ax.text(x, y - 0.006, desc, transform=ax.transAxes, ha="center", va="top",
                 fontsize=7.4, color="#666666", linespacing=1.3)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - 0.05, y + 0.006), xytext=(x0 + 0.05, y + 0.006),
                     transform=ax.transAxes, xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#999999", linewidth=1.1))
    ax.text(STAGE_X, y + 0.05, "these are not the same quantity — Sections 12 and 19–21",
             transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, color="#888888", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 17.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Stage 1: static exploitability ---
    y1 = 0.965
    stage_box(ax, y1, "STATIC EXPLOITABILITY\n5.14–5.44 expected seats\n(one-shot best response, opponent held passive)", BLUE)

    # --- Stage 2: mixed equilibrium ---
    y2 = 0.855
    spine_arrow(ax, y1, y2, "reciprocal optimization\n(Sections 3–5)")
    stage_box(ax, y2, "MIXED EQUILIBRIUM\nDistribution over 5–11 near-optimal portfolios per side\nno low-regret PURE solution found nearby (best pure regret 0.36–0.86 seats)", DARK, fontsize=10.2)
    ax.text(STAGE_X, y2 - 0.048,
             "2022: 11-portfolio support (75 D / 71 R races touched)   •   2024: 5-portfolio support (59 D / 52 R)",
             transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#666666", style="italic")

    # --- Stage 3: most opportunity eliminated ---
    y3 = 0.712
    spine_arrow(ax, y2, y3, "most apparent opportunity is competed away\n(Sections 6–9)")
    stage_box(ax, y3, "MOST OPPORTUNITY ELIMINATED\nObserved spending tracks a Cook-rating heuristic,\nnot the equilibrium or a unilateral optimum", MIDGRAY, fontsize=10.2)

    # --- Conceptual bridge: static / persistent / dynamic ---
    y_bridge = y3 - 0.11
    ax.plot([STAGE_X, STAGE_X], [y3 - 0.028, y_bridge + 0.065], color="#666666", linewidth=1.6,
             transform=ax.transAxes, zorder=1)
    concept_bridge(ax, y_bridge)

    # --- Stage 4: timing channel appears ---
    y4 = y_bridge - 0.13
    spine_arrow(ax, y_bridge - 0.05, y4, "but capital is irreversible — the opponent's\nresponse set shrinks over time (Sections 10–14)")
    stage_box(ax, y4, "TIMING CHANNEL APPEARS\nInitial K=3 pool: up to +0.155 expected seats\nfrom waiting (2024 R: FL-22, NC-06, NV-01)", ORANGE)
    ax.text(STAGE_X, y4 - 0.044, "four stress tests, Sections 15–18, applied in sequence:",
             transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#555555", style="italic")

    # --- Stress-test funnel, eliminating candidates one by one ---
    y_top = y4 - 0.09
    y_bottom = 0.17
    n_rows = 4
    ys = [y_top - i * (y_top - y_bottom) / (n_rows - 1) for i in range(n_rows)]
    # funnel taper: the horizontal tick narrows from x_top (wide, right after the
    # timing-channel box) to x_bottom (narrow, right before AZ-09) as the candidate
    # set is eliminated row by row -- makes the 4-to-1 narrowing visually legible.
    tick_widths = [0.30 - i * (0.30 - 0.09) / (n_rows - 1) for i in range(n_rows)]
    ax.plot([STAGE_X, STAGE_X], [y_top + 0.02, y_bottom], color="#666666", linewidth=1.6,
             transform=ax.transAxes, zorder=1)

    rows = [
        ("Stress test 1 — exclude the mechanical final-week floor",
         "+0.1553 → +0.0430  (2024 R, K=3)",
         "not a genuine mid-season signal at this pool size (2022 R: +0.0622 → +0.0037, FL-07 fully explained away)", False),
        ("Stress test 2 — widen the pool, K=3 → K≈15–21 (4-criterion union screen)",
         "new headline:  2024 R +0.1033   |   2024 D +0.0796",
         "both driven by the NC-01/06/13/14 (R) and NC-06/13/14 (D) cluster — a redistricting-unstable baseline, not yet trustworthy", False),
        ("Stress test 3 — exclude the redistricting-flagged NC cluster",
         "R: +0.1033 → +0.0492  |  D: +0.0796 → +0.0032",
         "D-side collapses to ≈null (CT-02 dominated by FL-27); R-side's NC leaders drop out — AZ-09 emerges as the new leader", False),
        ("Stress test 4 — monthly (8-date) → weekly (18-date) grid",
         "AZ-09: +0.0492 → +0.0492  (unchanged to 3 d.p.)",
         "trajectory confirmed smooth; a new FL-02 '22(R) value surfaces but is flat for 10 weeks then jumps — rejected, same character as the mechanical floor", True),
    ]
    for y, tick_w, (header, value, item, ok) in zip(ys, tick_widths, rows):
        eliminated_row(ax, y, tick_w, header, value, item, ok=ok)

    # --- Stage 5: AZ-09 survives ---
    y5 = 0.095
    spine_arrow(ax, y_bottom, y5, None)
    stage_box(ax, y5, "AZ-09 ’24 (R) SURVIVES\n+0.049 expected seats — smooth, gradual retention,\nclears every check applied", GREEN)

    # --- Final interpretive box ---
    y6 = 0.032
    spine_arrow(ax, y5, y6, None)
    ax.text(STAGE_X, y6, "EXISTENCE PROOF, NOT A TYPICAL PAYOFF",
             transform=ax.transAxes, ha="center", va="center",
             fontsize=11, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK, linewidth=1.6), zorder=3)
    ax.text(STAGE_X + STAGE_W / 2 + 0.04, y6,
             "2022 null: Theta_genuine ≈ 0 for both sides,\ndespite comparable (even larger) static structure",
             transform=ax.transAxes, ha="left", va="center", fontsize=8.2, color="#666666",
             style="italic", linespacing=1.4)

    fig.suptitle("From apparent static opportunity to one surviving timing opportunity",
                  fontsize=15, x=0.06, ha="left", fontweight="bold", y=0.998)
    fig.text(0.06, 0.985, "A sequential search for strategic timing value that survives reciprocal optimization and stress testing",
              ha="left", va="top", fontsize=10, color="#555555", style="italic")

    caption = ("Each stage's arrow names the mechanism that erodes the previous stage's headline number; the four stress-test "
               "rows show the actual before/after value at each check, and the verdict on what that check eliminated. AZ-09 is "
               "the one candidate, out of dozens tested across two cycles and both parties, that survives all four checks. See "
               "Sections 2–21 for the full derivation of every number shown here.")
    fig.text(0.5, 0.002, textwrap.fill(caption, width=108), ha="center", va="bottom", fontsize=8.0, color="#666666")

    fig.tight_layout(rect=(0, 0.012, 1, 0.975))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "research_arc_overview.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
