"""
Two static frames telling this project's headline result (project_spec.md
Section 12; docs/methodology.md's "Nash equilibrium: a real limit cycle"
and "Double-oracle mixed equilibrium" sections): reciprocal best-response
optimization does not settle on one deterministic portfolio -- it cycles --
and what resolves it is a probability distribution over a handful of
portfolios, not a single allocation.

Consumes ONLY src/visualization/data_adapter.load_nash_history and
load_double_oracle_support -- computes nothing itself (visuals/README.md's
design rule). Needs:
  - results/nash_equilibrium_<cycle>.json (a short surrogate-driven BR
    trajectory; generate via game/equilibrium.py::iterate_best_response
    with use_surrogate=True, saved as {"cycle": ..., "history": [...]})
  - results/double_oracle_<cycle>.json (scripts/double_oracle.py's output)

Chart axes are built by hand (manual coordinate scaling + Line/Rectangle
primitives), NOT ManimGL's `Axes` class: `Axes` renders as blank/invisible
in this environment (isolated and confirmed via a minimal repro -- a plain
`Axes()` with `self.add` produces nothing, while every other primitive
tested renders correctly) with no LaTeX install available to debug via its
usual tex-based tick-label path either. Manual primitives sidestep both
issues and are simple enough for two flat charts.

Static images only (no video): each scene ends on one frame -- render with
`-s -w` (skip animation playback, save the final frame, write to disk
without opening a window).

Render:
    manimgl visuals/scenes/best_response_dynamics.py BestResponseTrajectory -s -w --hd
    manimgl visuals/scenes/best_response_dynamics.py MixedEquilibriumSupport -s -w --hd
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from manimlib import *  # noqa: F401,F403 -- ManimGL scene API

from visualization.data_adapter import load_double_oracle_support, load_nash_history

D_COLOR = BLUE
R_COLOR = RED
AXIS_COLOR = GREY_B


class BestResponseTrajectory(Scene):
    cycle = 2024

    def construct(self):
        history = load_nash_history(self.cycle)
        e_seats_d = [h["e_seats_d"] for h in history]
        rounds = list(range(len(history)))

        title = Text(f"Best-response dynamics — {self.cycle}", font_size=40)
        title.to_edge(UP)
        self.add(title)

        # ---- manual chart box (see module docstring: Axes renders blank here) ----
        x0, x1 = -5.0, 5.0
        y0, y1 = -1.8, 1.8
        v_lo, v_hi = min(e_seats_d) - 0.1, max(e_seats_d) + 0.1

        def to_point(r: int, v: float) -> np.ndarray:
            x = x0 + (x1 - x0) * (r / max(rounds))
            y = y0 + (y1 - y0) * ((v - v_lo) / (v_hi - v_lo))
            return np.array([x, y, 0.0])

        axis_h = Line(np.array([x0, y0, 0]), np.array([x1, y0, 0]), color=AXIS_COLOR)
        axis_v = Line(np.array([x0, y0, 0]), np.array([x0, y1, 0]), color=AXIS_COLOR)
        self.add(axis_h, axis_v)

        # y-axis tick labels at the data min/max so the (small) oscillation band reads directly
        for v in (v_lo + 0.1, v_hi - 0.1):
            tick_y = to_point(0, v)[1]
            self.add(Line(np.array([x0 - 0.08, tick_y, 0]), np.array([x0, tick_y, 0]), color=AXIS_COLOR))
            self.add(Text(f"{v:.1f}", font_size=18).next_to(np.array([x0, tick_y, 0]), LEFT, buff=0.15))

        x_label = Text("best-response round", font_size=22)
        x_label.next_to(axis_h, DOWN, buff=0.3)
        y_label = Text("E[D seats]", font_size=22)
        y_label.next_to(axis_v, UP, buff=0.2)
        self.add(x_label, y_label)

        line = VMobject()
        line.set_points_as_corners([to_point(r, v) for r, v in zip(rounds, e_seats_d)])
        line.set_stroke(color=D_COLOR, width=4)
        self.add(line)

        caption = Text("Bounces. Never settles.", font_size=32, color=YELLOW)
        caption.next_to(axis_h, DOWN, buff=0.9)
        self.add(caption)
        self.wait(0.1)


class MixedEquilibriumSupport(Scene):
    cycle = 2024

    def construct(self):
        title = Text(f"Best-response dynamics — {self.cycle}", font_size=40)
        title.to_edge(UP)
        self.add(title)

        subtitle = Text("The double-oracle equilibrium: a distribution over portfolios", font_size=30)
        subtitle.next_to(title, DOWN, buff=0.35)
        self.add(subtitle)

        support = load_double_oracle_support(self.cycle)
        d_support = support["d_support"]
        weights = [s["weight"] for s in d_support]
        names = [f"Portfolio {chr(65 + i)}" for i in range(len(d_support))]

        chart_height, chart_width = 2.8, 9.0
        n_bars = len(weights)
        bar_width = chart_width / n_bars * 0.6
        max_w = max(weights)
        baseline = ORIGIN + DOWN * 1.6

        bars = VGroup()
        pct_labels = VGroup()
        name_labels = VGroup()
        for i, (w, name) in enumerate(zip(weights, names)):
            x = (i - (n_bars - 1) / 2) * (chart_width / n_bars)
            bar_h = chart_height * (w / max_w)
            bar = Rectangle(width=bar_width, height=bar_h, fill_color=D_COLOR,
                             fill_opacity=0.85, stroke_color=D_COLOR)
            bar.move_to(baseline + RIGHT * x + UP * bar_h / 2)
            bars.add(bar)
            pct_labels.add(Text(f"{w:.0%}", font_size=24).next_to(bar, UP, buff=0.12))
            name_labels.add(Text(name, font_size=18).next_to(bar, DOWN, buff=0.15))

        baseline_line = Line(baseline + LEFT * chart_width / 2, baseline + RIGHT * chart_width / 2,
                              color=AXIS_COLOR)
        self.add(baseline_line, bars, pct_labels, name_labels)

        caption = Text("No stable target list — a stable DISTRIBUTION over target lists.",
                        font_size=26, color=YELLOW)
        caption.next_to(name_labels, DOWN, buff=0.5)
        self.add(caption)
        self.wait(0.1)


if __name__ == "__main__":
    print(
        "Render (static frames only) with:\n"
        "  manimgl visuals/scenes/best_response_dynamics.py BestResponseTrajectory -s -w --hd\n"
        "  manimgl visuals/scenes/best_response_dynamics.py MixedEquilibriumSupport -s -w --hd\n"
        "Requires manimgl + ffmpeg installed, and results/nash_equilibrium_<cycle>.json + "
        "results/double_oracle_<cycle>.json to exist."
    )
