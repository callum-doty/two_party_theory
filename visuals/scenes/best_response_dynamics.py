"""
Best-response dynamics: (D_0, R_0) -> (D_1, R_1) -> ... -> (D*, R*)
(project_spec.md Section 12). The signature visual of the project.

NOT YET RUNNABLE: ManimGL (`manimlib`) is not installed on this machine --
see visuals/README.md's install step. Written against ManimGL's API
(distinct from Manim Community Edition); untested until that install
happens and `results/nash_equilibrium_<cycle>.json` exists (run
scripts/solve_nash.py first).

Consumes ONLY src/visualization/data_adapter.load_nash_history -- computes
nothing itself (visuals/README.md's design rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from manimlib import *  # noqa: F401,F403 -- ManimGL scene API

from visualization.data_adapter import load_nash_history


class BestResponseDynamics(Scene):
    cycle = 2024

    def construct(self):
        history = load_nash_history(self.cycle)
        e_seats_d = [h["e_seats_d"] for h in history]
        e_seats_r = [h["e_seats_r"] for h in history]
        rounds = list(range(len(history)))

        axes = Axes(
            x_range=[0, max(rounds), max(1, len(rounds) // 10)],
            y_range=[min(e_seats_d + e_seats_r) - 5, max(e_seats_d + e_seats_r) + 5, 10],
            axis_config={"include_tip": True},
        )
        axes.add_coordinate_labels()
        x_label = axes.get_x_axis_label("\\text{best-response round}")
        y_label = axes.get_y_axis_label("\\text{expected seats}")

        d_line = axes.get_line_graph(rounds, e_seats_d, line_color=BLUE, add_vertex_dots=True)
        r_line = axes.get_line_graph(rounds, e_seats_r, line_color=RED, add_vertex_dots=True)

        title = Text(f"Best-response convergence — {self.cycle}", font_size=36)
        title.to_edge(UP)

        self.play(Write(title))
        self.play(ShowCreation(axes), Write(x_label), Write(y_label))
        self.play(ShowCreation(d_line), ShowCreation(r_line), run_time=3)
        self.wait(2)


if __name__ == "__main__":
    print(
        "Render with: manimgl visuals/scenes/best_response_dynamics.py "
        "BestResponseDynamics -o   (requires manimgl + ffmpeg installed, "
        "and results/nash_equilibrium_2024.json to exist -- run "
        "scripts/solve_nash.py --cycle 2024 first)"
    )
