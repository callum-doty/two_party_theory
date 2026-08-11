"""
p_i(D_i, R_i) as a 3D surface for one illustrative race, with the observed
point and its two derivatives (MSG_D, MSG_R -- spec Section 6) drawn as
tangent lines. Reads ONLY results/payoff_surface_<district>_<cycle>.json
(scripts/build_payoff_surface_data.py) -- computes nothing itself, per
visuals/README.md's design rule.

Render (static final frame, per visuals/README.md):
    manimgl visuals/scenes/race_payoff_surface.py RacePayoffSurface -s --uhd -o
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from manimlib import *  # noqa: F401,F403

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
Z_HEIGHT = 4.0  # visual height mapped from p_win in [0, 1]


def load_surface(district_id: str, cycle: int) -> dict:
    path = RESULTS_DIR / f"payoff_surface_{district_id}_{cycle}.json"
    with open(path) as f:
        return json.load(f)


class PayoffSurface(Surface):
    """p_win(D, R) surface, D/R in dollars mapped onto axes' native units
    via the supplied ThreeDAxes coordinate-to-point function."""

    def __init__(self, interp: RegularGridInterpolator, axes: "ThreeDAxes", **kwargs):
        self.interp = interp
        self.axes = axes
        super().__init__(**kwargs)

    def uv_func(self, u: float, v: float) -> np.ndarray:
        p = float(self.interp([[u, v]])[0])
        return self.axes.c2p(u, v, p * Z_HEIGHT)


class RacePayoffSurface(ThreeDScene):
    district_id = "MI-08"
    cycle = 2024

    def construct(self):
        data = load_surface(self.district_id, self.cycle)
        d_grid = np.array(data["d_grid"])
        r_grid = np.array(data["r_grid"])
        p_grid = np.array(data["p_grid"])
        d_max, r_max = float(d_grid[-1]), float(r_grid[-1])

        interp = RegularGridInterpolator((d_grid, r_grid), p_grid, bounds_error=False, fill_value=None)

        axes = ThreeDAxes(
            x_range=(0, d_max, d_max / 4),
            y_range=(0, r_max, r_max / 4),
            z_range=(0, Z_HEIGHT, Z_HEIGHT / 4),
            width=6, height=6, depth=Z_HEIGHT,
        )

        surface = PayoffSurface(interp, axes, u_range=(0, d_max), v_range=(0, r_max), resolution=(41, 41))
        surface.set_color(BLUE_D)
        surface.set_opacity(0.85)
        mesh = SurfaceMesh(surface, resolution=(21, 21))
        mesh.set_stroke(WHITE, 0.5, opacity=0.25)

        x_label = Text("D spend ($)", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("R spend ($)", font_size=24).next_to(axes.y_axis.get_end(), UP)
        title = Text(f"{self.district_id} ({data['cook_rating']}, PVI {data['pvi']:+.1f}) — {self.cycle}",
                     font_size=32).to_corner(UL)
        title.fix_in_frame()

        d_obs, r_obs, p_obs = data["party_d_obs"], data["party_r_obs"], data["p_win_obs"]
        msg_d, msg_r = data["MSG_D_obs"], data["MSG_R_obs"]
        obs_point = axes.c2p(d_obs, r_obs, p_obs * Z_HEIGHT)
        dot = Sphere(radius=0.07).move_to(obs_point).set_color(YELLOW)

        # Tangent line lengths scaled for visibility (raw MSG is ~1e-8
        # seats/$, far too small to draw at native scale): each spans 18%
        # of the D-axis range, preserving relative sign/magnitude between
        # the two so a steeper curve visibly reads as a steeper line.
        arrow_len = 0.18 * d_max
        d_tip = axes.c2p(d_obs + arrow_len, r_obs, (p_obs + msg_d * arrow_len) * Z_HEIGHT)
        r_tip = axes.c2p(d_obs, r_obs + arrow_len, (p_obs - msg_r * arrow_len) * Z_HEIGHT)
        d_line = Line(obs_point, d_tip, color=GREEN, stroke_width=6)
        r_line = Line(obs_point, r_tip, color=RED, stroke_width=6)
        d_tip_dot = Sphere(radius=0.045).move_to(d_tip).set_color(GREEN)
        r_tip_dot = Sphere(radius=0.045).move_to(r_tip).set_color(RED)

        info = VGroup(
            Text(f"P(D wins) = {p_obs:.1%}", font_size=22, color=YELLOW),
            Text(f"MSG_D = {msg_d:+.2e} seats/$", font_size=22, color=GREEN),
            Text(f"MSG_R = {msg_r:+.2e} seats/$", font_size=22, color=RED),
            Text(f"D spend = ${d_obs:,.0f}   R spend = ${r_obs:,.0f}", font_size=20, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR)
        info.fix_in_frame()

        self.frame.reorient(-40, 65)
        self.add(axes, surface, mesh, x_label, y_label, title, info,
                  dot, d_line, r_line, d_tip_dot, r_tip_dot)
        self.wait(0.1)
