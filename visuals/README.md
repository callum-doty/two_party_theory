# Visualization layer (ManimGL)

ManimGL (`manimlib` v1.7.2) and `ffmpeg` are installed (2026-08-12).
`best_response_dynamics.py` is runnable end to end -- static frames only
(see below), the two rendered PNGs live in `visuals/renders/publication/`.

**No LaTeX distribution is installed.** Anything routing through ManimGL's
`Tex`/`MTex` (including `Axes.get_x_axis_label`/`get_y_axis_label` and
`BarChart`'s built-in tick labels) fails with `LaTeX Error: File
'standalone.cls' not found`. Use `Text`/`MarkupText` (font-rendered, no
LaTeX) for all on-scene text instead.

**`Axes` renders blank in this environment**, independent of the LaTeX
issue -- isolated with a minimal repro (a bare `Axes()` + `self.add`
produces nothing, while `Circle`, `Rectangle`, `Line`, and a hand-built
`VMobject` polyline all render correctly). Not debugged further; scenes
build charts by hand instead -- manual `(round, value) -> scene point`
linear scaling plus `Line`/`Rectangle`/`VMobject` primitives, exactly the
combination confirmed to work. See `best_response_dynamics.py`'s module
docstring for the repro details if revisiting this.

**Static images only, no rendered video, per this project's own
direction**: each scene's `construct()` ends on one frame with no `wait()`
loop or camera move; render with `-s -w` (skip animation playback, save
the last frame, write to disk without opening an interactive window) --
NOT `-o` (which opens the file after saving and expects a display).

## Design rule

Manim never calculates the scientific result. Every scene reads FROZEN
result files already written by `scripts/*.py` (`results/*.json`,
`results/*.csv`) -- never re-derives payoff, gradients, or equilibria
itself. `src/visualization/data_adapter.py` is the one place scene code is
allowed to touch `results/`; scenes call into it, not into `src/game/`
directly. This keeps `data -> estimation -> optimization -> frozen results
-> Manim` a one-way pipeline (see `docs/project_spec.md`'s addendum).

## Scene families -> theoretical objects

| Scene (`visuals/scenes/`) | Spec section | Reads |
|---|---|---|
| `race_payoff_surface.py` | §4, single race | one race's `p_i(D_i, R_i)` grid (precompute via `game/payoff.py`, freeze to `results/payoff_surface_<district>.json`) |
| `marginal_value.py` | §6 | same, tangent-plane slice |
| `best_response_dynamics.py` (RUNNABLE) | §12 | `results/nash_equilibrium_<cycle>.json`'s `history` field (scene `BestResponseTrajectory`) + `results/double_oracle_<cycle>.json` (scene `MixedEquilibriumSupport`) |
| `equilibrium_convergence.py` | §11-12 | same, full trajectory to `(D*, R*)` |
| `strategic_surplus_map.py` | §7-8 | `results/race_surplus_<cycle>.csv` |
| `exploitability_decomposition.py` | §14 | `results/persistent_value_<cycle>.json` |
| `budget_waterfill.py` | §7, KKT intuition | `results/race_surplus_<cycle>.csv`'s `MSG_D`/`lambda_D` |
| `dynamic_equilibrium.py` | future `Theta`/dynamic extension | not yet applicable -- static project only so far |

`visuals/configs/` holds per-scene render configs (resolution/fps/quality
presets); `visuals/assets/` holds any static images/fonts; renders land in
`visuals/renders/{preview,publication,video}/` (gitignored -- regenerate,
don't commit).

## Rendering `best_response_dynamics.py`

```bash
# 1. Regenerate the frozen inputs it reads (both already exist as of
#    2026-08-12, but stale after any payoff/equilibrium code change):
#    - results/nash_equilibrium_<cycle>.json: a short surrogate-driven BR
#      trajectory, via game/equilibrium.py::iterate_best_response
#      (use_surrogate=True), saved as {"cycle": ..., "history": [...]}
#    - results/double_oracle_<cycle>.json: scripts/double_oracle.py's output

# 2. Render both static frames (HD, no window, no video):
manimgl visuals/scenes/best_response_dynamics.py BestResponseTrajectory -s -w --hd
manimgl visuals/scenes/best_response_dynamics.py MixedEquilibriumSupport -s -w --hd
```

Note the explicit warning from 3b1b/manim: `manimgl` and Manim Community
Edition (`manim`) are separate packages with separate install instructions
that should not be mixed in the same environment -- this project uses
ManimGL specifically (`docs/project_spec.md`'s addendum).
