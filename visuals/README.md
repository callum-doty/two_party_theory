# Visualization layer (ManimGL)

Structure-only scaffold. **Neither ManimGL (`manimlib`) nor Manim Community
Edition is installed on this machine yet**, and neither is `ffmpeg` (both
require it). Install is a separate, deliberate follow-up step, not bundled
into this scaffold -- see "Getting ManimGL running" below when ready.

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
| `best_response_dynamics.py` | §12 | `results/nash_equilibrium_<cycle>.json`'s `history` field |
| `equilibrium_convergence.py` | §11-12 | same, full trajectory to `(D*, R*)` |
| `strategic_surplus_map.py` | §7-8 | `results/race_surplus_<cycle>.csv` |
| `exploitability_decomposition.py` | §14 | `results/persistent_value_<cycle>.json` |
| `budget_waterfill.py` | §7, KKT intuition | `results/race_surplus_<cycle>.csv`'s `MSG_D`/`lambda_D` |
| `dynamic_equilibrium.py` | future `Theta`/dynamic extension | not yet applicable -- static project only so far |

`visuals/configs/` holds per-scene render configs (resolution/fps/quality
presets); `visuals/assets/` holds any static images/fonts; renders land in
`visuals/renders/{preview,publication,video}/` (gitignored -- regenerate,
don't commit).

## Getting ManimGL running (when ready)

```bash
brew install ffmpeg pango
pip install manimgl
manimgl visuals/scenes/best_response_dynamics.py SceneName -o   # -o = save final frame as PNG
```

Note the explicit warning from 3b1b/manim: `manimgl` and Manim Community
Edition (`manim`) are separate packages with separate install instructions
that should not be mixed in the same environment.
