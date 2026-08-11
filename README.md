# Strategic Campaign Allocation

A two-player model of Democratic/Republican House campaign spending: both
committees allocate finite budgets simultaneously, and the question is
whether observed spending is close to a strategic equilibrium -- not
whether one side is spending optimally in isolation. Full formalism in
[`docs/project_spec.md`](docs/project_spec.md); implementation notes and
open questions in [`docs/methodology.md`](docs/methodology.md).

Separate project from, but built directly on top of, the prior
single-optimizer trilogy (`../Political Portfolio`): same public FEC/Census/
Cook/generic-ballot data, same estimated margin model and persuasion
ceiling, same validated nonlinear optimizer. What's new here is treating R
as an endogenous strategic player rather than a fixed or mechanically
reactive opponent -- see `docs/methodology.md` for exactly where this
project's math diverges from `backtest.optimizer.nash`'s existing
one-off Nash implementation.

## Layout

```
src/
    backtest/       # reused foundation (data pipeline, margin model, nonlinear
                     # optimizer, existing Nash solver) -- vendored unchanged
                     # from ../Political Portfolio/src/backtest
    estimation/      # thin re-exports of backtest's response/uncertainty model
    game/            # the new two-player layer: payoff, gradients, best
                     # response, equilibrium, exploitability, persistent value
    optimizer/       # nonlinear (re-export) + concave surrogate (D validated,
                     # R not yet)
    validation/      # synthetic-game (Level C) + historical-backtest orchestration
scripts/             # CLI entry points, one per src/ capability (see docs/project_spec.md §25)
data/                # raw/ + processed/ (reused), catalog/ (new data dictionary)
results/             # JSON/CSV outputs from scripts/*.py -- consumed by visuals/, not committed by default
figures/             # matplotlib static figures
visuals/             # Manim (ManimGL) scene scaffold -- see visuals/README.md
tests/               # fast unit + Level C tests by default; `pytest -m slow` for real-universe checks
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`config.yaml` (copied from the source project) points at `data/raw/` and
`data/processed/` already populated in this repo. `outputs/` and `results/`
are created on first run.

## Quickstart

```bash
python scripts/build_cycle_state.py --cycle 2024        # race universe + budgets summary
python scripts/compute_exploitability.py --cycle 2024   # headline RegretD/RegretR/E
python scripts/compute_race_surplus.py --cycle 2024      # (Z_D, Z_R) map -> results/ + figures/static/
python scripts/solve_nash.py --cycle 2024 --damping-theta 0.5 --max-rounds 40   # slow: tens of minutes
python scripts/compute_persistent_value.py --cycle 2024  # PSV for top-surplus races
```

`pytest` runs the fast suite (synthetic-game + unit tests) in a few seconds.
`pytest -m slow` additionally exercises the real 433-race 2024 universe
(~2 minutes).

## Status

Scaffolded 2026-08-11: the two-player math (`src/game/`) is implemented and
smoke-tested against the real 2024 universe -- `compute_exploitability.py`
reproduces the old trilogy's one-shot regret figures (RegretD ~2.85,
RegretR ~4.6 seats).

Also on 2026-08-11, two data-pipeline fixes, both in `docs/methodology.md`:

1. Audited how R's spending environment is built and closed a real data gap
   -- state party committees' 24K coordinated expenditures were only ever
   scanned for Democratic state parties (old project's `FINDINGS.md`
   Section 10.7 Gap 3). Added the Republican-side scan, ran it for both
   2022 and 2024. Small dollar impact by itself (NRCC 2024: $131.95M ->
   $132.11M).
2. **Bigger fix, caught by design review before any historical backtest
   ran**: `party_d`/`party_r` (the two-player game's actual decision
   variables) were "total minus candidate money," which includes outside-
   group independent expenditures DCCC/NRCC don't control. Redefined via
   `src/estimation/control_provenance.py` to be only each national
   committee's OWN coordinated + own-IE money. This moved DCCC's 2024
   budget from $465.2M to **$102.1M**, NRCC's from $132.1M to **$47.2M**
   (NRCC's own IEs alone are $48.4M -- 16x its coordinated spending, and
   were previously indistinguishable from Congressional Leadership Fund's
   or any other outside group's). RegretD/RegretR at the corrected, smaller
   budgets: 2.36 / 3.23 seats (were 2.85 / 4.61).

Not yet run: the full 2022/2024 historical backtest (spec §26's MVP), the
R-side surrogate validation, the D/R symmetry test, and the Manim renders
(`visuals/` is structure-only -- neither ManimGL nor ffmpeg are installed on
this machine yet).
