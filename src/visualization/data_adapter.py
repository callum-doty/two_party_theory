"""
The one place scene code (`visuals/scenes/*.py`) is allowed to read
`results/*.json` / `results/*.csv` -- keeps the Manim layer a pure
CONSUMER of already-computed results, never a place new science gets
computed (see `visuals/README.md`'s design rule). No Manim dependency here,
so this module is testable without ManimGL installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def load_nash_history(cycle: int) -> list[dict]:
    """Round-by-round (party_d, party_r, e_seats_d, e_seats_r) trajectory
    for best_response_dynamics.py / equilibrium_convergence.py, from
    scripts/solve_nash.py's output."""
    path = RESULTS_DIR / f"nash_equilibrium_{cycle}.json"
    with open(path) as f:
        return json.load(f)["history"]


def load_race_surplus(cycle: int) -> pd.DataFrame:
    """(Z_D, Z_R) per district for strategic_surplus_map.py, from
    scripts/compute_race_surplus.py's output."""
    path = RESULTS_DIR / f"race_surplus_{cycle}.csv"
    return pd.read_csv(path)


def load_persistent_value(cycle: int) -> dict:
    """V_uni / PSV / erosion / retention per race for
    exploitability_decomposition.py, from
    scripts/compute_persistent_value.py's output."""
    path = RESULTS_DIR / f"persistent_value_{cycle}.json"
    with open(path) as f:
        return json.load(f)


def load_double_oracle_support(cycle: int) -> dict:
    """D-side and R-side equilibrium support (portfolio index, label,
    mixture weight) for best_response_dynamics.py's mixed-equilibrium
    scene -- from scripts/double_oracle.py's output. Prefers a
    `_resumed` run over the primary one if both exist for this cycle
    (mirrors game/double_oracle.py::load_solved's convention, but this
    function stays JSON-only / Manim-import-free per this module's own
    design rule, so it doesn't import game/double_oracle.py itself)."""
    resumed = RESULTS_DIR / f"double_oracle_{cycle}_resumed.json"
    primary = RESULTS_DIR / f"double_oracle_{cycle}.json"
    path = resumed if resumed.exists() else primary
    with open(path) as f:
        meta = json.load(f)
    return {
        "d_support": sorted(meta["d_support"], key=lambda s: -s["weight"]),
        "r_support": sorted(meta["r_support"], key=lambda s: -s["weight"]),
        "value_e_seats_d": meta["value_e_seats_d"],
        "converged": meta.get("converged", True),
    }


def load_exploitability_table(cycles: list[int]) -> pd.DataFrame:
    """One row per cycle: Dem regret / GOP regret / total exploitability
    (spec §22 output #3), from scripts/compute_exploitability.py's output."""
    rows = []
    for cycle in cycles:
        path = RESULTS_DIR / f"exploitability_{cycle}.json"
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)
