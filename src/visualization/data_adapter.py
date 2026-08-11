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


def load_exploitability_table(cycles: list[int]) -> pd.DataFrame:
    """One row per cycle: Dem regret / GOP regret / total exploitability
    (spec §22 output #3), from scripts/compute_exploitability.py's output."""
    rows = []
    for cycle in cycles:
        path = RESULTS_DIR / f"exploitability_{cycle}.json"
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)
