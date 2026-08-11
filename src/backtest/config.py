"""Load and expose the central configuration."""

from __future__ import annotations
import os
from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent.parent  # repo root


def _load() -> dict:
    cfg_path = _ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_cfg = _load()


# ─── Convenience accessors ────────────────────────────────────────────────────

def raw_path(source: str) -> Path:
    return _ROOT / _cfg["paths"]["raw"][source]


def processed_path() -> Path:
    return _ROOT / _cfg["paths"]["processed"]


def outputs_path() -> Path:
    return (_ROOT / _cfg["paths"]["outputs"]).resolve()


def universe_cfg() -> dict:
    return _cfg["universe"]


def panel_cycles() -> list[int]:
    return _cfg["panel"]["cycles"]


def min_repeat_pairs() -> int:
    return _cfg["panel"]["min_repeat_challenger_pairs"]


def generic_ballot_2024() -> float:
    gb = _cfg["generic_ballot_2024"]
    if gb is None:
        raise ValueError("generic_ballot_2024 not set in config.yaml")
    return float(gb)


def generic_ballot_for_cycle(cycle: int) -> float:
    """Return the final pre-election generic ballot (D − R) for any cycle."""
    by_cycle = _cfg.get("generic_ballot_by_cycle", {})
    if cycle in by_cycle:
        return float(by_cycle[cycle])
    if cycle == 2024:
        return generic_ballot_2024()
    raise ValueError(f"No generic ballot defined for cycle {cycle}. "
                     f"Add to generic_ballot_by_cycle in config.yaml.")


def beta_rc_prior() -> dict:
    return _cfg.get("beta_rc", {})


def uncertainty_cfg() -> dict:
    return _cfg["uncertainty"]


def optimizer_cfg() -> dict:
    return _cfg["optimizer"]


def persuasion_ceiling_c_max() -> float:
    return float(_cfg["persuasion_ceiling"]["c_max"])


def floor_maturity_reference_dollars() -> float:
    return float(_cfg["persuasion_ceiling"]["floor_maturity_reference_dollars"])


def validation_cfg() -> dict:
    return _cfg["validation"]


def cook_win_probs() -> dict[str, float]:
    return _cfg["cook_win_probs"]


def outputs_cfg() -> dict:
    return _cfg["outputs"]


def budget_2026_projection_cfg() -> dict:
    """CPI-U inputs for backtest.model.budget.estimate_budget_2026(). See
    config.yaml's `budget_2026_projection:` block."""
    return _cfg["budget_2026_projection"]


def dynamic_cfg() -> dict:
    """Paper II (src/backtest/dynamic/) config block. See config.yaml's
    `dynamic:` section and docs/paper2_draft.md §3.2–3.3."""
    return _cfg["dynamic"]


def period_days() -> int:
    """Biweekly grid spacing (days). Single source of truth for the period
    length used by dynamic/periods.py, scripts/simulate_and_validate.py, and
    scripts/solve_bellman_lsm.py -- see config.yaml's `dynamic.period_days`."""
    return int(_cfg["dynamic"]["period_days"])


def election_day(cycle: int):
    """Return the cycle's general-election date as a datetime.date.

    Only cycles with an explicit `election_day_{cycle}` entry in config.yaml
    are supported (currently just the live 2026 cycle) -- historical cycles'
    election days are read directly from dated result/polling data elsewhere
    and don't need a config entry.
    """
    from datetime import date
    key = f"election_day_{cycle}"
    if key not in _cfg["dynamic"]:
        raise ValueError(f"No election_day defined for cycle {cycle}. "
                          f"Add dynamic.{key} to config.yaml.")
    return date.fromisoformat(_cfg["dynamic"][key])


def competitive_ratings() -> list[str]:
    return _cfg["universe"]["competitive_ratings"]


def live_cycle_ballot_exclusions(cycle: int) -> list[dict]:
    """Manually verified interim substitute for MIT's ballot-membership check
    on a cycle still in progress (see config.yaml's `live_cycle_ballot_exclusions`
    block and fec.py's `_ballot_last_names()`). Returns [] for any cycle not
    explicitly listed -- most cycles rely on MIT data instead and need no entry."""
    return _cfg.get("live_cycle_ballot_exclusions", {}).get(cycle, [])


def reload() -> None:
    """Re-read config.yaml from disk (useful after run_estimation writes beta_rc)."""
    global _cfg
    _cfg = _load()
