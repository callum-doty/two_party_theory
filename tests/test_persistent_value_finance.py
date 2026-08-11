"""Unit tests for game/persistent_value.py's budget-neutral financing helper
-- pure numpy logic, no model dependency, so these run fast."""

import numpy as np

from game.persistent_value import _finance_delta


def test_finances_delta_from_single_lowest_msg_race():
    party = np.array([100_000.0, 50_000.0, 200_000.0])
    msg = np.array([5e-6, 1e-6, 3e-6])  # race 1 has the lowest marginal value
    out = _finance_delta(party, msg, target_idx=0, delta=20_000.0, cap=1_000_000.0)
    assert out[0] == 120_000.0
    assert out[1] == 30_000.0          # financed entirely from race 1
    assert out[2] == 200_000.0         # untouched
    assert np.isclose(out.sum(), party.sum())  # budget-neutral


def test_cascades_when_lowest_race_cannot_cover_delta_alone():
    party = np.array([100_000.0, 10_000.0, 200_000.0])
    msg = np.array([5e-6, 1e-6, 3e-6])  # race 1 lowest, then race 2
    out = _finance_delta(party, msg, target_idx=0, delta=30_000.0, cap=1_000_000.0)
    assert out[0] == 130_000.0
    assert out[1] == 0.0               # fully drained
    assert out[2] == 180_000.0         # covers the remaining 20,000
    assert np.isclose(out.sum(), party.sum())


def test_respects_target_race_cap():
    party = np.array([90_000.0, 50_000.0])
    msg = np.array([5e-6, 1e-6])
    out = _finance_delta(party, msg, target_idx=0, delta=50_000.0, cap=100_000.0)
    assert out[0] == 100_000.0          # capped at 100k, only 10k actually taken
    assert out[1] == 40_000.0           # only 10k financed, not the full 50k
