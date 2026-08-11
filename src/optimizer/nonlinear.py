"""
The validated nonlinear optimizer (project_spec.md Section 12: "the existing
nonlinear optimizer should be the benchmark solver"). Re-exported unchanged
from backtest.optimizer.allocator -- this project does not re-derive the
SLSQP formulation, only gives it a stable import path under the new
project's own src/ layout.
"""

from __future__ import annotations

from backtest.optimizer.allocator import (  # noqa: F401
    OptimizerResult,
    nonlinear_expected_seats_at_party_dollars,
    optimize_nonlinear,
)
