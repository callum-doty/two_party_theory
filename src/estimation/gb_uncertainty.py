"""
Residual generic-ballot uncertainty by horizon -- the information-resolution
half of Theta = information option value + strategic flexibility option
value (2026-08-13 follow-up to game/strategic_window.py's response-delay
work, which built the SECOND term only).

Reuses the historical calibration already built for Paper III's Theta work
(scripts/estimate_gb_volatility.py) rather than refitting: pools 2018-2024
538 generic-ballot series (data/raw/generic_ballot/generic_ballot_
historical_538.csv), NOT the 2026-specific live-poll series that script
also loads -- this project's own docstring already validated that pooled
historical file is genuinely cross-cycle, so it needs no 2026 dependency
to reuse. The load/realized-vol logic is reimplemented here (not imported
from scripts/) rather than have src/ depend on scripts/ -- the dependency
direction runs the other way throughout this project (see backtest.data.
fec's DCCC_COMMITTEE_ID for the same reasoning).

USE CASE: at reference date t (days_before days before Election Day), a
committee's read on the national environment is noisy -- the EVENTUAL
generic ballot G_final can still deviate from whatever G_t implies, by an
amount whose standard deviation is exactly what this module estimates
(realized std of the h-day-ahead change in the historical series, h =
days_before). game/information_value.py uses this to simulate "how
different would today's best-looking race have looked under the noisy
date-t signal, vs. the TRUE final one."

NOT a parametric random-walk assumption taken on faith: Paper III's own
Section 5.3 checked whether std/sqrt(days) is roughly constant across
horizons (consistent with simple diffusion) and found it holds
approximately from 30-270 days but is NOT perfectly flat within any one
cycle (2022 realized vol per-sqrt-day actually rises with horizon: 0.21 at
30d -> 0.26 at 180d, the opposite of flat). Because of that, this module
does NOT fit a single per-sqrt-day constant and extrapolate -- it looks up
the DIRECTLY REALIZED std(delta_G) at the exact horizon requested from that
cycle's own historical series, only falling back to sqrt(time) interpolation
between two directly-observed horizons when the exact horizon isn't one of
the ones with enough historical pairs to trust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
GB_HISTORICAL_PATH = REPO_ROOT / "data" / "raw" / "generic_ballot" / "generic_ballot_historical_538.csv"
MIN_PAIRS = 30  # minimum realized (G_t, G_{t+h}) pairs before trusting a horizon's std


def load_historical_gb_series() -> dict[int, pd.Series]:
    """{cycle: G_t series indexed by date}, G_t = Dem% - Rep%. Same
    construction as scripts/estimate_gb_volatility.py::load_historical_series,
    reimplemented here to avoid a src/ -> scripts/ dependency."""
    df = pd.read_csv(GB_HISTORICAL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    series = {}
    for cycle, g in df.groupby("cycle"):
        piv = g.pivot_table(index="date", columns="candidate", values="pct_estimate")
        piv = piv.sort_index()
        gt = (piv["Democrats"] - piv["Republicans"]).asfreq("D").interpolate()
        series[int(cycle)] = gt
    return series


def _realized_std(series: pd.Series, horizon_days: int) -> tuple[float, int]:
    shifted = series.shift(-horizon_days)
    delta = (shifted - series).dropna()
    return float(delta.std()), int(len(delta))


def residual_gb_std(cycle: int, days_before: int, series: pd.Series | None = None) -> float:
    """std(G_final - G_t) for a committee standing `days_before` days
    before Election Day of `cycle` -- the realized historical volatility
    of the generic ballot over that many remaining days, from that SAME
    cycle's own series (not pooled across cycles, since Paper III's own
    check found real cycle-to-cycle differences in this volatility, e.g.
    2022 running meaningfully hotter than 2020). Falls back to the nearest
    horizon with >= MIN_PAIRS realized observations if days_before itself
    is too close to the series' end to have enough pairs."""
    if series is None:
        series = load_historical_gb_series()[cycle]
    std, n = _realized_std(series, days_before)
    if n >= MIN_PAIRS:
        return std
    # Fall back to the largest horizon <= days_before with enough pairs,
    # scaled by sqrt(time) to days_before -- only invoked at the very
    # longest requested horizons where the series itself runs out of room.
    for h in range(days_before - 1, 0, -1):
        std_h, n_h = _realized_std(series, h)
        if n_h >= MIN_PAIRS:
            return std_h * np.sqrt(days_before / h)
    raise ValueError(f"Not enough historical generic-ballot data to estimate volatility for cycle {cycle}")
