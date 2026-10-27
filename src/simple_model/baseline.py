"""
Job 1: baseline win probability.

    margin_i = a + b * PVI_i + c[status_i] + d * floor_share_i + GB
    P(D wins i) = Phi(margin_i / sigma)

floor_share_i = log((F_D + k) / (F_D + F_R + 2k)), the Democratic share of
the money neither national committee controls (candidate, state party,
outside groups; estimation/control_provenance.py), k = $10K. It is a
forecasting feature standing in for candidate strength, not a causal effect:
the causal effect of party money is job 2's business. Out of sample it moves
expected D seats from 205 to 211 (actual 214) in 2024 and from 206 to 215
(actual 215) in 2022.

Fit by OLS on contested races (|margin| < 80) with sigma = the residual SD.
The generic ballot enters with a fixed 1:1 pass-through (uniform swing).

PVI comes from simple_model/pvi.py, on the district lines each election
actually used, and each race's tier is re-derived from it. The universe's
own PVI (backtest/data/pvi.py) puts 2012-2020 races on 118th-Congress
lines; fitting on that flattens the PVI slope (1.06 margin points per PVI
point in the old model, ~1.8 on correct lines) and inflates sigma, which is
what made long shots look winnable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRAINING_CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024)
STATUSES = ("Incumbent", "Challenger", "Open")   # D-perspective, as in RaceRecord
UNCONTESTED_MARGIN = 80.0
FLOOR_PSEUDO_DOLLARS = 10_000.0


def races_frame(races, floor_r) -> pd.DataFrame:
    """RaceRecords (after apply_control_floor) plus its R-floor array -> the
    columns both jobs and the allocator need."""
    return pd.DataFrame([dict(
        district_id=r.district_id, cook_rating=r.cook_rating, status=r.incumb_status,
        pvi=r.pvi, gb=r.generic_ballot, cvap=r.cvap, outcome=r.outcome,
        floor_d=r.cand_d_total, floor_r=float(fr),
    ) for r, fr in zip(races, floor_r)])


def _design(df: pd.DataFrame) -> np.ndarray:
    status = df["status"].to_numpy()
    unknown = set(status) - set(STATUSES)
    if unknown:
        raise ValueError(f"unknown incumbency status: {sorted(unknown)}")
    fd = df["floor_d"].to_numpy(dtype=float) + FLOOR_PSEUDO_DOLLARS
    fr = df["floor_r"].to_numpy(dtype=float) + FLOOR_PSEUDO_DOLLARS
    return np.column_stack([
        np.ones(len(df)),
        df["pvi"].to_numpy(dtype=float),
        (status == "Incumbent").astype(float),
        (status == "Open").astype(float),
        np.log(fd / (fd + fr)),
    ])


@dataclass(frozen=True)
class Baseline:
    coef: np.ndarray            # intercept, PVI, D-incumbent, open seat, floor share (Challenger is the reference)
    sigma: float
    train_cycles: tuple[int, ...]
    n_train: int

    def margin(self, df: pd.DataFrame) -> np.ndarray:
        """Expected D margin (pp) at the status quo, including the national swing."""
        return _design(df) @ self.coef + df["gb"].to_numpy(dtype=float)

    def win_prob(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.margin(df) / self.sigma)


def fit_frame(df: pd.DataFrame, train_cycles: tuple[int, ...] = ()) -> Baseline:
    """Fit from a frame with columns pvi, status, floor_d, floor_r, gb, margin."""
    df = df[df["margin"].abs() < UNCONTESTED_MARGIN]
    y = df["margin"].to_numpy(dtype=float) - df["gb"].to_numpy(dtype=float)
    X = _design(df)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    return Baseline(coef=coef, sigma=sigma, train_cycles=tuple(train_cycles), n_train=len(y))


def load_cycle(cycle: int) -> tuple[list, pd.DataFrame]:
    """One cycle's control-floored races, and their frame with the actual D
    margin attached (NaN where no result)."""
    import dataclasses

    from backtest.data import elections
    from backtest.data.pvi import derive_rating
    from backtest.data.universe import build_universe
    from estimation.control_provenance import apply_control_floor

    from .pvi import load_pvi

    pvi = load_pvi(cycle)
    races = [dataclasses.replace(r, pvi=float(pvi[r.district_id]),
                                 cook_rating=derive_rating(float(pvi[r.district_id]), r.incumb_status))
             for r in build_universe(cycle=cycle) if r.district_id in pvi.index]
    races, floor_r = apply_control_floor(races, cycle)
    df = races_frame(races, floor_r)
    margins = elections.load_results(cycle).set_index("district_id")["margin_pp"]
    return races, df.assign(cycle=cycle, margin=df["district_id"].map(margins))


def fit(cycles: tuple[int, ...]) -> Baseline:
    df = pd.concat([load_cycle(c)[1] for c in cycles], ignore_index=True).dropna(subset=["margin"])
    return fit_frame(df, cycles)


TIERS = ("Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R")


def calibration_by_tier(p: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted vs. actual D win rate per tier, plus Brier score, for
    races with a known result."""
    known = df["margin"].notna().to_numpy()
    won = (df["margin"].to_numpy() > 0)[known]
    p, tier = p[known], df["cook_rating"].to_numpy()[known]
    rows = [dict(tier=t, n=int((tier == t).sum()), predicted=p[tier == t].mean(),
                 actual=won[tier == t].mean()) for t in TIERS if (tier == t).any()]
    out = pd.DataFrame(rows)
    out.attrs["brier"] = float(np.mean((p - won) ** 2))
    return out
