"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RaceRecord:
    """One row of the backtest universe — fully constructed before modelling."""

    district_id: str          # Daily Kos stable identifier, e.g. "TX-07"
    state: str
    district: int
    cook_rating: str          # "Safe D" … "Safe R"
    incumb_status: str        # "Incumbent" | "Challenger" | "Open"
    pvi: float                # signed: D-positive, e.g. D+3 → +3, R+8 → -8
    d_total: float            # total Democratic-aligned spend ($)
    r_total: float            # total Republican-aligned spend ($)
    cvap: int                 # citizen voting age population (ACS 2022 5-yr)
    generic_ballot: float     # pre-election 2024 D − R generic ballot average
    redistricting_flagged: bool = False
    outcome: str | None = None  # "D" | "R" | None (filled after results merge)
    cand_d_total: float = 0.0   # candidate-only D spend (floor; not DCCC-controllable)
    indiv_share: float = 0.0    # D candidate TTL_INDIV_CONTRIB / TTL_RECEIPTS (quality proxy)


@dataclass
class BetaRC:
    """Repeat-challenger spending coefficient, estimated from 2012–2022 panel."""

    estimate: float
    se: float
    n_pairs: int


@dataclass
class SigmaModel:
    """
    Fitted heteroskedastic σᵢ function estimated from 2012–2022 panel residuals.

    predict(pvi, incumb_status) → σᵢ in percentage-points of margin.
    """

    _coef: dict  # internal — use predict()

    def predict(self, abs_pvi: float, incumb_status: str, generic_ballot: float = 0.0) -> float:
        """Return fitted σᵢ for a given district type.

        generic_ballot: pre-election D−R GB average. Larger |GB| implies more
        volatile national environment and higher residual scatter.
        """
        a0 = self._coef["intercept"]
        a1 = self._coef["abs_pvi"]
        a2 = self._coef.get("is_open", 0.0)
        a3 = self._coef.get("is_challenger", 0.0)
        a4 = self._coef.get("abs_gb", 0.0)
        # Duan (1983) smearing factor: exp(fitted) alone recovers the median,
        # not the mean, of the log-normal |residual| — see estimation/sigma.py.
        smear = self._coef.get("smearing_factor", 1.0)
        is_open = 1.0 if incumb_status == "Open" else 0.0
        is_chall = 1.0 if incumb_status == "Challenger" else 0.0
        return float(smear * np.exp(
            a0 + a1 * abs_pvi + a2 * is_open + a3 * is_chall + a4 * abs(generic_ballot)
        ))


@dataclass
class FactorModel:
    """Factor covariance matrix for portfolio variance computation."""

    loadings: np.ndarray        # shape (n_races, n_factors)
    factor_cov: np.ndarray      # shape (n_factors, n_factors)
    district_ids: list[str]

    def race_covariance(self) -> np.ndarray:
        """Return full (n_races × n_races) covariance matrix Cov(Yᵢ, Yⱼ)."""
        return self.loadings @ self.factor_cov @ self.loadings.T


@dataclass
class ModelOutputs:
    """Per-race model quantities computed from the spending response surface."""

    district_id: str
    ratio: float          # D / (D + R)
    mu_hat: float         # fitted expected margin (percentage points)
    sigma_i: float        # margin uncertainty (percentage points)
    p_win: float          # Φ(μᵢ / σᵢ)
    msg_i: float          # marginal seat gain per $1M additional Democratic spend


@dataclass
class AllocationResult:
    """Per-race allocation comparison — one row of the primary output table."""

    district_id: str
    recommended_share: float   # sᵢ* / B
    observed_share: float      # D_total_i / B
    difference: float          # recommended − observed


@dataclass
class UncertaintyBundle:
    """
    Distribution of allocation recommendations across β_RC draws.

    recommended_shares_matrix: shape (K, n_races) — one row per draw.
    """

    district_ids: list[str]
    recommended_shares_matrix: np.ndarray   # (K, n_races)
    observed_shares: np.ndarray             # (n_races,)

    def median_share(self) -> np.ndarray:
        return np.median(self.recommended_shares_matrix, axis=0)

    def credible_interval(self, level: float = 0.83) -> tuple[np.ndarray, np.ndarray]:
        lo = (1 - level) / 2
        hi = 1 - lo
        return (
            np.quantile(self.recommended_shares_matrix, lo, axis=0),
            np.quantile(self.recommended_shares_matrix, hi, axis=0),
        )

    def prob_model_exceeds_dccc(self) -> np.ndarray:
        """P(recommended_share_k > observed_share) for each race."""
        return (self.recommended_shares_matrix > self.observed_shares[None, :]).mean(axis=0)
