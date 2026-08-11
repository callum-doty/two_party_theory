"""
The state-update operator f (paper §3.1, §3.3, docs/paper2_draft.md).

Section 3.1 leaves f fully generic as a theoretical object. This module
provides `EMAStateUpdater` — the concrete "f_baseline" the paper commits to
for its empirical sections (§3.3, §6, §7): a simple exponential moving
average, chosen to suppress raw period-to-period polling/spending noise,
not because it is the theoretically preferred choice (paper §8 states this
explicitly as an untested modeling choice). A Bayesian update or a formal
Kalman/particle filter is the natural future replacement; any such
replacement implements the same `StateUpdater` Protocol used here.

`compute_raw_snapshot()` is the only place this module touches Paper I's
estimation pipeline, and it does so purely by calling it:
`compute_outputs_batch()` (unmodified) re-derives mu/sigma from that
period's RaceRecord snapshot. Model coefficients (`coef`, `sigma_model`)
are fit once by `run_estimation.py` and held fixed across every period —
this module never re-fits them.
"""

from __future__ import annotations
import dataclasses
from datetime import date
from typing import Protocol

from ..types import RaceRecord, SigmaModel
from ..model.margin import MarginModelCoefficients
from ..model.win_prob import compute_outputs_batch
from .state import CampaignState, RaceState


def compute_raw_snapshot(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    period: int,
    period_date: date,
    generic_ballot_national: float,
    cash_on_hand_by_district: dict[str, float] | None = None,
) -> CampaignState:
    """Re-run Paper I's fitted pipeline (unmodified) on this period's
    RaceRecord snapshot to produce the pre-smoothing raw state.

    This is the "naive f" described in paper §3.3: `coef` and `sigma_model`
    are held fixed; only each race's `d_total`/`r_total` (and therefore its
    spend ratio) varies by period. `mu_hat`/`sigma_hat` are initialized to
    the raw values here — `EMAStateUpdater.update()` (or any other
    `StateUpdater`) is what actually smooths them against the prior period.

    `cash_on_hand_by_district` (district_id -> dollars) is optional and, if
    supplied, populates `RaceState.cash_on_hand_d` — previously an
    unconditional stub (this module's docstring history / data_catalog.md
    §2.7). This module stays decoupled from `backtest.data.fec` (dynamic/
    depends on data/, never the reverse, per the rest of this package's
    convention); the caller (e.g. `dynamic/simulate.py`'s historical
    harness) is responsible for building this dict via
    `fec.cash_on_hand_as_of()` and passing it in. Districts absent from the
    dict, or when the dict itself is None, leave `cash_on_hand_d` as None —
    the pre-existing, still-valid "no data for this district/period" state.
    """
    outputs = compute_outputs_batch(races, coef, sigma_model)
    race_states: dict[str, RaceState] = {}
    for out, race in zip(outputs, races):
        race_states[race.district_id] = RaceState(
            base=race,
            period=period,
            period_date=period_date,
            mu_hat=out.mu_hat,
            sigma_hat=out.sigma_i,
            mu_raw=out.mu_hat,
            sigma_raw=out.sigma_i,
            d_total_t=race.d_total,
            r_total_t=race.r_total,
            cand_d_total_t=race.cand_d_total,
            cash_on_hand_d=(
                cash_on_hand_by_district.get(race.district_id)
                if cash_on_hand_by_district is not None else None
            ),
        )
    return CampaignState(
        period=period,
        period_date=period_date,
        races=race_states,
        generic_ballot_national=generic_ballot_national,
    )


class StateUpdater(Protocol):
    """f(X_t, information_t) -> X_{t+1} (paper §3.1)."""

    def update(self, prev: CampaignState | None, raw_snapshot: CampaignState) -> CampaignState:
        """`raw_snapshot` carries this period's freshly re-estimated
        mu_raw/sigma_raw (see compute_raw_snapshot); `prev` carries the
        previous period's already-smoothed state, or None for the first
        period. Returns a new CampaignState with mu_hat/sigma_hat updated
        per the concrete rule."""
        ...


class EMAStateUpdater:
    """f_baseline (paper §3.3): exponential moving average.

        mu_hat_t    = lambda * mu_hat_{t-1}    + (1 - lambda) * mu_raw_t
        sigma_hat_t = lambda * sigma_hat_{t-1} + (1 - lambda) * sigma_raw_t

    `lam` (lambda) in (0, 1); higher = more inertia, less reactive to a
    single period's noise. Default 0.7 (paper §3.3's starting value),
    overridable via `config.yaml`'s `dynamic.ema_lambda`. The first period
    (prev=None) has no prior to smooth against: mu_hat_0 = mu_raw_0.
    """

    def __init__(self, lam: float = 0.7):
        if not (0.0 < lam < 1.0):
            raise ValueError(f"EMA lambda must be in (0, 1), got {lam}")
        self.lam = lam

    def update(self, prev: CampaignState | None, raw_snapshot: CampaignState) -> CampaignState:
        new_races: dict[str, RaceState] = {}
        for district_id, raw_rs in raw_snapshot.races.items():
            prev_rs = prev.races.get(district_id) if prev is not None else None
            if prev_rs is not None:
                mu_hat = self.lam * prev_rs.mu_hat + (1 - self.lam) * raw_rs.mu_raw
                sigma_hat = self.lam * prev_rs.sigma_hat + (1 - self.lam) * raw_rs.sigma_raw
            else:
                mu_hat = raw_rs.mu_raw
                sigma_hat = raw_rs.sigma_raw
            new_races[district_id] = dataclasses.replace(raw_rs, mu_hat=mu_hat, sigma_hat=sigma_hat)
        return dataclasses.replace(raw_snapshot, races=new_races)


# Explicitly deferred (paper §8's stated limitation) — not implemented here:
#   class KalmanStateUpdater(StateUpdater): ...
#   class ParticleFilterStateUpdater(StateUpdater): ...
