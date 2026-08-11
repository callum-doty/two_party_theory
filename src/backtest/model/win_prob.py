"""
Win probability and marginal seat gain computation.

P_win_i = Φ(μᵢ / σᵢ)

where Φ is the standard normal CDF, μᵢ is the fitted expected margin, and
σᵢ is the estimated margin uncertainty from the heteroskedastic model.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from ..types import RaceRecord, ModelOutputs, SigmaModel
from ..model.margin import MarginModelCoefficients, predict, predict_floor_margin
from ..model import ceiling
from .. import config


def compute_outputs(
    race: RaceRecord,
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    beta1_override: float | None = None,
) -> ModelOutputs:
    """
    Compute all model quantities for a single race.

    Parameters
    ----------
    beta1_override : used during β_RC uncertainty propagation draws
    """
    d_total = race.d_total
    r_total = race.r_total
    total = d_total + r_total

    if total <= 0:
        ratio = 0.5
    else:
        ratio = d_total / total

    ratio = np.clip(ratio, 1e-6, 1 - 1e-6)

    mu_raw = predict(
        pvi=race.pvi,
        incumb_status=race.incumb_status,
        generic_ballot=race.generic_ballot,
        ratio=ratio,
        coef=coef,
        beta1_override=beta1_override,
        total_spend=total,
        cvap=race.cvap,
        indiv_share=race.indiv_share,
    )

    sigma = sigma_model.predict(abs(race.pvi), race.incumb_status, race.generic_ballot)

    # Persuasion ceiling (model/ceiling.py, FINDINGS.md): cap the achievable
    # shift from the race's own candidate-only floor (no party money) at
    # C_i = c_max * persuadability(mu_floor, sigma) — prevents the log-ratio
    # gradient's 1/D blowup at near-zero floors from being read as a real,
    # unbounded spending effect.
    mu_floor = predict_floor_margin(
        pvi=race.pvi,
        incumb_status=race.incumb_status,
        generic_ballot=race.generic_ballot,
        cand_d_total=race.cand_d_total,
        r_total=r_total,
        coef=coef,
        beta1_override=beta1_override,
        cvap=race.cvap,
        indiv_share=race.indiv_share,
    )
    c_max = config.persuasion_ceiling_c_max()
    mu, grad_factor = ceiling.apply(mu_raw, mu_floor, sigma, c_max)

    p_win = float(norm.cdf(mu / sigma))

    msg = _marginal_seat_gain(
        mu=mu,
        sigma=sigma,
        pvi=race.pvi,
        incumb_status=race.incumb_status,
        d_total=d_total,
        r_total=r_total,
        coef=coef,
        beta1_override=beta1_override,
    ) * grad_factor

    return ModelOutputs(
        district_id=race.district_id,
        ratio=ratio,
        mu_hat=mu,
        sigma_i=sigma,
        p_win=p_win,
        msg_i=msg,
    )


def _marginal_seat_gain(
    mu: float,
    sigma: float,
    pvi: float,
    incumb_status: str,
    d_total: float,
    r_total: float,
    coef: MarginModelCoefficients,
    beta1_override: float | None = None,
) -> float:
    """
    MSG_i = φ(μᵢ/σᵢ) · (1/σᵢ) · (∂μᵢ/∂Dᵢ)

    ∂μᵢ/∂Dᵢ = [β₁ + β₂·|PVI_i| + β₃·incumb_i] · Rᵢ/(Dᵢ·(Dᵢ+Rᵢ))
              + α₄ · 1/(Dᵢ+Rᵢ)

    The α₄ term captures how adding a dollar increases total spending
    intensity log((D+R)/CVAP). Its contribution to MSG is small relative
    to the log-ratio term but ensures the gradient is exact.

    D and R must be passed separately (not just D+R): the log-ratio
    gradient is R/(D·(D+R)), not 1/(D+R) — these coincide only at D=R
    (spending parity) and diverge for any lopsided spending ratio.

    b1 mirrors predict()'s open-seat substitution: β₁ is replaced by
    coef.beta1_open for Open races when set, so MSG differentiates the
    same effective β that predict() used to produce mu. beta1_open takes
    priority over beta1_override for the same reason as predict(): the
    override carries beta_RC bootstrap-draw uncertainty, a different
    quantity from the separately-calibrated beta1_open.

    Returns MSG per dollar. Multiply by 1e6 externally for per-$1M.
    """
    total_spend = d_total + r_total
    if total_spend <= 0 or d_total <= 0:
        return 0.0

    if coef.beta1_open is not None and incumb_status == "Open":
        b1 = coef.beta1_open
    elif beta1_override is not None:
        b1 = beta1_override
    else:
        b1 = coef.beta1
    is_incumb = 1.0 if incumb_status == "Incumbent" else 0.0
    abs_pvi = abs(pvi)

    d_log_ratio_dd = r_total / (d_total * total_spend)
    d_mu_d_s = (b1 + coef.beta2 * abs_pvi + coef.beta3 * is_incumb) * d_log_ratio_dd
    # α₄ gradient: ∂log((D+R)/CVAP)/∂D = 1/(D+R)
    d_mu_d_s += coef.alpha4 / total_spend

    phi = float(norm.pdf(mu / sigma))
    return phi * (1.0 / sigma) * d_mu_d_s


def compute_floor_msg(
    race: RaceRecord,
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    beta1_override: float | None = None,
) -> float:
    """
    Marginal seat gain evaluated at the race's own candidate-only spending
    floor (D_i = cand_d_total, no party money added), rather than at the
    observed/current total spending level compute_outputs() uses.

    Rationale (efficiency-test redesign, see FINDINGS.md's discussion of the
    Spearman test's diminishing-returns confound): correlating observed
    party spending against MSG evaluated *at that same observed spending
    level* is mechanically biased toward a negative reading, because
    diminishing returns alone guarantees that races which already received
    more money show lower current-MSG, independent of whether the money was
    well targeted. MSG evaluated at the floor is fixed before any party
    dollar is committed and therefore cannot be contaminated by the spending
    decision under test.

    The persuasion ceiling's grad_factor is exactly 1.0 at the floor itself
    (delta = max(mu_raw - mu_floor, 0) = 0 when mu_raw = mu_floor), so no
    ceiling correction is needed here -- this is the ceiling's own anchor
    point.
    """
    mu_floor = predict_floor_margin(
        pvi=race.pvi,
        incumb_status=race.incumb_status,
        generic_ballot=race.generic_ballot,
        cand_d_total=race.cand_d_total,
        r_total=race.r_total,
        coef=coef,
        beta1_override=beta1_override,
        cvap=race.cvap,
        indiv_share=race.indiv_share,
    )
    sigma = sigma_model.predict(abs(race.pvi), race.incumb_status, race.generic_ballot)

    floor_dollars = max(race.cand_d_total, 1.0)
    r_dollars = max(race.r_total, 1.0)

    return _marginal_seat_gain(
        mu=mu_floor,
        sigma=sigma,
        pvi=race.pvi,
        incumb_status=race.incumb_status,
        d_total=floor_dollars,
        r_total=r_dollars,
        coef=coef,
        beta1_override=beta1_override,
    )


def compute_outputs_batch(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    beta1_override: float | None = None,
) -> list[ModelOutputs]:
    """Apply compute_outputs to every race in the list."""
    return [compute_outputs(r, coef, sigma_model, beta1_override) for r in races]
