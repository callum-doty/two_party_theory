"""
Nash equilibrium for adversarial NRCC response (FINDINGS.md Section 10.5's
"left as future work", Paper III's reduced-form eta scalar).

Replaces allocator.py's _reactive_r() -- a mechanical, one-sided reaction
rule (R_i = R_base + eta*max(0, D_i - D_obs)) -- with NRCC modeled as its own
constrained optimizer, and the joint fixed point of both committees'
simultaneous best-response problems solved via best-response dynamics
(Gauss-Seidel fictitious play): each round, D re-solves its full optimization
against R's most-recently-updated allocation, then R re-solves against D's
just-updated allocation. Neither side is privileged as a leader -- this is
the standard computational method for locating a Nash equilibrium of a
smooth, simultaneously-played game (as opposed to a Stackelberg/leader-
follower formulation, which this module deliberately does not implement).

Both sides share the SAME underlying probability model (the margin model +
persuasion ceiling already used everywhere else in this codebase) -- the
standard common-knowledge-of-the-model assumption implicit in any Nash
equilibrium: both players agree on what a given (D, R) allocation predicts
for each race's win probability, they just have opposed objectives over it.
D's payoff is E[D seats] = sum(Phi(mu/sigma)); R's payoff is
E[R seats] = n_races - E[D seats] = sum(1 - Phi(mu/sigma)) -- a zero-sum game
in seats.

Asymmetry, flagged explicitly rather than silently applied or silently
omitted: D's best response reuses optimizer/allocator.py's
optimize_nonlinear() unmodified, including its persuasion ceiling
(model/ceiling.py), which was empirically calibrated against DCCC's own
historical spending-response data (FINDINGS.md's persuasion-ceiling
section). R's best response uses a MIRRORED ceiling construction below
(_r_precompute/_r_mu_and_grad) -- same functional form, anchored at R's own
floor instead of D's, reusing the same c_max config value -- but this mirror
is NOT independently calibrated against NRCC-specific spending-response
data, since no such calibration exists in this project. It is included
because omitting any ceiling on R's side would let R's optimizer extrapolate
mu unboundedly downward as r -> 0, exactly the failure mode the D-side
ceiling exists to prevent (model/ceiling.py's own docstring); the specific
magnitude (c_max) is a stated assumption, not a validated NRCC-specific
figure, and should be read as such alongside any equilibrium found here.

Existence and uniqueness are NOT guaranteed for this non-convex game.
find_nash_equilibrium_multi_start() runs best-response dynamics from several
starting points and reports whether they agree, rather than silently
returning one candidate fixed point as "the" answer.

Scoring fix (2026-08-10), found via a logically-impossible negative regret
in a downstream analysis: best_response("R", ...)'s returned e_seats_own
used to be self-scored via the mirrored ceiling above -- but that mirrored
formula and the D-side formula (used for D's own e_seats_own AND for THIS
module's own final e_seats_d/e_seats_r in solve_best_response_dynamics)
diverge sharply on identical (D, R) data whenever both ceilings bind
simultaneously (~1/3 of races), even though the underlying uncapped mu_raw
is provably identical between them. best_response("R", ...) now re-scores
its result via the SAME D-side formula used everywhere else in this module
and the wider codebase, after SLSQP converges -- the mirrored formula
remains the SLSQP search's own internal objective/gradient (kept
deliberately: substituting D's one-sided ceiling formula there instead
would give R's optimizer a zero-gradient dead region throughout the exact
area it operates in, a worse failure mode than a scoring mismatch). This
does NOT change solve_best_response_dynamics()'s actual dollar allocations
or convergence trajectory -- its Gauss-Seidel stopping rule is purely
dollar-delta-based and its own final e_seats_d/e_seats_r already used the
D-side formula before this fix -- but it does mean every PER-ROUND
diagnostic in `history` is now trustworthy, not just the final round. See
best_response()'s own inline comment for the full derivation and the
concrete before/after numbers that surfaced this.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .. import config
from ..model import ceiling as ceiling_mod
from .allocator import _p_win_vec, _precompute_race_arrays, optimize_nonlinear

SCALE = 1_000_000.0  # matches allocator.py's SLSQP conditioning trick


@dataclass
class SideResult:
    """One side's best response for one round: its own party-money
    allocation and its own objective value."""
    party: np.ndarray            # (n,) this side's own party-money allocation ($)
    e_seats_own: float           # this side's own expected seats
    status: str


@dataclass
class NashResult:
    party_d: np.ndarray
    party_r: np.ndarray
    e_seats_d: float
    e_seats_r: float              # sanity check: e_seats_d + e_seats_r == n_races
    converged: bool
    n_iterations: int
    history: list[dict]           # per-round {round, max_delta_d, max_delta_r, e_seats_d, e_seats_r}
    multi_start_agreement: dict | None = field(default=None)


def _r_precompute(races, coef, sigma_model, cand_r_total: np.ndarray, d_current: np.ndarray) -> dict:
    """Mirror of allocator._precompute_race_arrays(), anchored at R's own
    floor (cand_r_total) instead of D's cand_d_total, with D held fixed at
    d_current this round. mu_const/c_spend/sigma/cvap/alpha4 are identical
    formulas to the D-side computation (race characteristics don't depend on
    which side is varying); mu_floor_r/C_r are R's own mirrored ceiling
    anchor -- see module docstring's asymmetry note."""
    n = len(races)
    pvi = np.array([r.pvi for r in races])
    abs_pvi = np.abs(pvi)
    incumb = np.array([1.0 if r.incumb_status == "Incumbent" else 0.0 for r in races])
    gb = np.array([r.generic_ballot for r in races])
    indiv_share = np.array([r.indiv_share for r in races])

    mu_const = (coef.alpha0 + coef.alpha1 * pvi + coef.alpha2 * incumb
                + coef.alpha3 * gb + coef.alpha5 * indiv_share)
    c_spend = coef.beta1 + coef.beta2 * abs_pvi + coef.beta3 * incumb
    if coef.beta1_open is not None:
        is_open = np.array([1.0 if r.incumb_status == "Open" else 0.0 for r in races])
        c_spend = np.where(is_open, coef.beta1_open + coef.beta2 * abs_pvi, c_spend)

    sigma = np.array([
        sigma_model.predict(abs_pvi[i], races[i].incumb_status, races[i].generic_ballot)
        for i in range(n)
    ])
    cvap = np.array([max(r.cvap, 1) for r in races])
    alpha4 = coef.alpha4

    d_fixed = np.maximum(np.asarray(d_current, dtype=float), 1.0)
    r_floor = np.maximum(np.asarray(cand_r_total, dtype=float), 1.0)
    t_floor = d_fixed + r_floor
    ratio_floor = np.clip(d_fixed / t_floor, 1e-15, 1 - 1e-15)
    mu_floor_r = mu_const + c_spend * np.log(ratio_floor) + alpha4 * np.log(t_floor / cvap)

    c_max = config.persuasion_ceiling_c_max()
    C_r = ceiling_mod.ceiling(mu_floor_r, sigma, c_max, maturity_factor=1.0)

    return dict(mu_const=mu_const, c_spend=c_spend, sigma=sigma, cvap=cvap, alpha4=alpha4,
                d_fixed=d_fixed, r_floor=r_floor, mu_floor_r=mu_floor_r, C_r=C_r)


def _r_mu_and_grad(party_r: np.ndarray, arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """mu_capped_r(party_r) and d(mu_capped_r)/d(party_r), mirroring
    allocator._apply_ceiling/_msg_vec but anchored at R's own floor.

    d(mu_raw)/d(r) = (alpha4 - c_spend)/t (module docstring's derivation) --
    at the live model's current alpha4=0.0, this is -c_spend/t, i.e. R
    spending strictly helps R (mu_raw strictly decreases), matching the
    intended sign. delta_r = max(mu_floor_r - mu_raw, 0) mirrors D-side's
    max(mu_raw - mu_floor, 0): D's ceiling caps mu from rising too far above
    its floor as D spends; R's mirror caps mu from falling too far below its
    own floor as R spends."""
    r = arrays["r_floor"] + party_r
    d = arrays["d_fixed"]
    t = d + r
    ratio = np.clip(d / t, 1e-15, 1 - 1e-15)
    mu_raw = arrays["mu_const"] + arrays["c_spend"] * np.log(ratio) + arrays["alpha4"] * np.log(t / arrays["cvap"])

    mu_floor_r, C_r = arrays["mu_floor_r"], arrays["C_r"]
    delta = np.maximum(mu_floor_r - mu_raw, 0.0)
    decay = np.exp(-delta / C_r)
    mu_capped = mu_floor_r - C_r * (1.0 - decay)

    d_mu_raw_d_partyr = (arrays["alpha4"] - arrays["c_spend"]) / t
    d_mu_capped_d_partyr = decay * d_mu_raw_d_partyr
    return mu_capped, d_mu_capped_d_partyr


def best_response(
    side: str,
    races: list,
    coef,
    sigma_model,
    *,
    opp_total_fixed: np.ndarray,
    own_party_budget: float,
    own_cap_fraction: float,
    cand_r_total: np.ndarray | None = None,
    x0: np.ndarray | None = None,
) -> SideResult:
    """One side's constrained best response, holding the OTHER side's total
    spend fixed at opp_total_fixed ($ per race).

    side="D": maximizes sum(Phi(mu/sigma)). Reuses optimizer/allocator.py's
    optimize_nonlinear() UNMODIFIED (eta=0.0 -- the mechanical, one-sided
    reaction rule this module replaces is irrelevant to a genuine
    simultaneous game) with races' r_total set to opp_total_fixed. This is
    already-validated D-side machinery; the equivalence to calling
    optimize_nonlinear() directly is checked in tests/test_nash.py.

    side="R": minimizes sum(Phi(mu/sigma)) (equivalently maximizes R's own
    expected seats, n_races - that sum), via the mirrored ceiling
    construction in _r_precompute/_r_mu_and_grad. cand_r_total (R's own
    candidate-committee floor, $ per race) is required for this side --
    NOT a RaceRecord field (RaceRecord only carries r_total, D's view of the
    opponent's total spend, not R's own floor/party split), so it is passed
    explicitly rather than added to the shared RaceRecord type, keeping this
    module's blast radius on the rest of the pipeline at zero."""
    n = len(races)
    if side == "D":
        races_t = [dataclasses.replace(r, r_total=float(opp_total_fixed[i])) for i, r in enumerate(races)]
        res = optimize_nonlinear(
            races_t, coef, sigma_model, budget=own_party_budget,
            cov_matrix=np.eye(n) * 1e-6, gamma=0.0, cap_fraction=own_cap_fraction,
            party_budget=own_party_budget, eta=0.0,
        )
        floors = np.array([r.cand_d_total for r in races])
        party = np.maximum(res.allocations - floors, 0.0)
        return SideResult(party=party, e_seats_own=res.expected_seats, status=res.status)

    if side == "R":
        if cand_r_total is None:
            raise ValueError("cand_r_total is required for side='R'")
        arrays = _r_precompute(races, coef, sigma_model, cand_r_total, d_current=opp_total_fixed)
        cap = own_cap_fraction * own_party_budget

        x0_dollars = x0 if x0 is not None else np.zeros(n)
        x0_s = np.clip(x0_dollars, 0.0, cap) / SCALE
        cap_s = cap / SCALE
        pb_s = own_party_budget / SCALE
        neg_ones = -np.ones(n)

        def neg_r_seats(xs: np.ndarray) -> float:
            mu_capped, _ = _r_mu_and_grad(xs * SCALE, arrays)
            p_r_win = 1.0 - norm.cdf(mu_capped / arrays["sigma"])
            return -float(p_r_win.sum())

        def neg_r_seats_grad(xs: np.ndarray) -> np.ndarray:
            mu_capped, d_mu_d_partyr = _r_mu_and_grad(xs * SCALE, arrays)
            phi = norm.pdf(mu_capped / arrays["sigma"])
            d_pdwin_d_partyr = (phi / arrays["sigma"]) * d_mu_d_partyr
            d_prwin_d_partyr = -d_pdwin_d_partyr
            return -(d_prwin_d_partyr * SCALE)

        def budget_slack(xs: np.ndarray) -> float:
            return float(pb_s - xs.sum())

        result = minimize(
            neg_r_seats, x0=x0_s, method="SLSQP", jac=neg_r_seats_grad,
            bounds=[(0.0, cap_s)] * n,
            constraints=[{"type": "ineq", "fun": budget_slack, "jac": lambda xs: neg_ones.copy()}],
            options={"maxiter": 3000, "ftol": 1e-12},
        )
        party = np.maximum(result.x * SCALE, 0.0)
        status = "optimal" if result.success else f"slsqp:{result.message}"

        # e_seats_own is scored via the D-SIDE formula (same one used
        # everywhere else -- _precompute_race_arrays/_p_win_vec, and the
        # SAME formula solve_best_response_dynamics's own final e_seats_d/
        # e_seats_r report), NOT via the mirrored-ceiling formula that
        # guided the SLSQP search above. The two disagree sharply: mu_raw is
        # identical between them (confirmed directly), but the two ceiling
        # anchors -- D's own floor (calibrated against real DCCC data) vs.
        # R's mirrored floor (not independently calibrated, see this
        # function's own docstring) -- clip differently whenever BOTH
        # ceilings bind, which happens for roughly a third of races. Found
        # 2026-08-10 via a logically-impossible NEGATIVE regret (a best
        # response scored worse than the baseline it was optimizing
        # against) -- purely a scoring-formula mismatch, not evidence the
        # search itself found a bad allocation: re-scored consistently, the
        # SAME allocation the mirrored-formula search finds already
        # achieves substantial, positive, real improvement (2024: 4.6
        # seats, self-reported mirrored score said only 0.26; 2022: 3.3
        # seats, self-reported score was an impossible -0.51).
        #
        # The mirrored formula is KEPT as the SLSQP objective/gradient
        # above (not replaced) because naively substituting D's ceiling
        # there instead would go flat (zero gradient) throughout the exact
        # region R's optimizer operates in -- D's ceiling only clips
        # mu_raw > mu_floor (D overspending its own floor); R's spending
        # pushes mu_raw BELOW D's floor, where D's one-sided formula
        # returns a constant (mu_floor) with zero gradient, which would be
        # a worse failure mode (no search signal at all) than a scoring
        # mismatch. The mirrored ceiling remains a reasonable, well-behaved
        # SEARCH landscape; only the REPORTED score is corrected here.
        #
        # This does NOT change solve_best_response_dynamics()'s actual
        # dollar allocations or its convergence trajectory: that function's
        # Gauss-Seidel stopping criterion is purely dollar-delta-based
        # (max_delta_d/max_delta_r), never touches e_seats_own, and its own
        # FINAL reported e_seats_d/e_seats_r already used this same D-side
        # formula before this fix. What changes is that solve_best_response
        # _dynamics()'s PER-ROUND history["e_seats_r"] (which reads
        # res_r.e_seats_own directly) is now trustworthy at every
        # intermediate round too, not just at the final one.
        d_total_fixed = np.asarray(opp_total_fixed, dtype=float)
        r_total_at_party = cand_r_total + party
        races_scored = [
            dataclasses.replace(r, d_total=float(d_total_fixed[i]), r_total=float(r_total_at_party[i]))
            for i, r in enumerate(races)
        ]
        scoring_arrays = _precompute_race_arrays(races_scored, coef, sigma_model, eta=0.0)
        floors_d_scored = np.array([r.cand_d_total for r in races_scored])
        party_d_fixed = np.maximum(d_total_fixed - floors_d_scored, 0.0)
        p_d_at_r_star = _p_win_vec(party_d_fixed, scoring_arrays)
        e_seats_r = float(n - p_d_at_r_star.sum())

        return SideResult(party=party, e_seats_own=e_seats_r, status=status)

    raise ValueError(f"side must be 'D' or 'R', got {side!r}")


def solve_best_response_dynamics(
    races: list,
    coef,
    sigma_model,
    cand_r_total: np.ndarray,
    party_budget_d: float,
    party_budget_r: float,
    cap_fraction_d: float = 0.15,
    cap_fraction_r: float = 0.15,
    init_party_d: np.ndarray | None = None,
    init_party_r: np.ndarray | None = None,
    damping_theta: float = 1.0,
    max_rounds: int = 100,
    tol_dollars: float = 10_000.0,
    track_races: bool = False,
) -> NashResult:
    """Gauss-Seidel best-response iteration (fictitious play) to a fixed
    point: each round, D re-solves against R's most-recent allocation, then
    R re-solves against D's just-updated allocation -- genuinely
    simultaneous, neither side privileged as a leader.

    damping_theta < 1.0 averages each round's best response with the
    previous round's allocation (party_new = theta*br + (1-theta)*party_old)
    -- the standard fictitious-play stabilization for games whose undamped
    best-response dynamics cycle rather than converge.

    track_races : if True, each history entry also carries this round's full
        per-race party_d/party_r allocation vectors (as lists, for JSON
        round-tripping), so callers can draw individual races' D_0 -> R_1 ->
        D_2 -> ... convergence paths (scripts/game_theory's trajectory work).
        Default False keeps history's shape exactly as it was for existing
        callers (solve_nash_equilibrium.py, tests/test_nash.py) -- this is
        additive, not a behavior change to any existing caller."""
    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    cand_r_total = np.asarray(cand_r_total, dtype=float)

    party_d = init_party_d.copy() if init_party_d is not None else np.maximum(d0 - floors_d, 0.0)
    party_r = init_party_r.copy() if init_party_r is not None else np.maximum(r0 - cand_r_total, 0.0)

    history: list[dict] = []
    converged = False
    it = 0
    for it in range(max_rounds):
        r_total_current = cand_r_total + party_r
        res_d = best_response(
            "D", races, coef, sigma_model, opp_total_fixed=r_total_current,
            own_party_budget=party_budget_d, own_cap_fraction=cap_fraction_d, x0=party_d,
        )
        party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
        d_total_after_d = floors_d + party_d_new

        res_r = best_response(
            "R", races, coef, sigma_model, opp_total_fixed=d_total_after_d,
            own_party_budget=party_budget_r, own_cap_fraction=cap_fraction_r,
            cand_r_total=cand_r_total, x0=party_r,
        )
        party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r

        max_delta_d = float(np.max(np.abs(party_d_new - party_d))) if n else 0.0
        max_delta_r = float(np.max(np.abs(party_r_new - party_r))) if n else 0.0
        entry = {
            "round": it, "max_delta_d": max_delta_d, "max_delta_r": max_delta_r,
            "e_seats_d": res_d.e_seats_own, "e_seats_r": res_r.e_seats_own,
        }
        if track_races:
            entry["party_d"] = party_d_new.tolist()
            entry["party_r"] = party_r_new.tolist()
        history.append(entry)

        party_d, party_r = party_d_new, party_r_new
        if max_delta_d < tol_dollars and max_delta_r < tol_dollars:
            converged = True
            break

    r_final_total = cand_r_total + party_r
    d_final_total = floors_d + party_d
    races_final = [dataclasses.replace(r, r_total=float(r_final_total[i])) for i, r in enumerate(races)]
    final_arrays = _precompute_race_arrays(races_final, coef, sigma_model, eta=0.0)
    p_d = _p_win_vec(party_d, final_arrays)
    e_seats_d = float(p_d.sum())
    e_seats_r = float(n - e_seats_d)

    return NashResult(
        party_d=party_d, party_r=party_r, e_seats_d=e_seats_d, e_seats_r=e_seats_r,
        converged=converged, n_iterations=it + 1, history=history,
    )


def find_nash_equilibrium_multi_start(
    races: list,
    coef,
    sigma_model,
    cand_r_total: np.ndarray,
    party_budget_d: float,
    party_budget_r: float,
    cap_fraction_d: float = 0.15,
    cap_fraction_r: float = 0.15,
    damping_theta: float = 1.0,
    max_rounds: int = 100,
    tol_dollars: float = 10_000.0,
) -> NashResult:
    """Runs solve_best_response_dynamics from 3 starting points (observed
    baseline, uniform-across-races, zero-party-spend) and reports whether
    they converge to the same fixed point, within tol_dollars*10 per race.
    Existence/uniqueness are NOT guaranteed for this non-convex game -- if
    the starts disagree, that is reported in the returned NashResult's
    multi_start_agreement field, not silently averaged away or discarded."""
    n = len(races)
    starts = {
        "observed": None,
        "uniform": (np.full(n, party_budget_d / max(n, 1)), np.full(n, party_budget_r / max(n, 1))),
        "zero": (np.zeros(n), np.zeros(n)),
    }
    per_start: dict[str, NashResult] = {}
    for name, init in starts.items():
        init_party_d, init_party_r = (None, None) if init is None else init
        per_start[name] = solve_best_response_dynamics(
            races, coef, sigma_model, cand_r_total, party_budget_d, party_budget_r,
            cap_fraction_d, cap_fraction_r, init_party_d=init_party_d, init_party_r=init_party_r,
            damping_theta=damping_theta, max_rounds=max_rounds, tol_dollars=tol_dollars,
        )

    names = list(starts.keys())
    max_pairwise_d = max(
        (float(np.max(np.abs(per_start[a].party_d - per_start[b].party_d)))
         for i, a in enumerate(names) for b in names[i + 1:]),
        default=0.0,
    )
    max_pairwise_r = max(
        (float(np.max(np.abs(per_start[a].party_r - per_start[b].party_r)))
         for i, a in enumerate(names) for b in names[i + 1:]),
        default=0.0,
    )
    converged_all = all(r.converged for r in per_start.values())
    agree = converged_all and max_pairwise_d < tol_dollars * 10 and max_pairwise_r < tol_dollars * 10

    base = per_start["observed"]
    base.multi_start_agreement = {
        "converged_all": converged_all,
        "agree_within_tolerance": agree,
        "max_pairwise_party_d_diff": max_pairwise_d,
        "max_pairwise_party_r_diff": max_pairwise_r,
        "per_start_converged": {k: v.converged for k, v in per_start.items()},
        "per_start_e_seats_d": {k: v.e_seats_d for k, v in per_start.items()},
        "per_start_n_iterations": {k: v.n_iterations for k, v in per_start.items()},
    }
    return base
