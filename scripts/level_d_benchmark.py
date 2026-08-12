#!/usr/bin/env python3
"""
Level D historical-behavior benchmark (project_spec.md Section 20): "Are
observed allocations closer to the estimated Nash equilibrium than to
reasonable alternative strategies?" Compares the OBSERVED allocation
against five benchmark strategy pairs -- equal allocation; Cook-category
heuristic; one-sided optimizer; mixed equilibrium (this project's
replacement for a nonexistent pure Nash point, see docs/methodology.md's
"Double-oracle mixed equilibrium" section); random feasible portfolios --
on both E[D seats] and L1 distance from the observed (D, R) profile. This
is the test H3 ("observed allocations are substantially closer to Nash
equilibrium than to unilateral optima") actually needs and, per
docs/results_2022_2024.md, was flagged as not yet built.

Each strategy is defined for BOTH sides (not just D) so that "E[D seats]"
means something well-defined: D and R each play THEIR OWN version of that
strategy against each other, not against the observed opponent. This
mirrors how every other pairwise comparison in this project works (BR_D
vs. BR_R, double-oracle's D-pool vs. R-pool, etc.) rather than mixing one
side's benchmark strategy against the other's observed allocation.

  - Equal: uniform allocation across all races, both sides.
  - Cook heuristic: proportional to Cook-category competitiveness weight
    (Toss-Up highest, Safe zero), same weights for both sides
    (game/benchmarks.py::cook_heuristic_allocation).
  - One-sided optimizer: BR_D(R_observed) vs. BR_R(D_observed) -- each
    side's one-shot unilateral best response, played against EACH OTHER
    (not against the observed opponent that produced them) -- distinct
    from this project's usual "regret" framing, which holds the opponent
    fixed at observed.
  - Mixed equilibrium: the double-oracle solve's LP value is the
    game-theoretically correct E[D seats] under the mixture (NOT
    recomputed from the mixture's expected/average portfolio, since
    p_win_shared is nonlinear in the opponent's spending -- see
    game/double_oracle.py's module docstring); L1 distance is measured
    against the mixture's expected portfolio (sum_j p_j * portfolio_j) as
    a descriptive summary of "the mixture's typical allocation," reported
    separately and not conflated with the value.
  - Random feasible: mean over K=20 independent Dirichlet-weighted random
    portfolios per side (game/benchmarks.py::random_feasible_allocation).

Requires results/double_oracle_{cycle}.json and its saved portfolio .npy
files (scripts/double_oracle.py) for the mixed-equilibrium row -- for 2022
specifically, results/double_oracle_2022_resumed.json's larger pool/support
(the converged extended run) is used if present, since the original
results/double_oracle_2022.json run did not converge.

Usage:
    python scripts/level_d_benchmark.py --cycle 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from build_cycle_state import build_cycle_state  # noqa: E402
from game import benchmarks  # noqa: E402
from game import best_response as br  # noqa: E402
from game import double_oracle as do  # noqa: E402
from game import payoff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("level_d_benchmark")

N_RANDOM_DRAWS = 20


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _load_double_oracle(results_dir: Path, cycle: int) -> dict | None:
    """Thin wrapper over game/double_oracle.py::load_solved -- adds the
    mixture's EXPECTED portfolio (sum_j p_j * portfolio_j), which is only
    a descriptive L1-distance summary here (see this script's module
    docstring), not something other callers of load_solved necessarily want."""
    solved = do.load_solved(results_dir, cycle)
    if solved is None:
        return None
    expected_d = sum(w * dp for w, dp in zip(solved["p"], solved["d_pool"]))
    expected_r = sum(w * rp for w, rp in zip(solved["q"], solved["r_pool"]))
    return {
        "value_e_seats_d": solved["value_e_seats_d"],
        "expected_party_d": expected_d, "expected_party_r": expected_r,
        "converged": solved["converged"], "source_file": solved["source_file"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Level D five-way historical-behavior benchmark")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--n-random-draws", type=int, default=N_RANDOM_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total = state["cand_r_total"]
    budget_d, budget_r = state["budget_d"], state["budget_r"]
    n = len(races)
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    floors_d = np.array([r.cand_d_total for r in races])
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    e_seats_d_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())

    rows = {}

    # 1. Equal allocation
    eq_d = benchmarks.equal_allocation(n, budget_d)
    eq_r = benchmarks.equal_allocation(n, budget_r)
    rows["equal"] = {
        "e_seats_d": float(payoff.p_win_shared(eq_d, eq_r, arrays).sum()),
        "l1_d": _l1(eq_d, party_d_obs), "l1_r": _l1(eq_r, party_r_obs),
    }

    # 2. Cook-category heuristic
    cook_d = benchmarks.cook_heuristic_allocation(races, budget_d, args.cap_fraction_d)
    cook_r = benchmarks.cook_heuristic_allocation(races, budget_r, args.cap_fraction_r)
    rows["cook_heuristic"] = {
        "e_seats_d": float(payoff.p_win_shared(cook_d, cook_r, arrays).sum()),
        "l1_d": _l1(cook_d, party_d_obs), "l1_r": _l1(cook_r, party_r_obs),
    }

    # 3. One-sided optimizer: BR_D(R_obs) vs. BR_R(D_obs), played against EACH OTHER
    logger.info("Solving one-shot unilateral best responses (exact SLSQP)…")
    res_d = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                     budget_d=budget_d, cap_fraction_d=args.cap_fraction_d)
    res_r = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                     budget_r=budget_r, cap_fraction_r=args.cap_fraction_r)
    rows["one_sided_optimizer"] = {
        "e_seats_d": float(payoff.p_win_shared(res_d.party, res_r.party, arrays).sum()),
        "l1_d": _l1(res_d.party, party_d_obs), "l1_r": _l1(res_r.party, party_r_obs),
    }

    # 4. Mixed equilibrium (double oracle)
    do = _load_double_oracle(REPO_ROOT / "results", args.cycle)
    if do is not None:
        logger.info(f"Loaded double-oracle mixed equilibrium from {do['source_file']} "
                     f"(converged={do['converged']})")
        rows["mixed_equilibrium"] = {
            "e_seats_d": do["value_e_seats_d"],  # LP value -- the correct value under the mixture
            "l1_d": _l1(do["expected_party_d"], party_d_obs),
            "l1_r": _l1(do["expected_party_r"], party_r_obs),
            "note": "e_seats_d is the double-oracle LP value; l1 is distance of the mixture's "
                    "EXPECTED portfolio (sum_j p_j * portfolio_j), a descriptive summary only "
                    "-- p_win_shared is nonlinear in the opponent, so this L1 point is not "
                    "itself what the mixture 'plays.'",
        }
    else:
        logger.warning(f"No double-oracle results found for cycle {args.cycle} -- skipping that row. "
                        f"Run scripts/double_oracle.py --cycle {args.cycle} first.")

    # 5. Random feasible portfolios (mean over K draws)
    rng = np.random.default_rng(args.seed)
    rand_e_seats, rand_l1_d, rand_l1_r = [], [], []
    for _ in range(args.n_random_draws):
        rd = benchmarks.random_feasible_allocation(n, budget_d, args.cap_fraction_d, rng)
        rr = benchmarks.random_feasible_allocation(n, budget_r, args.cap_fraction_r, rng)
        rand_e_seats.append(float(payoff.p_win_shared(rd, rr, arrays).sum()))
        rand_l1_d.append(_l1(rd, party_d_obs))
        rand_l1_r.append(_l1(rr, party_r_obs))
    rows["random_feasible"] = {
        "e_seats_d": float(np.mean(rand_e_seats)), "e_seats_d_std": float(np.std(rand_e_seats)),
        "l1_d": float(np.mean(rand_l1_d)), "l1_r": float(np.mean(rand_l1_r)),
        "n_draws": args.n_random_draws,
    }

    # Observed itself, for reference
    rows["observed"] = {"e_seats_d": e_seats_d_obs, "l1_d": 0.0, "l1_r": 0.0}

    logger.info(f"cycle={args.cycle} Level D benchmark (E[D seats] observed = {e_seats_d_obs:.3f}):")
    for name, r in rows.items():
        l1_total = r["l1_d"] + r["l1_r"]
        logger.info(f"  {name:22s} E[D seats]={r['e_seats_d']:8.3f}  "
                    f"L1_D=${r['l1_d']/1e6:7.1f}M  L1_R=${r['l1_r']/1e6:7.1f}M  "
                    f"L1_total=${l1_total/1e6:7.1f}M")

    ranked = sorted(
        [(name, r["l1_d"] + r["l1_r"]) for name, r in rows.items() if name != "observed"],
        key=lambda kv: kv[1],
    )
    logger.info(f"Closest to observed by combined L1 distance: {ranked[0][0]} (${ranked[0][1]/1e6:.1f}M), "
                f"then {', '.join(f'{n} (${d/1e6:.1f}M)' for n, d in ranked[1:])}")

    out = {"cycle": args.cycle, "config": {"cap_fraction_d": args.cap_fraction_d,
                                            "cap_fraction_r": args.cap_fraction_r,
                                            "n_random_draws": args.n_random_draws, "seed": args.seed},
           "rows": rows, "ranked_by_l1_total": [n for n, _ in ranked]}
    out_path = REPO_ROOT / "results" / f"level_d_benchmark_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
