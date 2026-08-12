#!/usr/bin/env python3
"""
Double-oracle mixed-strategy equilibrium for one cycle (docs' "Revised
order of work" #3; game/double_oracle.py has the solver itself).

Seeds the portfolio pool with every strategically-motivated allocation
already on hand for this cycle -- not just the generic {observed, uniform,
zero} -- since a better-informed seed pool means fewer expensive oracle
rounds:
  - observed, uniform, zero (the same three used throughout this project)
  - BR_D(R_observed) / BR_R(D_observed) (one-shot unilateral best responses,
    scripts/solve_best_responses.py's objects, recomputed here)
  - the best trajectory/basin-hop point from minimize_pure_exploitability.py,
    if that cycle's results/pure_exploitability_min_{cycle}.json exists
  - the fictitious-play time-average from fictitious_play.py, if that
    cycle's results/fictitious_play_{cycle}.json exists

Usage:
    python scripts/double_oracle.py --cycle 2024
    python scripts/double_oracle.py --cycle 2024 --max-rounds 25 --eps 0.01
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
from game import best_response as br  # noqa: E402
from game import double_oracle as do  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("double_oracle")


def _load_npy_pair(results_dir: Path, prefix_d: str, prefix_r: str, cycle: int):
    fd, fr = results_dir / f"{prefix_d}_{cycle}.npy", results_dir / f"{prefix_r}_{cycle}.npy"
    if fd.exists() and fr.exists():
        return np.load(fd), np.load(fr)
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Double-oracle mixed equilibrium for one cycle")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--eps", type=float, default=0.02, help="min improvement (seats) to keep growing the pool")
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total = state["cand_r_total"]
    budget_d, budget_r = state["budget_d"], state["budget_r"]
    n = len(races)

    floors_d = np.array([r.cand_d_total for r in races])
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)

    d_pool = [party_d_obs, np.full(n, budget_d / n), np.zeros(n)]
    r_pool = [party_r_obs, np.full(n, budget_r / n), np.zeros(n)]
    d_labels = ["observed", "uniform", "zero"]
    r_labels = ["observed", "uniform", "zero"]

    logger.info("Seeding pool with one-shot unilateral best responses…")
    res_d = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                     budget_d=budget_d, cap_fraction_d=args.cap_fraction_d)
    res_r = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                     budget_r=budget_r, cap_fraction_r=args.cap_fraction_r)
    d_pool.append(res_d.party); d_labels.append("BR_D(R_obs)")
    r_pool.append(res_r.party); r_labels.append("BR_R(D_obs)")

    results_dir = REPO_ROOT / "results"
    pd_min, pr_min = _load_npy_pair(results_dir, "pure_exploitability_min_party_d",
                                     "pure_exploitability_min_party_r", args.cycle)
    if pd_min is not None:
        d_pool.append(pd_min); d_labels.append("E_min_candidate")
        r_pool.append(pr_min); r_labels.append("E_min_candidate")
        logger.info("Seeded E_min candidate from minimize_pure_exploitability.py's saved output.")

    pd_fp, pr_fp = _load_npy_pair(results_dir, "fictitious_play_avg_party_d",
                                   "fictitious_play_avg_party_r", args.cycle)
    if pd_fp is not None:
        d_pool.append(pd_fp); d_labels.append("fictitious_play_avg")
        r_pool.append(pr_fp); r_labels.append("fictitious_play_avg")
        logger.info("Seeded fictitious-play time-average from fictitious_play.py's saved output.")

    logger.info(f"Starting double oracle with {len(d_pool)} D portfolios, {len(r_pool)} R portfolios: "
                f"D={d_labels}, R={r_labels}")

    result = do.double_oracle(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        args.cap_fraction_d, args.cap_fraction_r, d_pool, r_pool,
        max_rounds=args.max_rounds, eps=args.eps,
    )

    logger.info(f"Double oracle {'converged' if result.converged else 'hit max_rounds'} "
                f"after {result.n_rounds} rounds. Final pools: D={len(result.d_pool)}, R={len(result.r_pool)}")
    logger.info(f"Value (E[D seats] at equilibrium) = {result.value:.3f}")
    logger.info(f"D support: {result.d_support} (weights {result.p[result.d_support] if len(result.d_support) else []})")
    logger.info(f"R support: {result.r_support} (weights {result.q[result.r_support] if len(result.r_support) else []})")
    for g in result.gain_history:
        logger.info(f"  round {g['round']}: value={g['value']:.4f} d_gain={g['d_gain']:.4f} "
                    f"r_gain={g['r_gain']:.4f} lp_gap={g['lp_gap']:.2e} pool_sizes=({g['d_pool_size']},{g['r_pool_size']})")

    def _label(labels, idx):
        return labels[idx] if idx < len(labels) else f"oracle_added_{idx}"

    out = {
        "cycle": args.cycle,
        "config": {"max_rounds": args.max_rounds, "eps": args.eps,
                   "cap_fraction_d": args.cap_fraction_d, "cap_fraction_r": args.cap_fraction_r},
        "seed_d_labels": d_labels, "seed_r_labels": r_labels,
        "converged": result.converged, "n_rounds": result.n_rounds,
        "final_d_pool_size": len(result.d_pool), "final_r_pool_size": len(result.r_pool),
        "value_e_seats_d": result.value,
        "d_support": [{"index": i, "label": _label(d_labels, i), "weight": float(result.p[i])}
                      for i in result.d_support],
        "r_support": [{"index": i, "label": _label(r_labels, i), "weight": float(result.q[i])}
                      for i in result.r_support],
        "d_mixture_full": result.p.tolist(),
        "r_mixture_full": result.q.tolist(),
        "gain_history": result.gain_history,
    }
    out_path = results_dir / f"double_oracle_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    for i, dp in enumerate(result.d_pool):
        np.save(results_dir / f"double_oracle_d_portfolio_{i}_{args.cycle}.npy", dp)
    for i, rp in enumerate(result.r_pool):
        np.save(results_dir / f"double_oracle_r_portfolio_{i}_{args.cycle}.npy", rp)
    logger.info(f"Saved -> {out_path}")
    logger.info(f"SUMMARY cycle={args.cycle}: double-oracle support sizes D={len(result.d_support)}/"
                f"{len(result.d_pool)}, R={len(result.r_support)}/{len(result.r_pool)}, "
                f"value={result.value:.3f}")


if __name__ == "__main__":
    main()
