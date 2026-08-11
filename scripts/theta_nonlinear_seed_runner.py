#!/usr/bin/env python3
"""
Generic, parameterized driver for a single paired LP-vs-nonlinear replicate,
used to launch many independent replicates as separate OS processes (so
they run truly in parallel on separate cores, rather than sequentially
within one Python process as the earlier *_multiseed.py scripts did).

Usage:
    python3 theta_nonlinear_seed_runner.py --k 50 --seed 51 --scenario bootstrap
    python3 theta_nonlinear_seed_runner.py --k 30 --seed 1 --scenario eta_fit_2022
    python3 theta_nonlinear_seed_runner.py --k 30 --seed 1 --scenario eta_fit_2024

Reuses theta_nonlinear_multiseed.run_one_seed() (bootstrap scenario) and
run_one_seed_single_cycle() (single-cycle scenarios) unchanged -- both
already implement and verify the paired, common-random-numbers methodology.

Output: outputs/theta_nonlinear_seed_{scenario}_k{K}_seed{seed}.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
import theta_nonlinear_multiseed as tms

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenario", choices=["bootstrap", "eta_fit_2022", "eta_fit_2024"], required=True)
    args = ap.parse_args()

    lsm.K_PATHS = args.k
    tms.K_PATHS_REDUCED = args.k
    print(f"scenario={args.scenario} K_PATHS={args.k} seed={args.seed} "
          f"N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")

    if args.scenario == "bootstrap":
        result = tms.run_one_seed(args.seed)
    elif args.scenario == "eta_fit_2022":
        result = tms.run_one_seed_single_cycle(args.seed, 2022)
    else:
        result = tms.run_one_seed_single_cycle(args.seed, 2024)

    out_path = ROOT / f"outputs/theta_nonlinear_seed_{args.scenario}_k{args.k}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"scenario": args.scenario, "k_paths": args.k, "seed": args.seed,
                    "result": result}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"\nResult: Theta_LP(0)={result['theta0_lp']:+.4f}  "
          f"Theta_nonlinear(0)={result['theta0_nonlinear']:+.4f}  "
          f"Delta_allocator={result['delta_allocator']:+.4f}")


if __name__ == "__main__":
    main()
