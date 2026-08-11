#!/usr/bin/env python3
"""
Item (2) of Section 8.9's investigation plan, final rung of the K
progression (15 -> 30 -> 50 -> 100). K=15 (3 seeds), K=30 (in progress),
and K=50 (in progress) have all shown a tightly clustered, consistently
negative Delta_allocator that has not moved meaningfully as K increased.
Launched now, in parallel with the K=30-multiseed and K=50 jobs, since the
machine has ample spare CPU capacity (each of these jobs uses roughly 1.3
of 16 cores) and K=100 is the single most expensive step, best started as
early as possible rather than queued after the others finish.

Reuses theta_nonlinear_multiseed.run_one_seed() unchanged (the verified,
paired, common-random-numbers methodology) -- only K_PATHS and the seed
differ. 800 total nonlinear calls (100 paths x 8 periods) -- expect this to
be the longest-running job so far.

Output: outputs/theta_nonlinear_k100.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
import theta_nonlinear_multiseed as tms

ROOT = Path(__file__).parent.parent
K_PATHS_K100 = 100
SEED = 6


def main():
    lsm.K_PATHS = K_PATHS_K100
    tms.K_PATHS_REDUCED = K_PATHS_K100
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")
    print("Prior K=15 seeds: Delta_allocator = -5.164, -4.723, -4.278 (mean -4.722, SD 0.443)")
    print("Prior K=30 seed (20260731): Delta_allocator = -4.703")

    result = tms.run_one_seed(SEED)

    out_path = ROOT / "outputs/theta_nonlinear_k100.json"
    with open(out_path, "w") as f:
        json.dump({"k_paths": K_PATHS_K100, "seed": SEED, "result": result}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"\nK=100 result: Theta_LP(0)={result['theta0_lp']:+.4f}  "
          f"Theta_nonlinear(0)={result['theta0_nonlinear']:+.4f}  "
          f"Delta_allocator={result['delta_allocator']:+.4f}")


if __name__ == "__main__":
    main()
