#!/usr/bin/env python3
"""
Item (2) of Section 8.9's investigation plan: the first real step up the K
progression (15 -> 30 -> 50 -> 100), after three independent K=15 seeds
already showed a tightly clustered, consistently negative Delta_allocator
(Table 13e). K=30 roughly doubles the number of nonlinear calls (30x8=240
vs 15x8=120), so this is not free, but more seeds at K=15 cannot reduce the
per-seed noise the way an actual larger K can -- this is why the plan calls
for increasing K itself, not just adding more K=15 replicates.

Reuses theta_nonlinear_multiseed.run_one_seed() unchanged (already
implements the paired, common-random-numbers methodology verified
bit-identical before trusting) -- only K_PATHS and the seed differ.

Output: outputs/theta_nonlinear_k30.json
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
K_PATHS_K30 = 30
SEED = 20260731


def main():
    lsm.K_PATHS = K_PATHS_K30
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")
    print("Prior K=15 seeds (paired): Delta_allocator = -5.164, -4.723, -4.278 "
          "(mean -4.722, SD 0.443)")

    result = tms.run_one_seed(SEED)

    out_path = ROOT / "outputs/theta_nonlinear_k30.json"
    with open(out_path, "w") as f:
        json.dump({"k_paths": K_PATHS_K30, "seed": SEED, "result": result}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"\nK=30 result: Theta_LP(0)={result['theta0_lp']:+.4f}  "
          f"Theta_nonlinear(0)={result['theta0_nonlinear']:+.4f}  "
          f"Delta_allocator={result['delta_allocator']:+.4f}")


if __name__ == "__main__":
    main()
