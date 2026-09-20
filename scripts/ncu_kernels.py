"""Launch a single TriFast kernel repeatedly for Nsight Compute (ncu) profiling.

Usage:
    python scripts/ncu_kernels.py <kernel> [-n N]

Prerequisite launches (fwd, bwd_q) are run once first so that the profiled
kernel has valid inputs (o/mx/dn and delta). Then the target kernel is
launched 3 more times; ncu can profile these with `--launch-skip`.

The autotune cache must already be warm for the given N (run
scripts/bench_kernels.py first), otherwise ncu would profile every tuning
candidate.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bench_kernels import KERNELS, _make_launchers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=KERNELS)
    parser.add_argument("-n", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    # AlphaFold 3 kernel-benchmark shape, matching scripts/bench_kernels.py.
    launchers = _make_launchers(n=args.n, h=8, d=32, dtype=torch.bfloat16)

    # Populate o/mx/dn and delta so every backward kernel has valid inputs.
    launchers["fwd"]()
    launchers["bwd_q"]()
    torch.cuda.synchronize()

    # The launches ncu should actually profile.
    for _ in range(args.iters):
        launchers[args.kernel]()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
