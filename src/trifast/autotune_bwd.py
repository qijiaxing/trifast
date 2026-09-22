"""Autotune candidates for the fused backward kernel in `triton_bwd.py`.

The search space is much narrower than the forward's, and three of its edges are hard
walls rather than preferences. All measured on an H20-3e at n=1024, h=8, d=32, bf16,
against a 62.4 ms three-kernel baseline:

| BLOCK_J | BLOCK_K | warps | stages | fused ms | regs | spills |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64 | 4 | 2 | **39.8** | 250 | 0 |
| 64 | 64 | 4 | 3 | 41.6 | 255 | 0 |
| 64 | 64 | 4 | 4 | 41.9 | 255 | 8 |
| 64 | 64 | 4 | 1 | 43.6 | 255 | 10 |
| 32 | 64 | 4 | 2 | 49.1 | 208 | 0 |
| 128 | 64 | 4 | 2 | 60.8 | 255 | 160 |
| 64 | 128 | 4 | 2 | 71.4 | 255 | 246 |
| 64 | 64 | 8 | 2 | 84.6 | 216 | 0 |
| 64 | 32 | 4 | 3 | 119.9 | 255 | 72 |

1. **`BLOCK_K` must be >= 64 for bf16/fp16.** In the transposed `[k, j]` orientation
   `BLOCK_K` is the M dimension of the dV and dK dots, and M=32 falls off wgmma onto
   `mma.sync`: 119.9 ms against 39.8 ms, a 3x cliff rather than a gradient.
2. **`num_warps=4`.** Eight warps is 2.1x slower at the same tile -- these shapes want
   one warpgroup.
3. **Bigger tiles spill.** 128 in either axis blows past the register file; the kernel
   holds two `[k, j]` fp32 tiles plus dk, dv and the accumulator pointer tensors.

Capping registers with `maxnreg` to buy occupancy also loses: 168 spills 86 slots and
costs 12 ms, 128 spills 50 and costs 10 ms. The kernel runs at 250 registers and ~12.5 %
occupancy by design -- the win comes from doing five matmuls instead of nine, not from
latency hiding.
"""

import torch
import triton

from trifast.autotune_helpers import FORCE_TUNE

_bwd_fused_configs = [
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    # Small-tile fallback: the only candidate that fits when fp32 takes the `ieee` path
    # at DIM=128, and the sensible shape for n < 64.
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
]

if FORCE_TUNE:
    _bwd_fused_configs.extend(
        [
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3, maxnreg=168),
        ]
    )


def pinned_bwd_fused_config(dim: int, dtype: torch.dtype) -> dict:
    """The config traced/fake execution launches with, because it must not autotune.

    Autotuning a kernel that *accumulates* into its outputs is only safe if the tuner's
    `reset_to_zero` fires between the trials and the real launch. It does in eager; it
    does not when a cold cache is hit inside a compiled region, where the benchmarking
    trials add into the db and dq accumulators and left them ~400x too large (measured,
    n=96, bf16). A pinned config never benchmarks, so the traced path cannot hit this.

    The values are the measured winner at n=1024, h=8, d=32, bf16, and they are safe
    defaults elsewhere -- the two cliffs in the table above are both avoided. fp32 at
    DIM > 64 is the one shape that cannot use them: `input_precision="ieee"` stages both
    dot operands through shared memory, and 64x64 then asks for 255 KB of the 232 KB an
    SM has.
    """
    if dtype == torch.float32 and dim > 64:
        return {"BLOCK_J": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3}
    return {"BLOCK_J": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2}


def prune_bwd_fused_configs(configs, named_args, **kwargs):
    """Drop tiles that cannot be built for fp32 inputs at DIM > 64.

    `input_precision="ieee"` with fp32 operands disables the tensor cores entirely and
    stages both dot operands through shared memory, so DIM=128 at BLOCK_J=BLOCK_K=64
    asks for 255 KB of the 232 KB an SM has and fails to launch. Triton's autotuner
    prunes `OutOfResources` candidates on its own, but only after paying a compile for
    each; this skips them, and guarantees a survivor rather than an empty list.
    """
    if named_args["q_ptr"].dtype != torch.float32 or kwargs["DIM"] <= 64:
        return configs
    small = [c for c in configs if c.kwargs["BLOCK_J"] * c.kwargs["BLOCK_K"] <= 32 * 32]
    return small or configs
