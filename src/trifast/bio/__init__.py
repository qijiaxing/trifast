"""sm_90a triangle-attention forward kernels from uplifting-biomolecular-modeling.

Copied from common/opt_core/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/:

    cuda_b  <- cuda_b/  (M1, csrc/m1/: 3 consumer warpgroups, max-free streaming, staged bias)

Their cuda/ kernel (triattn_sm90.cuh) was measured ~10% slower than trifast's _fwd at every N
and dropped. Kernels here are forward only, bf16, D == 32, with one calling convention:

    out, lse, mx, dn = triangle_attention(q, k, v, bias, mask=None, scale=None)

q, k, v: [B, N, H, S, D] (stride(-1) == 1 and the other strides multiples of 8 elements, so a
permuted trifast [B, H, N, S, D] tensor needs no copy); bias: [B, 1, H, S, S] shared by the N
rows; mask: [B, N, 1, 1, S] bool, True = masked (trifast's convention). out is [B, N, H, S, D];
lse, mx, dn are [B, N, H, S] fp32 softmax statistics in trifast's convention (see cuda_b).

The extensions need CUTLASS headers when JIT-built: $CUTLASS_PATH, else /opt/cutlass, else the
copy bundled with flashinfer.
"""

from __future__ import annotations

import importlib.util
import os

import torch


class Unsupported(NotImplementedError):
    """Raised (before any work) for inputs a kernel does not serve, naming the reason."""


def cutlass_include() -> str:
    candidates = [os.environ.get("CUTLASS_PATH"), "/opt/cutlass"]
    spec = importlib.util.find_spec("flashinfer")
    if spec is not None and spec.origin is not None:
        candidates.append(os.path.join(os.path.dirname(spec.origin), "data", "cutlass"))
    for root in candidates:
        if root and os.path.isfile(os.path.join(root, "include", "cutlass", "cutlass.h")):
            return os.path.join(root, "include")
    raise Unsupported("bio: CUTLASS headers not found; set CUTLASS_PATH to a CUTLASS checkout")


# nvcc flags the kernel directories were sealed with.
NVCC_FLAGS = [
    "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
    "-gencode", "arch=compute_90a,code=sm_90a", "-DNDEBUG", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "--ftemplate-backtrace-limit=0", "-lineinfo", "-Xcompiler", "-Wno-psabi", "-diag-suppress", "177,550",
]


def tma_ok(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and all(int(s) % 8 == 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0


def check_device(t: torch.Tensor, who: str) -> None:
    if t.device.type != "cuda" or torch.cuda.get_device_capability(t.device) != (9, 0):
        raise Unsupported(f"{who}: sm_90a (H100/H200/H20) only")
