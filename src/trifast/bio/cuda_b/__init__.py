"""cuda_b: the M1 sm_90a triangle-attention forward from triattn_pkg/cuda_b/ (see trifast.bio).

Three consumer warpgroups (one pair row each), 64x32 S chunks streamed max-free, pair bias /
scale staged once per call in fp32 MMA-fragment order. CTA tiles the max-free pass cannot
finish are recomputed by an exact SAFE instantiation via a fix list. Fully-masked rows return
the uniform mean of v.

csrc/ holds the m1/ kernel and the fa3_utils.h it includes, changed from upstream in two ways:
the mask is True = masked (trifast's convention), and the kernel also returns trifast's softmax
statistics. Their prebuilt binaries therefore no longer apply; the extension is JIT-built on
first use.

Statistics, [B, N, H, S] fp32 each, with x = (scale * q.k + bias) * log2(e) the base-two logit:

    P = exp2(x - mx) / dn          (what trifast's backward reconstructs)
    lse = (mx + log2(dn)) * ln(2)  (natural-log logsumexp, as trifast's lse)

lse is exact. mx and dn are a consistent pair but NOT trifast's values: the max-free pass never
tracks the row max, so mx sits ~64 above it and dn is ~2^-64 (tiles recomputed by the SAFE pass
do hold the exact max). dn sums the bf16-rounded P that the PV product used. A fully-masked row
gets trifast's values for S logits of MASK_FILL: mx = MASK_FILL * log2(e), dn = S.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from trifast.bio import NVCC_FLAGS, Unsupported, check_device, cutlass_include, tma_ok
from trifast.torch import MASK_FILL

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
# Hot instantiations; each gets its SAFE (fix-list) partner, flags | 1024.
FLAGS = [0]


def _all_flags():
    base = {f & ~1024 for f in FLAGS}
    return sorted(base | {(f & ~256) | 1024 for f in base})


def _generate_sources(build_dir: str):
    inst = os.path.join(build_dir, "inst_m1")
    os.makedirs(inst, exist_ok=True)
    srcs, decls, rows = [], [], []
    for fl in _all_flags():
        body = (f'#include "launch_m1.cuh"\nnamespace triattn_m1 {{\n'
                f"void run_m1_{fl}(Args const& a) {{ launch_m1<Traits<{fl}>>(a); }}\n"
                f"int64_t smem_m1_{fl}() {{ return smem_bytes_m1<Traits<{fl}>>(); }}\n}}\n")
        path = os.path.join(inst, f"m1_{fl}.cu")
        if not os.path.exists(path) or open(path).read() != body:
            open(path, "w").write(body)
        srcs.append(path)
        decls.append(f"void run_m1_{fl}(Args const&); int64_t smem_m1_{fl}();")
        rows.append(f"    t[{fl}] = Entry{{&run_m1_{fl}, smem_m1_{fl}()}};")
    table = ("\n".join(decls) + "\nstatic std::map<int, Entry> make_table() {\n    std::map<int, Entry> t;\n"
             + "\n".join(rows) + "\n    return t;\n}\n")
    tpath = os.path.join(inst, "table_m1.inc")
    if not os.path.exists(tpath) or open(tpath).read() != table:
        open(tpath, "w").write(table)
    return srcs, inst


def _build(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import _get_build_directory, load

    name = "trifast_bio_cuda_b_ext"
    srcs, inst = _generate_sources(_get_build_directory(name, verbose))
    csrc = os.path.join(_HERE, "csrc", "m1")
    _EXT = load(
        name=name,
        sources=[os.path.join(csrc, "m1_binding.cu")] + srcs,
        extra_include_paths=[csrc, os.path.join(_HERE, "csrc"), inst, cutlass_include()],
        extra_cuda_cflags=NVCC_FLAGS,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


_FIX_TOTAL = {}  # device -> int32[2]: [CTA tiles recomputed by the SAFE pass, debug counter]
_MASK_COUNTS = {}  # device -> int32[2]: [irregular rows, fully-masked rows]


def _device_buffer(cache: dict, dev) -> torch.Tensor:
    key = torch.device(dev).index
    if key not in cache:
        cache[key] = torch.zeros(2, dtype=torch.int32, device=dev)
    return cache[key]


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (out, lse, mx, dn); mask is True where a key is masked."""
    if q.dim() != 5:
        raise Unsupported(f"cuda_b: q/k/v must be [B,N,H,S,D]; got rank {q.dim()}")
    B, N, H, S, D = q.shape
    check_device(q, "cuda_b")
    if D != 32:
        raise Unsupported(f"cuda_b: head dim {D} (only 32)")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Unsupported(f"cuda_b: q/k/v dtype {q.dtype} (only bf16)")
    if k.shape != q.shape or v.shape != q.shape:
        raise Unsupported("cuda_b: q, k, v must have identical shapes [B, N, H, S, 32]")
    if tuple(bias.shape) != (B, 1, H, S, S):
        raise Unsupported(f"cuda_b: bias shape {tuple(bias.shape)} (need [B, 1, H, S, S])")
    if mask is not None and tuple(mask.shape) != (B, N, 1, 1, S):
        raise Unsupported(f"cuda_b: mask shape {tuple(mask.shape)} (need [B, N, 1, 1, S])")
    ext = _build()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q, k, v = (t if tma_ok(t) else t.contiguous() for t in (q, k, v))
    maskw = rowkind = kcend = kcstart = keyany = rowkc0 = rowkc1 = None
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask != 0
        maskw, keyany, rowkind, kcend, kcstart, rowkc0, rowkc1 = ext.stage_mask(
            mask, _device_buffer(_MASK_COUNTS, q.device)
        )
    n_ctas = ((S + 127) // 128) * ((N + 2) // 3 + 1) * B * H
    # Fix list: [0] = count (zeroed by stage_bias), then CTA-tile triples.
    fix = torch.empty(1 + 3 * n_ctas, dtype=torch.int32, device=q.device)
    bias_staged = ext.stage_bias(bias, float(scale), keyany, fix)
    out = torch.empty(B, N, H, S, D, dtype=q.dtype, device=q.device)
    # The kernel takes one set of strides for all three statistics.
    lse = torch.empty(B, N, H, S, dtype=torch.float32, device=q.device)
    mx = torch.empty_like(lse)
    dn = torch.empty_like(lse)
    ext.fwd(
        q, k, v, bias_staged, float(scale), out, lse, mx, dn, float(MASK_FILL),
        fix, _device_buffer(_FIX_TOTAL, q.device),
        maskw, rowkind, kcend, kcstart, rowkc0, rowkc1, 0, None,
    )
    return out, lse, mx, dn


__all__ = ["triangle_attention", "Unsupported"]
