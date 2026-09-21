"""Opt-in triangle attention with an experimental atomic fused backward.

The original API is unchanged. Atomic reductions are not deterministic;
deterministic-algorithm mode is rejected, including warn-only mode. Only first
derivatives are supported. Compilation and device coverage must be validated
for the installed PyTorch/Triton versions before production use.
"""

import torch
from torch.autograd.function import once_differentiable

from trifast._fused_backward import fused_backward
from trifast._fused_forward import fused_forward_optimized as fused_forward


def _check_determinism():
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "triangle_attention_fused uses nondeterministic atomic reductions; "
            "use trifast.torch.triangle_attention in deterministic mode."
        )


def _backward_block_j(q):
    # D128 needs a smaller query tile: measured 1.8--2.0x versus 64x64
    # on H20 for BF16/FP16. The kernel retains its IEEE FP32 resource guard.
    return 32 if q.shape[-1] == 128 else 64


def _validate(q, k, v, bias, mask):
    if q.ndim != 5:
        raise ValueError("q must have shape [B,H,N,N,D]")
    bs, h, n, nk, d = q.shape
    if min(bs, h, n) <= 0 or nk != n or d not in (16, 32, 64, 128):
        raise ValueError(
            "require positive B,H,N, square attention axes and D in {16,32,64,128}"
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v must have identical shapes")
    if bias.shape != (bs, h, n, n) or mask.shape != (bs, n, n):
        raise ValueError("bias must be [B,H,N,N] and mask must be [B,N,N]")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("q must be float16, bfloat16 or float32")
    if any(x.dtype != q.dtype for x in (k, v, bias)):
        raise TypeError("q, k, v and bias must have identical dtype")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean; True replaces the score")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, bias, mask)):
        raise ValueError("all inputs must be on the same CUDA device")
    _check_determinism()


def triangle_attention_fused_bwd(
    do, q, k, v, bias, o, mx, dn, mask, *, centered_stats=False
):
    """Direct five-output backward matching the baseline's timed interface.

    Saved o/mx/dn must come from the matching forward. Set centered_stats=True
    for FP32 statistics from fused_forward; leave False for baseline statistics.
    Workspace clearing,
    contiguous copies, output casts and the zero boolean dmask are included.
    """
    _validate(q, k, v, bias, mask)
    for name, x in (("do", do), ("o", o)):
        if x.shape != q.shape or x.dtype != q.dtype or x.device != q.device:
            raise ValueError(f"{name} must match q's shape, dtype and device")
    for name, x in (("mx", mx), ("dn", dn)):
        if x.shape != q.shape[:-1] or x.dtype != torch.float32 or x.device != q.device:
            raise ValueError(f"{name} must be float32 [B,H,N,N] on q's device")
    dq, dk, dv, db = fused_backward(
        do,
        q,
        k,
        v,
        bias,
        o,
        mx,
        dn,
        mask,
        bj=_backward_block_j(q),
        bk=64,
        gi=1,
        warps=4,
        stages=3,
        centered_stats=centered_stats,
    )
    return dq, dk, dv, db, torch.zeros_like(mask)


class _FusedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask):
        o, _lse, mx, dn = fused_forward(q, k, v, bias, mask)
        ctx.save_for_backward(q, k, v, bias, mask, o, mx, dn)
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        _check_determinism()
        q, k, v, bias, mask, o, mx, dn = ctx.saved_tensors
        # Autograd expects None for the boolean mask; avoid allocating the
        # compatibility dmask output used by the direct five-output interface.
        dq, dk, dv, db = fused_backward(
            do,
            q,
            k,
            v,
            bias,
            o,
            mx,
            dn,
            mask,
            bj=_backward_block_j(q),
            centered_stats=(q.dtype == torch.float32),
        )
        return dq, dk, dv, db, None


def triangle_attention_fused(q, k, v, bias, mask, *, chunk_i=128):
    """Return attention output using shared-computation fused backward.

    By default, process i in chunks of 128 to bound FP32 dQ scratch. Set
    chunk_i=None to use the full workspace; positive integers select explicit
    chunk sizes. This is a fixed default, not a free-memory-based policy.
    Inputs may be noncontiguous. Masked scores use finite -10000 replacement.
    """
    if chunk_i is not None:
        # Delayed import avoids a cycle with the explicit low-memory helper,
        # which shares validation, forward, and the D128 tile selection.
        from trifast.fused_low_memory_api import triangle_attention_fused_low_memory

        return triangle_attention_fused_low_memory(q, k, v, bias, mask, chunk_i=chunk_i)
    _validate(q, k, v, bias, mask)
    return _FusedAttention.apply(q, k, v, bias, mask)


__all__ = ["triangle_attention_fused", "triangle_attention_fused_bwd"]
