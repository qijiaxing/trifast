"""Explicit low-memory helper, also used by the fused API default.

Chunking limits the FP32 dQ scratch allocation while retaining full FP32
reductions. Atomic reductions remain nondeterministic; only first gradients
are supported. This module is intentionally not exported from package init.
"""

import torch
from torch.autograd.function import once_differentiable

from trifast._fused_dispatch import fused_backward_dispatch
from trifast.fused_api import (
    _check_determinism,
    _validate,
    fused_forward,
)


class _LowMemoryAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask, chunk_i):
        o, _lse, mx, dn = fused_forward(q, k, v, bias, mask)
        ctx.save_for_backward(q, k, v, bias, mask, o, mx, dn)
        ctx.chunk_i = chunk_i
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        _check_determinism()
        q, k, v, bias, mask, o, mx, dn = ctx.saved_tensors
        dq, dk, dv, db = fused_backward_dispatch(
            do,
            q,
            k,
            v,
            bias,
            o,
            mx,
            dn,
            mask,
            chunk_i=ctx.chunk_i,
        )
        return dq, dk, dv, db, None, None


def triangle_attention_fused_low_memory(q, k, v, bias, mask, *, chunk_i=128):
    """Experimental opt-in attention with bounded FP32 dQ scratch.

    chunk_i must be a positive integer (bool is not accepted). Values larger
    than N use one chunk. Smaller chunks trade extra launches for less memory.
    """
    if isinstance(chunk_i, bool) or not isinstance(chunk_i, int):
        raise TypeError("chunk_i must be a positive integer, not bool")
    if chunk_i <= 0:
        raise ValueError("chunk_i must be positive")
    _validate(q, k, v, bias, mask)
    return _LowMemoryAttention.apply(q, k, v, bias, mask, chunk_i)


__all__ = ["triangle_attention_fused_low_memory"]
