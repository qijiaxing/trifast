"""Opaque dispatcher boundaries for dynamic-shape fused attention.

Python launch selection and the low-memory chunk loop execute inside these
operators, outside the compiled graph. The public autograd.Function supplies
first derivatives; these internal operators do not define higher derivatives.
"""

import torch

from trifast._fused_backward import fused_backward as _full_backward
from trifast._fused_chunked import fused_backward as _chunked_backward
from trifast._fused_forward import fused_forward_optimized


@torch.library.custom_op("trifast::fused_dispatch_forward", mutates_args=())
def fused_forward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output and matching finite-mask normalization statistics."""
    return fused_forward_optimized(q, k, v, bias, mask)


@fused_forward_dispatch.register_fake
def _forward_fake(q, k, v, bias, mask):
    # Explicit shape allocation also models noncontiguous inputs: the real
    # wrapper materializes its inputs and returns contiguous, nonalias outputs.
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    stats = tuple(
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        for _ in range(3)
    )
    return output, *stats


@torch.library.custom_op("trifast::fused_dispatch_backward", mutates_args=())
def fused_backward_dispatch(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    mx: torch.Tensor,
    dn: torch.Tensor,
    mask: torch.Tensor,
    chunk_i: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four gradients; zero selects full workspace, positive chunks i.

    mx/dn must come from fused_forward_dispatch for the same inputs. In FP32
    their centered-domain convention is selected here without changing masks.
    """
    if chunk_i < 0:
        raise ValueError("chunk_i must be zero (full workspace) or positive")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("fused attention uses nondeterministic atomic reductions")
    options = {
        "bj": 32 if q.shape[-1] == 128 else 64,
        "centered_stats": q.dtype == torch.float32,
    }
    inputs = (do, q, k, v, bias, output, mx, dn, mask)
    if chunk_i == 0:
        return _full_backward(*inputs, **options)
    return _chunked_backward(*inputs, chunk_i=chunk_i, **options)


@fused_backward_dispatch.register_fake
def _backward_fake(do, q, k, v, bias, output, mx, dn, mask, chunk_i=0):
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(bias.shape, dtype=bias.dtype, device=bias.device),
    )
