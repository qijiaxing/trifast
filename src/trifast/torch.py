import math
import triton
import torch
from jaxtyping import Bool, Float
from einops import rearrange
from torch.library import wrap_triton, triton_op
import triton.testing
from triton.tools.tensor_descriptor import TensorDescriptor

from trifast.triton import (
    _fwd,
    _fwd_pointer,
    _fwd_finalize,
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)

# Value the kernels substitute for a masked score. NOT torch.finfo(q.dtype).min, which
# is what the reference's masked_fill_ uses and what this file used to pass: the kernels
# convert scores to log2 units, and finfo(fp32).min * 1.4427 overflows fp32 to -inf.
#
# Unlike flex's identically-valued MASK_FILL (see flex/flex.py), nothing here depends on
# the magnitude being *large* either -- the stable fallback stores its normalization
# offset and denominator separately, so a fully-masked row never needs SENTINEL + log(N)
# to stay distinguishable from SENTINEL. The one requirement is that a masked key next
# to valid ones gets exactly zero weight. -1e4 leaves a large margin and matches flex.
#
# Lives in fp32 score space, so one value serves every input dtype.
MASK_FILL = -1e4
USE_FAST_PATH = False
USE_TMA_BIAS = True


@triton_op("trifast::triangle_attention", mutates_args={})
def _triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (o, lse, mx, dn).

    `lse` is the natural-log logsumexp, the same convention flex and protenix return,
    and is for callers and diagnostics only. The backward consumes `mx` (the base-two
    normalization offset) and `dn` (the corresponding denominator) instead.
    """
    sm_scale = q.shape[-1] ** -0.5

    bs, h, _, n, dim = q.shape

    # TODO: Should also allow flattening arbitrary batch dims.
    q = rearrange(q, "b h ... -> (b h) ...").contiguous()
    k = rearrange(k, "b h ... -> (b h) ...").contiguous()
    v = rearrange(v, "b h ... -> (b h) ...").contiguous()
    b = rearrange(b, "b h ... -> (b h) ...").contiguous()
    fwd_b = (
        (b * 1.4426950408889634).to(q.dtype)
        if USE_FAST_PATH and q.dtype == torch.bfloat16
        else b
    )
    mask = mask.contiguous()

    # e.g. batch x head
    bh = q.shape[0]
    can_use_tma_bias = (
        USE_TMA_BIAS
        and dim <= 64
        and type(fwd_b).__name__
        not in {
            "FakeTensor",
            "FunctionalTensor",
        }
    )
    if can_use_tma_bias:
        # on hopper, tma requires 16 bytes alignment
        bias_alignment = 16 // fwd_b.element_size()
        padded_n = triton.cdiv(n, bias_alignment) * bias_alignment
        padded_b = torch.nn.functional.pad(fwd_b, (0, padded_n - n))
        desc_b = TensorDescriptor.from_tensor(
            padded_b.reshape(bh * n, padded_n), block_shape=[64, 32]
        )
    else:
        desc_b = fwd_b

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    o = torch.zeros_like(q)
    # _fwd takes a single set of strides for these three, so keep them identical.
    lse = torch.zeros((bh, n, n), device=q.device, dtype=torch.float32)
    mx = torch.zeros_like(lse)
    dn = torch.zeros_like(lse)

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    fwd_kernel = _fwd if can_use_tma_bias else _fwd_pointer

    # fmt: off
    wrap_triton(fwd_kernel)[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse, mx, dn, lse.stride(0), lse.stride(1), lse.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        fwd_b, fwd_b.stride(0), fwd_b.stride(1), fwd_b.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        desc_b,
        neg_inf=MASK_FILL,
        sm_scale=sm_scale, N=n, H=h, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        USE_FAST_PATH=USE_FAST_PATH,
        USE_TMA_BIAS=can_use_tma_bias,
    )

    if USE_FAST_PATH:
        wrap_triton(_fwd_finalize)[(n, bh)](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse, mx, dn, lse.stride(0), lse.stride(1), lse.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            fwd_b, fwd_b.stride(0), fwd_b.stride(1), fwd_b.stride(2),
            mask, mask.stride(0), mask.stride(1), mask.stride(2),
            sm_scale=sm_scale, neg_inf=MASK_FILL, N=n, H=h, DIM=dim,
            BLOCK_J=64, BLOCK_K=32, num_warps=4, num_stages=3,
            USE_FAST_PATH=USE_FAST_PATH,
        )


    o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    lse = rearrange(lse, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    mx = rearrange(mx, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dn = rearrange(dn, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return o, lse, mx, dn


@triton_op(
    "trifast::triangle_attention_backward",
    mutates_args={},
)
def triangle_attention_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    o: torch.Tensor,
    mx: torch.Tensor,
    dn: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, h, *_ = q.shape

    # TODO: Should also allow flattening arbitrary batch dims.
    q = rearrange(q, "b h ... -> (b h) ...")
    k = rearrange(k, "b h ... -> (b h) ...")
    v = rearrange(v, "b h ... -> (b h) ...")
    b = rearrange(b, "b h ... -> (b h) ...")
    kernel_b = (
        (b * 1.4426950408889634).to(q.dtype)
        if USE_FAST_PATH and q.dtype == torch.bfloat16
        else b
    )
    o = rearrange(o, "b h ... -> (b h) ...")
    mx = rearrange(mx, "b h ... -> (b h) ...")
    dn = rearrange(dn, "b h ... -> (b h) ...")
    do = rearrange(do, "b h ... -> (b h) ...")

    bh, _, n, dim = q.shape
    sm_scale = dim**-0.5

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    # Every valid element of these outputs is overwritten by a non-atomic store.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty_like(b)
    dmask = torch.zeros_like(mask)  # Don't need grads, but torch expects a zero tensor

    # fp32, not q.dtype: delta enters the cancellation-prone (dsm_value - delta) that
    # _bwd_kv and _bwd_b read back, and rounding it to bf16 there was the most likely
    # reason db was the weakest of the five gradients. _bwd_q overwrites every element.
    d = torch.empty((bh, n, n), dtype=torch.float32, device=q.device)

    def q_grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    # fmt: off
    # NOTE: This also calculates delta for kv/b!
    wrap_triton(_bwd_q)[q_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        kernel_b, kernel_b.stride(0), kernel_b.stride(1), kernel_b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        USE_FAST_PATH=USE_FAST_PATH,
    )
    # fmt: on

    # Do the actual backward pass.
    def kv_grid(x):
        return (triton.cdiv(n, x["BLOCK_K"]), n, bh)

    # fmt: off
    wrap_triton(_bwd_kv)[kv_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        kernel_b, kernel_b.stride(0), kernel_b.stride(1), kernel_b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        USE_FAST_PATH=USE_FAST_PATH,
    )
    # fmt: on

    def b_grid(x):
        return (
            triton.cdiv(n, x["BLOCK_J"]),
            triton.cdiv(n, x["BLOCK_K"]),
            bh,
        )

    # fmt: off
    wrap_triton(_bwd_b)[b_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        kernel_b, kernel_b.stride(0), kernel_b.stride(1), kernel_b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        db, db.stride(0), db.stride(1), db.stride(2),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        USE_FAST_PATH=USE_FAST_PATH,
    )
    # fmt: on

    dq = rearrange(dq, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dk = rearrange(dk, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dv = rearrange(dv, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    db = rearrange(db, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return dq, dk, dv, db, dmask


def backwards(ctx, *grad: tuple[Float[torch.Tensor, "b h n n d"],]) -> tuple[
    Float[torch.Tensor, "b h n n d"],  # dq
    Float[torch.Tensor, "b h n n d"],  # dk
    Float[torch.Tensor, "b h n n d"],  # dv
    Float[torch.Tensor, "b h n n"],  # db
    Bool[torch.Tensor, "b n n"],  # dmask
]:
    do = grad[0]
    q, k, v, b, mask, o, mx, dn = ctx.saved_tensors
    dq, dk, dv, db, dmask = triangle_attention_bwd(
        do,
        q,
        k,
        v,
        b,
        o,
        mx,
        dn,
        mask,
    )

    return dq, dk, dv, db, dmask


def setup_context(ctx, inputs, output) -> None:
    q, k, v, b, mask, *_ = inputs
    # lse is deliberately not saved: it is a convenience for callers, and the backward
    # reads the unfused (mx, dn) pair instead.
    o, _lse, mx, dn = output

    ctx.save_for_backward(q, k, v, b, mask, o, mx, dn)


_triangle_attention.register_autograd(backwards, setup_context=setup_context)


def triangle_attention(
    q: Float[torch.Tensor, "b h n n d"],
    k: Float[torch.Tensor, "b h n n d"],
    v: Float[torch.Tensor, "b h n n d"],
    b: Float[torch.Tensor, "b h n n"],
    mask: Bool[torch.Tensor, "b n n"],
) -> Float[torch.Tensor, "b h n n d"]:
    o, *_ = _triangle_attention(q, k, v, b, mask)
    return o
