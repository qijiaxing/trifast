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
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)
from trifast.autotune_bwd import pinned_bwd_fused_config
from trifast.triton_bwd import (
    _bwd_bias_prep,
    _bwd_fused,
    _bwd_fused_tuned,
    _bwd_preprocess,
    _bwd_scale_cast,
)

# Traced/fake tensor execution (torch.compile, opcheck) differs from eager in two ways
# that matter to the kernels: it cannot build tensormaps, and it cannot be trusted to run
# an autotuner's reset_to_zero hook. Both paths branch on this.
_is_fake = lambda t: type(t).__name__ in {"FakeTensor", "FunctionalTensor"}

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
USE_TMA = True
USE_TMA_BIAS = True
# Route the bool mask through TMA as a bf16 copy instead of a per-iteration pointer
# load. Worth ~+2% on the forward; flip to False to fall back to the pointer load.
USE_TMA_MASK = True
# Route the fused backward's fp32 bias tile through TMA instead of a pointer load. Flip to
# False to fall back; both paths are exercised by scripts/bench_kernels.py, and the
# fake-tensor path takes the pointer load regardless.
USE_TMA_BWD_BIAS = True
# Run the backward as one fused kernel (plus two memory-bound passes) instead of the
# three kernels in triton.py, which each recompute the score tile. Flip to False to fall
# back to them; both paths are exercised by scripts/bench_kernels.py. The fused path
# accumulates db and dq with atomics, so those two gradients are no longer bitwise
# reproducible run to run -- the spread is ~1e-7 relative, far below one bf16 ulp.
USE_FUSED_BWD = True


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
    # Validate the layout before unpacking it. A wrongly-ordered 5-D tensor still
    # unpacks, so this used to fail silently rather than loudly: test_weight_updates
    # passed q as [b, n, n, h, d], which made the destructuring below read h=n and
    # n=1, and the kernel filled a single i slice and left 93.8% of the output zero.
    # torch._check raises eagerly and becomes a guard under torch.compile.
    torch._check(
        q.ndim == 5 and k.shape == q.shape and v.shape == q.shape,
        lambda: "q/k/v must all be [batch, heads, n, n, dim]; got "
        f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}",
    )
    torch._check(
        q.shape[2] == q.shape[3],
        lambda: "q/k/v are [batch, heads, n, n, dim], so dims 2 and 3 must match; "
        f"got {tuple(q.shape)} -- is the head axis in the wrong position?",
    )
    torch._check(
        b.ndim == 4 and b.shape == q.shape[:4],
        lambda: f"bias must be [batch, heads, n, n] = {tuple(q.shape[:4])}; "
        f"got {tuple(b.shape)}",
    )
    torch._check(
        mask.ndim == 3
        and mask.shape == (q.shape[0], q.shape[2], q.shape[3]),
        lambda: "mask must be [batch, n, n] = "
        f"{(q.shape[0], q.shape[2], q.shape[3])}; got {tuple(mask.shape)}",
    )

    sm_scale = q.shape[-1] ** -0.5

    bs, h, _, n, dim = q.shape

    # TODO: Should also allow flattening arbitrary batch dims.
    q = rearrange(q, "b h ... -> (b h) ...").contiguous()
    k = rearrange(k, "b h ... -> (b h) ...").contiguous()
    v = rearrange(v, "b h ... -> (b h) ...").contiguous()
    b = rearrange(b, "b h ... -> (b h) ...").contiguous()
    mask = mask.contiguous()

    # e.g. batch x head
    bh = q.shape[0]
    # Traced/fake tensor execution (torch.compile, opcheck) cannot build
    # tensormaps, so it falls back to the pointer kernel.
    #
    # Every pointer branch in `_fwd` is still reachable -- measured, not assumed, by
    # logging the flag combinations across the test matrix:
    #
    #   TMA  BIAS  MASK   reached by
    #    Y     Y     Y    the normal case, every tested shape with dim <= 64
    #    Y     N     Y    dim > 64, e.g. the (16, 4, 128) case in test_values
    #    N     Y     Y    dim where dim * element_size % 16 != 0, e.g. dim=4 bf16
    #    N     N     N    fake tensors, i.e. torch.compile and opcheck
    #
    # The last row is the one that cannot be designed away: TensorDescriptor needs a
    # real data pointer, and the descriptor construction below runs during fake
    # tracing. Since `_fwd_pointer` is `autotune(...)(_fwd.fn)` -- the same kernel
    # body -- the branches have to live here rather than in a separate kernel.
    # TMA needs 16-byte-aligned global strides; a contiguous [*, dim] inner
    # layout gives dim * element_size bytes per row.
    can_use_tma = (
        USE_TMA
        and dim * q.element_size() % 16 == 0
        and not _is_fake(q)
    )
    # The dim limit is a measured performance gate, not an alignment one -- the bias
    # box is [BLOCK_J, BLOCK_K] and does not depend on dim at all. Lifting it works
    # and is bit-identical, but at dim=128 the TMA bias is 4.0-4.2% *slower* than the
    # pointer load (99.4 vs 103.6 TFLOP/s, n=256 h=2, reproduced), because those
    # configs are already register-tight enough that the extra descriptor does not
    # pay. So dim > 64 keeps the pointer path deliberately.
    can_use_tma_bias = (
        USE_TMA_BIAS
        and dim <= 64
        and not _is_fake(b)
    )
    # Fake tensors cannot build a tensormap, as for q/k/v/o above. Nothing else is
    # needed: the flat [batch * n, padded_n] view built below addresses rows as
    # `batch * N + i`, which requires the mask to really be n x n, but the
    # torch._check calls at the top of this function already guarantee that
    # (mask.shape == (batch, q.shape[2], q.shape[3]) and q.shape[2] == q.shape[3],
    # and n is q.shape[3]). Reusing N for that row index rather than passing the
    # mask's own row count is deliberate -- the extra kernel argument measures ~2%
    # slower.
    can_use_tma_mask = USE_TMA_MASK and not _is_fake(mask)
    if can_use_tma_mask:
        # Why copy the mask at all: a bool tensor cannot go through TMA here. _fwd's
        # mask box must be >= 128 bytes or the pipelined loop faults with
        # cudaErrorMisalignedAddress -- 64 bytes is the only failing size, 128
        # through 512 all work, and a 1-byte box loads fine in a *standalone*
        # kernel, so this is a shared-memory alignment limit, not a TMA one.
        #
        # Why bf16 and not something wider: a 4-byte copy doubles the TMA traffic
        # and cancels the entire speedup. BLOCK_K=32 configs reach 128 bytes with a
        # 2-row box instead (see _fwd_descriptor_pre_hook).
        MASK_TMA_DTYPE = torch.bfloat16
        # Entries per 16 bytes, which is the row-pitch alignment TMA requires.
        mask_alignment = 16 // MASK_TMA_DTYPE.itemsize
        # Round the row length up so every row starts 16-byte aligned.
        padded_mask_n = triton.cdiv(n, mask_alignment) * mask_alignment
        # The widened copy. Only the truth of each entry is read, so 0.0/1.0 is
        # enough. Costs n**2 * 2 B (2 MB at n=1024) and ~35 us, once per call.
        wide_mask = torch.nn.functional.pad(
            mask.to(MASK_TMA_DTYPE), (0, padded_mask_n - n)
        )
        # Fold (batch, i) into a single row axis so the kernel can address a row as
        # `batch * N + i`. Rank-2 is load bearing: a rank-3 box over [batch, i, k]
        # gives up the whole speedup. block_shape is a placeholder that
        # _fwd_descriptor_pre_hook rewrites to [1, BLOCK_K] (or [2, BLOCK_K]).
        desc_mask = TensorDescriptor.from_tensor(
            wide_mask.reshape(mask.shape[0] * n, padded_mask_n), block_shape=[1, 32]
        )
    else:
        # Pointer fallback: _fwd indexes `mask` through its strides instead.
        desc_mask = mask
    if can_use_tma_bias:
        # on hopper, tma requires 16 bytes alignment
        bias_alignment = 16 // b.element_size()
        padded_n = triton.cdiv(n, bias_alignment) * bias_alignment
        padded_b = torch.nn.functional.pad(b, (0, padded_n - n))
        # The block_shape is a placeholder; _fwd_descriptor_pre_hook rewrites it
        # to [BLOCK_J, BLOCK_K] of the selected autotune config.
        desc_b = TensorDescriptor.from_tensor(
            padded_b.reshape(bh * n, padded_n), block_shape=[64, 32]
        )
    else:
        desc_b = b

    o = torch.zeros_like(q)
    if can_use_tma:
        # Rank-4 descriptors over the natural [bh, n, n, dim] layout. Boxes are
        # [1, 1, BLOCK_*, DIM]; the placeholder block_shape is rewritten by the
        # config pre-hook. The rank-4 box keeps each tile inside one (h, i)
        # slice, so rows >= n clip instead of wrapping into the next slice.
        desc_q = TensorDescriptor.from_tensor(q, block_shape=[1, 1, 64, 32])
        desc_k = TensorDescriptor.from_tensor(k, block_shape=[1, 1, 64, 32])
        desc_v = TensorDescriptor.from_tensor(v, block_shape=[1, 1, 64, 32])
        desc_o = TensorDescriptor.from_tensor(o, block_shape=[1, 1, 64, 32])
    else:
        desc_q, desc_k, desc_v, desc_o = q, k, v, o

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    # _fwd takes a single set of strides for these three, so keep them identical.
    lse = torch.zeros((bh, n, n), device=q.device, dtype=torch.float32)
    mx = torch.zeros_like(lse)
    dn = torch.zeros_like(lse)

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    # _fwd_pointer is the hook-free clone for traced/fake-tensor execution, so it is
    # only reachable when *no* descriptor was built. The mask now joins that vote.
    fwd_kernel = (
        _fwd
        if (can_use_tma or can_use_tma_bias or can_use_tma_mask)
        else _fwd_pointer
    )

    # fmt: off
    wrap_triton(fwd_kernel)[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse, mx, dn, lse.stride(0), lse.stride(1), lse.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        desc_b,
        desc_q, desc_k, desc_v, desc_o,
        desc_mask,
        neg_inf=MASK_FILL,
        sm_scale=sm_scale, N=n, H=h, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        USE_TMA=can_use_tma,
        USE_TMA_BIAS=can_use_tma_bias,
        USE_TMA_MASK=can_use_tma_mask,
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
    o = rearrange(o, "b h ... -> (b h) ...")
    mx = rearrange(mx, "b h ... -> (b h) ...")
    dn = rearrange(dn, "b h ... -> (b h) ...")
    do = rearrange(do, "b h ... -> (b h) ...")

    bh, _, n, dim = q.shape
    sm_scale = dim**-0.5

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    if USE_FUSED_BWD:
        # One kernel for all four gradients instead of three that each recompute the
        # score tile: 41.3 ms against 62.4 at n=1024, h=8, d=32, bf16. See
        # trifast/triton_bwd.py for why every tile is transposed to [k, j] and why db
        # and dq have to be atomic.
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq = torch.empty_like(q)
        dmask = torch.zeros_like(mask)

        # delta stays fp32.
        d = torch.empty((bh, n, n), dtype=torch.float32, device=q.device)

        # The two atomic accumulators, and the one place `torch.empty_like` would be
        # wrong: no store covers every element, the kernel only adds. The autotuner's
        # `reset_to_zero` does not help -- it fires during tuning, not on a
        # cached-config launch.
        #
        # db accumulates *transposed*, [bh, k, j], because ds is produced as [k, j] and
        # db's own layout is [bh, j, k]; a transposed atomic would scatter four bytes per
        # lane. dq accumulates in fp32 because a bf16 atomic would round once per k
        # block instead of once in total, and db was already the most
        # precision-sensitive of the five gradients.
        dbt = torch.zeros((bh, n, n), dtype=torch.float32, device=q.device)
        dq_acc = torch.zeros((bh, n, n, dim), dtype=torch.float32, device=q.device)

        # The fp32, inv_ln2-scaled, [bh, k, j] bias -- same orientation as dbt. This is a
        # *new* tensor, never a rebinding of `b`: the three-kernel path below still reads
        # the original input-dtype [bh, j, k] bias, and b.dtype is still needed for the db
        # cast. +33 MB at n=1024, against dq_acc's 1.07 GB.
        #
        # The last axis is padded to 16 elements (64 B) so every row starts 16-byte
        # aligned, which is what a TMA descriptor over this buffer will require.
        # `torch.zeros` rather than `empty` only so the pad is defined -- correctness does
        # not rest on it, `_bwd_j_block`'s `in_rangeT` is what bounds the reads.
        padded_n = triton.cdiv(n, 16) * 16
        b2t = torch.zeros((bh, n, padded_n), dtype=torch.float32, device=q.device)

        # delta = rowsum(o * do), and the bias pre-pass.
        # fmt: off
        wrap_triton(_bwd_preprocess)[(triton.cdiv(n, 64), n, bh)](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            d, d.stride(0), d.stride(1), d.stride(2),
            n, DIM=dim, BLOCK_J=64, num_warps=4,
        )
        wrap_triton(_bwd_bias_prep)[(triton.cdiv(n, 32), triton.cdiv(n, 32), bh)](
            b, b.stride(0), b.stride(1), b.stride(2),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            n, BLOCK=32, num_warps=4,
        )
        # fmt: on

        # Route the bias tile through TMA. The payoff is not registers -- taking the tile
        # out of the register file moves 255 -> ~239, nowhere near the <=170 that 3 CTAs/SM
        # needs. It is that the fp32 tile is otherwise staged to shared memory *from
        # registers*, which cost `LDS` 13.4 % of stall samples; TMA writes shared memory
        # directly from global instead.
        #
        # `_is_fake` because fake tensors cannot build a tensormap -- and that path also
        # launches the bare kernel with a pinned config, where no config pre_hook runs to
        # fix up the box. Both reasons point the same way: it keeps the pointer load.
        can_use_tma_bias = USE_TMA_BWD_BIAS and not _is_fake(b2t)
        if can_use_tma_bias:
            # Rank 2, folding (bh, k) into one row axis: a box is then [BLOCK_K, BLOCK_J]
            # with no rank-4 indexing, matching how _fwd views its own bias. The
            # block_shape here is a placeholder; _bwd_descriptor_pre_hook rewrites it to
            # the selected config. padded_n is already a multiple of 16 elements, so every
            # row satisfies TMA's 16-byte stride alignment without an F.pad.
            desc_b2t = TensorDescriptor.from_tensor(
                b2t.reshape(bh * n, padded_n), block_shape=[64, 64]
            )
        else:
            desc_b2t = b2t

        def fused_grid(x):
            return (triton.cdiv(n, x["BLOCK_K"]), n, bh)

        # Under tracing, take the kernel with a pinned config rather than the autotuner.
        # `_fwd_pointer` exists for the analogous reason (torch.compile rejects
        # autotuners carrying config hooks); here it is a correctness requirement, not a
        # compatibility one. See pinned_bwd_fused_config: a cold autotune cache inside a
        # compiled region leaves the db and dq accumulators ~400x too large, because the
        # benchmarking trials add into them and nothing zeroes them afterwards.
        if _is_fake(q):
            fused_kernel = _bwd_fused
            fused_cfg = pinned_bwd_fused_config(dim, q.dtype)
        else:
            fused_kernel = _bwd_fused_tuned
            fused_cfg = {}

        # fmt: off
        wrap_triton(fused_kernel)[fused_grid](
            d, d.stride(0), d.stride(1), d.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
            mask, mask.stride(0), mask.stride(1), mask.stride(2),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            desc_b2t,
            dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dbt, dbt.stride(0), dbt.stride(1), dbt.stride(2),
            dq_acc, dq_acc.stride(0), dq_acc.stride(1), dq_acc.stride(2), dq_acc.stride(3),
            sm_scale=sm_scale,
            neg_inf=MASK_FILL,
            N=n, H=h, DIM=dim,
            CLOSEST_N=CLOSEST_N,
            NEED_DB=True, NEED_DQ=True,
            USE_TMA_BIAS=can_use_tma_bias,
            **fused_cfg,
        )
        # fmt: on

        # dq = (dq_acc * sm_scale) in the input dtype. sm_scale is folded in here rather
        # than per tile in the kernel: 268 M multiplies instead of bh*n^3*DIM/BLOCK_K.
        numel = dq_acc.numel()
        wrap_triton(_bwd_scale_cast)[(triton.cdiv(numel, 4096),)](
            dq_acc, dq, sm_scale, numel, BLOCK=4096, num_warps=4
        )
        db = dbt.transpose(1, 2).contiguous().to(b.dtype)

        dq = rearrange(dq, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        dk = rearrange(dk, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        dv = rearrange(dv, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        db = rearrange(db, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        return dq, dk, dv, db, dmask

    # --- The original three-kernel path, kept reachable via USE_FUSED_BWD for A/B. ---

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
        b, b.stride(0), b.stride(1), b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
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
        b, b.stride(0), b.stride(1), b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
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
        b, b.stride(0), b.stride(1), b.stride(2),
        mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        db, db.stride(0), db.stride(1), db.stride(2),
        sm_scale=sm_scale,
        neg_inf=MASK_FILL,
        H=h, N=n, DIM=dim,
        CLOSEST_N=CLOSEST_N,
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
