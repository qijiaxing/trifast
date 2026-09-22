"""Bucketed BF16/FP16 forward retaining upstream TMA transfers.

Logical N and its dependent pitches remain runtime i64 scalars. The returned
mx/dn use the uncentered log2 convention of the fused low-precision backward.
FP32 callers must use the centered pointer forward instead. This module copies
upstream's forward mechanics without changing the baseline implementation.
"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.library import triton_op, wrap_triton
from triton.tools.tensor_descriptor import TensorDescriptor

from trifast.autotune import autotune
from trifast.autotune_helpers import (
    _fwd_configs,
    _fwd_descriptor_pre_hook,
    _fwd_pointer_configs,
    prune_fwd_configs,
)
from trifast.torch import MASK_FILL

USE_TMA = True
USE_TMA_BIAS = True
USE_TMA_MASK = True


# Keep this pool independent of the upstream baseline and pointer fallback.
# Clone the upstream pool (eight standard configs plus optional FORCE_TUNE
# entries), preserving descriptor setup hooks.
_fused_tma_configs = [
    triton.Config(
        kwargs=dict(config.kwargs),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        maxnreg=config.maxnreg,
        pre_hook=config.pre_hook,
    )
    for config in _fwd_configs
]
_d32_register_capped_config = triton.Config(
    {"BLOCK_J": 64, "BLOCK_K": 64},
    num_warps=4,
    num_stages=2,
    maxnreg=96,
    pre_hook=_fwd_descriptor_pre_hook,
)
_fused_tma_configs.append(_d32_register_capped_config)


def _prune_fused_tma_configs(configs, named_args, **kwargs):
    configs = prune_fwd_configs(configs, named_args, **kwargs)
    arguments = {**named_args, **kwargs}
    if arguments["DIM"] != 32:
        configs = [
            config for config in configs if config is not _d32_register_capped_config
        ]
    return configs

_RUNTIME_ARGS = [
    "N",
    "stride_oh",
    "stride_om",
    "stride_lh",
    "stride_lm",
    "stride_qh",
    "stride_qm",
    "stride_kh",
    "stride_km",
    "stride_vh",
    "stride_vm",
    "stride_bh",
    "stride_bm",
    "stride_maskh",
    "stride_maskm",
]


# fmt: off
@triton.jit
def _tma_kv_block(
    # Loop-carried online-softmax state, returned updated.
    acc, sm_denom, scores_max,
    q_block,
    # Pointer-path pointers, already advanced to start_k by the caller.
    kt_ptrs, b_ptrs, v_ptrs, mask_ptrs,
    desc_b, desc_k, desc_v, desc_mask,
    mask_j, k_idxs,
    start_h, start_i, start_j, start_k, mask_start_h,
    sm_scale, neg_inf2, inv_ln2, N,
    input_dtype: tl.constexpr,
    DIM: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TMA: tl.constexpr, USE_TMA_BIAS: tl.constexpr, USE_TMA_MASK: tl.constexpr,
    K_MASKED: tl.constexpr,
):
    """One k-block of the online softmax.

    ``K_MASKED`` says whether this block can run past ``N``. The caller peels the
    loop so that only the final partial block needs it, which lets every full block
    skip three [j,k]-sized selects and the [k] range compare -- pure waste whenever
    ``k + BLOCK_K <= N``, i.e. always when ``N % BLOCK_K == 0``.
    """
    if K_MASKED:
        mask_k = (k_idxs + start_k) < N
        in_range = mask_j[:, None] & mask_k[None, :] # [j,k]
    else:
        # Every column is in range; only the j >= N rows still need killing, and
        # `mask_j` alone is what `in_range` collapses to.
        in_range = tl.broadcast_to(mask_j[:, None], (BLOCK_J, BLOCK_K))

    if USE_TMA:
        kt_block = desc_k.load([start_h.to(tl.int32), start_i.to(tl.int32), start_k.to(tl.int32), 0]).reshape(BLOCK_K, DIM).T  # [d,k]
    elif K_MASKED:
        kt_block = tl.load(kt_ptrs, mask_k[None, :], other=0)  # [d,k]
    else:
        kt_block = tl.load(kt_ptrs)  # [d,k]
    if USE_TMA_BIAS:
        b_block = desc_b.load([(start_h * N + start_j).to(tl.int32), start_k.to(tl.int32)]).to(tl.float32)
    else:
        b_block = tl.load(b_ptrs, in_range, other=0).to(tl.float32)  # [j,k]

    # By TMA lands the row in shared memory once per CTA and broadcasts from there.
    if USE_TMA_MASK:
        # desc_mask is the widened mask flattened to [batch * N, padded_n], so a
        # row is (batch, i). Reusing N here rather than passing the mask's own
        # row count is deliberate -- the extra argument costs the whole speedup.
        # torch.py's shape checks guarantee the mask really is n x n, so N is right.
        mask_row = mask_start_h * N + start_i
        if BLOCK_K >= 64:
            # bf16 * BLOCK_K >= 128 bytes, the minimum this load tolerates.
            m_block = (desc_mask.load([mask_row.to(tl.int32), start_k.to(tl.int32)]).reshape(BLOCK_K) != 0) # [k]
        else:
            # BLOCK_K=32 would give a 64-byte box, the one size that faults with
            # a misaligned address, so fetch two i rows (128 bytes) instead.
            pair = desc_mask.load([mask_row.to(tl.int32), start_k.to(tl.int32)]).reshape(2, BLOCK_K)
            # Keep row 0 and discard row 1 (the next i). Masking then summing
            # over axis 0 is how to select a row without a dynamic slice, and
            # costs ~2 ops per lane on a tensor this small.
            m_block = tl.sum(
                tl.where(tl.arange(0, 2)[:, None] == 0, pair, 0.0), axis=0
            ) != 0
    elif K_MASKED:
        # Load the [k] key mask.
        # a rank-1 [k] tensor feeding a [j,k] broadcast is assigned slice<dim=0, parent=#mma>,
        # a layout *replicated* along the projected j dimension
        # -- all four warps hold the same columns,
        # and eight lanes within a warp share each column.
        # So the pointer path below issues eight 2-byte LDGs per warp per iteration.
        m_block = (tl.load(mask_ptrs, mask_k, other=1, cache_modifier=".cg") != 0) # [k]
    else:
        m_block = (tl.load(mask_ptrs, cache_modifier=".cg") != 0) # [k]

    # P = Q [BLOCK_J, D] @ K^T [D, BLOCK_K]
    scores = tl.dot(q_block, kt_block, input_precision="ieee")  # [j,k]
    # Online Softmax
    scores = scores * sm_scale + b_block
    scores *= inv_ln2 # 1.0 / ln(2), [j,k]
    scores = tl.where(m_block[None, :], neg_inf2, scores)
    # Only key padding affects a valid query's reduction. Invalid query rows
    # are independent and discarded by the masked/TMA output stores, so avoid
    # carrying their predicate through every softmax operation in full blocks.
    if K_MASKED:
        # Padding is not a real masked key; finite sentinel padding could raise
        # the maximum above extremely negative valid scores and cause underflow.
        scores = tl.where(mask_k[None, :], scores, -float("inf"))
    block_max = tl.maximum(scores_max, tl.max(scores, axis=1)) # [j]
    exp_scores = tl.math.exp2(scores - block_max[:, None])     # [j,k]
    if K_MASKED:
        exp_scores = tl.where(mask_k[None, :], exp_scores, 0.0)
    exp_scale = tl.math.exp2(scores_max - block_max)
    sm_denom = sm_denom * exp_scale + tl.sum(exp_scores, axis=1)
    acc = acc * exp_scale[:, None]
    scores_max = block_max
    # Load V
    if USE_TMA:
        v_block = desc_v.load([start_h.to(tl.int32), start_i.to(tl.int32), start_k.to(tl.int32), 0]).reshape(BLOCK_K, DIM)  # [k,d]
    elif K_MASKED:
        v_block = tl.load(v_ptrs, mask_k[:, None], other=0)  # [k,d]
    else:
        v_block = tl.load(v_ptrs)  # [k,d]
    # P fp32 -> bf16
    exp_scores = exp_scores.to(input_dtype)  # [j,k]

    # O = P @ V
    acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]
    return acc, sm_denom, scores_max


@autotune(
    configs=_fused_tma_configs,
    key=["H", "DIM", "CLOSEST_N"],
    prune_configs_by={"early_config_prune": _prune_fused_tma_configs},
    cache_name="fused_tma_runtime_bucket_vector_store_v4",
)
@triton.jit(do_not_specialize=_RUNTIME_ARGS)
def _fused_tma(
    o_ptr, stride_oh: tl.int64, stride_om: tl.int64, stride_on: tl.constexpr, stride_od: tl.constexpr,
    # lse/mx/dn are allocated side by side in torch.py -- same shape, dtype and layout
    # -- so one set of strides serves all three.
    lse_ptr, mx_ptr, dn_ptr, stride_lh: tl.int64, stride_lm: tl.int64, stride_ln: tl.constexpr,
    q_ptr, stride_qh: tl.int64, stride_qm: tl.int64, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    k_ptr, stride_kh: tl.int64, stride_km: tl.int64, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    v_ptr, stride_vh: tl.int64, stride_vm: tl.int64, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    b_ptr, stride_bh: tl.int64, stride_bm: tl.int64, stride_bn: tl.constexpr,
    mask_ptr, stride_maskh: tl.int64, stride_maskm: tl.int64, stride_maskn: tl.constexpr,
    desc_b,
    desc_q, desc_k, desc_v,
    # Widened mask, flattened to [batch * N, padded_n]; `mask_ptr` above stays for
    # the pointer fallback. Built in trifast.torch._triangle_attention.
    desc_mask,
    sm_scale,
    neg_inf,
    N: tl.int64,   # N is varing during training
    H: tl.constexpr,   # (TODO) Heads is constant
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TMA: tl.constexpr = False,
    USE_TMA_BIAS: tl.constexpr = False,
    USE_TMA_MASK: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty

    # Eager launches type Python-float args as fp32, but torch.compile's triton
    # integration binds them fp64, and an fp64 sentinel/scale promotes the
    # whole score chain (and the accumulator) to fp64. Downcast once so both
    # paths are numerically identical; in eager this is a no-op.
    sm_scale = sm_scale.to(tl.float32)
    neg_inf = neg_inf.to(tl.float32)

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1).to(tl.int64)  # Parallelize along i
    pid_h = tl.program_id(2).to(tl.int64)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2)

    # The sentinel in the same log2 units as the scores it replaces. Spelled identically
    # in all four kernels, so every one of them substitutes the same bits.
    neg_inf2 = neg_inf * inv_ln2

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0 # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_kt_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_kt_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn) # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]

    l_off = (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln) # [j]
    lse_ptrs = lse_ptr + l_off
    mx_ptrs = mx_ptr + l_off
    dn_ptrs = dn_ptr + l_off

    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)
    mask_ptrs= base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn) # [k]

    base_o_ptr = o_ptr + ((start_h * N + start_i) * N * DIM)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32) # [j]
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    if USE_TMA:
        # The TMA path loads q/k/v through rank-4 descriptors over the natural
        # [bh, n, n, dim] layout, one (h, i) slice per box row.
        q_block = desc_q.load([start_h.to(tl.int32), start_i.to(tl.int32), start_j.to(tl.int32), 0]).reshape(BLOCK_J, DIM) # [j,d]
    else:
        q_block = tl.load(q_ptrs, mask_j[:, None], other=0)  # [j,d]

    # Peel the k loop: full blocks cannot run past N, so they skip the range compare
    # and three [j,k] selects that are no-ops there. `n_full` is a runtime value, so
    # this stays one kernel variant per CLOSEST_N bucket -- no new specialization.
    n_full = (N // BLOCK_K) * BLOCK_K
    for start_k in tl.range(0, n_full, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        acc, sm_denom, scores_max = _tma_kv_block(
            acc, sm_denom, scores_max, q_block,
            kt_ptrs, b_ptrs, v_ptrs, mask_ptrs,
            desc_b, desc_k, desc_v, desc_mask,
            mask_j, k_idxs,
            start_h, start_i, start_j, start_k, mask_start_h,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            USE_TMA, USE_TMA_BIAS, USE_TMA_MASK,
            K_MASKED=False,
        )
        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    # The ragged tail, at most one block, and only when N is not a multiple of
    # BLOCK_K. The pointers were left pointing at it by the loop above.
    if n_full < N:
        acc, sm_denom, scores_max = _tma_kv_block(
            acc, sm_denom, scores_max, q_block,
            kt_ptrs, b_ptrs, v_ptrs, mask_ptrs,
            desc_b, desc_k, desc_v, desc_mask,
            mask_j, k_idxs,
            start_h, start_i, start_j, n_full, mask_start_h,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            USE_TMA, USE_TMA_BIAS, USE_TMA_MASK,
            K_MASKED=True,
        )


    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    # A regular vector store keeps output initialization visible to initcheck.
    # TMA reads retain the upstream data-movement optimization.
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    # Backward reconstructs probabilities as exp2(scores - mx) / dn.
    tl.store(mx_ptrs, scores_max, mask=mask_j)
    lse = (scores_max + tl.math.log2(sm_denom)) * ln2
    tl.store(lse_ptrs, lse, mask=mask_j)
    tl.store(dn_ptrs, sm_denom, mask=mask_j)
# fmt: on


_fused_tma_pointer = autotune(
    configs=_fwd_pointer_configs,
    key=["H", "DIM", "CLOSEST_N"],
    prune_configs_by={"early_config_prune": prune_fwd_configs},
    cache_name="fused_tma_pointer_runtime_bucket_vector_store_v3",
)(_fused_tma.fn)


@triton_op("trifast::fused_forward_tma_bucket", mutates_args={})
def fused_forward_tma(
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
        lambda: (
            "q/k/v must all be [batch, heads, n, n, dim]; got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        ),
    )
    torch._check(
        q.shape[2] == q.shape[3],
        lambda: (
            "q/k/v are [batch, heads, n, n, dim], so dims 2 and 3 must match; "
            f"got {tuple(q.shape)} -- is the head axis in the wrong position?"
        ),
    )
    torch._check(
        b.ndim == 4 and b.shape == q.shape[:4],
        lambda: (
            f"bias must be [batch, heads, n, n] = {tuple(q.shape[:4])}; "
            f"got {tuple(b.shape)}"
        ),
    )
    torch._check(
        mask.ndim == 3 and mask.shape == (q.shape[0], q.shape[2], q.shape[3]),
        lambda: (
            "mask must be [batch, n, n] = "
            f"{(q.shape[0], q.shape[2], q.shape[3])}; got {tuple(mask.shape)}"
        ),
    )

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "TMA fused forward supports BF16/FP16 only; FP32 requires centered statistics"
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
    # Descriptor coordinates are signed i32 even though pointer arithmetic and
    # logical strides are i64. Bias/mask flatten batch/head with a logical row.
    # Leave room for the largest descriptor tile and pitch alignment; fall back
    # to wide pointer addressing rather than truncating a large row coordinate.
    descriptor_i32_limit = (1 << 31) - 1 - 256
    descriptor_coords_safe = (
        bh * n <= descriptor_i32_limit and bs * n <= descriptor_i32_limit
    )
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
    _is_fake = lambda t: type(t).__name__ in {"FakeTensor", "FunctionalTensor"}
    # TMA needs 16-byte-aligned global strides; a contiguous [*, dim] inner
    # layout gives dim * element_size bytes per row.
    can_use_tma = (
        USE_TMA
        and descriptor_coords_safe
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
        USE_TMA_BIAS and descriptor_coords_safe and dim <= 64 and not _is_fake(b)
    )
    # Fake tensors cannot build a tensormap, as for q/k/v above. Nothing else is
    # needed: the flat [batch * n, padded_n] view built below addresses rows as
    # `batch * N + i`, which requires the mask to really be n x n, but the
    # torch._check calls at the top of this function already guarantee that
    # (mask.shape == (batch, q.shape[2], q.shape[3]) and q.shape[2] == q.shape[3],
    # and n is q.shape[3]). Reusing N for that row index rather than passing the
    # mask's own row count is deliberate -- the extra kernel argument measures ~2%
    # slower.
    can_use_tma_mask = USE_TMA_MASK and descriptor_coords_safe and not _is_fake(mask)
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
        wide_mask = mask.to(MASK_TMA_DTYPE)
        if padded_mask_n != n:
            wide_mask = torch.nn.functional.pad(wide_mask, (0, padded_mask_n - n))
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
        padded_b = b
        if padded_n != n:
            padded_b = torch.nn.functional.pad(b, (0, padded_n - n))
        # The block_shape is a placeholder; _fwd_descriptor_pre_hook rewrites it
        # to [BLOCK_J, BLOCK_K] of the selected autotune config.
        desc_b = TensorDescriptor.from_tensor(
            padded_b.reshape(bh * n, padded_n), block_shape=[64, 32]
        )
    else:
        desc_b = b

    o = torch.empty_like(q)
    if can_use_tma:
        # Rank-4 descriptors over the natural [bh, n, n, dim] layout. Boxes are
        # [1, 1, BLOCK_*, DIM]; the placeholder block_shape is rewritten by the
        # config pre-hook. The rank-4 box keeps each tile inside one (h, i)
        # slice, so rows >= n clip instead of wrapping into the next slice.
        desc_q = TensorDescriptor.from_tensor(q, block_shape=[1, 1, 64, 32])
        desc_k = TensorDescriptor.from_tensor(k, block_shape=[1, 1, 64, 32])
        desc_v = TensorDescriptor.from_tensor(v, block_shape=[1, 1, 64, 32])
    else:
        desc_q, desc_k, desc_v = q, k, v

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    # _fwd takes a single set of strides for these three, so keep them identical.
    lse = torch.empty((bh, n, n), device=q.device, dtype=torch.float32)
    mx = torch.empty_like(lse)
    dn = torch.empty_like(lse)

    CLOSEST_N = 1 << (n - 1).bit_length()

    # _fwd_pointer is the hook-free clone for traced/fake-tensor execution, so it is
    # only reachable when *no* descriptor was built. The mask now joins that vote.
    fwd_kernel = (
        _fused_tma
        if (can_use_tma or can_use_tma_bias or can_use_tma_mask)
        else _fused_tma_pointer
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
        desc_q, desc_k, desc_v,
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
