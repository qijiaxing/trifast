"""Online-softmax forward with shape/stride specialization and fully written outputs.

The baseline forward already fuses QK, bias/masking, softmax and PV. This copy
preserves that algorithm, adds a measured 64x64 tile configuration, and fixes
scalar promotion in Inductor and padding loads. The original baseline stays unchanged.
"""

import math

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.library import triton_op, wrap_triton

from trifast.autotune import autotune
from trifast.torch import MASK_FILL

# This pointer kernel has its own measured candidates. Upstream TMA configs
# carry descriptor hooks and register caps that do not belong to this kernel.
_optimized_configs = [
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]


# fmt: off
@autotune(
    configs=_optimized_configs,
    # Equal power-of-two buckets can have different tail costs (e.g. 768/800).
    # Including N also gives old persisted bucket-only entries a distinct key.
    key=["H", "DIM", "CLOSEST_N", "N"],
)
@triton.jit
def _fwd_fused_optimized(
    o_ptr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    # lse/mx/dn are allocated side by side in torch.py -- same shape, dtype and layout
    # -- so one set of strides serves all three.
    lse_ptr, mx_ptr, dn_ptr, stride_lh: tl.constexpr, stride_lm: tl.constexpr, stride_ln: tl.constexpr,
    q_ptr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    k_ptr, stride_kh: tl.constexpr, stride_km: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    v_ptr, stride_vh: tl.constexpr, stride_vm: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    b_ptr, stride_bh: tl.constexpr, stride_bm: tl.constexpr, stride_bn: tl.constexpr,
    mask_ptr, stride_maskh: tl.constexpr, stride_maskm: tl.constexpr, stride_maskn: tl.constexpr,
    sm_scale: tl.constexpr,
    neg_inf: tl.constexpr,
    N: tl.constexpr, H: tl.constexpr, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    CENTERED: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty
    # Inductor can lower Python float arguments as fp64. Keep the entire
    # online softmax and its dot accumulator in fp32 in eager and compiled mode.
    scale_fp32 = tl.cast(sm_scale, tl.float32)
    sentinel_fp32 = tl.cast(neg_inf, tl.float32)

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1).to(tl.int64)  # Parallelize along i
    pid_h = tl.program_id(2).to(tl.int64)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2)

    # The sentinel in the same log2 units as the scores it replaces. Spelled identically
    # in all four kernels, so every one of them substitutes the same bits.
    neg_inf2 = sentinel_fp32 * inv_ln2

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

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    if N % BLOCK_J == 0:
        mask_j = tl.full((BLOCK_J,), True, tl.int1)
    else:
        mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None], other=0)  # [j,d]
    if CENTERED:
        # Remove a shared per-query offset before adding small dot products.
        # At bias ~ MASK_FILL, adding in the unshifted FP32 domain loses
        # significant score differences before softmax ever sees them.
        shift = tl.load(base_b_ptr + j_idxs * stride_bm, mask_j, other=0).to(tl.float32)
        centered_sentinel = (sentinel_fp32 - shift) * inv_ln2

    for start_k in tl.range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        if N % BLOCK_K == 0:
            mask_k = tl.full((BLOCK_K,), True, tl.int1)
        else:
            mask_k = (k_idxs + start_k) < N
        in_range = mask_j[:, None] & mask_k[None, :] # [j,k]

        kt_block = tl.load(kt_ptrs, mask_k[None, :], other=0)  # [d,k]
        b_block = tl.load(b_ptrs, in_range, other=0).to(tl.float32)  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, other=1, cache_modifier=".cg") != 0 # [k]

        scores = tl.dot(q_block, kt_block, input_precision="ieee")  # [j,k]
        if CENTERED:
            scores = scores * scale_fp32 + (b_block - shift[:, None])
        else:
            scores = scores * scale_fp32 + b_block
        scores *= inv_ln2 # 1.0 / ln(2), [j,k]
        # Real masked keys keep the finite replacement score. Padding keys
        # must never affect the maximum, even when all true scores < MASK_FILL.
        if CENTERED:
            scores = tl.where(m_block[None, :], centered_sentinel[:, None], scores)
        else:
            scores = tl.where(m_block[None, :], neg_inf2, scores)
        scores = tl.where(in_range, scores, -float("inf"))

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        exp_scores = tl.math.exp2(scores - block_max[:, None])  # [j,k]
        # Keep padding probabilities explicitly zero. Finite sentinel scores
        # belong only to real masked keys, including fully masked rows.
        exp_scores = tl.where(mask_k[None, :], exp_scores, 0.0)

        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + tl.sum(exp_scores, 1)  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None], other=0)  # [k,d]
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        scores_max = block_max

        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn


    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    # The backward recomputes each softmax weight as exp2(scores - mx) / dn. Storing the
    # running max and the denominator separately, rather than the single fused
    # lse = mx*ln2 + log(dn), is what makes that recomputation exact on a row where
    # every key is masked.
    #
    # There mx *is* the sentinel, and mx + log(dn) cannot hold both magnitudes in fp32:
    # at the old sentinel of finfo(fp32).min the log(dn) term rounds away entirely, and
    # even at a moderate -1e4 it survives to only ~4 significant digits. The backward
    # then recovers a weight of 1 instead of 1/N (or, as shipped, 0 -- the sentinel was
    # inserted after the log2 conversion, so this line's `* ln2` scaled it by 0.693 and
    # exp2 underflowed). mx and dn each carry one magnitude, so nothing cancels:
    # scores - mx is exactly 0 on such a row and dn is exactly N.
    # CENTERED mx is log2 of the score after subtracting bias[bh,j,0].
    # Backward must use the same centered domain; dn is unchanged by the shift.
    tl.store(mx_ptrs, scores_max, mask=mask_j)
    tl.store(dn_ptrs, sm_denom, mask=mask_j)

    # Natural-log logsumexp, for callers and diagnostics. Nothing reads it back; it is
    # the pair above that the backward consumes.
    lse = (scores_max * ln2) + tl.log(sm_denom)
    if CENTERED:
        lse += shift

    tl.store(lse_ptrs, lse, mask=mask_j)
# fmt: on


@triton_op("trifast::fused_attention_forward_optimized", mutates_args={})
def fused_forward_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (o, lse, mx, dn).

    For FP32 inputs, mx stores the row maximum in the centered log2 domain
    (the bias at key zero has been subtracted). Its backward must set
    centered_stats=True. BF16/FP16 retain the original unshifted mx domain.

    `lse` is the natural-log logsumexp, the same convention flex and protenix return,
    and is for callers and diagnostics only. The backward consumes `mx` (the softmax row
    max, in log2 units) and `dn` (the softmax denominator) instead -- see _fwd for why
    the unfused pair is what makes a fully-masked row's gradient exact.
    """
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

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    # Every valid element has a unique writer; no input is read from these buffers.
    o = torch.empty_like(q)
    lse = torch.empty((bh, n, n), device=q.device, dtype=torch.float32)
    mx, dn = torch.empty_like(lse), torch.empty_like(lse)

    CLOSEST_N = 2 ** math.ceil(math.log2(n))

    # fmt: off
    wrap_triton(_fwd_fused_optimized)[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse, mx, dn, lse.stride(0), lse.stride(1), lse.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        neg_inf=MASK_FILL,
        sm_scale=sm_scale, N=n, H=h, DIM=dim,
        CLOSEST_N=CLOSEST_N,
        CENTERED=(q.dtype == torch.float32),
    )


    o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    lse = rearrange(lse, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    mx = rearrange(mx, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dn = rearrange(dn, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return o, lse, mx, dn
