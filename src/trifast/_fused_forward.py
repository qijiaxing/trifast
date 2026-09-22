"""Experimental bias-only power-of-two physical padding; runtime logical N.

N stays runtime; each bucket/dtype/D/H configuration tunes a small tile set.
The persistent cache namespace is separate from shape-specialized forward.
"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.library import triton_op, wrap_triton

from trifast.autotune import autotune
from trifast.torch import MASK_FILL


# N is a runtime i64 scalar; its alignment and value-one cases do not
# specialize. Dense strides are derived from N and DIM inside the kernel.
# fmt: off
@triton.jit
def _fwd_kv_block(q_block, kt_ptrs, b_ptrs, mask_ptrs, v_ptrs,
                  mask_j, k_idxs, start_k, N, scores_max, sm_denom, acc,
                  shift, centered_sentinel, neg_inf2, scale_fp32,
                  CENTERED: tl.constexpr, K_MASKED: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    input_dtype: tl.constexpr = q_block.dtype
    inv_ln2: tl.constexpr = 1.4426950408889634
    if K_MASKED:
        mask_k = k_idxs + start_k < N
    else:
        mask_k = tl.full((BLOCK_K,), True, tl.int1)
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
    block_max = tl.where(mask_j, block_max, 0.)
    exp_scores = tl.math.exp2(scores - block_max[:, None])  # [j,k]
    # Keep padding probabilities explicitly zero. Finite sentinel scores
    # belong only to real masked keys, including fully masked rows.
    exp_scores = tl.where(in_range, exp_scores, 0.0)

    exp_scale = tl.where(mask_j, tl.math.exp2(scores_max - block_max), 0.)  # [j]

    sm_denom = sm_denom * exp_scale + tl.sum(exp_scores, 1)  # [j]

    acc = acc * exp_scale[:, None]  # [j,d]
    v_block = tl.load(v_ptrs, mask_k[:, None], other=0)  # [k,d]
    exp_scores = exp_scores.to(input_dtype)  # [j,k]

    acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

    scores_max = block_max

    return scores_max, sm_denom, acc


_bucket_forward_configs = [
    triton.Config({"BLOCK_J": bj, "BLOCK_K": bk}, num_warps=4, num_stages=stages)
    for bj, bk in ((32, 32), (64, 32), (64, 64))
    for stages in (1, 3)
]


def _prune_bucket_forward(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    # IEEE FP32 lowering at wide D has much higher staging requirements.
    # Dtype is explicit in the key, independent of tensor/fake-tensor handling.
    if args["DTYPE_ID"] == 2 and args["DIM"] >= 64:
        return [c for c in configs if c.kwargs["BLOCK_J"] == 32
                and c.kwargs["BLOCK_K"] == 32 and c.num_stages == 1]
    return configs


@autotune(
    configs=_bucket_forward_configs,
    key=["H", "DIM", "CLOSEST_N", "DTYPE_ID"],
    prune_configs_by={"early_config_prune": _prune_bucket_forward},
    cache_name="padded_bucket_v1",
)
@triton.jit(do_not_specialize=["N"])
def _fwd_fused_optimized(
    o_ptr, lse_ptr, mx_ptr, dn_ptr, q_ptr, k_ptr, v_ptr, b_ptr, mask_ptr,
    sm_scale: tl.constexpr, neg_inf: tl.constexpr,
    N: tl.int64, H: tl.constexpr, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr, DTYPE_ID: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    CENTERED: tl.constexpr = False,
):
    # The wrapper materializes dense layouts. Derive their canonical strides
    # here so contiguous channel/key axes stay visible to vectorization.
    # Size-one axes only index zero, so canonicalizing their strides is exact.
    stride_qh = N.to(tl.int64) * N * DIM
    stride_qm = N.to(tl.int64) * DIM
    stride_qn: tl.constexpr = DIM
    stride_qd: tl.constexpr = 1
    stride_kh, stride_km = stride_qh, stride_qm
    stride_kn: tl.constexpr = DIM
    stride_kd: tl.constexpr = 1
    stride_vh, stride_vm = stride_qh, stride_qm
    stride_vn: tl.constexpr = DIM
    stride_vd: tl.constexpr = 1
    stride_oh, stride_om = stride_qh, stride_qm
    stride_on: tl.constexpr = DIM
    stride_od: tl.constexpr = 1
    stride_lh, stride_lm = N.to(tl.int64) * N, N.to(tl.int64)
    stride_ln: tl.constexpr = 1
    stride_bh = N.to(tl.int64) * CLOSEST_N
    stride_bm: tl.constexpr = CLOSEST_N
    stride_bn: tl.constexpr = 1
    stride_maskh, stride_maskm = N.to(tl.int64) * N, N.to(tl.int64)
    stride_maskn: tl.constexpr = 1
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

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None], other=0)  # [j,d]
    if CENTERED:
        # Remove a shared per-query offset before adding small dot products.
        # At bias ~ MASK_FILL, adding in the unshifted FP32 domain loses
        # significant score differences before softmax ever sees them.
        shift = tl.load(base_b_ptr + j_idxs * stride_bm, mask_j, other=0).to(tl.float32)
        centered_sentinel = (sentinel_fp32 - shift) * inv_ln2

    if not CENTERED:
        shift = tl.full((BLOCK_J,), 0., tl.float32)
        centered_sentinel = tl.full((BLOCK_J,), 0., tl.float32)
    full_end = (N // BLOCK_K) * BLOCK_K
    for start_k in tl.range(0, full_end, BLOCK_K):
        scores_max, sm_denom, acc = _fwd_kv_block(
            q_block, kt_ptrs, b_ptrs, mask_ptrs, v_ptrs,
            mask_j, k_idxs, start_k, N, scores_max, sm_denom, acc,
            shift, centered_sentinel, neg_inf2, scale_fp32,
            CENTERED, False, BLOCK_K)
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn
    if full_end < N:
        scores_max, sm_denom, acc = _fwd_kv_block(
            q_block, kt_ptrs, b_ptrs, mask_ptrs, v_ptrs,
            mask_j, k_idxs, full_end, N, scores_max, sm_denom, acc,
            shift, centered_sentinel, neg_inf2, scale_fp32,
            CENTERED, True, BLOCK_K)

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


@triton_op("trifast::fused_attention_forward_padded_bucket", mutates_args={})
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

    closest_n = 1 << (n - 1).bit_length()
    # Only bias receives a bucket-wide physical pitch. Q/K/V and output stay
    # at logical N. Padding allocation/copy are part of each wrapper call.
    if closest_n != n:
        b = torch.nn.functional.pad(b, (0, closest_n - n))
    dtype_id = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[q.dtype]

    # fmt: off
    wrap_triton(_fwd_fused_optimized)[grid](
        o, lse, mx, dn, q, k, v, b, mask,
        neg_inf=MASK_FILL,
        sm_scale=sm_scale, N=n, H=h, DIM=dim,
        CLOSEST_N=closest_n, DTYPE_ID=dtype_id,
        CENTERED=(q.dtype == torch.float32),
    )


    o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    lse = rearrange(lse, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    mx = rearrange(mx, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dn = rearrange(dn, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return o, lse, mx, dn
