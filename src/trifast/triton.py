import triton
import triton.testing
import triton.language as tl
from trifast.autotune import autotune
from trifast.autotune_helpers import (
    _fwd_configs,
    prune_fwd_configs,
    _bwd_kv_configs,
    _bwd_q_configs,
    _bwd_b_configs,
)

# Every kernel evaluates scores in base-two units. For BF16, torch.py converts the
# comparatively small bias tensor once, outside the hot O(N^3) loop; forward and backward
# both receive that same rounded tensor. Scaling q or k instead would round before the dot
# and change the computed attention scores.
#
# The BF16 forward uses zero as a fixed softmax offset.
# This removes the
#  - block-max,
#  - accumulator-rescaling,
#  - running-max work
# from every K tile.
# The result is identical whenever the exponentials and denominator are finite.
# `_fwd_finalize` checks those denominators and reruns only exceptional tiles
# with the stable online recurrence.
# It also handles fully-masked rows,
# whose finite-sentinel semantics produce mean(V).
#
# The mask is a torch bool tensor, which loads as uint8, so every `m_block` load ends in
# `!= 0`. Triton deprecates a non-boolean tl.where condition and will reject it in a future
# release.


# fmt: off
@autotune(
    configs=_fwd_configs,
    key=["H", "DIM", "CLOSEST_N"],
    prune_configs_by={"early_config_prune": prune_fwd_configs},
)
@triton.jit
def _fwd(
    o_ptr, stride_oh, stride_om, stride_on, stride_od,
    # lse/mx/dn are allocated side by side in torch.py -- same shape, dtype and layout
    # -- so one set of strides serves all three.
    lse_ptr, mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mask_ptr, stride_maskh, stride_maskm, stride_maskn,
    sm_scale,
    neg_inf,
    N,   # N is varing during training
    H,   # (TODO) Heads is constant
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_FAST_PATH: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_h = tl.program_id(2)  # Parallelize along h

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

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    if USE_FAST_PATH:
        scores_max = tl.zeros([BLOCK_J], dtype=tl.float32)
    else:
        scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]

    for start_k in tl.range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N
        in_range = mask_j[:, None] & mask_k[None, :] # [j,k]

        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
        b_block = tl.load(b_ptrs,  in_range).to(tl.float32)  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0 # [k]

        # P = Q @ K
        scores = tl.dot(q_block, kt_block, input_precision="ieee")  # [j,k]
        if USE_FAST_PATH:
            # BF16 fast path:
            #   - the caller converts bias to base-two units once. The
            #   - Use zero as fixed softmax offset.
            scores = scores * (sm_scale * inv_ln2) + b_block
            exp_scores = tl.math.exp2(scores)
            exp_scores = tl.where(in_range & ~m_block[None, :], exp_scores, 0.0)
            sm_denom += tl.sum(exp_scores, 1)
        else:
            scores = scores * sm_scale + b_block
            scores *= inv_ln2 # 1.0 / ln(2), [j,k]
            scores = tl.where(m_block[None, :], neg_inf2, scores)
            scores = tl.where(in_range, scores, neg_inf2)
            block_max = tl.maximum(scores_max, tl.max(scores, 1))
            exp_scores = tl.math.exp2(scores - block_max[:, None])
            exp_scores = tl.where(mask_k[None, :], exp_scores, 0.0)
            # (TODO) no need to update acc if scores_max == block_max
            exp_scale = tl.math.exp2(scores_max - block_max)
            sm_denom = sm_denom * exp_scale + tl.sum(exp_scores, 1)
            acc = acc * exp_scale[:, None]
            scores_max = block_max
        # Load V
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
        # P fp32 -> bf16
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        # O = P @ V
        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn


    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    # Backward reconstructs probabilities as exp2(scores - mx) / dn. The BF16 fast
    # path uses zero as mx; the stable path uses its running maximum.
    if not USE_FAST_PATH:
        tl.store(mx_ptrs, scores_max, mask=mask_j)
        lse = (scores_max + tl.math.log2(sm_denom)) * ln2
        tl.store(lse_ptrs, lse, mask=mask_j)
    tl.store(dn_ptrs, sm_denom, mask=mask_j)

    # The fast path materializes LSE in the post-processing kernel.
# fmt: on


@triton.jit
def _fwd_finalize(
    o_ptr,
    stride_oh,
    stride_om,
    stride_on,
    stride_od,
    lse_ptr,
    mx_ptr,
    dn_ptr,
    stride_lh,
    stride_lm,
    stride_ln,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    mask_ptr,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    sm_scale,
    neg_inf,
    N,
    H,
    DIM: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_FAST_PATH: tl.constexpr = False,
):
    """Materialize LSE and stably recompute rows outside the BF16 fast path's range."""
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)
    batch = pid_h // H
    input_dtype = q_ptr.dtype.element_ty
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    neg_inf2 = neg_inf * inv_ln2

    j_idxs = tl.arange(0, BLOCK_J)
    k_idxs = tl.arange(0, BLOCK_K)
    d_idxs = tl.arange(0, DIM)

    for start_j in tl.range(0, N, BLOCK_J):
        mask_j = start_j + j_idxs < N
        stat_offset = (
            pid_h * stride_lh + pid_i * stride_lm + (start_j + j_idxs) * stride_ln
        )
        row_max = tl.load(mx_ptr + stat_offset, mask=mask_j)
        row_denom = tl.load(dn_ptr + stat_offset, mask=mask_j)
        invalid = (
            (row_denom == 0.0) | (row_denom == float("inf")) | (row_denom != row_denom)
        )

        if USE_FAST_PATH and tl.sum(invalid.to(tl.int32), axis=0) != 0:
            q_ptrs = (
                q_ptr
                + pid_h * stride_qh
                + pid_i * stride_qm
                + (start_j + j_idxs[:, None]) * stride_qn
                + d_idxs[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=mask_j[:, None])
            fixed_max = tl.full([BLOCK_J], -float("inf"), tl.float32)
            fixed_denom = tl.zeros([BLOCK_J], tl.float32)
            fixed_acc = tl.zeros([BLOCK_J, DIM], tl.float32)

            k_base = k_ptr + pid_h * stride_kh + pid_i * stride_km
            v_base = v_ptr + pid_h * stride_vh + pid_i * stride_vm
            b_base = b_ptr + pid_h * stride_bh
            mask_base = mask_ptr + batch * stride_maskh + pid_i * stride_maskm
            for start_k in tl.range(0, N, BLOCK_K):
                mask_k = start_k + k_idxs < N
                in_range = mask_j[:, None] & mask_k[None, :]
                k = tl.load(
                    k_base
                    + d_idxs[:, None] * stride_kd
                    + (start_k + k_idxs[None, :]) * stride_kn,
                    mask=mask_k[None, :],
                )
                b = tl.load(
                    b_base
                    + (start_j + j_idxs[:, None]) * stride_bm
                    + (start_k + k_idxs[None, :]) * stride_bn,
                    mask=in_range,
                ).to(tl.float32)
                key_mask = (
                    tl.load(mask_base + (start_k + k_idxs) * stride_maskn, mask=mask_k)
                    != 0
                )
                scores = tl.dot(q, k, input_precision="ieee") * (sm_scale * inv_ln2) + b
                scores = tl.where(key_mask[None, :], neg_inf2, scores)
                scores = tl.where(in_range, scores, neg_inf2)
                new_max = tl.maximum(fixed_max, tl.max(scores, axis=1))
                p = tl.math.exp2(scores - new_max[:, None])
                p = tl.where(mask_k[None, :], p, 0.0)
                alpha = tl.math.exp2(fixed_max - new_max)
                fixed_denom = fixed_denom * alpha + tl.sum(p, axis=1)
                fixed_acc *= alpha[:, None]
                v = tl.load(
                    v_base
                    + (start_k + k_idxs[:, None]) * stride_vn
                    + d_idxs[None, :] * stride_vd,
                    mask=mask_k[:, None],
                )
                fixed_acc = tl.dot(
                    p.to(input_dtype), v, fixed_acc, input_precision="ieee"
                )
                fixed_max = new_max

            o_ptrs = (
                o_ptr
                + pid_h * stride_oh
                + pid_i * stride_om
                + (start_j + j_idxs[:, None]) * stride_on
                + d_idxs[None, :] * stride_od
            )
            tl.store(
                o_ptrs,
                (fixed_acc / fixed_denom[:, None]).to(input_dtype),
                mask=mask_j[:, None] & invalid[:, None],
            )
            row_max = tl.where(invalid, fixed_max, row_max)
            row_denom = tl.where(invalid, fixed_denom, row_denom)
            tl.store(mx_ptr + stat_offset, row_max, mask=mask_j & invalid)
            tl.store(dn_ptr + stat_offset, row_denom, mask=mask_j & invalid)

        lse = (row_max + tl.math.log2(row_denom)) * ln2
        tl.store(lse_ptr + stat_offset, lse, mask=mask_j)


# fmt: off
@autotune(
    configs=_bwd_kv_configs,
    key=["H", "DIM", "CLOSEST_N"],
    reset_to_zero=["dk_ptr", "dv_ptr"],
)
@triton.jit
def _bwd_kv(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_FAST_PATH: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty

    # program id
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    # The sentinel in log2 units; see the module docstring.
    neg_inf2 = neg_inf * inv_ln2

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = 0 # we iterate over j, so each pid starts at 0
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)   # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None,:] * stride_vn)  # [d,k]

    l_off = (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln) # [j]
    mx_ptrs = mx_ptr + l_off
    dn_ptrs = dn_ptr + l_off

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs= base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn) # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd) # [k,d]

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd) # [k,d]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N

    # load k/v once per pid
    vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
    kt_block = tl.load(kt_ptrs, mask_k[None, :]) # [d,k]
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0 # [k]

    # accumulate over j for dk/dv
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        mask_j = (j_idxs + start_j) < N
        in_range = mask_j[:, None] & mask_k[None, :] # [j,k]

        q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
        b_block = tl.load(b_ptrs, in_range).to(tl.float32) # [j,k]

        scores = tl.dot(q_block, kt_block, input_precision="ieee") # [j,k]
        if USE_FAST_PATH:
            scores = scores * (sm_scale * inv_ln2) + b_block
        else:
            scores = (scores * sm_scale + b_block) * inv_ln2
        scores = tl.where(m_block[None, :], neg_inf2, scores)
        scores = tl.where(in_range, scores, neg_inf2)

        row_max = tl.load(mx_ptrs, mask=mask_j) # [j]
        row_denom = tl.load(dn_ptrs, mask=mask_j, other=1.0) # [j]
        sm_value = tl.math.exp2(scores - row_max[:, None]) / row_denom[:, None] # [j,k]
        sm_value = tl.where(in_range, sm_value, 0.0)

        do = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
        # dv wants the weights as the forward used them, masked keys included: on a
        # fully-masked row those are the 1/N that produced o = mean(v).
        dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee") # [k,d]

        delta = tl.load(d_ptrs, mask_j) # [j]

        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        # d(score)/dk is zero wherever the forward *replaced* the score with the
        # sentinel, so this is the backward of _fwd's two tl.where calls. Masked keys
        # used to drop out on their own because sm_value underflowed to 0 there; that
        # stops being true once a fully-masked row correctly gets sm_value = 1/N, and
        # without this dk would pick up a gradient the reference sends to zero.
        dscores = tl.where(m_block[None, :], 0.0, dscores)
        dscores = tl.where(in_range, dscores, 0.0)
        dscores = dscores.to(input_dtype) # [j,k]

        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee") # [k,d]

        # increment pointers
        q_ptrs += BLOCK_J * stride_qn
        d_ptrs += BLOCK_J * stride_dn
        b_ptrs += BLOCK_J * stride_bm
        mx_ptrs += BLOCK_J * stride_ln
        dn_ptrs += BLOCK_J * stride_ln
        do_ptrs += BLOCK_J * stride_don


    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None])
    tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])
# fmt: on


# fmt: off
@autotune(
    configs=_bwd_q_configs,
    key=["H", "DIM", "CLOSEST_N"],
    reset_to_zero=["dq_ptr", "d_ptr"],
)
@triton.jit
def _bwd_q(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    mask_ptr, stride_maskh, stride_maskm, stride_maskn,
    o_ptr, stride_oh, stride_om, stride_on, stride_od,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dq_ptr, stride_dqh, stride_dqm, stride_dqn, stride_dqd,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,  BLOCK_K: tl.constexpr,
    USE_FAST_PATH: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)

    # The sentinel in log2 units; see the module docstring.
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

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd) # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr  + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn) # [d,k]

    l_off = (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln) # [j]
    mx_ptrs = mx_ptr + l_off
    dn_ptrs = dn_ptr + l_off

    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)
    mask_ptrs= base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn) # [k]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    base_dq_ptr = dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
    dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs[None, :] * stride_dqd) # [j,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
    row_max = tl.load(mx_ptrs, mask_j) # [j]
    row_denom = tl.load(dn_ptrs, mask_j, other=1.0) # [j]
    do_block = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
    o_block = tl.load(o_ptrs, mask_j[:, None]) # [j,d]

    # delta feeds the cancellation-prone (dsm_value - delta) -- the softmax jacobian
    # makes sum_k of that identically zero -- so both the product and the store stay in
    # fp32. It used to be rounded to the input dtype on the way to _bwd_kv and _bwd_b.
    delta = tl.sum(o_block.to(tl.float32) * do_block.to(tl.float32), axis=1) # [j]

    tl.store(d_ptrs, delta, mask=mask_j)

    dq_block = tl.zeros([BLOCK_J, DIM], dtype=tl.float32)

    # iterte over k for dq = \sum_{k} ds_{jk} k_{k}
    for start_k in range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N
        in_range = mask_j[:, None] & mask_k[None, :] # [j,k]

        b_block = tl.load(b_ptrs, in_range).to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0 # [k]
        k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]

        scores = tl.dot(q_block, tl.trans(k_block), input_precision="ieee") # [j,k]
        if USE_FAST_PATH:
            scores = scores * (sm_scale * inv_ln2) + b_block
        else:
            scores = (scores * sm_scale + b_block) * inv_ln2
        scores = tl.where(m_block[None, :], neg_inf2, scores)  # [j,k]
        scores = tl.where(in_range, scores, neg_inf2)

        sm_value = tl.math.exp2(scores - row_max[:, None]) / row_denom[:, None] # [j,k]
        sm_value = tl.where(in_range, sm_value, 0.0)

        vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
        dsm_value = tl.dot(do_block, vt_block, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        # Backward of _fwd's two tl.where calls; see _bwd_kv for why it is needed.
        dscores = tl.where(m_block[None, :], 0.0, dscores)
        dscores = tl.where(in_range, dscores, 0.0)
        dscores = dscores.to(input_dtype) # [j,k]

        dq_block += tl.dot(dscores, k_block, input_precision="ieee")

        k_ptrs += BLOCK_K * stride_kn
        vt_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    # d(score)/dq is sm_scale * k. k_block is no longer pre-scaled, so the factor lands
    # here instead -- the same shape as _bwd_kv's dk_block *= sm_scale.
    dq_block *= sm_scale
    tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])
# fmt: on


# fmt: off
@autotune(
    configs=_bwd_b_configs,
    key=["H", "DIM", "CLOSEST_N"],
    reset_to_zero=["db_ptr"],
)
@triton.jit
def _bwd_b(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    db_ptr, stride_dbh, stride_dbm, stride_dbn,
    sm_scale,
    neg_inf,
    H, N, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_FAST_PATH: tl.constexpr = False,
):
    input_dtype = q_ptr.dtype.element_ty
    BLOCK_I: tl.constexpr = 1

    # program id
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    # The sentinel in log2 units; see the module docstring.
    neg_inf2 = neg_inf * inv_ln2

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = 0 # we iterate over j, so each pid starts at 0
    start_j = pid_j * BLOCK_J
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd  # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]

    l_off = (start_h * stride_lh) + (j_idxs * stride_ln) # [j]
    mx_ptrs = mx_ptr + l_off
    dn_ptrs = dn_ptr + l_off

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs= base_mask_ptr + (k_idxs * stride_mn) # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_db_ptr = db_ptr + (pid_h * stride_dbh)
    db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)

    base_d_ptr = d_ptr + (start_h * stride_dh)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N
    mask_j = j_idxs < N
    in_range = mask_j[:, None] & mask_k[None, :] # [j,k]

    db_block = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)

    # loop over i
    for start_i in range(0, N, BLOCK_I):
        start_i = tl.multiple_of(start_i, BLOCK_I)
        q_block = tl.load(q_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        k_block = tl.load(k_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]

        b_block = tl.load(b_ptrs, in_range, cache_modifier=".cg").to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0 # [k]

        scores = tl.dot(q_block, tl.trans(k_block), input_precision="ieee") # [j,k]
        if USE_FAST_PATH:
            scores = scores * (sm_scale * inv_ln2) + b_block
        else:
            scores = (scores * sm_scale + b_block) * inv_ln2
        scores = tl.where(m_block[None, :], neg_inf2, scores)
        scores = tl.where(in_range, scores, neg_inf2)

        row_max = tl.load(mx_ptrs, mask=mask_j, cache_modifier=".cg") # [j]
        row_denom = tl.load(dn_ptrs, mask=mask_j, other=1.0, cache_modifier=".cg") # [j]
        sm_score = tl.math.exp2(scores - row_max[:, None]) / row_denom[:, None] # [j,k]
        sm_score = tl.where(in_range, sm_score, 0.0)

        do = tl.load(do_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        delta = tl.load(d_ptrs, mask_j, cache_modifier=".cg") # [j]

        v_block = tl.load(v_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]
        dsm_value = tl.dot(do, tl.trans(v_block), input_precision="ieee") # [j,k]

        dscores = sm_score * (dsm_value - delta[:, None]) # [j,k]
        # Backward of _fwd's two tl.where calls; see _bwd_kv for why it is needed.
        dscores = tl.where(m_block[None, :], 0.0, dscores)
        dscores = tl.where(in_range, dscores, 0.0)

        db_block += dscores

        # increment pointers
        q_ptrs += stride_qm * BLOCK_I
        k_ptrs += stride_km * BLOCK_I
        v_ptrs += stride_vm * BLOCK_I
        mx_ptrs += stride_lm * BLOCK_I
        dn_ptrs += stride_lm * BLOCK_I
        mask_ptrs += stride_mm * BLOCK_I
        d_ptrs += stride_dm * BLOCK_I
        do_ptrs += stride_dom * BLOCK_I

    tl.store(db_ptrs, db_block.to(input_dtype), mask=in_range)
# fmt: on
