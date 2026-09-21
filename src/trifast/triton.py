import triton
import triton.testing
import triton.language as tl
from trifast.autotune import autotune
from trifast.autotune_helpers import (
    _fwd_configs,
    _fwd_pointer_configs,
    prune_fwd_configs,
    _bwd_kv_configs,
    _bwd_q_configs,
    _bwd_b_configs,
)

# Every kernel evaluates scores in base-two units.
#
# The sentinel is finite (MASK_FILL converted to log2 units below), so a fully-masked
# row degenerates to uniform weights and produces mean(V) instead of NaNs.
#
# The mask is a torch bool tensor, which loads as uint8, so every `m_block` load ends in
# `!= 0`. Triton deprecates a non-boolean tl.where condition and will reject it in a future
# release.


# fmt: off
@triton.jit
def _fwd_kv_block(
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
        kt_block = desc_k.load([start_h, start_i, start_k, 0]).reshape(BLOCK_K, DIM).T  # [d,k]
    elif K_MASKED:
        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
    else:
        kt_block = tl.load(kt_ptrs)  # [d,k]
    if USE_TMA_BIAS:
        b_block = desc_b.load([start_h * N + start_j, start_k]).to(tl.float32)
    else:
        b_block = tl.load(b_ptrs, in_range).to(tl.float32)  # [j,k]

    # By TMA lands the row in shared memory once per CTA and broadcasts from there.
    if USE_TMA_MASK:
        # desc_mask is the widened mask flattened to [batch * N, padded_n], so a
        # row is (batch, i). Reusing N here rather than passing the mask's own
        # row count is deliberate -- the extra argument costs the whole speedup,
        # so torch.py instead gates the descriptor on the mask being n x n.
        mask_row = mask_start_h * N + start_i
        if BLOCK_K >= 64:
            # bf16 * BLOCK_K >= 128 bytes, the minimum this load tolerates.
            m_block = (desc_mask.load([mask_row, start_k]).reshape(BLOCK_K) != 0) # [k]
        else:
            # BLOCK_K=32 would give a 64-byte box, the one size that faults with
            # a misaligned address, so fetch two i rows (128 bytes) instead.
            pair = desc_mask.load([mask_row, start_k]).reshape(2, BLOCK_K)
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
        m_block = (tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0) # [k]
    else:
        m_block = (tl.load(mask_ptrs, cache_modifier=".cg") != 0) # [k]

    # P = Q [BLOCK_J, D] @ K^T [D, BLOCK_K]
    scores = tl.dot(q_block, kt_block, input_precision="ieee")  # [j,k]
    # Online Softmax
    scores = scores * sm_scale + b_block
    scores *= inv_ln2 # 1.0 / ln(2), [j,k]
    scores = tl.where(m_block[None, :], neg_inf2, scores)
    scores = tl.where(in_range, scores, neg_inf2)
    block_max = tl.maximum(scores_max, tl.max(scores, axis=1)) # [j]
    exp_scores = tl.math.exp2(scores - block_max[:, None])     # [j,k]
    if K_MASKED:
        # Columns past N must not reach sm_denom. They are already neg_inf2, which
        # underflows exp2 to 0 for any row that has a live key -- but a fully masked
        # row has block_max == neg_inf2, so exp2(0) == 1 and the padding would count.
        # That combination (ragged N + fully masked row) is what this select is for.
        exp_scores = tl.where(mask_k[None, :], exp_scores, 0.0)
    # (TODO) no need to update acc if scores_max == block_max
    exp_scale = tl.math.exp2(scores_max - block_max)
    sm_denom = sm_denom * exp_scale + tl.sum(exp_scores, axis=1)
    acc = acc * exp_scale[:, None]
    scores_max = block_max
    # Load V
    if USE_TMA:
        v_block = desc_v.load([start_h, start_i, start_k, 0]).reshape(BLOCK_K, DIM)  # [k,d]
    elif K_MASKED:
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
    else:
        v_block = tl.load(v_ptrs)  # [k,d]
    # P fp32 -> bf16
    exp_scores = exp_scores.to(input_dtype)  # [j,k]

    # O = P @ V
    acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]
    return acc, sm_denom, scores_max


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
    desc_b,
    desc_q, desc_k, desc_v, desc_o,
    # Widened mask, flattened to [batch * N, padded_n]; `mask_ptr` above stays for
    # the pointer fallback. Built in trifast.torch._triangle_attention.
    desc_mask,
    sm_scale,
    neg_inf,
    N,   # N is varing during training
    H,   # (TODO) Heads is constant
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

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32) # [j]
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    # The TMA path loads q/k/v through rank-4 descriptors over the natural
    # [bh, n, n, dim] layout, one (h, i) slice per box row. Rows beyond n are
    # clipped by the tensormap (zero-filled loads, no-op stores), so they
    # never wrap into the neighbouring i slice the way a flat 2D view would.
    # Clipped rows are the j >= N tail this kernel already treats as garbage:
    # `in_range` kills their scores and every store is masked or clipped.
    if USE_TMA:
        q_block = desc_q.load([start_h, start_i, start_j, 0]).reshape(BLOCK_J, DIM)
    else:
        q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]

    # Peel the k loop: full blocks cannot run past N, so they skip the range compare
    # and three [j,k] selects that are no-ops there. `n_full` is a runtime value, so
    # this stays one kernel variant per CLOSEST_N bucket -- no new specialization.
    n_full = (N // BLOCK_K) * BLOCK_K
    for start_k in tl.range(0, n_full, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        acc, sm_denom, scores_max = _fwd_kv_block(
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
        acc, sm_denom, scores_max = _fwd_kv_block(
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
    if USE_TMA:
        desc_o.store(
            [start_h, start_i, start_j, 0],
            final_output.reshape(1, 1, BLOCK_J, DIM),
        )
    else:
        tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    # Backward reconstructs probabilities as exp2(scores - mx) / dn.
    tl.store(mx_ptrs, scores_max, mask=mask_j)
    lse = (scores_max + tl.math.log2(sm_denom)) * ln2
    tl.store(lse_ptrs, lse, mask=mask_j)
    tl.store(dn_ptrs, sm_denom, mask=mask_j)
# fmt: on


_fwd_pointer = autotune(
    configs=_fwd_pointer_configs,
    key=["H", "DIM", "CLOSEST_N"],
    prune_configs_by={"early_config_prune": prune_fwd_configs},
    cache_name="_fwd_pointer",
)(_fwd.fn)


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
):
    input_dtype = q_ptr.dtype.element_ty

    # See _fwd: keep scalar args fp32 even when compiled launches bind them
    # fp64.
    sm_scale = sm_scale.to(tl.float32)
    neg_inf = neg_inf.to(tl.float32)

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
):
    input_dtype = q_ptr.dtype.element_ty

    # See _fwd: keep scalar args fp32 even when compiled launches bind them
    # fp64.
    sm_scale = sm_scale.to(tl.float32)
    neg_inf = neg_inf.to(tl.float32)

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
):
    input_dtype = q_ptr.dtype.element_ty
    BLOCK_I: tl.constexpr = 1

    # See _fwd: keep scalar args fp32 even when compiled launches bind them
    # fp64.
    sm_scale = sm_scale.to(tl.float32)
    neg_inf = neg_inf.to(tl.float32)

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
