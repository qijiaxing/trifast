"""Fused backward kernels.

The three original backward kernels (`_bwd_q`, `_bwd_kv`, `_bwd_b` in `triton.py`) each
recompute `scores = q·k^T * sm_scale + bias` and `dp = do·v^T`, rebuild `p` from
`(mx, dn)` and rebuild `ds` -- nine matmuls and three softmax epilogues where five and
one would do. `_bwd_fused` does all four gradients in one pass.

Two structural choices carry the speedup, both measured on an H20-3e at
n=1024, h=8, d=32, bf16:

1. **Every score tile is computed transposed, as `[k, j]`.** dV needs `p^T` and dK needs
   `ds^T`; in the `[j, k]` orientation those are `tl.trans` of a *computed fp32
   accumulator*, which lowers to a shared-memory round trip (`local_alloc` +
   `local_load`, four stmatrix and four ldmatrix, two extra barriers). That is what puts
   `_bwd_kv` at 196 registers and 12.45 % occupancy. Transposing a *loaded* tile instead
   is free -- it becomes `ttg.memdesc_trans`, pure metadata on the shared-memory operand
   descriptor that lands as the wgmma transpose bit, with zero ldmatrix/stmatrix and the
   same register count. Flipping the orientation alone takes the dk/dv work from
   `_bwd_kv`'s 30.6 ms to 26.0 ms at 211 registers.

2. **`db` and `dq` are accumulated with fp32 atomics.** `db` reduces over the triangle
   axis `i`, which this kernel carries in its grid, so a cross-CTA reduction is
   unavoidable -- that is the entire reason `_bwd_b` existed as a third kernel. `dq`
   reduces over `k`, which the grid splits. fp32 atomics sustain ~940 G updates/s here,
   close to plain-store speed, but only with `sem="relaxed"`: Triton's default is
   `acq_rel`, which is 2.4x slower (9.1 -> 22.1 ms on the db traffic alone).

`dq` is the one place where the transposed orientation loses, and it loses twice over.
`dq^T[d,j] = dot(trans(k), ds^T)` has M = DIM, and M < 64 is not selected for wgmma, so
at DIM=32 it demotes to `mma.sync` and costs 5.3 ms more end to end; at DIM=64 it faults
outright with an out-of-range shared address, because `k_blk` is already a wgmma operand
untransposed and asking for both of its layouts overruns the allocation. So dq pays one
accumulator transpose, `dot(trans(ds), k)`, and lands in its natural `[j, d]` layout.
"""

import triton
import triton.language as tl

from trifast.autotune import autotune
from trifast.autotune_bwd import _bwd_fused_configs, prune_bwd_fused_configs


# fmt: off
@triton.jit
def _bwd_preprocess(
    o_ptr, stride_oh, stride_om, stride_on, stride_od,      # INPUT [BH, N, N, D]
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod, # INPUT [BH, N, N, D]
    d_ptr, stride_dh, stride_dm, stride_dn,                 # OUTPUT [BH, N, N]
    N,
    DIM: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """delta[i, j] = sum_d o[i, j, d] * do[i, j, d], in fp32.

    Each program computes one dot product of [:D] * [:D]

    """
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    j_idxs = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)
    mask_j = j_idxs < N

    o_ptrs = (o_ptr + pid_h * stride_oh + pid_i * stride_om
              + j_idxs[:, None] * stride_on + d_idxs[None, :] * stride_od)
    do_ptrs = (do_ptr + pid_h * stride_doh + pid_i * stride_dom
               + j_idxs[:, None] * stride_don + d_idxs[None, :] * stride_dod)

    o_block = tl.load(o_ptrs, mask_j[:, None]).to(tl.float32)   # [D]
    do_block = tl.load(do_ptrs, mask_j[:, None]).to(tl.float32) # [D]
    delta = tl.sum(o_block * do_block, axis=1)  # [1], fp32

    d_ptrs = d_ptr + pid_h * stride_dh + pid_i * stride_dm + j_idxs * stride_dn
    tl.store(d_ptrs, delta, mask=mask_j)


@triton.jit
def _bwd_scale_cast(
    src_ptr, dst_ptr, scale, n_elem,
    BLOCK: tl.constexpr,
):
    """dst = (src * scale).to(dst.dtype), over flat contiguous tensors.

    Turns the fp32 dq accumulator into the bf16 gradient. `sm_scale` is folded in here
    rather than in `_bwd_fused` on purpose: applying it per partial tile would cost
    bh*n^3*DIM/BLOCK_K multiplies (4.3e9 at n=1024) against 268 M here, and the sum is
    linear so the result is the same up to fp32 rounding.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = off < n_elem
    val = tl.load(src_ptr + off, in_range).to(tl.float32) * scale
    tl.store(dst_ptr + off, val.to(dst_ptr.dtype.element_ty), in_range)


@triton.jit
def _bwd_j_block(
    # Loop-carried accumulators, returned updated.
    dk, dv,
    # Per-CTA tiles, hoisted out of the j loop by the caller.
    k_blk, v_blk, m_blk, mask_k,
    # Pointers, already advanced to start_j by the caller.
    q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
    j_idxs, start_j,
    sm_scale, neg_inf2, inv_ln2, N,
    input_dtype: tl.constexpr,
    DIM: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    NEED_DB: tl.constexpr, NEED_DQ: tl.constexpr,
    J_MASKED: tl.constexpr, K_EXACT: tl.constexpr,
):
    """One j-block of the backward, with every tile in `[k, j]` orientation.

    ``J_MASKED`` says whether this block can run past ``N``; the caller peels the loop so
    only the final partial block needs it. ``K_EXACT`` says this CTA's k tile is entirely
    in range. When neither holds, ``in_rangeT``, the ``[j]`` range compare and three
    ``[k, j]`` selects are provably no-ops, and ``RANGED`` drops all of them -- worth
    2.6 ms of 42.4 at n=1024. The backward gets more out of this than the forward did
    because it carries four selects per score element rather than two.
    """
    RANGED: tl.constexpr = J_MASKED or not K_EXACT

    if J_MASKED:
        mask_j = (j_idxs + start_j) < N                       # [j]
        if K_EXACT:
            in_rangeT = tl.broadcast_to(mask_j[None, :], (BLOCK_K, BLOCK_J))
        else:
            in_rangeT = mask_k[:, None] & mask_j[None, :]      # [k,j]
    elif not K_EXACT:
        # Every j is in range; only this CTA's ragged k rows still need killing, and
        # `mask_k` alone is what `in_rangeT` collapses to.
        in_rangeT = tl.broadcast_to(mask_k[:, None], (BLOCK_K, BLOCK_J))

    if J_MASKED:
        q_block = tl.load(q_ptrs, mask_j[:, None])             # [j,d]
        do_block = tl.load(do_ptrs, mask_j[:, None])           # [j,d]
        row_max = tl.load(mx_ptrs, mask_j)                     # [j]
        # other=1.0: a masked-out row must not divide by zero. With the `in_rangeT`
        # select on pT below this is belt and braces, but dropping either lets a NaN
        # reach the dv matmul, where NaN * 0 poisons the accumulator.
        row_denom = tl.load(dn_ptrs, mask_j, other=1.0)        # [j]
        delta = tl.load(d_ptrs, mask_j)                        # [j]
    else:
        q_block = tl.load(q_ptrs)
        do_block = tl.load(do_ptrs)
        row_max = tl.load(mx_ptrs)
        row_denom = tl.load(dn_ptrs)
        delta = tl.load(d_ptrs)
    # The bias tile is the one load that needs `in_rangeT` rather than `mask_j`: it is
    # indexed (j, k), so a ragged k tile reads past the end of the last bh slice.
    if RANGED:
        b_block = tl.load(bt_ptrs, in_rangeT).to(tl.float32)   # [k,j]
    else:
        b_block = tl.load(bt_ptrs).to(tl.float32)

    # dP^T does not depend on the score tile, so issue it first; scheduling it between
    # pT and dsT would put a third live [k,j] fp32 tile on the critical path.
    dpT = tl.dot(v_blk, tl.trans(do_block), input_precision="ieee")   # [k,j]

    scoresT = tl.dot(k_blk, tl.trans(q_block), input_precision="ieee")  # [k,j]
    scoresT = (scoresT * sm_scale + b_block) * inv_ln2
    # Substituting the sentinel after the log2 conversion, exactly as _fwd does. A
    # differently scaled sentinel makes exp2(scoresT - row_max) underflow on a fully
    # masked row and loses the documented mean(V) gradient.
    scoresT = tl.where(m_blk[:, None], neg_inf2, scoresT)
    if RANGED:
        scoresT = tl.where(in_rangeT, scoresT, neg_inf2)

    smT = tl.math.exp2(scoresT - row_max[None, :]) / row_denom[None, :]  # [k,j]
    if RANGED:
        smT = tl.where(in_rangeT, smT, 0.0)

    # dv wants the weights as the forward used them, masked keys included: on a fully
    # masked row those are the 1/N that produced o = mean(v).
    dv = tl.dot(smT.to(input_dtype), do_block, dv, input_precision="ieee")  # [k,d]

    dsT = smT * (dpT - delta[None, :])                         # [k,j]
    # d(score)/dk is zero wherever the forward *replaced* the score with the sentinel,
    # so this is the backward of _fwd's two selects. Masked keys do not drop out on
    # their own once a fully-masked row correctly gets smT = 1/N.
    dsT = tl.where(m_blk[:, None], 0.0, dsT)
    if RANGED:
        dsT = tl.where(in_rangeT, dsT, 0.0)

    # db is the fp32 dsT, before the bf16 cast that dk and dq take -- matching _bwd_b,
    # which accumulated fp32 and rounded once. It has to sit *after* the two selects
    # above, not between them and the cast: the `m_blk` zeroing is what keeps a fully
    # masked row's 1/N weights out of db, dk and dq, and it is the one guard here whose
    # absence is a wrong answer rather than a rounding.
    if NEED_DB:
        if RANGED:
            tl.atomic_add(dbt_ptrs, dsT, mask=in_rangeT, sem="relaxed")
        else:
            tl.atomic_add(dbt_ptrs, dsT, sem="relaxed")

    ds16 = dsT.to(input_dtype)                                 # [k,j]
    dk = tl.dot(ds16, q_block, dk, input_precision="ieee")     # [k,d]

    if NEED_DQ:
        # [j,k] @ [k,d] -> [j,d]. This is the one dot that does *not* want the
        # transposed orientation. `dq^T[d,j] = dot(trans(k_blk), ds16)` would avoid the
        # accumulator transpose, but its M is DIM, and M < 64 drops off wgmma onto
        # mma.sync: measured 5.3 ms slower end to end at DIM=32. At DIM=64 it is worse
        # than slow -- `k_blk` is already a wgmma operand untransposed, and asking for
        # both layouts of it faults with an out-of-range shared address
        # (compute-sanitizer, DIM=64, bf16). So pay the one transpose and get dq in its
        # natural layout, which also saves the epilogue a transpose.
        dq_tile = tl.dot(tl.trans(ds16), k_blk, input_precision="ieee")
        if J_MASKED:
            tl.atomic_add(
                dq_ptrs, dq_tile,
                mask=tl.broadcast_to(mask_j[:, None], (BLOCK_J, DIM)),
                sem="relaxed",
            )
        else:
            tl.atomic_add(dq_ptrs, dq_tile, sem="relaxed")

    return dk, dv


@triton.jit
def _bwd_j_loop(
    k_blk, v_blk, m_blk, mask_k,
    q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
    stride_qn, stride_don, stride_bm, stride_ln, stride_dn, stride_dbtj, stride_dqj,
    j_idxs,
    sm_scale, neg_inf2, inv_ln2, N,
    input_dtype: tl.constexpr,
    DIM: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    NEED_DB: tl.constexpr, NEED_DQ: tl.constexpr,
    K_EXACT: tl.constexpr,
):
    """The whole j loop for one k tile, peeled, returning (dk, dv).

    Instantiated twice by ``_bwd_fused`` on a uniform per-CTA branch over whether this
    CTA's k tile is ragged. ``K_EXACT`` cannot be decided on the host: it depends on
    ``BLOCK_K``, which the autotuner picks, and ``N`` is deliberately a runtime value so
    that varying crop sizes reuse one compiled kernel per ``CLOSEST_N`` bucket. Deciding
    it here is also strictly better than a host-side flag would be -- only the one ragged
    CTA pays, instead of every CTA whenever ``N % BLOCK_K != 0``.
    """
    dk = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # Peel the j loop: full blocks cannot run past N. `n_full` is a runtime value, so
    # this stays one kernel variant per CLOSEST_N bucket.
    n_full = (N // BLOCK_J) * BLOCK_J
    for start_j in tl.range(0, n_full, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        dk, dv = _bwd_j_block(
            dk, dv,
            k_blk, v_blk, m_blk, mask_k,
            q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
            j_idxs, start_j,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            J_MASKED=False, K_EXACT=K_EXACT,
        )
        q_ptrs += BLOCK_J * stride_qn
        do_ptrs += BLOCK_J * stride_don
        bt_ptrs += BLOCK_J * stride_bm
        mx_ptrs += BLOCK_J * stride_ln
        dn_ptrs += BLOCK_J * stride_ln
        d_ptrs += BLOCK_J * stride_dn
        if NEED_DB:
            dbt_ptrs += BLOCK_J * stride_dbtj
        if NEED_DQ:
            dq_ptrs += BLOCK_J * stride_dqj

    # The ragged tail, at most one block. The pointers were left pointing at it.
    if n_full < N:
        dk, dv = _bwd_j_block(
            dk, dv,
            k_blk, v_blk, m_blk, mask_k,
            q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
            j_idxs, n_full,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            J_MASKED=True, K_EXACT=K_EXACT,
        )
    return dk, dv


@triton.jit
def _bwd_fused(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    # OUTPUT
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    dbt_ptr, stride_dbth, stride_dbtk, stride_dbtj,
    dq_ptr, stride_dqh, stride_dqm, stride_dqj, stride_dqd,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    NEED_DB: tl.constexpr, NEED_DQ: tl.constexpr,
):
    """dq, dk, dv and db in one pass. Grid (cdiv(N, BLOCK_K), N, bh).

    pid0 is the k tile and varies fastest, so the k tiles of one `i` run together and
    share q/do/bias in L2. A CTA owns one k tile and walks j, keeping dk and dv in
    registers for a single plain store at the end -- those two stay exact and
    deterministic. dq and db are atomic; see the module docstring.

    `dbt_ptr` is a **transposed** fp32 accumulator, `[bh, k, j]`, because dsT is
    produced as `[k, j]` and db's own layout is `[bh, j, k]`; a transposed atomic would
    scatter 4 bytes per lane. The caller transposes and casts it afterwards. `dq_ptr` is
    an fp32 accumulator in dq's own `[bh, i, j, d]` layout, so the epilogue only has to
    scale and cast it.
    """
    input_dtype = q_ptr.dtype.element_ty

    # See _fwd: keep scalar args fp32 even when compiled launches bind them fp64.
    sm_scale = sm_scale.to(tl.float32)
    neg_inf = neg_inf.to(tl.float32)

    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    # The sentinel in log2 units; see the triton.py module docstring.
    neg_inf2 = neg_inf * inv_ln2

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_k = pid_k * BLOCK_K

    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)
    mask_k = k_idxs < N

    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)  # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd)  # [k,d]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd)  # [k,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)  # [j,d]

    # The bias tile, read transposed. The contiguous axis (k) is the tile's row axis, so
    # this coalesces into vector loads; a host-side pre-transposed copy measured no
    # faster and costs a 33 MB pass.
    bt_ptrs = (b_ptr + (start_h * stride_bh)
               + (j_idxs[None, :] * stride_bm) + (k_idxs[:, None] * stride_bn))  # [k,j]

    l_off = (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln)  # [j]
    mx_ptrs = mx_ptr + l_off
    dn_ptrs = dn_ptr + l_off

    d_ptrs = d_ptr + (start_h * stride_dh) + (start_i * stride_dm) + (j_idxs * stride_dn)  # [j]

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs = base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn)  # [k]

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)  # [k,d]

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd)  # [k,d]

    # Only build the accumulator pointer tensors that are actually used: a rank-2 64-bit
    # pointer tensor is 32 registers per thread on its own at [64, 32].
    if NEED_DB:
        dbt_ptrs = (dbt_ptr + (start_h * stride_dbth)
                    + (k_idxs[:, None] * stride_dbtk) + (j_idxs[None, :] * stride_dbtj))  # [k,j]
    else:
        dbt_ptrs = dbt_ptr
    if NEED_DQ:
        dq_ptrs = (dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
                   + (j_idxs[:, None] * stride_dqj) + (d_idxs[None, :] * stride_dqd))  # [j,d]
    else:
        dq_ptrs = dq_ptr

    # k, v and the key mask are loop-invariant for this CTA. The mask being hoisted is
    # what makes the forward's 32x LDG replication a non-issue here: it is one [k] load
    # per CTA, not one per iteration, and its [:, None] broadcast is split across warps
    # rather than replicated.
    k_blk = tl.load(k_ptrs, mask_k[:, None])  # [k,d]
    v_blk = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
    m_blk = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0  # [k]

    # Uniform per-CTA branch: everything but the last k tile is in range, and the
    # in-range instantiation drops the [k] compare, three [k,j] selects and the store
    # predicate. Worth 2.6 ms of 42.4 at n=1024 -- and n here is a multiple of BLOCK_K,
    # so the ragged side is dead code that only the odd crop size reaches.
    #
    # d(score)/dk is sm_scale * q, and k was not pre-scaled, so the factor lands on dk
    # here -- once per CTA, as in _bwd_kv. dq takes the same factor in _bwd_scale_cast.
    if start_k + BLOCK_K <= N:
        dk, dv = _bwd_j_loop(
            k_blk, v_blk, m_blk, mask_k,
            q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
            stride_qn, stride_don, stride_bm, stride_ln, stride_dn,
            stride_dbtj, stride_dqj,
            j_idxs,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            K_EXACT=True,
        )
        tl.store(dk_ptrs, (dk * sm_scale).to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))
    else:
        dk, dv = _bwd_j_loop(
            k_blk, v_blk, m_blk, mask_k,
            q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
            stride_qn, stride_don, stride_bm, stride_ln, stride_dn,
            stride_dbtj, stride_dqj,
            j_idxs,
            sm_scale, neg_inf2, inv_ln2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            K_EXACT=False,
        )
        tl.store(dk_ptrs, (dk * sm_scale).to(input_dtype), mask_k[:, None])
        tl.store(dv_ptrs, dv.to(input_dtype), mask_k[:, None])
# fmt: on


# The autotuned entry point. `_bwd_fused` itself stays a bare JITFunction so
# `scripts/proto_bwd_fused.py` can pin configs, the same split as
# `_fwd_pointer = autotune(...)(_fwd.fn)`.
#
# `reset_to_zero` covers the two atomic accumulators during tuning -- without it every
# candidate would add to the previous one's result. It does *not* fire on a
# cached-config launch (see autotune.py: the reset pre_hook runs only inside the
# cache-miss branch), so `torch.py` still has to allocate them with `torch.zeros`.
_bwd_fused_tuned = autotune(
    configs=_bwd_fused_configs,
    key=["H", "DIM", "CLOSEST_N"],
    prune_configs_by={"early_config_prune": prune_bwd_fused_configs},
    reset_to_zero=["dbt_ptr", "dq_ptr"],
)(_bwd_fused)
