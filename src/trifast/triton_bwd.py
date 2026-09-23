"""Fused backward kernels.

The three original backward kernels (`_bwd_q`, `_bwd_kv`, `_bwd_b` in `triton.py`) each
recompute `scores = q·k^T * sm_scale + bias` and `dp = do·v^T`, rebuild `p` from
`(mx, dn)` and rebuild `ds` -- nine matmuls and three softmax epilogues where five and
one would do. `_bwd_fused` does all four gradients in one pass.

Four choices carry the speedup -- two structural (1, 2) and two about how the bias tile
reaches the kernel (3, 4). All measured on an H20-3e at n=1024, h=8, d=32, bf16:

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

3. **The bias reaches the kernel as fp32, pre-scaled and pre-transposed** -- `_bwd_bias_prep`
   builds `b2t[bh, k, j] = b[bh, j, k] * inv_ln2` once. `_bwd_fused` then reads it in its
   natural `[k, j]` orientation and folds it in with a single FFMA.

   This is worth a paragraph because the obvious reading of the profile is the wrong one.
   ncu's PC sampling puts **22.95 % of all not-issued stall samples on the old bias load**,
   but only 0.45 % of that is `LDG`: the read itself always coalesced into
   `ld.global.v4.b32`. The cost was everything after it -- `PRMT 12.28 %` and `CS2R 6.09 %`,
   which is `arith.extf` bf16->fp32 across 4096 tile elements every j-iteration, one PRMT
   per element against a CS2R-materialised zero, plus a `#blocked -> #mma` layout
   conversion. So the fix is not a better load; it is to stop widening.

   Pre-scaling by `inv_ln2` on the host turns `(scoresT * sm_scale + b) * inv_ln2` into
   `scoresT * (sm_scale * inv_ln2) + b2`, one FFMA where there were two multiplies and an
   add. That also *reduces* rounding -- one FFMA plus two precomputed constants against
   mul->add->mul.

   The transpose is not cosmetic and not for coalescing: it is what lets a TMA descriptor
   read the tile. `TensorDescriptor` asserts `strides[-1] == 1`, so TMA cannot describe a
   transposed view, and `.T` on a *loaded* tile is only free when that tile feeds a wgmma
   as an operand (point 1 above). The bias tile is added to an fp32 accumulator, so a
   transpose there would lower to a real layout conversion -- reintroducing exactly what
   this change removes. Storing `b2t` transposed means no transpose anywhere, and it
   matches the `[bh, k, j]` layout `dbt` already uses.

4. **That fp32 bias tile then goes through TMA** (`USE_TMA_BIAS`, gated host-side on
   `USE_TMA_BWD_BIAS` and `not _is_fake`). Widening the tile made Triton stage it in
   shared memory *from registers*; a descriptor writes shared memory straight from global
   instead, so the store half disappears: `shared_st` 172.0 M -> 138.4 M instructions,
   and spills 10 -> 4.

   The measured gain is **not** where it was predicted, which is worth recording. `LDS`
   did not fall -- it rose (397.9 M -> 509.3 M), because the tile still has to be read out
   of shared memory into `#mma` and it is twice the bytes it used to be. What actually
   improved is `IMAD`, 12.8 % -> 7.2 % of stall samples: almost all of that is
   `IMAD.MOV.U32` register marshalling, which eases once the tile is no longer competing
   for the register file. 39.81 -> 37.69 ms clock-locked, 7.34e9 -> 6.67e9 instructions.

   This is also not an occupancy win, and never could have been: taking the tile out of
   registers moves 255 -> ~239, against the <=170 that 3 CTAs/SM needs.
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
def _bwd_bias_prep(
    b_ptr, stride_bh, stride_bm, stride_bn,       # INPUT  [BH, N, N], (j, k)
    b2t_ptr, stride_b2h, stride_b2k, stride_b2j,  # OUTPUT [BH, N, PADDED_N], (k, j)
    N,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr = 1.4426950408889634,     # = 1.0 / ln(2)
):
    """b2t[h, k, j] = b[h, j, k] * SCALE, in `b2t`'s dtype.

    Transposes, widens and scales the bias in one pass so `_bwd_fused` can fold it in with
    a single FFMA and no `extf`. See point 3 of the module docstring for why all three
    happen here rather than in the kernel.

    A tiled transpose, `[BLOCK, BLOCK]` per program: both the read and the write stay
    coalesced along their own contiguous axis. Bandwidth bound and tiny -- 16 MB read plus
    33 MB written at n=1024, against a ~35 ms kernel.

    Only the valid `j < N` region is written; the caller allocates `b2t` with
    `torch.zeros` so the `[N, PADDED_N)` columns stay zero. That padding exists for
    16-byte row alignment, which a TMA descriptor over this buffer would require -- it is
    *not* what makes `_bwd_fused`'s reads safe. `in_rangeT` is (see `_bwd_j_block`), and
    it has to be: at `BLOCK_J=64, N=17` a j tile runs to 63, well past `PADDED_N = 32`,
    and would otherwise read the next k row rather than any pad.
    """
    # `SCALE` is `inv_ln2` for `_bwd_fused`, which wants the bias already in log2 units.
    # `SCALE=1.0` leaves the bias untouched, which is what a *narrower* `b2t` needs: at
    # bf16, rounding `b * inv_ln2` to 8 mantissa bits puts a 2e-3 relative error into an
    # exponent, and the backward's recomputed weights then disagree with the forward's by
    # ~1 %. An unscaled bf16 `b2t` is a bit-exact copy of the bf16 input instead.
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    j_idxs = pid_j * BLOCK + tl.arange(0, BLOCK)
    k_idxs = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask_j = j_idxs < N
    mask_k = k_idxs < N

    b_ptrs = (b_ptr + pid_h * stride_bh
              + j_idxs[:, None] * stride_bm + k_idxs[None, :] * stride_bn)  # [j,k]
    b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)

    b2t_ptrs = (b2t_ptr + pid_h * stride_b2h
                + k_idxs[:, None] * stride_b2k + j_idxs[None, :] * stride_b2j)  # [k,j]
    tl.store(b2t_ptrs, tl.trans(b_block) * SCALE, mask_k[:, None] & mask_j[None, :])


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
    desc_b2t, bt_row,
    j_idxs, start_j,
    s2, neg_inf2, N,
    input_dtype: tl.constexpr,
    DIM: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    NEED_DB: tl.constexpr, NEED_DQ: tl.constexpr,
    J_MASKED: tl.constexpr, K_EXACT: tl.constexpr,
    USE_TMA_BIAS: tl.constexpr,
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
    # Already fp32 and already scaled by inv_ln2 -- `_bwd_bias_prep` did both, which is
    # where the old `.to(tl.float32)` here went (module docstring, point 3).
    #
    # This is still the one load that needs `in_rangeT` rather than `mask_j`, and the
    # transpose makes it matter more, not less. `b2t` rows are `(bh, k)` and `PADDED_N`
    # wide, so an out-of-range *j* no longer runs off the end of a row into padding -- it
    # runs into the next k row. At `BLOCK_J=64, N=17` that is j up to 63 against a 32-wide
    # row. `in_rangeT` is the only thing standing between that and a wrong answer; the
    # zeroed pad is not (see `_bwd_bias_prep`). Out-of-range *k* still overruns the bh
    # slice, as before.
    #
    # The unmasked branch is safe by construction: `RANGED=False` means the caller proved
    # j + BLOCK_J <= n_full <= N and start_k + BLOCK_K <= N.
    #
    # The TMA path cannot mask at the load, so it does not: out-of-range lanes get either
    # the zero fill (columns past PADDED_N) or a neighbouring slice's values (rows past
    # this bh slice), and both are discarded by the `in_rangeT` select on `scoresT` below.
    # That select already exists and is unconditional under `RANGED`, which is exactly how
    # `_fwd` handles its own TMA bias box. It is a `tl.where`, not arithmetic, so even a
    # NaN read cannot leak.
    if USE_TMA_BIAS:
        b_block = desc_b2t.load([bt_row, start_j])             # [k,j]
    elif RANGED:
        b_block = tl.load(bt_ptrs, in_rangeT)                  # [k,j]
    else:
        b_block = tl.load(bt_ptrs)

    # dP^T does not depend on the score tile, so issue it first; scheduling it between
    # pT and dsT would put a third live [k,j] fp32 tile on the critical path.
    dpT = tl.dot(v_blk, tl.trans(do_block), input_precision="ieee")   # [k,j]

    scoresT = tl.dot(k_blk, tl.trans(q_block), input_precision="ieee")  # [k,j]
    # One FFMA. `s2` is sm_scale * inv_ln2 and `b_block` is already inv_ln2-scaled, so
    # this is the old `(scoresT * sm_scale + b_block) * inv_ln2` with both constants
    # folded -- and one rounding instead of three.
    scoresT = scoresT * s2 + b_block
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
    desc_b2t, bt_row,
    stride_qn, stride_don, stride_b2j, stride_ln, stride_dn, stride_dbtj, stride_dqj,
    j_idxs,
    s2, neg_inf2, N,
    input_dtype: tl.constexpr,
    DIM: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
    NEED_DB: tl.constexpr, NEED_DQ: tl.constexpr,
    K_EXACT: tl.constexpr,
    USE_TMA_BIAS: tl.constexpr,
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
            desc_b2t, bt_row,
            j_idxs, start_j,
            s2, neg_inf2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            J_MASKED=False, K_EXACT=K_EXACT,
            USE_TMA_BIAS=USE_TMA_BIAS,
        )
        q_ptrs += BLOCK_J * stride_qn
        do_ptrs += BLOCK_J * stride_don
        bt_ptrs += BLOCK_J * stride_b2j
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
            desc_b2t, bt_row,
            j_idxs, n_full,
            s2, neg_inf2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            J_MASKED=True, K_EXACT=K_EXACT,
            USE_TMA_BIAS=USE_TMA_BIAS,
        )
    return dk, dv


@triton.jit
def _bwd_fused(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b2t_ptr, stride_b2h, stride_b2k, stride_b2j,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    desc_b2t,
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
    USE_TMA_BIAS: tl.constexpr = False,
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

    `desc_b2t` is a rank-2 TMA descriptor over the same `b2t`, viewed `[bh*N, PADDED_N]`
    so a row is `(bh, k)` and a box is `[BLOCK_K, BLOCK_J]` at `[start_h*N + start_k,
    start_j]`. Under `USE_TMA_BIAS` it replaces the pointer load; otherwise it is ignored
    and may be any value (the host passes the tensor itself, as `_fwd` does). The box
    shape has to match the selected config, which a per-config `pre_hook` rewrites --
    see `autotune_bwd.py`.

    `b2t_ptr` is the fp32, inv_ln2-scaled, `[bh, k, j]` bias that `_bwd_bias_prep` builds
    -- the same orientation as `dbt_ptr`. Its last axis may be padded past N; the kernel
    never relies on that, it only relies on the pad being zero. Module docstring, point 3.
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
    # The score scale in log2 units. The bias carries its own inv_ln2 from
    # `_bwd_bias_prep`, so the two together collapse the score epilogue to one FFMA.
    s2 = sm_scale * inv_ln2

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

    # Row of this CTA's bias box in the flattened [bh*N, PADDED_N] view the descriptor
    # sees. Uses N, not PADDED_N: only the column pitch was padded.
    bt_row = start_h * N + start_k

    # The bias tile, read straight: `b2t` is already stored `[bh, k, j]`, so j -- the axis
    # the loop walks -- is the contiguous one and each row of the tile is a vector load.
    bt_ptrs = (b2t_ptr + (start_h * stride_b2h)
               + (k_idxs[:, None] * stride_b2k) + (j_idxs[None, :] * stride_b2j))  # [k,j]

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
            desc_b2t, bt_row,
            stride_qn, stride_don, stride_b2j, stride_ln, stride_dn,
            stride_dbtj, stride_dqj,
            j_idxs,
            s2, neg_inf2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            USE_TMA_BIAS=USE_TMA_BIAS,
            K_EXACT=True,
        )
        tl.store(dk_ptrs, (dk * sm_scale).to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))
    else:
        dk, dv = _bwd_j_loop(
            k_blk, v_blk, m_blk, mask_k,
            q_ptrs, do_ptrs, bt_ptrs, mx_ptrs, dn_ptrs, d_ptrs, dbt_ptrs, dq_ptrs,
            desc_b2t, bt_row,
            stride_qn, stride_don, stride_b2j, stride_ln, stride_dn,
            stride_dbtj, stride_dqj,
            j_idxs,
            s2, neg_inf2, N,
            input_dtype,
            DIM, BLOCK_J, BLOCK_K,
            NEED_DB, NEED_DQ,
            USE_TMA_BIAS=USE_TMA_BIAS,
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
