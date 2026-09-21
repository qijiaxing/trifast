"""Five-dot fused backward with preprocessing and wide stride arithmetic.

Each program owns one (batch, head, i, key tile) and loops over all query
tiles. P and dScores are shared by all four gradients (five dot products).
dK/dV have a unique writer. dQ and the shared pair-bias gradient use
FP32 atomic reductions; no cubic probability tensor is materialized.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _preprocess(
    O,
    DO,
    V,
    M,
    DELTA,
    COUNT,
    DIFF,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
    DOH: tl.constexpr,
    DOI: tl.constexpr,
    DOJ: tl.constexpr,
    DOD: tl.constexpr,
):
    tile = tl.program_id(0)
    i = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)
    j = tile * BJ + tl.arange(0, BJ)
    d = tl.arange(0, D)
    off = ((bh * N + i) * N + j[:, None]) * D + d[None, :]
    doff = (
        bh * DOH
        + i.to(tl.int64) * DOI
        + j[:, None].to(tl.int64) * DOJ
        + d[None, :].to(tl.int64) * DOD
    )
    o = tl.load(O + off, j[:, None] < N, 0).to(tl.float32)
    do = tl.load(DO + doff, j[:, None] < N, 0).to(tl.float32)
    delta = tl.sum(o * do, 1)
    tl.store(DELTA + (bh * N + i) * N + j, delta, j < N)
    if tile == 0:
        all_k = tl.arange(0, triton.next_power_of_2(N))
        masked = tl.load(M + ((bh // H) * N + i) * N + all_k, all_k < N, 1) != 0
        count = tl.sum(((all_k < N) & ~masked).to(tl.int32), 0)
        tl.store(COUNT + bh * N + i, count)
        difference = tl.zeros((D,), tl.float32)
        if count == 1:
            single_k = tl.max(tl.where((all_k < N) & ~masked, all_k, 0), 0)
            single_v = tl.load(V + ((bh * N + i) * N + single_k) * D + d).to(tl.float32)
            for start in range(0, N, BK):
                k = start + tl.arange(0, BK)
                values = tl.load(
                    V + ((bh * N + i) * N + k[:, None]) * D + d[None, :],
                    k[:, None] < N,
                    0,
                ).to(tl.float32)
                difference += tl.sum(
                    tl.where(k[:, None] < N, single_v[None, :] - values, 0.0), 0
                )
        tl.store(DIFF + (bh * N + i) * D + d, difference)


@triton.jit
def _fused_bwd_k_owned(
    Q,
    K,
    V,
    B,
    DELTA,
    COUNT,
    DIFF,
    DO,
    MX,
    DN,
    M,
    DQ,
    DK,
    DV,
    DB,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
    DOH: tl.constexpr,
    DOI: tl.constexpr,
    DOJ: tl.constexpr,
    DOD: tl.constexpr,
    CENTERED: tl.constexpr = False,
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    i = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)
    d = tl.arange(0, D)
    koff = ((bh * N + i) * N + k[:, None]) * D + d[None, :]
    kk = tl.load(K + koff, k[:, None] < N, 0)
    v = tl.load(V + koff, k[:, None] < N, 0)
    masked = tl.load(M + ((bh // H) * N + i) * N + k, k < N, 1) != 0
    singleton = tl.load(COUNT + bh * N + i) == 1
    owns_singleton = singleton & (tl.sum(((k < N) & ~masked).to(tl.int32), 0) > 0)
    singleton_diff = tl.load(DIFF + (bh * N + i) * D + d, owns_singleton, 0)
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)
    dtype: tl.constexpr = Q.dtype.element_ty
    scale: tl.constexpr = D**-0.5
    invln2: tl.constexpr = 1.4426950408889634
    for start_j in range(0, N, BJ):
        j = start_j + tl.arange(0, BJ)
        valid = (j[:, None] < N) & (k[None, :] < N)
        qoff = ((bh * N + i) * N + j[:, None]) * D + d[None, :]
        boff = bh * N * N + j[:, None].to(tl.int64) * N + k[None, :]
        bias = tl.load(B + boff, valid, 0).to(tl.float32)
        if CENTERED:
            shift = tl.load(B + bh * N * N + j.to(tl.int64) * N, j < N, 0).to(
                tl.float32
            )
        else:
            shift = tl.full((BJ,), 0.0, tl.float32)
        sentinel2 = (-1.0e4 - shift) * invln2
        q = tl.load(Q + qoff, j[:, None] < N, 0)
        doff = (
            bh * DOH
            + i.to(tl.int64) * DOI
            + j[:, None].to(tl.int64) * DOJ
            + d[None, :].to(tl.int64) * DOD
        )
        do = tl.load(DO + doff, j[:, None] < N, 0)
        delta = tl.load(DELTA + (bh * N + i) * N + j, j < N, 0)
        mx = tl.load(MX + (bh * N + i) * N + j, j < N, 0)
        dn = tl.load(DN + (bh * N + i) * N + j, j < N, 1)
        s = (
            tl.dot(q, tl.trans(kk), input_precision="ieee") * scale
            + (bias - shift[:, None])
        ) * invln2
        s = tl.where(masked[None, :] | ~valid, sentinel2[:, None], s)
        p = tl.exp2(s - mx[:, None]) / dn[:, None]
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = tl.where(valid & ~masked[None, :], p * (dp - delta[:, None]), 0.0)
        if singleton:
            if N == 1:
                ds = tl.full((BJ, BK), 0.0, tl.float32)
            else:
                sentinel_p = tl.exp2(sentinel2 - mx) / dn
                stable_dp = tl.sum(do.to(tl.float32) * singleton_diff[None, :], 1)
                ds = tl.where(
                    valid & ~masked[None, :], p * (sentinel_p * stable_dp)[:, None], 0.0
                )
        ds_low = ds.to(dtype)
        dq = tl.dot(ds_low, kk, input_precision="ieee") * scale
        dk += tl.dot(tl.trans(ds_low), q, input_precision="ieee")
        dv += tl.dot(tl.trans(p).to(dtype), do, input_precision="ieee")
        tl.atomic_add(DQ + qoff, dq, j[:, None] < N, sem="relaxed")
        tl.atomic_add(DB + boff, ds, valid, sem="relaxed")
    tl.store(DK + koff, dk * scale, k[:, None] < N)
    tl.store(DV + koff, dv, k[:, None] < N)


def fused_backward(
    do,
    q,
    k,
    v,
    b,
    o,
    mx,
    dn,
    mask,
    *,
    bj=64,
    bk=64,
    gi=1,
    warps=4,
    stages=3,
    centered_stats=False,
):
    """Return dQ, dK, dV and dBias; gi is a reserved benchmark argument.

    Clearing dQ/dBias and the dQ cast are included in this call. dK/dV are
    directly stored in the input dtype after full FP32 query accumulation.
    """
    bs, h, n, _, d = q.shape
    # IEEE FP32 dot lowering needs substantially more staging storage than
    # tensor-core BF16/FP16. Keep the broad dtype/D contract within SM90 limits.
    if q.dtype == torch.float32 and d >= 64:
        bj, bk, stages = min(bj, 32), min(bk, 32), 1
    q, k, v, b, o, mx, dn, mask = [
        x.contiguous() for x in (q, k, v, b, o, mx, dn, mask)
    ]
    # Preserve spatial transposes and non-unit channel strides. reshape copies
    # only when collapsing batch/head cannot be represented as a view.
    do = do.reshape(bs * h, n, n, d)
    delta = torch.empty((bs * h, n, n), dtype=torch.float32, device=q.device)
    count = torch.empty((bs * h, n), dtype=torch.int32, device=q.device)
    difference = torch.empty((bs * h, n, d), dtype=torch.float32, device=q.device)
    _preprocess[(triton.cdiv(n, bj), n, bs * h)](
        o,
        do,
        v,
        mask,
        delta,
        count,
        difference,
        n,
        h,
        d,
        bj,
        bk,
        *do.stride(),
        num_warps=4,
        num_stages=1,
    )
    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk = torch.empty_like(q)
    dv = torch.empty_like(q)
    db = torch.zeros(b.shape, dtype=torch.float32, device=q.device)
    _fused_bwd_k_owned[(triton.cdiv(n, bk), n, bs * h)](
        q,
        k,
        v,
        b,
        delta,
        count,
        difference,
        do,
        mx,
        dn,
        mask,
        dq,
        dk,
        dv,
        db,
        n,
        h,
        d,
        bj,
        bk,
        *do.stride(),
        CENTERED=centered_stats,
        num_warps=warps,
        num_stages=stages,
    )
    return dq.to(q.dtype), dk, dv, db.to(b.dtype)
