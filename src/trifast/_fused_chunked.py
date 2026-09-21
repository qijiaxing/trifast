"""Experimental bounded FP32 dQ workspace; frozen public API is unchanged.

Chunk i, keeping all key-owner reductions for a query in FP32 before one cast.
dBias remains global FP32 across chunks. Extra launches trade time for memory.
"""

import torch
import triton
import triton.language as tl

from trifast._fused_backward import _prepare_do, _preprocess


@triton.jit(do_not_specialize=["N", "I_START", "CHUNK_I"])
def _chunk_bwd_k_owned_body(
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
    N: tl.int64,
    H: tl.constexpr,
    D: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
    I_START: tl.int64,
    CHUNK_I: tl.int64,
    CENTERED: tl.constexpr = False,
    ALIGNED: tl.constexpr = False,
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    k_valid = tl.full((BK,), True, tl.int1) if ALIGNED else k < N
    local_i = tl.program_id(1)
    i = local_i + I_START
    bh = tl.program_id(2).to(tl.int64)
    d = tl.arange(0, D)
    koff = ((bh * N + i) * N + k[:, None]) * D + d[None, :]
    kk = tl.load(K + koff, k_valid[:, None], 0)
    v = tl.load(V + koff, k_valid[:, None], 0)
    masked = tl.load(M + ((bh // H) * N + i) * N + k, k_valid, 1) != 0
    singleton = tl.load(COUNT + bh * N + i) == 1
    owns_singleton = singleton & (tl.sum((k_valid & ~masked).to(tl.int32), 0) > 0)
    singleton_diff = tl.load(DIFF + (bh * N + i) * D + d, owns_singleton, 0)
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)
    dtype: tl.constexpr = Q.dtype.element_ty
    scale: tl.constexpr = D**-0.5
    invln2: tl.constexpr = 1.4426950408889634
    for start_j in range(0, N, BJ):
        j = start_j + tl.arange(0, BJ)
        j_valid = tl.full((BJ,), True, tl.int1) if ALIGNED else j < N
        valid = (j_valid[:, None]) & (k_valid[None, :])
        qoff = ((bh * N + i) * N + j[:, None]) * D + d[None, :]
        boff = bh * N * N + j[:, None].to(tl.int64) * N + k[None, :]
        bias = tl.load(B + boff, valid, 0).to(tl.float32)
        if CENTERED:
            shift = tl.load(B + bh * N * N + j.to(tl.int64) * N, j_valid, 0).to(
                tl.float32
            )
        else:
            shift = tl.full((BJ,), 0.0, tl.float32)
        sentinel2 = (-1.0e4 - shift) * invln2
        q = tl.load(Q + qoff, j_valid[:, None], 0)
        doff = qoff
        do = tl.load(DO + doff, j_valid[:, None], 0)
        delta = tl.load(DELTA + (bh * N + i) * N + j, j_valid, 0)
        mx = tl.load(MX + (bh * N + i) * N + j, j_valid, 0)
        dn = tl.load(DN + (bh * N + i) * N + j, j_valid, 1)
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
        tl.atomic_add(
            DQ + ((bh * CHUNK_I + local_i) * N + j[:, None]) * D + d[None, :],
            dq,
            j_valid[:, None],
            sem="relaxed",
        )
        tl.atomic_add(DB + boff, ds, valid, sem="relaxed")
    tl.store(DK + koff, dk * scale, k_valid[:, None])
    tl.store(DV + koff, dv, k_valid[:, None])


@triton.jit(do_not_specialize=["N", "I_START", "CHUNK_I"])
def _chunk_bwd_k_owned(
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
    N: tl.int64,
    H: tl.constexpr,
    D: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
    I_START: tl.int64,
    CHUNK_I: tl.int64,
    CENTERED: tl.constexpr = False,
):
    ALIGNMENT: tl.constexpr = max(BJ, BK)
    if N % ALIGNMENT == 0:
        _chunk_bwd_k_owned_body(
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
            tl.multiple_of(N, ALIGNMENT),
            H,
            D,
            BJ,
            BK,
            I_START,
            CHUNK_I,
            CENTERED,
            ALIGNED=True,
        )
    elif N % 8 == 0:
        _chunk_bwd_k_owned_body(
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
            tl.multiple_of(N, 8),
            H,
            D,
            BJ,
            BK,
            I_START,
            CHUNK_I,
            CENTERED,
            ALIGNED=False,
        )
    else:
        _chunk_bwd_k_owned_body(
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
            N,
            H,
            D,
            BJ,
            BK,
            I_START,
            CHUNK_I,
            CENTERED,
            ALIGNED=False,
        )


@triton.jit(do_not_specialize=["N", "C", "I_START", "COUNT"])
def _flush_dq(
    W,
    DQ,
    N: tl.int64,
    D: tl.constexpr,
    C: tl.int64,
    I_START: tl.int64,
    COUNT: tl.int64,
    BLOCK: tl.constexpr,
):
    bh = tl.program_id(1).to(tl.int64)
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = x < COUNT.to(tl.int64) * N * D
    value = tl.load(W + bh * C * N * D + x, valid, 0)
    tl.store(DQ + (bh * N + I_START) * N * D + x, value, valid)


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
    chunk_i=128,
):
    bs, h, n, _, d = q.shape
    if chunk_i < 1:
        raise ValueError("chunk_i must be positive")
    c = min(chunk_i, n)
    if q.dtype == torch.float32 and d >= 64:
        bj, bk, stages = min(bj, 32), min(bk, 32), 1
    q, k, v, b, o, mx, dn, mask = [
        x.contiguous() for x in (q, k, v, b, o, mx, dn, mask)
    ]
    do = _prepare_do(do, bs, h, n, d)
    delta = torch.empty((bs * h, n, n), device=q.device, dtype=torch.float32)
    count = torch.empty((bs * h, n), device=q.device, dtype=torch.int32)
    difference = torch.empty((bs * h, n, d), device=q.device, dtype=torch.float32)
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
        num_warps=4,
        num_stages=1,
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(q)
    dv = torch.empty_like(q)
    db = torch.zeros(b.shape, device=q.device, dtype=torch.float32)
    scratch = torch.empty((bs * h, c, n, d), device=q.device, dtype=torch.float32)
    for start in range(0, n, c):
        length = min(c, n - start)
        scratch.zero_()
        _chunk_bwd_k_owned[(triton.cdiv(n, bk), length, bs * h)](
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
            scratch,
            dk,
            dv,
            db,
            n,
            h,
            d,
            bj,
            bk,
            start,
            c,
            CENTERED=centered_stats,
            num_warps=warps,
            num_stages=stages,
        )
        _flush_dq[(triton.cdiv(length * n * d, 1024), bs * h)](
            scratch, dq, n, d, c, start, length, 1024, num_warps=4
        )
    return dq, dk, dv, db.to(b.dtype)
