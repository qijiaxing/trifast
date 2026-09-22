"""Experimental bounded FP32 dQ workspace; frozen public API is unchanged.

Chunk i, keeping all key-owner reductions for a query in FP32 before one cast.
dBias remains global FP32 across chunks. Extra launches trade time for memory.
"""

import torch
import triton
import triton.language as tl

from trifast._fused_backward import (
    _BUCKET_CONFIGS,
    _jblock,
    _prepare_do,
    _preprocess,
    _prune_bucket_stages,
)
from trifast.autotune import autotune


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
    CLOSEST_N: tl.constexpr,
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
    scale: tl.constexpr = D**-0.5
    dq_row = DQ + (bh * CHUNK_I + local_i) * N * D
    full_j = (N // BJ) * BJ
    for start_j in range(0, full_j, BJ):
        dk, dv = _jblock(
            Q,
            B,
            DELTA,
            DO,
            MX,
            DN,
            dq_row,
            DB,
            kk,
            v,
            k,
            k_valid,
            masked,
            singleton,
            singleton_diff,
            dk,
            dv,
            start_j,
            i,
            bh,
            N,
            D,
            BJ,
            BK,
            CLOSEST_N,
            CENTERED,
            J_MASKED=False,
        )
    if full_j < N:
        start_j = full_j
        dk, dv = _jblock(
            Q,
            B,
            DELTA,
            DO,
            MX,
            DN,
            dq_row,
            DB,
            kk,
            v,
            k,
            k_valid,
            masked,
            singleton,
            singleton_diff,
            dk,
            dv,
            start_j,
            i,
            bh,
            N,
            D,
            BJ,
            BK,
            CLOSEST_N,
            CENTERED,
            J_MASKED=True,
        )
    tl.store(DK + koff, dk * scale, k_valid[:, None])
    tl.store(DV + koff, dv, k_valid[:, None])


@autotune(
    configs=_BUCKET_CONFIGS,
    key=["H", "D", "BJ", "BK", "CENTERED", "CLOSEST_N"],
    reset_to_zero=["DQ"],
    restore_value=["DB"],
    prune_configs_by={"early_config_prune": _prune_bucket_stages},
    cache_name="chunk_bwd_k_owned_padded_bucket_peel_v1",
)
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
    CLOSEST_N: tl.constexpr,
    CENTERED: tl.constexpr = False,
):
    # ALIGNED describes this key tile only; query tails are peeled separately.
    if (tl.program_id(0) + 1) * BK <= N:
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
            CLOSEST_N,
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
            CLOSEST_N,
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
            CLOSEST_N,
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
    if warps != 4 or stages not in (1, 2, 3):
        raise ValueError("bucket tuning requires warps=4 and stages in {1,2,3}")
    # stages is retained for compatibility; the bucket tuner chooses 1/2/3.
    bs, h, n, _, d = q.shape
    if chunk_i < 1:
        raise ValueError("chunk_i must be positive")
    c = min(chunk_i, n)
    if q.dtype == torch.float32 and d >= 64:
        bj, bk, stages = min(bj, 32), min(bk, 32), 1
    q, k, v, b, o, mx, dn, mask = [
        x.contiguous() for x in (q, k, v, b, o, mx, dn, mask)
    ]
    bucket = 1 << (n - 1).bit_length()
    if bucket != n:
        b = torch.nn.functional.pad(b, (0, bucket - n))
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
            CLOSEST_N=bucket,
            CENTERED=centered_stats,
        )
        _flush_dq[(triton.cdiv(length * n * d, 1024), bs * h)](
            scratch, dq, n, d, c, start, length, 1024, num_warps=4
        )
    return dq, dk, dv, db[..., :n].to(b.dtype).contiguous()
