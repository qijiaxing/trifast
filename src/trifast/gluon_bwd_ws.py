"""The fused backward again, this time with the loads in their own warpgroup.

`gluon_bwd.py` is at 0.95x of `triton_bwd.py` and its remaining deficit is one number:
`long_scoreboard` is **41.6 % of not-issued samples** against the Triton kernel's 16.8 %.
That module's docstring names the fix and declines to build it. This is it.

## Why a producer partition, and not another schedule tweak

The body of `gluon_bwd.py` is a single serial dependency chain -- `wait_group` -> mma ->
epilogue -> `STS` -> mma -> atomic -> mma -- with **no independent work anywhere in it**.
Occupancy is pinned at 2 CTAs/SM by registers *and* shared memory, which is 8 warps/SM on
4 schedulers, i.e. **2 warps per scheduler, both running that same chain**. Measured
consequence: each warp issues ~800 instructions per j-iteration and takes ~5,400 cycles to
do it, and `issued / elapsed` comes out at **0.30 IPC** -- 70 % of issue slots go empty.
A GPU hides latency with other warps that have independent work, and there were none.

So the six source-level fixes tried in `gluon_bwd.py` all failed for the same reason: they
each removed an instruction from a chain whose cost is not instructions. Splitting the chain
is a different kind of change -- it is the only one that gives the schedulers something else
to issue.

Scope, interface and preconditions are `gluon_bwd.py`'s, minus its `STAGES <= N // BLOCK_J`
requirement: there is no read-ahead prologue here, so short rows need no special casing.

## It works: 34.44 ms, 1.01x `triton_bwd.py`, 1.06x `gluon_bwd.py`

n=1024, h=8, d=32, bf16, H20-3e, `STAGES=3`, `maxnreg=116`. Across shapes:

| | triton | gluon_bwd | gluon_bwd_ws | |
| --- | --- | --- | --- | --- |
| n=512 | 4.56 | 4.75 | **4.68** | 0.97x triton |
| n=1024 | 34.67 | 36.48 | **34.44** | **1.01x** triton, 1.06x gluon |
| n=2048 | 269.80 | 284.83 | **263.53** | **1.02x** triton, 1.08x gluon |

Gradients are unchanged from `gluon_bwd.py`: dk and dv bit-identical to the three-kernel
reference, dq worst 2.51e-3.

**`maxnreg=116` is not a tuning knob, it is the configuration** -- see `WS_MAXNREG`. At the
default the kernel runs at 0.62x, because the CTA reserves the whole register file.

## What it bought, and what it cost

Against `gluon_bwd.py` at the same shape (n=512 ncu, not-issued samples), the consumer's own
stalls fell roughly as predicted:

| opcode | gluon_bwd | ws | |
| --- | --- | --- | --- |
| `FENCE.VIEW.ASYNC.S` | 20419 | **8637** | one fewer fence, and no L1TEX drain behind it |
| `WARPGROUP.DEPBAR.LE` | 30064 | **13251** | operands are ready when the mma issues |
| `MUFU.RCP` | 5735 | **1930** | |
| `FADD` | 12255 | **7576** | the epilogue is no longer waiting on loads |
| `REDG` | 14863 | **6930** | |

And warp specialization charged most of it back:

| opcode | ws | |
| --- | --- | --- |
| `@!P1` | 34821 (18.2 %) | the two mbarrier spin loops per iteration |
| `USETMAXREG.DEALLOC.CTAPOOL` | 28859 (15.1 %) | `setmaxnreg` in the producer |
| `BAR.SYNC.DEFER_BLOCKING` | 17628 (9.2 %) | up from 8200 |
| `LDS.U8` | 5441 (2.9 %) | **one instruction**: the partition-id read |

Net: 191217 samples against 202608, occupancy 12.5 % -> 25 %, `Compute (SM)` 50.9 -> 51.9 %,
and 5.6 % less wall clock. A modest win made of two large offsetting effects, which is worth
knowing before anyone tries this on a kernel with a shorter dependency chain to hide.

Two things it could not fix, and one it must not be asked to:

- **Fences 2 and 3 stay.** They order the consumer's *own* register->shared stores (`p_s`,
  `ds_s`) against its own wgmma. Warp specialization does not touch that.
- **The matmuls cannot be factored out.** `reference/01-attention-forward.py` gives them
  their own partition, and **that is a Blackwell-only trick**: `tcgen05` accumulators live in
  tensor memory, shared between partitions. On Hopper a wgmma accumulator is registers owned
  by the issuing warpgroup. Hopper warp specialization is limited to load-producer /
  math-consumer / store-epilogue.
- **`USETMAXREG` at 15.1 % of samples is not 15 % of time.** Dropping `worker_num_regs`
  removes the instruction and costs 34.44 -> 37.49 ms. Those samples are producer warps
  parked in the dealloc, and a parked producer does not stop the consumer from issuing. This
  is the clearest example in either module of why sample share must not be read as time.

## Dead ends, measured

1. **Staging the three `[j]` vectors in the producer: 75 ms, a 2x regression.** This was the
   single largest item the analysis predicted -- ~30 % of `long_scoreboard` -- and the third
   independent refutation of the idea (`gluon_bwd.py` records the other two). They are 256 B
   each; the producer has to `LDG` then `STS` them synchronously before it can arrive, which
   triples its per-iteration latency and destroys the run-ahead that makes the partition
   work. They stay on the consumer.
2. **A narrower producer.** `PROD_WARPS=1` is 42.44 ms with 28 spills, `=2` is 58.69. The
   4-warp producer also reuses `gluon_bwd.py`'s copy layouts verbatim.
3. **An 8-warp consumer** to halve per-thread register demand, which would have made
   `maxnreg` a non-issue: **Triton caps a warp-specialized kernel at 256 threads**
   (`out of resource: threads, Required: 384, Hardware limit: 256`), so 8 + 4 is unavailable.
4. **Dropping `setmaxnreg`** (item above): 37.49 ms.

## What the smoke test settled first (`scripts/proto_gluon_ws.py smoke`)

- `ttgl.warp_specialize` **works on sm90**, with 1-, 2- and 4-warp workers.
- **Each partition needs layouts whose `warpsPerCTA` matches that partition's own warp
  count**, not the parent's.
- **Triton pads the worker to a full warpgroup on Hopper.** `PROD_WARPS=1` still reports
  `Block Size 256`. So warp specialization always costs 8 warps/CTA here, and that -- not the
  producer's own registers -- is what forces `maxnreg` down to 116.
- `kernel.n_regs` reports the max over partitions (255 even for a trivial kernel), so it says
  nothing about the split. Occupancy has to come from ncu's `Block Limit Registers`.
- The handshake below is the one that works. The producer blocks on its *own* copies with
  `wait_group(0)` and then signals a plain single-count arrival, rather than using
  `async_copy.mbarrier_arrive`, which is a per-thread `cp.async.mbarrier.arrive` and would
  need the barrier's arrival count to match the number of *threads* issuing copies. The
  producer has nothing else to do with those cycles, and it still runs STAGES-1 iterations
  ahead of the consumer.

## What the smoke test settled first (scripts/proto_gluon_ws.py has it as `smoke`)

- `ttgl.warp_specialize` **works on sm90**, with 1-, 2- and 4-warp workers.
- **Each partition needs layouts whose `warpsPerCTA` matches that partition's own warp
  count**, not the parent's. This is why the producer here is 4 warps: it reuses `BL_D` and
  `BL_J` verbatim. A 1-warp producer needs its own `[1, 1]` layouts, and 4x the `LDGSTS` per
  thread.
- `kernel.n_regs` reports the max over partitions (255 for a trivial kernel), so it says
  nothing about the split. Occupancy has to come from ncu's `Block Limit Registers`.
- The handshake below is the one that works. The producer blocks on its *own* copies with
  `wait_group(0)` and then signals a plain single-count arrival, rather than using
  `async_copy.mbarrier_arrive`, which is a per-thread `cp.async.mbarrier.arrive` and would
  need the barrier's arrival count to match the number of *threads* issuing copies. The
  producer has nothing else to do with those cycles, and it still runs STAGES-1 iterations
  ahead of the consumer.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

from trifast.gluon_bwd import _acc_layout

# `maxnreg` is not a tuning knob here, it is the configuration. It is a *launch* option, so
# callers have to pass it; `_gl_bwd_ws` cannot set it for them.
#
# With 8 warps / 256 threads per CTA, 2 CTAs/SM needs `maxnreg * 256 * 2 <= 65536`, i.e.
# `<= 128`. Within that ceiling the measured curve is not monotone and not explained by
# spills or occupancy -- 96/100/104/108 are 39.7/43.0/39.7/39.5 ms and 120/124/128 are
# 38.3/38.7/39.1, while **112 and 116 are 34.9/34.6**. All of 112..128 are 2 CTAs/SM with
# zero spills, so this is a ptxas scheduling boundary, not a resource one. Two adjacent
# values reproducing at two different stage depths is what makes it a plateau worth
# standing on rather than another spike of the kind `gluon_bwd.py` records for `maxnreg`.
WS_MAXNREG = 116

# fmt: off


@gluon.jit
def _ws_produce(
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    b2t_ptr, stride_b2h, stride_b2k, stride_b2j,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    q_s, do_s, b_s, full, empty,
    n_iters, pid_h, pid_i, start_k,
    DIM: ttgl.constexpr, BLOCK_J: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
    STAGES: ttgl.constexpr, PROD_WARPS: ttgl.constexpr,
):
    """Copies, and nothing else. Never reads a value the consumer computes.

    Runs in its own warpgroup, so every instruction here is independent work the schedulers
    can issue while the consumer is stalled -- which is the entire point.
    """
    bias_dtype: ttgl.constexpr = b2t_ptr.dtype.element_ty
    # `gluon_bwd.py`'s copy layouts, with `warpsPerCTA` retargeted to *this partition's*
    # warp count -- the one hard rule the smoke test turned up. 16 bytes per thread along
    # the contiguous axis either way; a narrower producer just makes more passes.
    BL_D: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [PROD_WARPS, 1], [1, 0])
    BL_J: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 128 // bias_dtype.primitive_bitwidth], [4, 8], [PROD_WARPS, 1], [1, 0])

    jd_r = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(1, BL_D))[:, None]
    kd_c = ttgl.arange(0, DIM, ttgl.SliceLayout(0, BL_D))[None, :]
    bj_r = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, BL_J))[:, None] + start_k
    bj_c = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(0, BL_J))[None, :]

    q_base = q_ptr + pid_h * stride_qh + pid_i * stride_qm
    do_base = do_ptr + pid_h * stride_doh + pid_i * stride_dom
    b2_base = b2t_ptr + pid_h * stride_b2h

    idx = 0
    phase = 0
    for it in range(n_iters):
        sj = it * BLOCK_J
        # The slot is free once the consumer's last wgmma over it has retired.
        mbarrier.wait(empty.index(idx), phase)
        async_copy.async_copy_global_to_shared(
            q_s.index(idx), q_base + (jd_r + sj) * stride_qn + kd_c * stride_qd)
        async_copy.async_copy_global_to_shared(
            do_s.index(idx), do_base + (jd_r + sj) * stride_don + kd_c * stride_dod)
        async_copy.async_copy_global_to_shared(
            b_s.index(idx), b2_base + bj_r * stride_b2k + (bj_c + sj) * stride_b2j)
        async_copy.commit_group()
        # Block on our own copies, then signal. See the module docstring on why this rather
        # than `async_copy.mbarrier_arrive`. The producer is ~15x faster than the consumer
        # per iteration, so serialising it here costs nothing and it still runs ahead.
        async_copy.wait_group(0)
        mbarrier.arrive(full.index(idx), count=1)
        idx += 1
        if idx == STAGES:
            idx = 0
            phase = phase ^ 1


@gluon.jit
def _ws_consume(
    d_ptr, stride_dh, stride_dm, stride_dn,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    dbt_ptr, stride_dbth, stride_dbtk, stride_dbtj,
    dq_ptr, stride_dqh, stride_dqm, stride_dqj, stride_dqd,
    k_s, v_s, p_s, ds_s, q_s, do_s, b_s, full, empty,
    m_blk, sm_scale, neg_inf2, n_iters, pid_h, pid_i, start_k,
    DIM: ttgl.constexpr, BLOCK_J: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
    STAGES: ttgl.constexpr,
    NEED_DB: ttgl.constexpr, NEED_DQ: ttgl.constexpr,
    BIAS_SCALED: ttgl.constexpr, CONS_WARPS: ttgl.constexpr,
):
    """The five matmuls and the softmax epilogue. Touches global memory only to write.

    Body-for-body the same schedule as `gluon_bwd.py._gl_bwd_fused` -- two independent mmas
    with a partial drain between them, `ds16` staged once and transposed for free, dQ issued
    before dK so its atomic runs underneath dK. The differences are that the staged tiles
    arrive by mbarrier instead of `wait_group`, and that the pre-mma fence is gone.
    """
    input_dtype: ttgl.constexpr = k_ptr.dtype.element_ty
    # Spreading the score tiles over 8 warps instead of 4 halves the per-thread register
    # demand, which is the only lever that makes warp specialization affordable here --
    # see the module docstring's register arithmetic.
    MMA_J: ttgl.constexpr = _acc_layout(BLOCK_J, CONS_WARPS)
    MMA_D: ttgl.constexpr = _acc_layout(DIM, CONS_WARPS)
    BL_D: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [CONS_WARPS, 1], [1, 0])

    inv_ln2: ttgl.constexpr = 1.4426950408889634
    s2 = sm_scale.to(ttgl.float32) * inv_ln2

    k_of_j = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, MMA_J)) + start_k
    j_of_k = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(0, MMA_J))
    kd_r = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, BL_D))[:, None] + start_k
    kd_c = ttgl.arange(0, DIM, ttgl.SliceLayout(0, BL_D))[None, :]

    l_base = mx_ptr + pid_h * stride_lh + pid_i * stride_lm
    dn_base = dn_ptr + pid_h * stride_lh + pid_i * stride_lm
    d_base = d_ptr + pid_h * stride_dh + pid_i * stride_dm

    dk = ttgl.zeros([BLOCK_K, DIM], ttgl.float32, layout=MMA_D)
    dv = ttgl.zeros([BLOCK_K, DIM], ttgl.float32, layout=MMA_D)

    idx = 0
    phase = 0
    for it in range(n_iters):
        start_j = it * BLOCK_J
        # Replaces `wait_group(STAGES-1)` *and* the fence that followed it: the tiles were
        # written by the producer warpgroup, and an mbarrier arrival orders them for us.
        mbarrier.wait(full.index(idx), phase)
        q_c = q_s.index(idx)
        do_c = do_s.index(idx)

        # --- the two independent matmuls, scoresT first ---------------------------------
        t_s = warpgroup_mma(k_s, q_c.permute([1, 0]),
                            warpgroup_mma_init(ttgl.zeros([BLOCK_K, BLOCK_J],
                                                          ttgl.float32, layout=MMA_J)),
                            is_async=True)
        t_p = warpgroup_mma(v_s, do_c.permute([1, 0]),
                            warpgroup_mma_init(ttgl.zeros([BLOCK_K, BLOCK_J],
                                                          ttgl.float32, layout=MMA_J)),
                            is_async=True)

        # These stay on the consumer. Handing them to the producer is the most obvious
        # remaining win and it is a **2x regression**; see the dead-end list.
        row_max = ttgl.load(l_base + (j_of_k + start_j) * stride_ln)      # [j]
        row_denom = ttgl.load(dn_base + (j_of_k + start_j) * stride_ln)   # [j]
        delta = ttgl.load(d_base + (j_of_k + start_j) * stride_dn)        # [j]
        b_blk = b_s.index(idx).load(MMA_J).to(ttgl.float32)

        # Partial drain: scoresT is ready, dpT is not.
        scoresT = warpgroup_mma_wait(num_outstanding=1, deps=(t_s,))
        if BIAS_SCALED:
            scoresT = scoresT * s2 + b_blk
        else:
            scoresT = (scoresT * sm_scale + b_blk) * inv_ln2
        scoresT = ttgl.where(m_blk[:, None], neg_inf2, scoresT)
        pT = ttgl.exp2(scoresT - row_max[None, :]) / row_denom[None, :]   # [k,j]
        p_s.store(pT.to(input_dtype))

        dpT = warpgroup_mma_wait(num_outstanding=0, deps=(t_p,))

        dq_ptrs = (dq_ptr + pid_h * stride_dqh + pid_i * stride_dqm
                   + (ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(1, MMA_D))[:, None]
                      + start_j) * stride_dqj
                   + ttgl.arange(0, DIM, ttgl.SliceLayout(0, MMA_D))[None, :] * stride_dqd)

        fence_async_shared()
        t_v = warpgroup_mma(p_s, do_c, warpgroup_mma_init(dv), is_async=True)
        dsT = pT * (dpT - delta[None, :])                                 # [k,j]
        dsT = ttgl.where(m_blk[:, None], 0.0, dsT)
        dv = warpgroup_mma_wait(num_outstanding=0, deps=(t_v,))

        if NEED_DB:
            dbt_ptrs = (dbt_ptr + pid_h * stride_dbth + k_of_j[:, None] * stride_dbtk
                        + (j_of_k[None, :] + start_j) * stride_dbtj)
            ttgl.atomic_add(dbt_ptrs, dsT, sem="relaxed")

        ds_s.store(dsT.to(input_dtype))
        fence_async_shared()
        if NEED_DQ:
            t_q = warpgroup_mma(ds_s.permute([1, 0]), k_s,
                                warpgroup_mma_init(ttgl.zeros([BLOCK_J, DIM],
                                                              ttgl.float32, layout=MMA_D)),
                                is_async=True)
        t_k = warpgroup_mma(ds_s, q_c, warpgroup_mma_init(dk), is_async=True)
        if NEED_DQ:
            dq_tile = warpgroup_mma_wait(num_outstanding=1, deps=(t_q,))
            ttgl.atomic_add(dq_ptrs, dq_tile, sem="relaxed")
        dk = warpgroup_mma_wait(num_outstanding=0, deps=(t_k,))

        # Release the slot only now. `t_k` reads `q_c` and `t_v` reads `do_c`, so the slot
        # is not reusable until the last of those has *retired* -- not merely been issued.
        # Arriving any earlier lets the producer overwrite a live wgmma operand.
        mbarrier.arrive(empty.index(idx), count=1)
        idx += 1
        if idx == STAGES:
            idx = 0
            phase = phase ^ 1

    dk_ptrs = (dk_ptr + pid_h * stride_dkh + pid_i * stride_dkm
               + kd_r * stride_dkn + kd_c * stride_dkd)
    dv_ptrs = (dv_ptr + pid_h * stride_dvh + pid_i * stride_dvm
               + kd_r * stride_dvn + kd_c * stride_dvd)
    ttgl.store(dk_ptrs, ttgl.convert_layout((dk * sm_scale).to(input_dtype), BL_D))
    ttgl.store(dv_ptrs, ttgl.convert_layout(dv.to(input_dtype), BL_D))


@gluon.jit
def _gl_bwd_ws(
    d_ptr, stride_dh, stride_dm, stride_dn,                        # delta [BH,N,N]
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b2t_ptr, stride_b2h, stride_b2k, stride_b2j,                   # [BH,N,PADDED_N]
    mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,                        # bool mask [B,N,N]
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    # OUTPUT
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    dbt_ptr, stride_dbth, stride_dbtk, stride_dbtj,                # fp32 [BH,K,J]
    dq_ptr, stride_dqh, stride_dqm, stride_dqj, stride_dqd,        # fp32 [BH,N,N,D]
    sm_scale, neg_inf,
    N, H,
    DIM: ttgl.constexpr,
    BLOCK_J: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
    NEED_DB: ttgl.constexpr, NEED_DQ: ttgl.constexpr,
    STAGES: ttgl.constexpr = 3,
    BIAS_SCALED: ttgl.constexpr = False,
    PROD_WARPS: ttgl.constexpr = 4,
    PROD_REGS: ttgl.constexpr = 40,
    CONS_WARPS: ttgl.constexpr = 4,
):
    """Allocate, prime the channel, then split into producer and consumer.

    Same interface and same preconditions as `gluon_bwd._gl_bwd_fused` -- grid
    `(N // BLOCK_K, N, bh)`, `N % BLOCK_J == 0`, `N % BLOCK_K == 0` -- except that this one
    has no prologue reading ahead, so `STAGES <= N // BLOCK_J` is *not* required here. The
    producer simply never runs more iterations than there are.
    """
    input_dtype: ttgl.constexpr = q_ptr.dtype.element_ty
    bias_dtype: ttgl.constexpr = b2t_ptr.dtype.element_ty

    MMA_J: ttgl.constexpr = _acc_layout(BLOCK_J, CONS_WARPS)
    SH_D: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, DIM], input_dtype)
    SH_J: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_J], input_dtype)
    SH_B: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_J], bias_dtype)
    BL_D: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [CONS_WARPS, 1], [1, 0])

    pid_k = ttgl.program_id(0)
    pid_i = ttgl.program_id(1)
    pid_h = ttgl.program_id(2)

    inv_ln2: ttgl.constexpr = 1.4426950408889634
    neg_inf2 = neg_inf.to(ttgl.float32) * inv_ln2
    start_k = pid_k * BLOCK_K
    mask_h = pid_h // H
    n_iters = N // BLOCK_J

    kd_r = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, BL_D))[:, None] + start_k
    kd_c = ttgl.arange(0, DIM, ttgl.SliceLayout(0, BL_D))[None, :]

    # Loop-invariant per-CTA tiles. These stay on the default warps: they are read by the
    # consumer's wgmma, they are loaded once, and leaving them here keeps the producer a
    # pure copy loop with no barrier of its own.
    k_base = k_ptr + pid_h * stride_kh + pid_i * stride_km
    v_base = v_ptr + pid_h * stride_vh + pid_i * stride_vm
    k_s = ttgl.allocate_shared_memory(
        input_dtype, [BLOCK_K, DIM], SH_D,
        ttgl.load(k_base + kd_r * stride_kn + kd_c * stride_kd))
    v_s = ttgl.allocate_shared_memory(
        input_dtype, [BLOCK_K, DIM], SH_D,
        ttgl.load(v_base + kd_r * stride_vn + kd_c * stride_vd))

    m_ptrs = (m_ptr + mask_h * stride_mh + pid_i * stride_mm
              + (ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, MMA_J)) + start_k) * stride_mn)
    m_blk = ttgl.load(m_ptrs) != 0                                     # [k]

    q_s = ttgl.allocate_shared_memory(input_dtype, [STAGES, BLOCK_J, DIM], SH_D)
    do_s = ttgl.allocate_shared_memory(input_dtype, [STAGES, BLOCK_J, DIM], SH_D)
    b_s = ttgl.allocate_shared_memory(bias_dtype, [STAGES, BLOCK_K, BLOCK_J], SH_B)
    p_s = ttgl.allocate_shared_memory(input_dtype, [BLOCK_K, BLOCK_J], SH_J)
    ds_s = ttgl.allocate_shared_memory(input_dtype, [BLOCK_K, BLOCK_J], SH_J)

    # One channel for all three staged tiles: they are produced together and consumed
    # together, so they need one pair of barriers, not three.
    full = mbarrier.allocate_mbarrier(batch=STAGES)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    for i in ttgl.static_range(STAGES):
        mbarrier.init(full.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
        # Prime: every slot starts free, so the producer's first `wait(empty, phase=0)`
        # falls straight through. Both partitions then run identical (idx, phase) counters
        # from (0, 0), which is what keeps the parities in step.
        mbarrier.arrive(empty.index(i), count=1)

    # `k_s` and `v_s` were filled by a generic store, so the consumer's wgmma needs a
    # generic->async proxy fence for them. Once, here: they are loop-invariant, and the
    # default partition runs on these same warps. The per-iteration staged tiles need no
    # such fence -- they are `cp.async`-written and mbarrier-ordered.
    fence_async_shared()

    # The (function, args) list has to be written inline here: binding it to a local makes
    # the Gluon frontend try to convert the `GluonJITFunction` to a tensor, and binding just
    # the arg tuples fails differently with "`_semantic` argument must be provided outside
    # of JIT functions".
    #
    # `worker_num_regs` is worth passing even though `.maxnreg` already caps the consumer:
    # dropping it costs 34.44 -> 37.49 ms. Its `USETMAXREG.DEALLOC.CTAPOOL` is 15.1 % of
    # not-issued *samples*, which is a good illustration of why samples are not time -- they
    # are producer warps parked in the dealloc, and a parked producer does not block the
    # consumer from issuing.
    ttgl.warp_specialize([
        (_ws_consume, (
        d_ptr, stride_dh, stride_dm, stride_dn,
        k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
        mx_ptr, dn_ptr, stride_lh, stride_lm, stride_ln,
        dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
        dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
        dbt_ptr, stride_dbth, stride_dbtk, stride_dbtj,
        dq_ptr, stride_dqh, stride_dqm, stride_dqj, stride_dqd,
        k_s, v_s, p_s, ds_s, q_s, do_s, b_s, full, empty,
        m_blk, sm_scale, neg_inf2, n_iters, pid_h, pid_i, start_k,
        DIM, BLOCK_J, BLOCK_K, STAGES, NEED_DB, NEED_DQ, BIAS_SCALED,
        CONS_WARPS)),
        (_ws_produce, (
        q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
        b2t_ptr, stride_b2h, stride_b2k, stride_b2j,
        do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
        q_s, do_s, b_s, full, empty,
        n_iters, pid_h, pid_i, start_k,
        DIM, BLOCK_J, BLOCK_K, STAGES, PROD_WARPS)),
    ], [PROD_WARPS], [PROD_REGS])

    for i in ttgl.static_range(STAGES):
        mbarrier.invalidate(full.index(i))
        mbarrier.invalidate(empty.index(i))
# fmt: on


def gluon_ws_launch_opts() -> dict:
    """The non-negotiable launch options for `_gl_bwd_ws`.

    `num_warps` must match `CONS_WARPS`, and `maxnreg` must be `WS_MAXNREG` -- at the
    default 256 the kernel runs at 0.62x instead of 1.01x, because the CTA reserves the
    whole register file and drops to 1 CTA/SM.
    """
    return dict(num_warps=4, num_stages=1, maxnreg=WS_MAXNREG)


def gluon_ws_supported(n: int, dim: int, dtype: torch.dtype,
                       block_j: int = 64, block_k: int = 64) -> bool:
    """Whether `_gl_bwd_ws` can run this shape.

    The same fast-path restrictions as `gluon_bwd.gluon_bwd_supported`, minus its
    `STAGES <= N // BLOCK_J` precondition: there is no read-ahead prologue here.
    """
    return (
        dtype in (torch.bfloat16, torch.float16)
        and n % block_j == 0
        and n % block_k == 0
        and dim % 8 == 0
        and block_k == 64
        and triton.runtime.driver.active.get_current_target().backend == "cuda"
    )
