"""The fused backward, hand-scheduled in Gluon.

`triton_bwd.py` is at 1.73x over the three-kernel path and its largest remaining stall is
not memory and not arithmetic: **28 % of not-issued samples are `WARPGROUP` sync**, in three
`WARPGROUP.DEPBAR.LE gsb0, 0x0` per j-iteration. Each of those drains the wgmma pipe to
*zero* outstanding. Two of the five matmuls -- `scoresT = k·qT` and `dpT = v·doT` -- are
independent, and the softmax epilogue depends only on the first, so the right schedule is to
issue both, release `scoresT` while `dpT` is still in flight, and run the epilogue underneath
it. `tl.dot` cannot express that: Triton's wait-insertion pass picks `pendings = 0` and no
source ordering changes its mind.

Gluon can. `warpgroup_mma_wait(num_outstanding=1)` is a partial drain, and the SASS is
exactly the intended shape:

    WARPGROUP.ARRIVE ; HGMMA (scoresT) x2
    WARPGROUP.ARRIVE ; HGMMA (dpT)     x2
    WARPGROUP.DEPBAR.LE gsb0, 0x1     <- releases scoresT, dpT still computing
    MUFU.EX2 ...                      <- epilogue hides dpT

Note the issue order is the *opposite* of `triton_bwd.py`'s, whose comment says to issue
`dpT` first. That was right there -- with two full drains, issuing the tile nobody needs yet
first keeps one fewer live fp32 tile on the critical path. Here it would be wrong: the
partial wait releases the *first-issued* mma, and the one we want early is `scoresT`.

Two further things fall out of writing the schedule by hand rather than inferring it.

**`ds16` is staged in shared memory once and used twice.** dK wants it as an A operand
`[k, j]`, dQ wants it transposed `[j, k]`. From a shared-memory descriptor the transpose is
`permute`, i.e. `memdesc_trans` -- free metadata, the wgmma transpose bit. `triton_bwd.py`
instead converts the same register tile twice, once to `dot_op` for dK and once through
`#linear` + `tt.trans` for dQ. It also means dQ is `[BJ, DIM]` with M = BJ = 64, so it stays
on wgmma without the accumulator transpose that module's docstring has to pay for.

**The `[j]` vectors are loaded straight into `SliceLayout(0, MMA)`.** `mx`, `dn` and `delta`
broadcast along k, so that slice layout *is* their natural home; loading into it directly
skips the per-iteration shared-memory staging Triton's pipeliner gives them (it allocates
three `memdesc<1x64xf32>` buffers and barriers on them).

Scope: the k-exact, j-exact fast path in bf16/fp16 -- the shape the benchmark and AlphaFold 3
actually run (`N % BLOCK == 0`). Ragged tiles, fp32 inputs and autotuning stay in
`triton_bwd.py`. `scripts/proto_gluon_bwd.py` checks this against it and times both.

## It is correct, it does what it was built to do, and it is still slightly slower

**36.48 ms against `triton_bwd.py`'s 34.67 at n=1024, h=8, d=32, bf16 -- 0.95x**, from 39.03
(0.89x) before the bf16 bias below. The ratio holds across shapes: 0.96x at n=512, 0.95x at
n=2048. `triton_bwd.py` remains the shipping kernel.

The gradients are now *more* accurate than the fp32 pre-scaled path they replace: **dk and dv
come out bit-identical to the three-kernel reference** at every shape `check` covers, and dq's
worst relative error drops 2.59e-3 -> 2.51e-3. See the score-epilogue note on `BIAS_SCALED`.

### Where the 39.03 went: the staged bias, in bf16

The first ncu read of this kernel blamed `long_scoreboard` on the bias *load latency*. That
was wrong in a way worth recording, because three plausible fixes follow from it and all
three are measured duds (below). The bias tile's real cost is **throughput, not latency**:
at 16 KB of fp32 per iteration it drove `L1/TEX Cache Throughput` to 76.2 % against the
Triton kernel's 65.6 %, which was the top utilization metric in both. It also set the
shared-memory budget, and at 32 KB/stage three stages was the ceiling.

Staging it in bf16 halves both. `L1/TEX` 76.2 -> 72.6 %, `mio_throttle` -35 %,
`short_scoreboard` -36 %, and `STAGES=5` now fits at 106.5 KB -- still 2 CTAs/SM, with four
iterations of lookahead instead of two. 39.03 -> 36.48.

Two things make this pay here but not in `triton_bwd.py`, whose docstring records a bf16 bias
as a dead end:

- **There is no layout conversion.** `b_s.load(MMA_J)` lands the tile directly in the score
  accumulator's own layout, so widening is only a widening. What cost Triton 18.4 % of stall
  samples was the `#blocked -> #mma` conversion attached to its `extf`, not the `extf`.
- **The tile must be staged *unscaled*.** `_bwd_bias_prep`'s `SCALE=1.0` makes `b2t` a
  bit-exact transposed copy of the bf16 input; inv_ln2 moves into the kernel as one FMUL
  (measured cost: 0.2 %). Pre-scaling in bf16 instead rounds `b * inv_ln2` to 8 mantissa
  bits, and **that error lands in an exponent** -- dq 2.6e-3 -> 1.8e-2, against a 2e-2
  tolerance. Same speed, seven times the error. At fp32 the rounding was invisible, which is
  why the prep kernel pre-scaled in the first place.

And a bank-conflict fix comes free with it. A warp's read of the score tile covers 8 rows;
at fp32 the default NVMMA 128 B swizzle sends pairs of those rows to the same banks, and the
profile showed 16 `LDS.64` each at exactly 2x the ideal wavefront count (12.4 % of all shared
wavefronts). At bf16 the same swizzle spreads 8 rows across all 32 banks exactly once. Fixing
it *directly* at fp32, with a hand-built `SwizzledSharedLayout(vec=8, per_phase=1,
max_phase=8)`, works -- and is worth 0.25 %, i.e. nothing, because shared bandwidth was never
the binding constraint.

The one thing the bf16 bias does cost is the kernel's `PRMT 0` property: widening 32 elements
per thread is 16 `PRMT`. The wgmma schedule this module exists for is untouched -- still 5
`WARPGROUP.ARRIVE`, 16 `HGMMA`, three full drains and two partial ones.

### Everything the design predicted, it delivered

Per-iteration stalls, cycles per issued instruction, at the 39.03 ms fp32-bias point:

| | triton | gluon | |
| --- | --- | --- | --- |
| `gmma` | 0.06 | 0.06 | the wgmma sync argument is *won* -- 2 partial drains, 5 groups |
| `short_scoreboard` | 0.90 | 0.60 | |
| `barrier` | 1.20 | 0.88 | |
| `mio_throttle` | 0.79 | 0.79 | |
| **`long_scoreboard`** | **0.67** | **2.09** | the entire deficit |
| instructions | 6.67e9 | 6.43e9 | 3.6 % fewer |
| `PRMT` (layout shuffles) | many | **0** | 16 after the bf16 bias -- the widening itself |

So: better on instruction count, better on shared-memory pressure, better on barriers, no
register layout shuffles at all -- and 3.1x worse on waiting for global data, which costs
more than all of it. `long_scoreboard` is still 41.6 % of not-issued samples after the bf16
bias, so this remains the axis to attack.

`triton_bwd.py` reads the same tile through a TMA descriptor. **That was tried, and TMA made
it worse -- 54.71 ms, 0.63x, with `long_scoreboard` going 2.09 -> 4.72.** A TMA copy is one
descriptor-driven transfer issued by a single thread; many parallel `LDGSTS` cover latency
better when the prefetch distance is short and there is a single warpgroup with nothing else
to run. `STAGES=3` does not rescue it (54.73). What makes it pay in `triton_bwd.py` is not
TMA itself but where Triton's pipeliner *places* the issue relative to the wait, interleaved
with the other loads. Reproducing that by hand needs a producer/consumer split --
`ttgl.warp_specialize`, a separate warpgroup doing nothing but copies -- which is still the
real next step and a much bigger one than swapping a load.

### Six measured dead ends, from reading the stall profile instruction by instruction

The `long_scoreboard` samples cluster on instructions that look like the problem and are not.
All six of these were built and timed against 39.03 ms; none is worth keeping.

1. **Hoisting the three `[j]` vector loads to the top of the body** -- 41.42 ms, registers
   235 -> 254. `LDG.E.64` plus the epilogue that consumes it (`FADD s - row_max`, the
   `FSETP`/`FSEL` of `/row_denom`) is the largest single `long_scoreboard` cluster, so buying
   them the `wait_group` + fence + mma-issue window looks free. It is not: registers are the
   binding constraint at 2 CTAs/SM and extending liveness at all costs more than the latency
   it hides. Deleting the loads outright (`NO_VEC`) is worth only 1.2 %, so there was never
   much there.
2. **Cutting the three `fence_async_shared()` per iteration down to one.** ptxas lowers
   `fence.proxy.async.shared::cta` to four dummy `LDS` + `MEMBAR.ALL.CTA` +
   `FENCE.VIEW.ASYNC.S`, the three fences carry 10.3 % of not-issued samples all by
   themselves, all `long_scoreboard`, and Triton's hot loop pays exactly one (its other
   fifteen are in cold code). Removing the pre-mma fence -- legal, because `q_c`/`do_c` are
   `cp.async`-written and `cp.async.wait_group` + `bar.sync` is sufficient; only the
   generic-store `k_s`/`v_s` need it, once, before the loop -- is 39.17 ms, i.e. nothing.
   Merging the other two by storing `p_s` and `ds_s` together and issuing all three matmuls
   at once is 41.87 ms, worse. **The fence is where the stall lands, not what causes it**;
   move it and the wait moves with it.
3. **`REDG.E.ADD.F32x4` for the two atomics** -- 69.31 ms, 0.50x. Gluon gets `F32x2` because
   a wgmma accumulator layout gives each thread *pairs* of columns; Triton gets `F32x4` via a
   `convert_layout`, and `REDG` is 7.3 % of samples here against 2.3 % there. Doing the same
   `convert_layout` costs a shared-memory round trip *and* 16 KB of scratch, which pushed the
   kernel to 1 CTA/SM. This is precisely the layout-shuffle cost the kernel exists to avoid.
4. **Trading registers for a 3rd CTA/SM.** At `STAGES=2` (73.7 KB) 3 CTAs needs only
   `regs <= 170`, and `maxnreg=168` does reach it: 38.51 ms with 14 spills, a 3.1 % gain over
   the same-stages baseline. But `maxnreg=170` with *fewer* spills (4) is 48.19 ms, and 164 /
   160 / 152 / 144 give 38.87 / 40.79 / 41.21 / 43.26. That is not a cost model, it is a
   ptxas lottery; nothing here is safe to ship.
5. **`STAGES=4`, at any bias dtype** -- 42.78 ms at bf16 (vs 36.48 at 5 and 37.96 at 3),
   registers 186 instead of 218, shared memory well inside the 2-CTA budget. Unexplained
   scheduling cliff. Stage depth has to be measured, not reasoned about: 2, 3, 5 are fine and
   4 is not.
6. **Bounds-checking the prologue in the kernel** -- see the prologue comment. Both forms
   cost ~14 ms.

The through-line: **every attempt to reduce work by adding a runtime decision loses more to
ptxas re-scheduling than it saves.** The one change that worked removed bytes without adding
a single instruction or branch. Ablating the bias entirely is the sharpest illustration --
`NO_BIAS` deletes 16 KB of loads per iteration and runs *slower*, 45.88 ms at 179 registers.

Also worth recording because it cost an hour: `tma.async_copy_global_to_shared` does **not**
set the barrier's expected transaction count, and a TMA arrival signals *through* that
counter. Omit `mbarrier.expect(bar, bytes)` and the kernel deadlocks outright with the GPU
pinned at 100 %, no error. Triton's own pipeliner emits `ttng.barrier_expect %bar, 16384`
before every `async_tma_copy_global_to_local`. `expect` lives in `hopper.mbarrier`;
`ampere.mbarrier` has only allocate/init/wait/arrive and will let you build the deadlock.
Device-side `tma.make_tensor_descriptor` also needs `triton.set_allocator` for its tensormap
scratch, which is global process state and a wart for library code.

Three more measured non-obvious results from building it:

- **A first cut with plain `ttgl.load` + `.store()` and no pipeline ran at 47.08 ms**, with
  `long_scoreboard` at 3.93. Gluon gives you the wgmma schedule and takes away Triton's
  software pipeliner; winning the sync argument is worth nothing until you have replaced it.
- **Putting the three `[j]` vectors through the pipeline as well is a regression**,
  39.03 -> 41.11 ms with registers 235 -> 248. They are 256 B each; the staging buffers and
  extra live values cost more than the latency they hide. They stay synchronous, issued after
  the mmas so their latency overlaps.

And one hard constraint, the same one `autotune_bwd.py` records: shared memory above ~117 KB
is 1 CTA/SM and the kernel collapses (fp32 bias at `STAGES=4`, 120 KB, 55.5 ms; bf16 bias at
`STAGES=6`, 120 KB, 55.4 ms). That ceiling is what the bf16 bias buys room under: `STAGES=5`
lands at 106.5 KB.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

@gluon.constexpr_function
def _acc_layout(n_dim: int, num_warps: int = 4):
    """`NVMMADistributedLayout` for a `[64, n_dim]` fp32 wgmma accumulator.

    Hopper wgmma is m64nNk16 for 16-bit operands. This reproduces what Triton's own layout
    picker chooses for the same dot (`_mmav3_acc_layout` in the triton-to-gluon translator):
    `instr_shape = [16, n, 16]` for the largest valid `n` dividing the N extent, and
    `warps_per_cta` grown along M first. It has to be a `constexpr_function` rather than a
    `gluon.jit` one -- layouts are compile-time values, not kernel values.
    """
    # Valid wgmma N for floating point, descending.
    valid_n = [256 - 8 * i for i in range(32)]
    m = 16
    m_warps = max(64 // m, 1)
    n_warps = max(num_warps // m_warps, 1)
    max_n = max(n_dim // n_warps, 8)
    n = next(x for x in valid_n if n_dim % x == 0 and x <= max_n)
    warps = [4, 1]
    shape_per_warp = [16, n]
    while warps[0] * warps[1] < num_warps:
        if 64 > shape_per_warp[0] * warps[0]:
            warps[0] *= 2
        else:
            warps[1] *= 2
    return ttgl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=warps, instr_shape=[m, n, 16]
    )


# fmt: off
@gluon.jit
def _gl_bwd_fused(
    d_ptr, stride_dh, stride_dm, stride_dn,                        # delta [BH,N,N]
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b2t_ptr, stride_b2h, stride_b2k, stride_b2j,                   # fp32 [BH,N,PADDED_N]
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
    STAGES: ttgl.constexpr = 5,
    # Is the staged bias already multiplied by inv_ln2 (`_bwd_bias_prep`'s default), or raw?
    # Raw is what a bf16 `b2t` needs; see the note at the score epilogue.
    BIAS_SCALED: ttgl.constexpr = False,
):
    """dq, dk, dv, db in one pass, k-exact and j-exact. Grid (N // BLOCK_K, N, bh).

    Caller guarantees `N % BLOCK_K == 0`, `N % BLOCK_J == 0` and `STAGES <= N // BLOCK_J`;
    there is no ragged path and no `in_rangeT`. The mask select on `m_blk` stays, because
    that is semantics rather than bounds -- it is what gives a fully-masked row its
    `mean(V)` gradient.
    """
    input_dtype: ttgl.constexpr = q_ptr.dtype.element_ty
    # The staged bias takes whatever dtype the caller allocated `b2t` as. `_bwd_bias_prep`
    # casts on store, so bf16 costs nothing host-side and halves both the shared-memory
    # footprint of the pipeline and the per-iteration L1 traffic of its largest tile.
    bias_dtype: ttgl.constexpr = b2t_ptr.dtype.element_ty

    # --- layouts -----------------------------------------------------------------------
    # Score-space accumulators, [BLOCK_K, BLOCK_J].
    MMA_J: ttgl.constexpr = _acc_layout(BLOCK_J)
    # Gradient accumulators, [*, DIM].
    MMA_D: ttgl.constexpr = _acc_layout(DIM)
    # Shared staging. `[*, DIM]` for the four input tiles, `[BLOCK_K, BLOCK_J]` for the two
    # score tiles we hand back to the tensor cores.
    SH_D: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, DIM], input_dtype)
    SH_J: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_J], input_dtype)
    # At bf16 the default NVMMA 128 B swizzle is already conflict-free for this reader: a
    # warp's `LDS.32` covers 8 rows x 4 bytes, and the swizzle sends those 8 rows to 8
    # disjoint 16-byte chunks = all 32 banks exactly once. At fp32 the same rows pair up
    # two-to-a-bank, which is why the profile of the fp32 tile showed 16 `LDS.64` each at
    # exactly twice the ideal wavefront count.
    SH_B: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, BLOCK_J], bias_dtype)

    # Coalesced layouts for global traffic. 8 contiguous elements per thread along the last
    # axis is one 16-byte vector at 16-bit, which is what the [*, DIM] tiles want.
    BL_D: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    # 16 bytes per thread along j, whatever the bias dtype is: 4 fp32 or 8 bf16.
    BL_J: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 128 // bias_dtype.primitive_bitwidth], [4, 8], [4, 1], [1, 0])

    # --- ids ---------------------------------------------------------------------------
    pid_k = ttgl.program_id(0)
    pid_i = ttgl.program_id(1)
    pid_h = ttgl.program_id(2)

    inv_ln2: ttgl.constexpr = 1.4426950408889634
    neg_inf2 = neg_inf.to(ttgl.float32) * inv_ln2
    # sm_scale folded into log2 units, as in triton_bwd.py. Only usable when the staged bias
    # carries its own inv_ln2 already, which is the fp32 `b2t` case; see `BIAS_SCALED`.
    s2 = sm_scale.to(ttgl.float32) * inv_ln2

    start_k = pid_k * BLOCK_K
    mask_h = pid_h // H

    # --- index vectors, each in the layout of its consumer ------------------------------
    # k along the rows of a [k, j] tile, and along the rows of a [k, DIM] tile.
    k_of_j = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, MMA_J)) + start_k
    kd_r = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, BL_D))[:, None] + start_k
    kd_c = ttgl.arange(0, DIM, ttgl.SliceLayout(0, BL_D))[None, :]
    jd_r = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(1, BL_D))[:, None]
    # j along the columns of a [k, j] tile, and as a [j] vector broadcast over k.
    j_of_k = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(0, MMA_J))
    bj_r = ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, BL_J))[:, None] + start_k
    bj_c = ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(0, BL_J))[None, :]


    # --- loop-invariant per-CTA tiles ---------------------------------------------------
    k_base = k_ptr + pid_h * stride_kh + pid_i * stride_km
    v_base = v_ptr + pid_h * stride_vh + pid_i * stride_vm
    k_s = ttgl.allocate_shared_memory(
        input_dtype, [BLOCK_K, DIM], SH_D,
        ttgl.load(k_base + kd_r * stride_kn + kd_c * stride_kd))
    v_s = ttgl.allocate_shared_memory(
        input_dtype, [BLOCK_K, DIM], SH_D,
        ttgl.load(v_base + kd_r * stride_vn + kd_c * stride_vd))

    # The key mask is [k] and loop-invariant, so it is one load per CTA. Broadcast over j.
    m_ptrs = (m_ptr + mask_h * stride_mh + pid_i * stride_mm
              + (ttgl.arange(0, BLOCK_K, ttgl.SliceLayout(1, MMA_J)) + start_k) * stride_mn)
    m_blk = ttgl.load(m_ptrs) != 0                                     # [k]

    # Per-iteration staging. q, do and the bias tile are `cp.async`-ed STAGES deep, which
    # is the piece Gluon does *not* give for free: `tl.dot`-based Triton runs its own
    # software pipeliner over the j loop, and a first cut of this kernel that used plain
    # `ttgl.load` + `.store()` measured `long_scoreboard` at 3.93 cycles per issued
    # instruction against the Triton kernel's 0.67 -- 47 ms against 34.7. Winning the wgmma
    # sync argument is worth nothing if the global loads are exposed.
    q_s = ttgl.allocate_shared_memory(input_dtype, [STAGES, BLOCK_J, DIM], SH_D)
    do_s = ttgl.allocate_shared_memory(input_dtype, [STAGES, BLOCK_J, DIM], SH_D)
    b_s = ttgl.allocate_shared_memory(bias_dtype, [STAGES, BLOCK_K, BLOCK_J], SH_B)

    # Separate [k, j] buffers for pT (dV's A operand) and dsT (dK/dQ's). Sharing one costs
    # 8 KB less but needs a barrier between the dV mma and the dsT store, and any WAR
    # hazard on a wgmma operand buffer makes ptxas drain the pipe -- which is the whole
    # thing this kernel exists to avoid.
    p_s = ttgl.allocate_shared_memory(input_dtype, [BLOCK_K, BLOCK_J], SH_J)
    ds_s = ttgl.allocate_shared_memory(input_dtype, [BLOCK_K, BLOCK_J], SH_J)

    dk = ttgl.zeros([BLOCK_K, DIM], ttgl.float32, layout=MMA_D)
    dv = ttgl.zeros([BLOCK_K, DIM], ttgl.float32, layout=MMA_D)

    q_base = q_ptr + pid_h * stride_qh + pid_i * stride_qm
    do_base = do_ptr + pid_h * stride_doh + pid_i * stride_dom
    b2_base = b2t_ptr + pid_h * stride_b2h
    l_base = mx_ptr + pid_h * stride_lh + pid_i * stride_lm
    dn_base = dn_ptr + pid_h * stride_lh + pid_i * stride_lm
    d_base = d_ptr + pid_h * stride_dh + pid_i * stride_dm

    n_iters = N // BLOCK_J

    # Prologue: fill STAGES-1 buffers so the steady state always has one ready.
    #
    # `sj` is a compile-time constant here, and it has to stay one. The prologue reaches
    # j = (STAGES-2)*BLOCK_J, so `STAGES <= N // BLOCK_J` is a *precondition* -- otherwise
    # these copies run off the end of `q`, `do` and `b2t`. It is enforced by the caller
    # (`launch_gluon` clamps), not here, because both in-kernel guards were measured and
    # both are disastrous: `if pre < n_iters: <copy>` and `min(pre, n_iters-1) * BLOCK_J`
    # each cost 36.5 -> ~50 ms, with registers 218 -> 178. Making the prologue's offsets
    # depend on a runtime value at all denies ptxas the immediate-offset form and it
    # re-allocates the whole loop around it. This kernel's schedule is that brittle.
    for pre in range(STAGES - 1):
        sj = pre * BLOCK_J
        async_copy.async_copy_global_to_shared(
            q_s.index(pre), q_base + (jd_r + sj) * stride_qn + kd_c * stride_qd)
        async_copy.async_copy_global_to_shared(
            do_s.index(pre), do_base + (jd_r + sj) * stride_don + kd_c * stride_dod)
        async_copy.async_copy_global_to_shared(
            b_s.index(pre), b2_base + bj_r * stride_b2k + (bj_c + sj) * stride_b2j)
        async_copy.commit_group()

    for it in range(n_iters):
        start_j = it * BLOCK_J
        cur = it % STAGES

        nxt = it + STAGES - 1
        if nxt < n_iters:
            sj = nxt * BLOCK_J
            slot = nxt % STAGES
            async_copy.async_copy_global_to_shared(
                q_s.index(slot), q_base + (jd_r + sj) * stride_qn + kd_c * stride_qd)
            async_copy.async_copy_global_to_shared(
                do_s.index(slot), do_base + (jd_r + sj) * stride_don + kd_c * stride_dod)
            async_copy.async_copy_global_to_shared(
                b_s.index(slot), b2_base + bj_r * stride_b2k + (bj_c + sj) * stride_b2j)
        async_copy.commit_group()
        # One group per iteration, so leaving STAGES-1 outstanding is exactly "the oldest
        # has landed".
        async_copy.wait_group(STAGES - 1)

        q_c = q_s.index(cur)
        do_c = do_s.index(cur)

        # --- the two independent matmuls, scoresT first ---------------------------------
        fence_async_shared()
        t_s = warpgroup_mma(k_s, q_c.permute([1, 0]),
                            warpgroup_mma_init(ttgl.zeros([BLOCK_K, BLOCK_J],
                                                          ttgl.float32, layout=MMA_J)),
                            is_async=True)
        t_p = warpgroup_mma(v_s, do_c.permute([1, 0]),
                            warpgroup_mma_init(ttgl.zeros([BLOCK_K, BLOCK_J],
                                                          ttgl.float32, layout=MMA_J)),
                            is_async=True)

        # The three [j] vectors are read *after* the mmas are in flight, so their latency
        # overlaps them. Putting them through the cp.async pipeline as well was measured and
        # is a regression -- 39.0 -> 41.1 ms, registers 235 -> 248. They are 256 B each; the
        # staging buffers and the extra live values cost more than the latency they hide.
        row_max = ttgl.load(l_base + (j_of_k + start_j) * stride_ln)      # [j]
        row_denom = ttgl.load(dn_base + (j_of_k + start_j) * stride_ln)   # [j]
        delta = ttgl.load(d_base + (j_of_k + start_j) * stride_dn)        # [j]
        # `_bwd_bias_prep` already delivered the bias transposed to [bh, k, j], so it comes
        # out of shared memory straight in the score tile's own layout. The `.to` is
        # therefore *only* a widening -- no `#blocked -> #mma` conversion is attached to it,
        # and that conversion, not the `extf`, is what made a bf16 bias a dead end in
        # `triton_bwd.py`. It is a no-op when `b2t` is already fp32.
        b_blk = b_s.index(cur).load(MMA_J).to(ttgl.float32)

        # Partial drain: scoresT is ready, dpT is not. Everything between here and the
        # second wait runs underneath the dpT matmul.
        scoresT = warpgroup_mma_wait(num_outstanding=1, deps=(t_s,))

        if BIAS_SCALED:
            scoresT = scoresT * s2 + b_blk
        else:
            # `b2t` holds the untouched bias, so widening it is exact and the recomputed
            # scores agree with the forward's bit for bit. inv_ln2 moves into the kernel as
            # one extra FMUL per element; pre-scaling a *bf16* `b2t` instead costs 2.6e-3
            # -> 1.8e-2 on dq, because that rounding error lands in an exponent.
            scoresT = (scoresT * sm_scale + b_blk) * inv_ln2
        # The sentinel goes in after the log2 conversion, as _fwd does, so that a fully
        # masked row's exp2(scoresT - row_max) is exp2(0) and dv comes out as mean(V).
        scoresT = ttgl.where(m_blk[:, None], neg_inf2, scoresT)
        pT = ttgl.exp2(scoresT - row_max[None, :]) / row_denom[None, :]   # [k,j]

        # dV wants the weights as the forward used them, masked keys included.
        p_s.store(pT.to(input_dtype))

        dpT = warpgroup_mma_wait(num_outstanding=0, deps=(t_p,))

        dq_ptrs = (dq_ptr + pid_h * stride_dqh + pid_i * stride_dqm
                   + (ttgl.arange(0, BLOCK_J, ttgl.SliceLayout(1, MMA_D))[:, None]
                      + start_j) * stride_dqj
                   + ttgl.arange(0, DIM, ttgl.SliceLayout(0, MMA_D))[None, :] * stride_dqd)

        fence_async_shared()
        t_v = warpgroup_mma(p_s, do_c, warpgroup_mma_init(dv), is_async=True)
        dsT = pT * (dpT - delta[None, :])                                 # [k,j]
        # d(score)/dk is zero wherever the forward replaced the score with the sentinel.
        dsT = ttgl.where(m_blk[:, None], 0.0, dsT)
        dv = warpgroup_mma_wait(num_outstanding=0, deps=(t_v,))

        if NEED_DB:
            # fp32, before the bf16 cast dk and dq take, matching _bwd_b.
            dbt_ptrs = (dbt_ptr + pid_h * stride_dbth + k_of_j[:, None] * stride_dbtk
                        + (j_of_k[None, :] + start_j) * stride_dbtj)
            ttgl.atomic_add(dbt_ptrs, dsT, sem="relaxed")

        # One shared store, two reads: [k, j] for dK and its free transpose for dQ.
        ds_s.store(dsT.to(input_dtype))
        fence_async_shared()
        # dQ is issued *before* dK for the same reason scoresT is issued before dpT: wgmma
        # retires in issue order, so `num_outstanding=1` releases the *first* of the two.
        # dQ is the one needed early (the atomic consumes its tile right here) while dK only
        # has to be ready by the end of the body, so the dq atomic runs underneath dK.
        #
        # Getting this backwards is not merely slow, it is wrong, and the SASS says so
        # loudly: issuing dK first and then asking for `num_outstanding=1` on dQ reads dQ's
        # accumulator before its mma retires, and ptxas responds by abandoning the group
        # structure altogether -- ARRIVE/HGMMA/DEPBAR-to-zero around every single HGMMA,
        # 16 instead of 5, which is worse than what `tl.dot` gives.
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

    # d(score)/dk carries sm_scale, applied once per CTA. dq takes it in _bwd_scale_cast.
    dk_ptrs = (dk_ptr + pid_h * stride_dkh + pid_i * stride_dkm
               + kd_r * stride_dkn + kd_c * stride_dkd)
    dv_ptrs = (dv_ptr + pid_h * stride_dvh + pid_i * stride_dvm
               + kd_r * stride_dvn + kd_c * stride_dvd)
    ttgl.store(dk_ptrs, ttgl.convert_layout((dk * sm_scale).to(input_dtype), BL_D))
    ttgl.store(dv_ptrs, ttgl.convert_layout(dv.to(input_dtype), BL_D))
# fmt: on


def gluon_bwd_supported(n: int, dim: int, dtype: torch.dtype,
                        block_j: int = 64, block_k: int = 64) -> bool:
    """Whether `_gl_bwd_fused` can run this shape.

    It is the fast path only: 16-bit inputs, exact tiles, and DIM a multiple of the 16-byte
    vector the `[*, DIM]` blocked layout assumes.
    """
    return (
        dtype in (torch.bfloat16, torch.float16)
        and n % block_j == 0
        and n % block_k == 0
        and dim % 8 == 0
        and block_k == 64
        and triton.runtime.driver.active.get_current_target().backend == "cuda"
    )


def gluon_bwd_stages(n: int, block_j: int = 64, want: int = 5) -> int:
    """How deep to stage the j pipeline for this `N`.

    `_gl_bwd_fused`'s prologue reads `STAGES-1` j tiles ahead with compile-time offsets and
    does not bounds-check them, so this is a precondition rather than a tuning knob: see the
    prologue comment for why the guard cannot live in the kernel.
    """
    return max(1, min(want, n // block_j))
