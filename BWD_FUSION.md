# Fusing the backward — H20-3e, N=1024, H=8, D=32, BF16

The backward ran three kernels — `_bwd_q` (dq + delta), `_bwd_kv` (dk, dv), `_bwd_b` (db)
— and each one independently recomputed `scores = q·kᵀ * sm_scale + bias` and
`dp = do·vᵀ`, rebuilt `p` from `(mx, dn)` and rebuilt `ds`. **Nine matmuls and three
softmax epilogues where five and one would do.** `_bwd_b` existed only because `db` is a
reduction over the triangle axis `i`, which the other two kernels carry in their grid.

`src/trifast/triton_bwd.py` replaces all three with one kernel plus three memory-bound
passes. Repro:

```
python scripts/bench_kernels.py                    # per-kernel and end-to-end, both paths
python scripts/proto_bwd_fused.py check            # correctness matrix vs the old path
python scripts/proto_bwd_fused.py bench | ablate    # config sweep, per-piece cost
pytest tests/unit/test_bwd_fused.py                # 32 cases, fused vs three-kernel
```

`bench` sweeps the TMA bias path and repeats the top two configs with the pointer load
(marked `pointer-bias`), so the last two rows are the A/B for point 4 below.
`trifast.torch.USE_TMA_BWD_BIAS` flips it globally.

Every timing below was taken on an idle GPU — this device is shared, and
`scripts/bench_kernels.py` now refuses to measure if anyone else is using it.

## Result

```
TriFast backward algorithmic throughput — BF16 (TFLOP/s)
┌──────┬───────────────┬───────┬─────────┐
│  N   │ Three kernels │ Fused │ Speedup │
├──────┼───────────────┼───────┼─────────┤
│  512 │         41.64 │ 71.14 │   1.71x │
│  640 │         42.68 │ 72.89 │   1.71x │
│  768 │         43.16 │ 74.20 │   1.72x │
│  800 │         40.77 │ 65.98 │   1.62x │
│ 1024 │         44.00 │ 76.23 │   1.73x │
└──────┴───────────────┴───────┴─────────┘
```

The `Fused` column includes the fp32 bias pre-pass and its TMA read, both described in
[The bias tile](#the-bias-tile-and-why-the-first-read-of-this-profile-was-wrong). The
progression at n=1024, each step measured against the same reference:

| | n=1024 fused | speedup |
| --- | --- | --- |
| original | 67.60 | 1.54x |
| + fp32 pre-scaled, pre-transposed bias | 73.44 | 1.67x |
| + TMA on that bias | **76.23** | **1.73x** |

The `Three kernels` column is untouched across all three runs (43.99 → 44.02 → 44.00 at
n=1024), which is the sanity check that the reference did not drift.

End to end, allocations and all three helper passes included. Both columns are counted
against the same five matmuls -- the work a backward *has* to do -- so they compare how
fast each path delivers the same gradients rather than how fast it runs its own instruction
mix; the three-kernel path performs nine. In wall clock: **62.4 -> 35.4 ms at n=1024**, of
which the fused kernel is 34.7 ms, the delta preprocess 0.26, the dq cast plus db transpose
0.46, and the bias pre-pass 0.02.

`n=800` gains least for the same reason the forward does: `800 % 64 == 32` leaves a ragged
tile that still pays for masking.

**Why 1.54x and not the 1.8x that 9 → 5 matmuls implies.** (This section predates the fp32
bias change and its TMA read, which together took the ratio to 1.73x — past the 1.57x this
accounting predicted, because it treated the per-matmul efficiency gap as all atomics and
register pressure when a third of it was the bias widening. The shape of the argument still
holds; the 12.5 % figure does not.) The three kernels together
sustain 79.2 TFLOP/s over the 9 matmuls they actually perform — which is where the table's
43.99 comes from, `79.2 × 5/9` — while the fused kernel sustains 69.1 over its 5. That
12.5 % lower per-matmul efficiency is the atomics (additive, see below) and the register
pressure: `1.8 × 0.875 = 1.57`, against a measured 1.54x.

## What was wrong

| kernel | ms | matmuls | reg/thread | occupancy | issue slots | DRAM |
| --- | --- | --- | --- | --- | --- | --- |
| `_bwd_q` | 16.78 | 3 | 79 | 37.4 % | 79.6 % | 3.8 % |
| `_bwd_kv` | 30.58 | 4 | **196** | **12.45 %** | **44.4 %** | 2.1 % |
| `_bwd_b` | 15.10 | 2 | 128 | 24.5 % | 78.9 % | 12.7 % |

`_bwd_q` and `_bwd_b` were instruction-issue bound, like the forward. `_bwd_kv` was
neither that nor bandwidth bound: at 196 registers it got 2 CTAs/SM and stalled at 4.48
warp-cycles per issued instruction. The cause was two lines,
`tl.dot(tl.trans(sm_value), do)` and `tl.dot(tl.trans(dscores), q)` — transposes of
**computed fp32 accumulators**.

## The two structural changes

(The bias tile is a third and a fourth; it has
[its own section](#the-bias-tile-and-why-the-first-read-of-this-profile-was-wrong).)

**1. Every score tile is computed transposed, as `[k, j]`.** dV needs `pᵀ` and dK needs
`dsᵀ`. In the `[j, k]` orientation those are transposes of an MMA accumulator, which lower
to a shared-memory round trip: `local_alloc` + `local_load`, four `stmatrix`, four
`ldmatrix`, and two extra `bar.sync`. Transposing a *loaded* tile costs nothing at all —
it becomes `ttg.memdesc_trans`, pure metadata on the shared-memory operand descriptor that
lands as the wgmma transpose bit, with zero ldmatrix/stmatrix and an identical register
count:

```
%b  = ttg.local_alloc      : tensor<64x32xbf16, #blocked> -> memdesc<64x32xbf16, #shared>
%bT = ttg.memdesc_trans %b {order = array<i32: 1, 0>} -> memdesc<32x64xbf16, #shared1>
      ttng.warp_group_dot %a, %bT, %cst
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, ...}>
```

So the kernel loads `q`, `do`, `k`, `v` and the bias tile in their natural orientations and
lets `tl.trans` handle the rest. Flipping the orientation alone takes the dk/dv work from
30.6 ms to **26.0 ms at 211 registers**. Loading a tile twice instead of transposing it is
*worse* — 26.9 ms and 60 % more shared memory.

**2. `db` and `dq` are accumulated with fp32 atomics.** `db` reduces over `i`, which is a
grid axis here, so a cross-CTA reduction is unavoidable — that is precisely why `_bwd_b`
existed. `dq` reduces over `k`, which the grid splits. `db` accumulates into a
**transposed** `[bh, k, j]` fp32 buffer because `ds` is produced as `[k, j]` and a
transposed atomic would scatter four bytes per lane; `dq` accumulates in fp32 (+1.07 GB at
n=1024) because a bf16 atomic would round once per k block instead of once in total, and
db was already the most precision-sensitive of the five gradients.

`dk` and `dv` stay register accumulators with a single plain store, so they remain exact
and deterministic — and in fact bit-identical to the old kernels in almost every tested
shape.

## What each piece costs

`proto_bwd_fused.py ablate` prices each gradient by flipping its constexpr off. It sweeps
`num_stages=2`, so these are **not** at the shipping config (`num_stages=3`) — they are
internally comparable, not comparable to the totals above. At
BLOCK_J = BLOCK_K = 64, `num_warps=4`, `num_stages=2`, with the fp32 TMA bias:

| | ms | Δ | reg |
| --- | --- | --- | --- |
| dk + dv (4 matmuls) | 22.51 | — | 246 |
| + db atomic | 31.66 | +9.15 | 242 |
| + dq (5th matmul and its atomic) | 37.91 | +6.25 | 255, 10 spills |

Both fusions still pay for themselves several times over: db deletes a 15.10 ms kernel and
dq a 16.78 ms one. The pre-bias numbers, for reference, were 26.02 / 31.88 / 39.77 — so the
bias change is worth ~3.5 ms on the dk/dv base alone, before any gradient is added.

Two results from the original sweep that the bias change does not affect, kept because they
are the reason the kernel is shaped the way it is: dq in the transposed orientation cost
+5.31 ms (255 regs, 18 spills), and dropping the peeled fast path cost +2.62 ms.

## Atomics

Measured standalone at the real update counts, `db` being 8.59e9 fp32 updates into a 32 MB
buffer with 1024-way cross-CTA collision per address:

| | G updates/s | ms |
| --- | --- | --- |
| `tl.atomic_add`, `sem="relaxed"` | 941 | 9.12 |
| `tl.atomic_add`, default `sem` (`acq_rel`) | 388 | 22.13 |
| plain `tl.store` (the floor) | 1171 | 7.34 |
| TMA `desc.atomic_add` | 939 | 9.15 |

Three things follow. **`sem="relaxed"` is mandatory, not an optimization** — Triton's
default is `acq_rel` and it is 2.4x slower. Atomics run at 80 % of store throughput, so the
traffic itself is affordable on this card. And **TMA reduce is dead even**, so it buys
nothing: the path is L2-bandwidth bound, not issue bound.

The uncomfortable part: **the atomics do not overlap with the math.** db costs +5.86 ms in
the kernel against 9.12 ms standalone, so roughly a third hides and the rest is additive.
Raising occupancy is the only lever, and this kernel does not have the registers to spare.

## Things that look right and are not

- **dq in the transposed orientation.** `dqᵀ[d,j] = dot(trans(k), dsᵀ)` avoids the one
  remaining accumulator transpose, so it looks strictly better. Its M is DIM, and M < 64 is
  not selected for wgmma: at DIM=32 it demotes to `mma.sync` and costs 5.3 ms. At DIM=64 it
  faults — `k_blk` is already a wgmma operand untransposed, and asking for both of its
  layouts gives `Out-of-range shared or local address` under compute-sanitizer. Pay the
  accumulator transpose; it also lands dq in its natural layout, which saves the epilogue a
  transpose.
- **`num_warps=8`.** 84.6 ms against 39.8 at the same tile — 2.1x slower. These shapes want
  one warpgroup.
- **Capping registers to buy occupancy.** `maxnreg=168` spills 86 slots and costs 12 ms;
  `maxnreg=128` spills 50 and costs 10 ms. The kernel runs at 250 registers and ~12.5 %
  occupancy *by design*; the win is the matmul count, not latency hiding.
- **`BLOCK_K = 32`.** 119.9 ms against 39.8 — a 3x cliff, not a gradient. BLOCK_K is the M
  dimension of the dV and dK dots in this orientation, so below 64 they leave wgmma too.
- **fp32 at DIM=128 with 64×64 tiles.** `input_precision="ieee"` disables the tensor cores
  and stages both dot operands through shared memory: 255,488 B requested against a
  232,448 B limit. `prune_bwd_fused_configs` keeps only 32×32 there. (Worth noting the old
  `_bwd_kv` spills 1342 slots on that shape, so this was already a sore spot.)
- **Autotuning a kernel that accumulates.** This one is the trap worth remembering. The
  autotuner benchmarks each candidate config by launching it many times, and every launch
  adds another copy into the db and dq accumulators. `reset_to_zero` is what keeps that out
  of the answer, and it fires in eager — but a **cold cache inside a compiled region** left
  dq and db ~400x too large. Not subtly wrong: a large multiple, which is why
  `tests/unit/test_bwd_fused.py::test_autotuning_does_not_corrupt_the_accumulators` exists.
  Traced and fake-tensor execution therefore takes `_bwd_fused` with a pinned config rather
  than the autotuner (`pinned_bwd_fused_config`), the same split as `_fwd`/`_fwd_pointer`
  but for correctness rather than compatibility.

## Correctness

`dk` and `dv` come out **bit-identical** to the three-kernel path in most tested shapes —
the score matmul reduces over the same `d` axis in the same order, so `ds` is unchanged.
`dq` differs by up to 2.5e-3 relative in bf16 (one bf16 ulp is 3.9e-3) because its
k-reduction now happens in atomic order, and `db` by up to 6.4e-4 in bf16 and ~3e-7 in
fp32. Bitwise run-to-run reproducibility of dq and db is gone; that is the cost of the
atomics, and it is far below one ulp of the output dtype.

Verified across n ∈ {16, 17, 64, 65, 96, 100, 128, 130, 200, 800, 1024},
DIM ∈ {16, 32, 64, 128}, fp32/fp16/bf16, random / absent / fully-masked-row masks, and both
tile shapes: 419 existing tests (including all 8 fully-masked cases), 32 new ones, and
`opcheck`.

## The profile of the fused kernel

`ncu --set full --kernel-name regex:_bwd_fused --launch-skip 1 --launch-count 1 -o
bwd_fused -f python scripts/ncu_kernels.py bwd_fused -n 1024`, against the two kernels it
replaces. ncu locks the SM clock to 1.63 GHz where the benchmark runs at ~1.79, so its
44.06 ms is the same speed as the benchmark's 39.7 ms.

| | fused | `_bwd_kv` | `_bwd_q` |
| --- | --- | --- | --- |
| Duration (clock-locked) | 44.06 ms | 32.87 ms | 18.26 ms |
| Registers / thread | 255 | 196 | 79 |
| Achieved occupancy | 12.47 % | 12.45 % | 37.38 % |
| Issue slots busy | 34.1 % | 44.4 % | 79.6 % |
| Compute (SM) throughput | 47.4 % | 50.9 % | 79.6 % |
| Memory throughput (SoL) | **55.2 %** | 33.8 % | 40.3 % |
| DRAM throughput | 2.6 % | 2.1 % | 3.8 % |
| L2 hit rate | 95.6 % | 89.7 % | 87.9 % |
| ALU pipe | 19.7 % | 34.6 % | 61.2 % |
| Warp inst / cycle (SM) | 106.5 | 138.6 | 248.4 |

The instruction pressure that made the old kernels issue-bound is gone — the fused kernel
issues 106 instructions per cycle against `_bwd_q`'s 248, and its ALU pipe is a third of
`_bwd_q`'s, because one softmax epilogue now serves five matmuls instead of three epilogues
serving nine. What replaced it is **latency**: 255 registers gives 2 CTAs/SM, 8 warps, and
ncu's own verdict is "all compute pipelines are under-utilized… doesn't issue enough warps".
2749 GFLOP in 44.06 ms is 62.4 TFLOP/s against a 130 TFLOP/s HGMMA ceiling at that clock,
i.e. 48 % — where the forward reaches 72 %.

Memory throughput at 55 % with DRAM at 2.6 % is the atomic traffic, and the counters confirm
the model exactly:

```
smsp__inst_executed_op_global_red.sum              100,663,296
l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum  1,610,612,736   -> 51.5 GB of L2
```

12.88e9 updates (8.59e9 for db, 4.29e9 for dq) ÷ 32 lanes ÷ 4 elements each = exactly
100,663,296 instructions, so Triton is emitting `red.global.add.v4.f32` and the sector count
is 51.5 GB with **zero** amplification — perfectly coalesced, entirely L2-resident. Nothing
to win on the addressing side; the only lever is issuing more warps so it overlaps.

### The same sections after the bias change

Both columns are the 8-section collection above, same command, idle GPU.

| | before | after |
| --- | --- | --- |
| Duration (clock-locked) | 44.10 ms | **37.74 ms** |
| Executed instructions | 7.730e9 | **6.671e9** |
| Compute (SM) throughput | 47.4 % | 55.4 % |
| Memory throughput (= L1/TEX) | 55.1 % | **67.2 %** |
| DRAM throughput | 2.6 % | 3.1 % |
| L2 hit rate | 95.6 % | 96.1 % |
| Issue slots busy | 34.1 % | 34.4 % |
| Warp cycles / issued inst | 5.84 | 5.79 |
| Registers / thread | 255 | 255 |
| Achieved occupancy | 12.46 % | 12.46 % |
| Active / eligible warps per sched | 1.99 / 0.45 | 1.99 / 0.46 |
| Block limit (registers / smem) | 2 / 3 | 2 / **2** |
| Dynamic shared memory | 41.98 KB | 83.97 KB |

Stall reasons, in cycles per issued instruction:

| | before | after |
| --- | --- | --- |
| barrier | 1.31 | 1.20 |
| **long_scoreboard** | 1.14 | **0.67** |
| selected (i.e. useful issue) | 1.00 | 1.00 |
| short_scoreboard | 0.79 | 0.90 |
| **mio_throttle** | 0.40 | **0.79** |
| wait | 0.60 | 0.61 |
| not_selected | 0.32 | 0.34 |

**Occupancy, registers and eligible warps are all unchanged.** The 14 % came from executing
14 % fewer instructions, not from hiding latency better — which is what the change claimed
and is worth having confirmed rather than assumed. `long_scoreboard` fell 41 %: the bias
load left the critical path, TMA being async. It was paid for in MIO pressure —
`mio_throttle` doubled and `short_scoreboard` rose — which is the `LDS` increase showing up
as a stall reason rather than only as an instruction count.

**`Block Limit Shared Mem` dropped 3 → 2, so shared memory now co-binds with registers.**
Before the change registers alone capped occupancy and 42 KB of 233 KB was idle; at 84 KB
both limits are 2 CTAs/SM. Nothing is lost today, but shared memory has stopped being free,
which matters for the two remaining leads that want to spend it (`num_stages=4`, and the
smem accumulator bank in lead 5).

**Two traps when re-profiling this kernel.** ncu's multi-pass replay makes autotune trial
timings meaningless, so a **cold** autotune cache under ncu can select the wrong config:
one 20-pass run here profiled `num_stages=2` (41.06 ms, 58.37 KB) instead of the shipping
`3` (37.74 ms, 83.97 KB), and five repeats with a warm cache all gave `3`. Warm the cache
with `scripts/bench_kernels.py` first and check `Dynamic Shared Memory Per Block` identifies
the config you meant to measure. And at `num_stages=3` ptxas reorders hard enough that
**line-level attribution smears** — 48.6 % of samples land on the softmax line and 24.5 % on
the loop header, where the scheduler parks its waits rather than where the work is. Compare
opcodes, not lines.

## The bias tile, and why the first read of this profile was wrong

Everything above this section was derived from `--set full` section summaries. Those say
the kernel is latency bound and point at occupancy, which is true but not actionable. Adding
**PC sampling** (`--section SourceCounters --import-source yes`) attributes stalls to
individual instructions, and it puts **22.95 % of all not-issued samples on a single line**:
the bias load.

```
ncu --kernel-name regex:_bwd_fused --launch-skip 1 --launch-count 1 \
    --section SourceCounters --import-source yes -o bwd_src -f \
    python scripts/ncu_kernels.py bwd_fused -n 1024
```

| line | % stalls | dominant opcodes |
| --- | --- | --- |
| `b_block = tl.load(bt_ptrs).to(tl.float32)` | **22.95** | PRMT 12.28, CS2R 6.09, STS 1.62, **LDG 0.45** |
| `dq_tile = tl.dot(tl.trans(ds16), k_blk)` | 14.76 | WARPGROUP 14.11 |
| `tl.atomic_add(dbt_ptrs, dsT, ...)` | 12.73 | **IMAD.MOV 9.52**, STS 1.27, REDG 0.86 |
| `dpT = tl.dot(v_blk, tl.trans(do_block))` | 12.40 | WARPGROUP |
| `ds16 = dsT.to(input_dtype)` | 8.26 | F2FP 3.79, STSM 2.24, LDSM 2.24 |
| `scoresT = tl.dot(k_blk, tl.trans(q_block))` | 7.42 | WARPGROUP |
| `tl.atomic_add(dq_ptrs, dq_tile, ...)` | 5.68 | |

**`LDG` is 0.45 %.** The load was never the problem — the "not worth investigating" entry
that measured its coalescing was right about coalescing and wrong about the conclusion. The
cost sat between the load and the FMA: `arith.extf` bf16→fp32 over 4096 tile elements
every j-iteration, one `PRMT` per element against a `CS2R`-materialised zero, plus a
`#blocked → #mma` layout conversion. 18.4 % of the kernel, in two opcodes that do no
arithmetic at all.

So `_bwd_bias_prep` widens, scales and transposes the bias once, host-side, into
`b2t[bh, k, j]` fp32 (+33 MB, 0.02 ms). Pre-scaling by `inv_ln2` also collapses
`(scoresT * sm_scale + b) * inv_ln2` into one FFMA. Measured A/B in one session on an idle
GPU, `BLOCK_J = BLOCK_K = 64`, `num_warps = 4`:

| | bf16 bias | fp32 pre-scaled | |
| --- | --- | --- | --- |
| best config | s2, **40.55 ms** | s3, **36.53 ms** | **−9.9 %** |
| end to end, all passes | 41.28 ms | 37.28 ms | −9.7 % |
| clock-locked (ncu) | 44.10 ms | 39.81 ms | |
| registers / spills | 255 / 0 | 255 / 10 | |
| FMUL per score element | 3.03 | **2.03** | exactly as predicted |
| total instructions | 7.73e9 | 7.34e9 | −5.0 % |
| SM throughput | 47.4 % | 52.5 % | |

And the mechanism, by stall share, with the TMA step (below) in the third column:

| opcode | before | fp32 bias | + TMA |
| --- | --- | --- | --- |
| PRMT | 12.28 % | **0.00 %** | 0.00 % |
| CS2R | 6.17 % | **0.05 %** | 0.07 % |
| F2FP | 3.91 % | 0.33 % | 0.45 % |
| **IMAD** | 12.70 % | 12.82 % | **7.16 %** |
| STS | 5.90 % | 4.47 % | 5.46 % |
| **LDS** | 5.18 % | **13.36 %** | 13.48 % |
| WARPGROUP | 28.82 % | 26.41 % | 28.08 % |

The widening is gone outright. **The trade is `LDS`**: an fp32 tile is twice the bytes, so
Triton stages it in shared memory instead of shuffling it in registers — which is also why
`num_stages` flipped from 2 to 3 (the fp32 tile makes deeper pipelining pay; see
`autotune_bwd.py`) and why shared memory went 42 → 71 KB. Registers stay pinned at 255, so
the extra stage costs no occupancy.

### Then TMA on the same buffer

`b2t` is stored `[bh, k, j]` precisely so a descriptor can read it: `TensorDescriptor`
asserts `strides[-1] == 1`, so TMA cannot describe a transposed view, and `.T` on this tile
is not free the way it is on a wgmma operand (it feeds an fp32 add, not a wgmma). A rank-2
descriptor over `b2t.reshape(bh*N, PADDED_N)` with a `[BLOCK_K, BLOCK_J]` box, offset
`[start_h*N + start_k, start_j]`, does it with no transpose anywhere. **34.70 ms against
36.59 at the same config, −5.2 %**, and spills 10 → 4.

**It did not work the way it was predicted to, which is the part worth keeping.** The
theory was that TMA removes the `LDS` traffic above. It does not: `LDS` *rose*, 397.9 M →
509.3 M instructions, because the tile still has to be read out of shared memory into
`#mma` and is twice the bytes it used to be. What TMA actually removed was the *store* half
— `shared_st` 172.0 M → 138.4 M — and, mostly, `IMAD`: 12.8 % → 7.2 % of stall samples,
nearly all `IMAD.MOV.U32` register marshalling, which eases once the tile stops competing
for the register file. Total instructions 7.34e9 → 6.67e9, 39.81 → 37.69 ms clock-locked.

Three implementation traps, all of which bite silently:

1. **Attach the box pre-hook per config (`config.pre_hook`), never `autotune(pre_hook=…)`.**
   Triton installs the `reset_to_zero` hook only when the constructor's `pre_hook` is None,
   so passing one stops `dbt`/`dq` being zeroed between autotune trials — a ~400x wrong
   answer, which is what `test_autotuning_does_not_corrupt_the_accumulators` guards.
2. **A stale on-disk config entry can come back without its pre-hook** and launch the
   placeholder box. `autotune.py` re-identifies the live `Config` to restore the
   non-serializable hook, but only if it still matches one; invalidate the cache when the
   config list changes.
3. **The traced/fake path must keep the pointer load.** Fake tensors cannot build a
   tensormap, and that path launches the bare kernel with `pinned_bwd_fused_config`, where
   no config pre-hook runs at all. `scripts/proto_bwd_fused.py` has the same problem for
   the same reason and sets the box by hand.

One note on reading the table above (the profiling traps themselves are in
[the section summaries](#the-same-sections-after-the-bias-change)). The db atomic's 12.73 %
is **`IMAD.MOV.U32`, not memory**: 201 M register moves, because `REDG.ADD.F32.128` wants
data and address contiguous and at 255 registers ptxas has no allocation freedom. The atomic
micro-benchmark below measures the traffic correctly; the traffic is not the cost.

## The decomposition is at a local optimum, and there is a conservation law

Worth writing down, because the 14 ms of atomics is the obvious thing to attack and three of
the obvious attacks are provably dead.

A CTA tiles two of the three axes `(j, k, i)` and *loops* the third. The looped axis's
gradient reduces inside registers and is free; the other two are split across CTAs and must be
atomic. With `E_g` elements and `R_g` contributing CTAs per gradient, updates are `E_g · R_g`:

| gradient | elements | contributing CTAs | updates |
| --- | --- | --- | --- |
| `db[j,k]` | `bh·n²` | `n/IC` | `bh·n³/IC` |
| `dq[i,j,:]` | `bh·n²·D` | `n/BK` | `bh·n³·D/BK` |
| `dk`,`dv[i,k,:]` | `bh·n²·D` each | `n/BJ` | `2·bh·n³·D/BJ` |

`T = bh·n³ · (1/IC + D/BK + 2D/BJ)`, minus the looped axis's term. At `D=32, BJ=BK=64, IC=1`:

| CTA loops | free | pays | `T / bh·n³` | fp32 scratch |
| --- | --- | --- | --- | --- |
| **j** (shipped) | dk, dv | dq 0.5 + db 1.0 | **1.5** | 1.1 GB |
| i (`_bwd_b`'s shape) | db | dq 0.5 + dk,dv 1.0 | **1.5** | 3.2 GB |
| k | dq | dk,dv 1.0 + db 1.0 | 2.0 | 2.2 GB |

**Making `db` free costs exactly what `db` cost.** The two candidates tie at 12.9e9 updates and
the shipped one wins on scratch. The law is tight enough that hybrids do not escape it either:
covering two `i` per CTA but atomically reducing `dk`/`dv` for the second one trades 4.3e9 db
updates for 4.3e9 dk/dv updates, exactly.

**Persistent scheduling does not help.** `T` counts (CTA, output-element) incidences, which the
tiling fixes — not the launch shape. Its only effects here would be launch-overhead
amortisation (131,072 CTAs over 40 ms, noise) and L2 locality (already 95.6 % hit). It is only
a vehicle for register accumulation across `i`, and an inner loop is a simpler vehicle.

**`BLOCK_I > 1` is the one term that is not conserved, and it is register-dead.** Driving
`1/IC` down would cut atomics 33 % at IC=2 (12.885e9 → 8.590e9) *and* load sectors 24 %
(1.159e9 → 0.889e9, since the bias tile is i-invariant and is half the 16.75 KB loaded per
j-iteration) — worth ~2.9 ms. It needs `IC` copies of the `dk`/`dv` accumulators, which are
live values that cannot be rematerialised. Measured, by keeping a second accumulator pair live
across the j loop at `BJ=BK=64, w4, s2`:

| | regs | spills |
| --- | --- | --- |
| dk/dv only, one pair (baseline) | 222 | 0 |
| dk/dv only, **two pairs** | 252 | 0 |
| everything, one pair (shipped) | **255** | 0 |
| everything, **two pairs** | 255 | **22** |
| everything − db atomic, two pairs | 255 | **8** |

The second pair costs **+30 registers**, and 2 CTAs/SM at 128 threads allows only
`65536/(2·128) = 256`. So IC=2 spills at the fast config *even if* the db atomic were removed
entirely — and removing it is worth only 12 registers (255 → 243), not the 64 that sizing
`dbt_ptrs` as an i64 tile would suggest; the compiler strength-reduces it. `BLOCK_J=32` does fit
(226 → 238, no spills) but starts 8.75 ms behind (49.30 vs 40.55 ms), which is more than IC=2
can return. **Do not re-attempt this without first finding ~32 registers elsewhere.**

## Leads, in descending value

1. **Gate db on `ctx.needs_input_grad`.** `NEED_DB` is already a constexpr; threading it
   through hands back the whole 5.86 ms whenever the bias needs no gradient. No register cost,
   no numerical change — the best remaining lead by a distance.
2. **The duplicated j loop costs 0.8 ms.** `_bwd_fused` instantiates `_bwd_j_loop` twice, on
   a uniform branch over whether this CTA's k tile is ragged, which is worth 2.6 ms but
   costs 0.8 in code size and register pressure. A cheaper way to specialize would net the
   difference.
3. ~~**TMA on `b2t`.**~~ **Done** — see [Then TMA on the same
   buffer](#then-tma-on-the-same-buffer). Worth −5.2 %. The entry this replaces predicted
   the win would come from removing `LDS`; it came from `IMAD` register marshalling
   instead, and `LDS` went up.
4. **The wgmma drains.** `WARPGROUP` is 28.1 % of stalls, in three
   `WARPGROUP.DEPBAR.LE gsb0, 0x0` per j-iteration — one after `dpT`, one after `scoresT`,
   one after `dq_tile`, each draining to zero outstanding. `dpT` and `scoresT` are
   independent and never overlap. The source already issues them adjacently and consumes
   them late, so there is no obvious source-level fix; this is a Triton pipelining question
   or a Gluon one.
5. **Shared-memory accumulators are the real ceiling, and Triton cannot express them.** This
   one got *harder*, not easier. Shared memory used to be the idle resource — 42 KB of
   233 KB, with registers alone capping occupancy. At 84 KB it now co-binds at 2 CTAs/SM
   (`Block Limit Shared Mem` 3 → 2), so a `dk`/`dv` accumulator bank has to fit in what is
   left rather than in a third of the SM. It is still the lever that would make `IC=4-8`
   free of register cost, which is where the atomics actually collapse, but the budget it
   draws on is now shared with the bias tile — and reverting to the pointer bias to get that
   space back costs 5.2 %.
   `allocate_shared_memory` exists in this Triton's Gluon frontend
   (`triton/experimental/gluon/language/_core.py:487`), at the cost of hand-written layouts and
   pipelining on an experimental API. Cluster/DSMEM reduction of `db` is *not* available even
   there — Gluon's Hopper cluster surface is only `arrive`/`wait`
   (`.../nvidia/hopper/cluster.py`), with no remote-shared addressing or cluster reduction.

## Not worth investigating

- **DRAM.** The fused kernel moves ~4.3 GB at n=1024 against a 35 ms runtime; DRAM
  throughput is 3.1 % of peak.
- **Pointer arithmetic, including the rank-2 pointer tensors.** Real 64-bit address math
  (`IMAD.WIDE`) is **0.47 %** of stall samples. The `IMAD` line in the tables above looks
  alarming at 7–13 %, but it is almost entirely `IMAD.MOV.U32` — register moves issued on
  the FMA pipe, i.e. a register-pressure symptom, not addressing. Nothing to win by
  reshaping how the pointers are built.
- **The atomics' memory traffic.** `REDG` itself is 1.9–2.9 % of stalls, DRAM is 2.6 %, L2
  hit is 95.6 %, and the sector count shows zero amplification. The db atomic's 12.7 % is
  the `IMAD.MOV` marshalling above, because `REDG.ADD.F32.128` needs data and address
  contiguous and at 255 registers ptxas has no freedom to place them.
- **The mask.** In this orientation `mask[i, k]` is loop-invariant, so it is one `[k]` load
  per CTA rather than one per iteration, and its `[:, None]` broadcast is split across warps
  rather than replicated. The forward's 32x LDG amplification simply does not arise.
- ~~**The bias transposed load.**~~ and ~~**Keeping the bias tile in bf16 into the
  FMA.**~~ **Both of these were wrong, and they were wrong in an instructive way** — see
  [The bias tile](#the-bias-tile-and-why-the-first-read-of-this-profile-was-wrong). The
  claims themselves hold up: the transposed read really does coalesce into
  `ld.global.v4.b32`, and keeping the tile bf16 *into the FMA* really does cost registers.
  What was wrong was concluding there was nothing there. Both entries measured the load
  and the FMA and never looked at the `extf` between them, which was 18.4 % of the
  kernel's stall samples.

---

# How the fused kernel is parallelized

Triangle attention is `N` independent attention problems per `(batch, head)`: for each row
`i`, every `j` attends over every `k`, with a bias `b[j, k]` that is **shared across all
`i`** and a mask `m[i, k]` that is shared across all `j` and all heads.

```
o[i, j, :] = Σ_k softmax_k( q[i,j,:]·k[i,k,:] * sm_scale + b[j,k] ) · v[i,k,:]
```

That `i` axis is where the parallelism comes from, and the fact that the bias has no `i`
axis is where all the difficulty comes from.

## The four gradients reduce over three different axes

| gradient | one element per | reduces over | how this kernel gets it |
| --- | --- | --- | --- |
| `dv[i,k,:]` | `(i, k, d)` | `j` | the CTA's own loop — register accumulator, **exact** |
| `dk[i,k,:]` | `(i, k, d)` | `j` | same |
| `dq[i,j,:]` | `(i, j, d)` | `k` | split across k-tile CTAs — **fp32 atomic** |
| `db[j,k]` | `(j, k)` | **`i`** | split across the grid's `i` axis — **fp32 atomic** |

No single loop nest can make all four free: `dk`/`dv` want `j` innermost, `dq` wants `k`
innermost, and `db` wants `i` innermost. Exactly one of the three can be the inner loop, so
the other two reductions have to cross CTA boundaries. That is the whole design problem, and
it is why the original code had three kernels — each one picked a different inner loop.

## The decomposition

```
grid = (cdiv(N, BLOCK_K),  N,  bh)          # = (16, 1024, 8) = 131,072 CTAs at n=1024
         pid0: k-tile      pid1: i   pid2: batch*head
```

One CTA owns **one k-tile of one attention problem**: 64 keys, for a single `(bh, i)`. It
loads `k`, `v` and the key mask once, then walks all of `j`:

```
CTA (pid_k, i, bh), 4 warps:
  k_blk[64,32], v_blk[64,32], m_blk[64]     <- loaded once, loop-invariant
  dk[64,32], dv[64,32]                      <- fp32 register accumulators
  for j_block in 0 .. N/BLOCK_J:            <- 16 iterations at n=1024
      ... 5 matmuls, one softmax epilogue ...
      dk += ...  ;  dv += ...               <- stays in registers
      atomic_add(db[j_block, k-tile])       <- 1024 CTAs (all i) collide here
      atomic_add(dq[i, j_block])            <- 16 CTAs (all k-tiles) collide here
  store dk, dv                              <- plain stores, written exactly once
```

`pid0` is the fastest-varying dimension, so the 16 CTAs that share an `(bh, i)` slice are
co-resident and read the same `q`, `do` and bias tiles out of L2 — measured 95.6 % L2 hit
rate, with DRAM at 2.6 %.

Scale at n=1024: 131,072 CTAs against 78 SMs holding 2 each, so ~840 waves. Load balance is
trivially even — the "triangle" in triangle attention is the pair-representation structure,
not a causal mask, so every `(j, k)` pair is computed and every CTA runs the same 16
iterations. And `i` alone supplies 1024-way parallelism, which is why there is no need to
split `j` as well.

## Why not a different inner loop

| CTA owns | inner loop | free | atomic | atomic updates at n=1024 |
| --- | --- | --- | --- | --- |
| **k-tile, one `i`** (chosen) | `j` | dk, dv | dq, db | 4.3e9 + 8.6e9 = **12.9e9** |
| j-tile, one `i` | `k` | dq | dk, dv, db | 8.6e9 + 8.6e9 = 17.2e9 |
| `(j,k)` tile, all `i` (`_bwd_b`'s shape) | `i` | db | dq, dk, dv | 12.9e9, in 3.2 GB of fp32 scratch |

The middle row is worse because `dk` and `dv` are *two* tensors and each is `DIM` wide, so
atomically reducing them costs twice what reducing `dq` does. The bottom row makes `db` free
— it is what `_bwd_b` did, and why that kernel existed — but then all three of the
`[…, DIM]` gradients need atomics and fp32 scratch, and the CTA loses the `k`/`v` reuse that
makes the loop cheap. The chosen row puts the two *widest* gradients in registers and pays
atomics for the narrow one (`db`, a `[j,k]` tile with no `d` axis) plus the one that is
cheapest to split (`dq`, whose contribution count falls as `1/BLOCK_K`).

The counts above are not estimates: `smsp__inst_executed_op_global_red.sum` is exactly
100,663,296 = 12.88e9 / 32 lanes / 4 elements per lane.

## What one j-block computes

Everything is held in `[k, j]` orientation so that dV and dK need no accumulator transpose
(see the top of this document). Six steps at `BLOCK_J = BLOCK_K = 64, DIM = 32`, five of
them matmuls — the ones carrying an `M/N/K` annotation:

```
      loaded:  k_blk[64,32]  v_blk[64,32]  q[64,32]  do[64,32]  b2t[64,64] fp32
               mx,dn,delta[64]

  (1) sT[64,64]  = k_blk @ trans(q)         M=BK  N=BJ  K=DIM     recompute the scores
      sT         = sT * s2 + b2t, then the mask selects            one FFMA; s2 and b2t
                                                                  both carry inv_ln2
  (2) pT[64,64]  = exp2(sT - mx) / dn                             rebuild p, no online softmax
  (3) dpT[64,64] = v_blk @ trans(do)        M=BK  N=BJ  K=DIM     independent of (1)
      dsT[64,64] = pT * (dpT - delta)                             softmax jacobian
  (4) dv[64,32] += pT   @ do                M=BK  N=DIM K=BJ
  (5) dk[64,32] += dsT  @ q                 M=BK  N=DIM K=BJ
  (6) dq[64,32]  = trans(dsT) @ k_blk       M=BJ  N=DIM K=BK      the one transpose paid for
```

The dependency graph is shallow, which is what lets one epilogue serve five matmuls:

```
   k,q ──(1)──> sT ──> pT ──┬──(4)──> dv
                            ├──> dsT ──┬──(5)──> dk
   v,do ─(3)──> dpT ────────┘          ├──(6)──> dq   (atomic)
                                       └───────> db   (atomic, fp32, before the bf16 cast)
```

`dpT` has no dependence on the score tile, so it is issued first — scheduling it between
`pT` and `dsT` would put a third live `[k,j]` fp32 tile on the critical path, and at 255
registers there is no room for one.

All five have the same `M·N·K` — two are `BK×BJ×DIM` and three are `BK×DIM×BJ`, which
coincide at `BLOCK_J = BLOCK_K` — so a j-block is 5 × 2·64·64·32 = 1.31 MFLOP, and
1.31 MFLOP × 16 j-blocks × 131,072 CTAs = **2749 GFLOP** for the whole backward. Five
matmuls' worth, against the nine the three original kernels performed between them.

Two details that fall out of the orientation. `p` is rebuilt straight from the forward's
`(mx, dn)` with a single `exp2`, so the backward has no online-softmax bookkeeping at all.
And `m_blk` is indexed `[i, k]`, which makes it **loop-invariant** for a CTA that owns a
k-tile — one `[k]` load per CTA instead of one per iteration, which is why the forward's
32x mask-load amplification never appears here.

## The three helper passes

| kernel | grid | what each program does | bound by |
| --- | --- | --- | --- |
| `_bwd_preprocess` | `(cdiv(N,64), N, bh)` = 131,072 | `delta[i,j] = Σ_d o·do` for a `[64, DIM]` tile | bandwidth: 1.11 GB at 4.19 TB/s, 0.26 ms |
| `_bwd_bias_prep` | `(cdiv(N,32), cdiv(N,32), bh)` = 8,192 | `b2t[k,j] = b[j,k] * inv_ln2`, fp32, a `[32,32]` tiled transpose | bandwidth: 16 MB read + 33 MB written, 0.02 ms |
| `_bwd_scale_cast` | flat, `cdiv(268M, 4096)` = 65,536 | `dq = (dq_f32 * sm_scale)` cast to bf16 | bandwidth: 1.6 GB, 0.46 ms with the db transpose |

`delta` is a batched inner product, not a GEMM — 0.48 FLOP/byte against this card's ~35
FLOP/byte balance point, so it runs at the HBM roofline and no matmul formulation helps.
It exists as a separate pass because `_bwd_fused` owns a *k* tile: computing `delta` inline
would mean reloading `o` once per k-block, 16x the traffic. (Substituting the algebraically
equivalent `delta[j] = Σ_k p·dp` does not work either — a CTA only sees its own k-tile, and
`ds` needs the *complete* delta before it can be formed.)
