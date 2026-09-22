# Fusing the backward — H20-3e, N=1024, H=8, D=32, BF16

The backward ran three kernels — `_bwd_q` (dq + delta), `_bwd_kv` (dk, dv), `_bwd_b` (db)
— and each one independently recomputed `scores = q·kᵀ * sm_scale + bias` and
`dp = do·vᵀ`, rebuilt `p` from `(mx, dn)` and rebuilt `ds`. **Nine matmuls and three
softmax epilogues where five and one would do.** `_bwd_b` existed only because `db` is a
reduction over the triangle axis `i`, which the other two kernels carry in their grid.

`src/trifast/triton_bwd.py` replaces all three with one kernel plus two memory-bound
passes. Repro:

```
python scripts/bench_kernels.py                    # per-kernel and end-to-end, both paths
python scripts/proto_bwd_fused.py check            # correctness matrix vs the old path
python scripts/proto_bwd_fused.py bench | ablate    # config sweep, per-piece cost
pytest tests/unit/test_bwd_fused.py                # 32 cases, fused vs three-kernel
```

Every timing below was taken on an idle GPU — this device is shared, and
`scripts/bench_kernels.py` now refuses to measure if anyone else is using it.

## Result

```
TriFast backward algorithmic throughput — BF16 (TFLOP/s)
┌──────┬───────────────┬───────┬─────────┐
│  N   │ Three kernels │ Fused │ Speedup │
├──────┼───────────────┼───────┼─────────┤
│  512 │         41.57 │ 62.43 │   1.50x │
│  640 │         42.69 │ 64.45 │   1.51x │
│  768 │         43.14 │ 65.47 │   1.52x │
│  800 │         40.76 │ 60.10 │   1.47x │
│ 1024 │         43.99 │ 67.60 │   1.54x │
└──────┴───────────────┴───────┴─────────┘
```

End to end, allocations and both helper passes included. Both columns are counted against
the same five matmuls -- the work a backward *has* to do -- so they compare how fast each
path delivers the same gradients rather than how fast it runs its own instruction mix; the
three-kernel path performs nine. In wall clock: **62.5 -> 40.7 ms at n=1024**, of which the
fused kernel is 39.7 ms, the delta preprocess 0.26 and the dq cast plus db transpose 0.46.

`n=800` gains least for the same reason the forward does: `800 % 64 == 32` leaves a ragged
tile that still pays for masking.

**Why 1.54x and not the 1.8x that 9 → 5 matmuls implies.** The three kernels together
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

At BLOCK_J = BLOCK_K = 64, `num_warps=4`, `num_stages=2`:

| | ms | Δ | reg |
| --- | --- | --- | --- |
| dk + dv (4 matmuls) | 26.02 | — | 211 |
| + db atomic | 31.88 | +5.86 | 232 |
| + dq (5th matmul and its atomic) | 39.77 | +7.89 | 250 |
| …with dq in the transposed orientation | 45.08 | +5.31 | 255, 18 spills |
| …without the peeled fast path | 42.39 | +2.62 | 255 |

Both fusions pay for themselves several times over: db costs 5.86 ms and deletes a 15.10 ms
kernel; dq costs 7.89 ms and deletes a 16.78 ms one.

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

## Leads, in descending value

1. **TMA for the loop-carried tiles.** A `[DIM, BLOCK_J]` 64-bit pointer tensor is 32
   registers per thread on its own, and this kernel carries several. Every load is in the
   natural orientation with only `tl.trans` applied afterwards, so the forward's descriptor
   machinery applies directly. This is the one lever likely to move 250 registers, and
   registers are what the atomics need in order to hide.
2. **`BLOCK_I = 2`** — one CTA covering two `i` values halves the db atomic count *and*
   halves the bias re-read (17.2 GB → 8.6 GB; the forward priced the bias load at 8.4 %).
   It needs a second dk/dv accumulator pair, so it is only viable at BLOCK_J=32.
3. **Gate db on `ctx.needs_input_grad`.** `NEED_DB` is already a constexpr; threading it
   through would hand back the whole 5.86 ms whenever the bias needs no gradient.
4. **The duplicated j loop costs 0.8 ms.** `_bwd_fused` instantiates `_bwd_j_loop` twice, on
   a uniform branch over whether this CTA's k tile is ragged, which is worth 2.6 ms but
   costs 0.8 in code size and register pressure. A cheaper way to specialize would net the
   difference.

## Not worth investigating

- **DRAM.** The fused kernel moves ~4.3 GB at n=1024 against a 40 ms runtime.
- **The bias transposed load.** `jj[None,:]*N + kk[:,None]` already coalesces into
  `ld.global.v4.b32`; a host-side pre-transposed copy measured no faster and costs a pass.
- **The mask.** In this orientation `mask[i, k]` is loop-invariant, so it is one `[k]` load
  per CTA rather than one per iteration, and its `[:, None]` broadcast is split across warps
  rather than replicated. The forward's 32x LDG amplification simply does not arise.

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
      loaded:  k_blk[64,32]  v_blk[64,32]  q[64,32]  do[64,32]  bT[64,64]  mx,dn,delta[64]

  (1) sT[64,64]  = k_blk @ trans(q)         M=BK  N=BJ  K=DIM     recompute the scores
      sT         = (sT * sm_scale + bT) * inv_ln2, then the mask selects
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

## The two helper passes

| kernel | grid | what each program does | bound by |
| --- | --- | --- | --- |
| `_bwd_preprocess` | `(cdiv(N,64), N, bh)` = 131,072 | `delta[i,j] = Σ_d o·do` for a `[64, DIM]` tile | bandwidth: 1.11 GB at 4.19 TB/s, 0.26 ms |
| `_bwd_scale_cast` | flat, `cdiv(268M, 4096)` = 65,536 | `dq = (dq_f32 * sm_scale)` cast to bf16 | bandwidth: 1.6 GB, 0.46 ms with the db transpose |

`delta` is a batched inner product, not a GEMM — 0.48 FLOP/byte against this card's ~35
FLOP/byte balance point, so it runs at the HBM roofline and no matmul formulation helps.
It exists as a separate pass because `_bwd_fused` owns a *k* tile: computing `delta` inline
would mean reloading `o` once per k-block, 16x the traffic. (Substituting the algebraically
equivalent `delta[j] = Σ_k p·dp` does not work either — a CTA only sees its own k-tile, and
`ds` needs the *complete* delta before it can be formed.)
