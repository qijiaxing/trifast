# `_fwd` Nsight Compute analysis — H20-3e, N=1024, H=8, D=32, BF16

Profiled with `ncu 2025.3.1` on the config the autotuner selects for this shape:
`BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=3` (TMA path for q/k/v/o/bias).

Repro:

```
python scripts/bench_kernels.py -k fwd                 # baseline: ~91-93 TFLOP/s at N=1024
ncu --set full --kernel-name regex:_fwd --launch-skip 1 --launch-count 1 \
    -o fwd_full -f python scripts/ncu_kernels.py fwd -n 1024
```

Note on absolute times: ncu locks the SM clock to 1.63 GHz while the benchmark runs at
~1.79 GHz, so ncu reports 12.99 ms where `do_bench` reports 11.81 ms. Ratios and pipe
utilisation percentages are unaffected.

## Speed of light

| Metric | Value |
| --- | --- |
| Duration (ncu, clock-locked) | 12.99 ms |
| Grid / block | (16, 1024, 8) × 128 thr, 131072 CTAs |
| Registers / thread | 91 (no spills) |
| Dynamic shared / block | 45.14 KB |
| Achieved occupancy | 31.1 % (20 of 64 warps) — capped by **both** registers (5 blocks) and shared mem (5 blocks) |
| Issue slots busy | 79.1 % |
| **BF16 tensor pipe** | **64.9 %** of peak |
| ALU pipe (int/logic) | 60.3 % |
| FMA pipe (fp32) | 27.0 % |
| XU pipe (MUFU/exp2) | 34.8 % |
| LSU pipe | 11.7 % |
| DRAM throughput | 3.6 % |
| L2 hit rate | 95.0 % |

Use `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active`
(64.9 %) for tensor-core utilisation, **not** the `sm__pipe_tensor_cycles_active`
that the SoL section implies (16.2 %) — the latter normalises against the FP8/sparse
peak. 64.9 % is consistent with the measured 93 TFLOP/s against a 78 SM × 1.79 GHz ×
1024 FLOP/SM/cycle ≈ 143 TFLOP/s HGMMA ceiling.

**This kernel is not memory bound and not tensor bound. It is instruction-issue bound.**

## Where the instructions go

5,230,034,944 warp instructions — 39,902 per CTA, 2,494 per k-iteration, i.e.
**19.5 instructions per score element**, of which **0.19 are HGMMA (0.96 % of the
total)**. Everything else is softmax, masking, bias and addressing.

SASS opcode histogram (top, % of executed instructions):

```
FADD  11.0%   FSEL  10.5%   IMAD   8.5%   FMUL  8.3%   FMNMX 5.8%
FFMA   5.7%   MUFU   5.5%   ISETP  4.1%   LOP3  4.1%   PRMT  3.9%
...                                                    HGMMA 0.96%
```

Attributed to `src/trifast/triton.py` lines (via `nvdisasm -g` joined against ncu's
per-instruction counts; the scheduler smears a few lines into their neighbours):

| line | % inst | % stall | source |
| --- | --- | --- | --- |
| 148 | 20.2 | 16.7 | `scores = tl.where(in_range, scores, neg_inf2)` |
| 140 | 11.2 | 10.4 | `m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0` |
| 150 | 10.7 | 12.7 | `exp_scores = tl.math.exp2(scores - block_max[:, None])` |
| 137 | 10.1 | 4.6 | `b_block = desc_b.load(...).to(tl.float32)` |
| 129 | 8.3 | 6.9 | `mask_k = (k_idxs + start_k) < N` |
| 127 | 6.1 | 5.2 | `for start_k in tl.range(0, N, BLOCK_K)` (uniform addressing) |
| 145 | 5.6 | 3.3 | `scores = scores * sm_scale + b_block` |
| 151 | 5.5 | 9.7 | `exp_scores = tl.where(mask_k[None, :], exp_scores, 0.0)` |
| 146 | 5.3 | 5.8 | `scores *= inv_ln2` |
| standard.py:170/263 | 9.8 | 9.6 | `tl.max` / `tl.sum` reduction helpers |
| 143 + 166 | **2.0** | 7.7 | **the two `tl.dot` calls** |

## Root cause: D=32 leaves nothing to hide behind

Each score element buys only `2 · D = 64` tensor FLOP from QK^T and 64 from PV — 128
FLOP, or 0.125 SM-cycles of HGMMA. The softmax epilogue around it costs ~19
instructions on the ALU/FMA/XU pipes, and those pipes are ~4–8× narrower than the
tensor core. At D=128 the same epilogue would be amortised over 4× more tensor work;
at D=32 it lands squarely on the critical path. That is the structural reason the
kernel sits at 65 % of HGMMA peak with 79 % of issue slots already busy.

## Measured cost of each piece

Ablation (`BLOCK_J=64, BLOCK_K=64, warps=4, stages=3`; each row removes one piece, so
most rows are numerically wrong and exist only to price the removed work):

| variant | TFLOP/s | gain |
| --- | --- | --- |
| baseline (correct) | 90.9 | — |
| no `tl.where(in_range, ...)` | 92.4 | 1.7 % |
| no `tl.where(mask_k, exp_scores, 0)` | 95.0 | 4.6 % |
| no separate `scores *= inv_ln2` | 92.8 | 2.1 % |
| **no `exp2`** | 92.6 | **1.9 %** |
| no `m_block` load + its `where` | 103.5 | **13.9 %** |
| …`where` + broadcast only, no load | 103.9 | 14.4 % |
| …mask load unpredicated, no `.cg` | 94.4 | 3.9 % |
| no bias load / convert / add | 106.9 | **17.7 %** |
| …bias TMA load hoisted out of the k loop | 98.5 | 8.4 % |

Two conclusions that overturn the obvious guesses:

1. **`exp2` is not the problem.** Despite XU at 34.8 %, deleting the transcendental
   buys only 1.9 %. Do not spend effort on the softmax exponential.
2. **The boolean mask is the single most expensive input, and it is the *load*, not
   the `where`** (14.4 % for load+where vs 13.9 % for removing both — the `where`
   itself is free). See the next section for why.

## The mask load is pathological

### How it is used

`mask` is `[batch, n, n]` `bool` — 1 MB at n=1024, and **one mask per batch item, not
per head** (`mask_start_h = pid_h // H`). The kernel builds a rank-1 pointer vector
once (line 105) and re-loads a `[BLOCK_K]` slice of it on every k-iteration:

```python
mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn)  # [k]
...
m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0   # [k]      line 140
scores  = tl.where(m_block[None, :], neg_inf2, scores)            # [j,k]    line 148
mask_ptrs += BLOCK_K * stride_maskn                               #          line 172
```

One 64-byte load per iteration, consumed by exactly one broadcast `tl.where`.

### What the compiler does with it

TTGIR assigns that rank-1 tensor the layout of its *consumer* — a projection of the
wgmma accumulator:

```
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
%m_block_128 = tt.load %m_block, %mask_k_117 cacheModifier = cg
             : tensor<64x!tt.ptr<i8>, #ttg.slice<{dim = 0, parent = #mma}>>
```

`slice<dim=0, parent=#mma>` means "the `[64,64]` accumulator layout with the row
dimension projected away", and that layout is *replicated* along the projected
dimension. Two independent factors multiply:

- `warpsPerCTA = [4, 1]` splits the 64 rows across the 4 warps but gives every warp
  all 64 columns. Slicing rows away leaves all 4 warps holding the **identical** 64
  mask bytes → **4× cross-warp redundancy.**
- Within a warp, lane *l* owns column pairs `{2(l%4)+8c, 2(l%4)+1+8c}` for `c=0..7`,
  so the 8 lanes sharing `l%4` hold the same columns → **8× intra-warp redundancy.**

The result in SASS — and these are the **only** non-TMA global loads in the entire
kernel; q/k/v/o/bias all arrive via `UTMALDG`:

```
8 × @!P LDG.E.U16.STRONG.GPU R??, desc[UR36][R20.64+...]
  each: 8,388,608 executions, 32/32 lanes active, 1 L2 sector per request
```

8 two-byte loads per warp per iteration × 32 lanes × 4 warps = **2048 bytes fetched
to read 64 distinct bytes — 32× amplification**, exactly the 4 × 8 above. Kernel
totals:

```
smsp__sass_inst_executed_op_global_ld.sum        67,108,864   (the mask, 8 static insts)
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   67,108,864   -> 2.15 GB of sectors
  ...of which "ideal"                            67,108,864   (coalescing is fine!)
smsp__sass_inst_executed_op_tma_ld.sum            6,422,528   (all of q/k/v/o/bias)
```

**2.15 GB of L2 sector traffic to read a 1 MB tensor, and 10× more requests than
q/k/v/o/bias combined.** Note `L2 Theoretical Sectors Global == Ideal`: ncu considers
the coalescing optimal. The waste is not misalignment, it is *redundancy* — the same
64 bytes requested 32 times per CTA-iteration, on top of the 128 CTAs (16 j-blocks ×
8 heads) that each re-read the same `mask[0, i, :]` row.

`.cg` compiles to `STRONG.GPU`, which bypasses L1 — that is why `L1/TEX Hit Rate` is
**0 %** for the whole kernel and all 2.15 GB lands on L2 (95 % hit, so DRAM stays
quiet at 3.6 %). The cost is request *throughput* and issue slots, not bandwidth or
latency: these 8 instructions carry only **4.3 % of the kernel's stall samples** while
deleting them is worth 13.9 %.

### What does *not* fix it

| attempted fix | gain |
| --- | --- |
| drop `.cg` only | +0.2 % |
| drop the `mask_k` predicate only | **−2.3 %** |
| drop both | +3.8 % |
| add `tl.max_contiguous` / `tl.multiple_of`, assume `stride_maskn == 1` | +3.0 % |
| load the `uint8` mask through a TMA descriptor | fails: `cudaErrorMisalignedAddress` |

Individually the micro-tweaks are noise or worse; only removing the predicate *and*
`.cg` together lets ptxas widen the loads, and even then it recovers a quarter of the
cost. None of them touch the 32× replication, because that is a property of the
layout, not of the addressing. TMA on the raw mask is rejected outright — TMA cannot
handle a 1-byte element type with a 64-wide inner box.

This is why the fix has to change the *element type and the load path*: a bf16 mask
staged through TMA (recommendation B) lands in shared memory once per CTA and is
broadcast from there, replacing all 8 LDG.

### Why the element type has to widen

TMA is not fundamentally limited to 2-byte types — a standalone probe loads the
`bool` tensor as a zero-copy `uint8` view happily at every box width from 16 to 256
bytes. But the same descriptor inside this kernel's software-pipelined loop
(`num_stages=3`, alongside four other TMA tiles) fails with
`cudaErrorMisalignedAddress`, and widening the box did not rescue it. A zero-copy
`uint8` path is therefore *plausible but unproven*; bf16 is the version that is
measured and working. Cost of widening, at n=1024:

```
bool[1,1024,1024] -> bf16 mask:   35 us,  +2.1 MB
  vs one _fwd launch (11.8 ms):   0.3 %        (against ~7 % saved)
```

And it is a one-time cost for the whole step, not per kernel: `_fwd` (line 140),
`_bwd_kv` (460) and `_bwd_b` (589) all load the mask with the identical
`tl.load(mask_ptrs, mask_k, cache_modifier=".cg") != 0` inside their k loops, so one
converted tensor serves all of them. (`_bwd_q` already hoists its mask load out of
the loop at line 291 and so does not have this pathology.)

## Recommended changes

Both are semantics-preserving and were verified **bit-identical** to the current
kernel (`max abs diff == 0` for `o` and `lse`) on a random 50/50 mask *and* on an
input with a fully-masked row:

| change | gain |
| --- | --- |
| **A. Aligned-N peeling** — `N // BLOCK_K` full blocks with `K_MASKED=False` plus at most one ragged tail. Full blocks skip the `[k]` range compare and three `[j,k]` selects. | **+8.7 % (implemented, measured in-tree)** |
| **B. Mask through TMA, keeping `tl.where`** — materialise the mask host-side as bf16, load it with a rank-2 `[1, BLOCK_K]` TMA box over the flat `[batch·n, padded_n]` view, and keep the existing select. Kills the 67 M LDG. | **+2.3 % (implemented, measured in-tree)** |

> **Do not trust the ablation harness's absolute gains.** The stripped-down kernel in
> the ablation table has a ~2.5 % slower pointer baseline than the real `_fwd`, so it
> overstates every improvement. B measured **+5.7 % in the harness and +2.2 % in the
> real kernel**; assume A's +6.5 % is similarly optimistic until it is implemented and
> measured in-tree. Always confirm against `scripts/bench_kernels.py`, on an idle GPU
> — a concurrent process on the same device silently cost 28 % in one measurement
> round here.

The gap between "delete the mask load entirely" (+13.9 %) and "re-route it through
TMA" (+2.3 %) is the TMA load's own cost: a `[1, BLOCK_K]` box is a degenerate
128-byte transfer that still pays a full barrier and a pipeline stage every iteration,
so it hands back roughly 12 of the 14 points. **TMA is a poor fit for a load this
small.** The remaining headroom is in not loading the mask per k-iteration at all —
it is loop-invariant in `j` and only `BLOCK_K` wide — but Triton offers no way to
dynamically slice a register-resident row, so that needs either a manual
shared-memory staging or hoisting the row into registers for a statically-known
number of k-blocks.

**The 2.3 % is fragile: three incidental choices each erase all of it.** Every one of
these was measured on an idle GPU, interleaved against the pointer path in the same
process, min-of-N:

| variant | gain over pointer |
| --- | --- |
| bf16, rank-2 box, row indexed by `N` — **what is implemented** | **+2.3 %** |
| int32 instead of bf16 (doubles TMA bytes) | −1.1 % |
| rank-3 `[1, 1, BLOCK_K]` box over `[batch, i, k]` | +0.1 % |
| rank-2, but row indexed by an extra `mask_rows` scalar argument | +0.1 % |

The last one is the least intuitive: passing the mask's true row count as a new kernel
argument, instead of reusing the `N` already in the signature, costs the entire win.
Reproduced twice in both directions. So the implementation reuses `N` and instead
*gates* on the mask really being `n x n` (`can_use_tma_mask`), letting the odd
`[b, n, n, h, d]` q layout fall back to the pointer path rather than paying for a
general row index.

**Do not replace the select with an addition.** Folding the sentinel in additively
(`scores + madd` instead of `tl.where(m, neg_inf2, scores)`) is tempting — it is
2 percentage points faster, +16.3 % combined — but it silently breaks the documented
"fully-masked row → `mean(V)`" behaviour. In the select form a fully-masked row has
every score at exactly `neg_inf2`, so the softmax is uniform; in the additive form it
has `s_i + neg_inf2`, and the max-subtraction recovers `softmax(s_i)` — i.e. the row
behaves as if the mask were absent. Measured on a deliberately fully-masked row:

```
select via TMA     max abs diff vs baseline: 0.0000e+00
additive via TMA   max abs diff vs baseline: 1.1288e+00
```

A random mask does not expose this (no row is ever fully masked), so the ablation
table's `SAFE-add` row and any test suite without a fully-masked case will both call
it correct. The 2 % is not worth the semantic change; keep the `where`, which the
ablation shows is free anyway (14.4 % for load+where vs 13.9 % for removing both).
`tests/unit/test_trifast.py::test_fully_masked_row_{values,semantics}` now cover this
(8 cases); they pass on the select form and fail on all 8 with the additive form.

### Two hazards the TMA mask path runs into

**1. The mask box must be at least 128 bytes.** Measured, one process per cell so a
fault cannot poison the next:

| mask dtype | BLOCK_K | box bytes | result |
| --- | --- | --- | --- |
| bf16 | 32 | 64 | **`cudaErrorMisalignedAddress`** |
| bf16 | 64 / 128 | 128 / 256 | OK |
| fp32, int32 | 32 / 64 / 128 | 128 / 256 / 512 | OK |

64 bytes is the *only* size that fails, and a 1-byte box loads fine in a standalone
kernel, so this is a shared-memory alignment limit inside the pipelined loop rather
than a TMA restriction. It bites during **autotuning**, which must try `BLOCK_K=32`,
so it is a hard crash on a cold cache — not a slow path. Two ways out, and only one
of them is free:

- Widen to a 4-byte element (`32 * 4 == 128`). Safe, but doubles the TMA traffic and
  gives back the entire win: **92.17 vs 93.21 TFLOP/s, a 1 % net regression.**
- Keep bf16 and give narrow tiles a **two-row box** (`[1, 2, BLOCK_K]`), discarding
  the second row. Keeps the 128-byte floor and the +2.2 %. This is what is
  implemented; `_fwd_descriptor_pre_hook` picks the box shape per config and the
  kernel branches on the `BLOCK_K` constexpr.

**2. Do not assume the mask's last dimension equals `n`.** `_triangle_attention`
derives `n` via `bs, h, _, n, dim = q.shape`, and `test_weight_updates` passes `q` as
`[b, n, n, h, d]`, so that destructuring yields **`n=1` against a `[b, 16, 16]`
mask**. The pointer path survives because it only ever uses mask *strides*. A flat
`[batch·n, n]` view does not — hence the rank-3 descriptor over `[batch, i, k]`, which
needs no agreement between the mask's shape and `n`.

The **bias** descriptor has this same latent bug and is not fixed here: `padded_b.reshape(bh * n, padded_n)`
in `trifast/torch.py` raises `RuntimeError: shape '[16, 8]' is invalid for input of
size 2048` on those shapes. `test_weight_updates` (3 cases) fails identically on
unmodified `master`, so it is pre-existing, but it does mean that entry point is
currently broken for non-`[b, h, n, n, d]` q layouts.

Folding `inv_ln2` into `sm_scale` is **not** worth pursuing. It prices at 2.1 % in
isolation, but doing it correctly requires pre-scaling `q` (a bf16 rounding change),
and once A and B are in it is worth 0.1 %.

## Further leads, in descending value

1. **Amortise the bias across `i` (ceiling ~8.4 %).** `bias` is indexed `(h, j, k)`
   and is completely independent of `pid_i`, yet the grid is `(j_blocks, i, bh)`, so
   1024 CTAs load the identical `[64, 64]` tile. Hoisting the TMA load out of the k
   loop — i.e. having one CTA process several `i` slices — prices at +8.4 %. The
   catch: two accumulators and two score tiles would push past 91 registers, and
   occupancy is *already* register- and shared-memory-limited at 31 %, so this could
   easily be a wash. Test before committing.
2. **Bias convert + add costs a further ~9.3 %** (17.7 % total − 8.4 % load). The
   `bf16 → fp32` conversion shows up as the 3.9 % `PRMT` line; unavoidable while the
   FFMA needs fp32 operands, but it shrinks with any scheme that touches the bias
   fewer times.
3. **Occupancy.** 31.1 % with `No Eligible` at 20.9 %. Both the register cap (91) and
   45.14 KB of shared memory bind at 5 blocks/SM. Once A and B remove ~4 tile-passes
   of live fp32 state, re-sweep `num_stages` and `num_warps` — the current 4/3 choice
   was tuned against the old instruction mix and the balance will have moved.
4. **Re-tune after any of the above.** The tile sweep at N=1024 shows the shape is
   sensitive: `BLOCK_K=64` beats `BLOCK_K=32` by 16 % (93.1 vs 80.3 TFLOP/s with
   `maxnreg=None`), and the reduction helpers (`tl.max`/`tl.sum`, 9.8 % of
   instructions) amortise better at larger `BLOCK_K`. `BLOCK_K=128` was worse (85.2).

## Not worth investigating

- **DRAM / L2.** 3.6 % DRAM, 95 % L2 hit. Nothing to win.
- **Register spilling.** Zero local-memory traffic.
- **Shared-memory bank conflicts.** 2.75 M against 448 M shared wavefronts (0.6 %).
- **`exp2` / the XU pipe.** 1.9 %, see above.


---

# Follow-up: results after the two changes landed

Both recommendations are now implemented and measured on an idle GPU. Final forward
throughput, against the baseline this document opened with:

| N | baseline | +TMA mask | +peeling | total |
| --- | --- | --- | --- | --- |
| 512 | 87.70 | 89.21 | **95.53** | +8.9 % |
| 640 | 89.89 | 91.56 | **98.82** | +9.9 % |
| 768 | 91.32 | 92.91 | **100.83** | +10.4 % |
| 800 | 84.96 | 86.48 | **91.21** | +7.4 % |
| 1024 | 92.99 | 95.18 | **103.46** | **+11.3 %** |

`n=800` gains least because `800 % 64 == 32` leaves a ragged tail block that still
pays for the masking; the other four sizes are exact multiples of `BLOCK_K=64`.

## How the profile moved

| Metric | before | after |
| --- | --- | --- |
| **BF16 tensor pipe** | 64.9 % | **72.2 %** |
| ALU pipe | 60.3 % | 44.6 % |
| XU pipe (exp2) | 34.8 % | 38.7 % |
| Warp instructions | 5.23 G | 4.56 G (−13 %) |
| **Non-TMA global loads** | 67,108,864 | **0** |
| Registers / thread | 91 | 118 |
| Achieved occupancy | 31.1 % | 24.9 % |

Two things worth noting. The mask was the *only* non-TMA global load in the kernel, so
that counter is now exactly zero. And occupancy went **down** (registers 91 → 118,
which costs one block per SM) while throughput went up 11 % — the instruction
reduction more than paid for the lost latency hiding. That makes register pressure the
obvious next lead: the peeled tail inlines a second copy of the block body, and
getting registers back under ~104 would restore the fifth block per SM.

## The estimate-vs-reality gap, again

The ablation harness said A was worth +6.5 % and B +5.7 %. Reality: **A +8.7 %, B
+2.3 %.** The harness got the *ranking* wrong, not just the magnitudes — it badly
understated the peeling and overstated the mask. Treat its numbers as a screening
signal only; anything load-bearing has to be measured in the real kernel.

## Alternatives measured and rejected

The two questions worth asking about the mask path, answered with numbers rather than
reasoning. All measured at pinned `BLOCK_J=64, BLOCK_K=64, warps=4, stages=3`, n=1024,
interleaved against the shipped path in one process, min-of-N, GPU idle, and all
producing **bit-identical** output.

**Does the mask have to be bf16? Could it be int8?** It can be int8 — an earlier claim
in this document that TMA rejects 1-byte types was wrong; that attempt used a
`[1, BLOCK_K]` box, which at `BLOCK_K=64` is 64 bytes, the one failing size. But int8
loses badly:

| mask element type | TFLOP/s | vs bf16 |
| --- | --- | --- |
| bf16, `[1, 64]` box, 2 MB copy + ~35 µs | **103.38** | — |
| uint8, `[2, 64]` box, **zero-copy** `view(torch.uint8)` | 87.22 | **−15.6 %** |

The 128-byte floor is the whole story. bf16 gets exactly the 64 entries it needs from a
128-byte box; uint8 needs `128 // BLOCK_K` rows, so it materialises 128 entries and
reduces them down to 64. Same bytes moved, twice the register traffic and an extra
reduction — far more expensive than the 2 MB copy and 35 µs it saves.

**Does it have to be TMA? Why not a plain load?** After peeling, the plain load is
unpredicated for free, so this is the fairest comparison available:

| mask fetch | TFLOP/s | vs pointer |
| --- | --- | --- |
| TMA bf16 descriptor | **103.39** | +3.3 % |
| plain `tl.load`, unpredicated, `.cg` | 100.05 | — |
| plain `tl.load`, unpredicated, no `.cg` | 100.07 | +0.0 % |

TMA still wins, and by *more* than before peeling (+3.3 % vs +2.3 %). `.cg` turns out
to be irrelevant — the earlier harness finding that dropping the predicate and `.cg`
together was worth +3.8 % was entirely the predicate, which peeling now removes
anyway. So the descriptor keeps earning its complexity.
