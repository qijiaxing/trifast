# TriFast

This is TriFast fork, which is optimized for Hopper GPU.

`src/trifast/triton.py`: the fwd kernel, and the three original bwd kernels.
`src/trifast/triton_bwd.py`: the fused bwd kernel.
`src/trifast/torch.py`: `triangle_attention` torch op.
`src/trifast/autotune_helpers.py`, `src/trifast/autotune_bwd.py`: triton config autotune candidates.
`python scripts/bench_kernels.py` can be used to run benchmark. It refuses to measure while
another job is using the GPU.

## Perf

H20 GPU, H = 8, D = 32.

### Forward optimizations

The forward kernel is tuned for the crop-size-varying training workload:

- `N` is a runtime value, while `CLOSEST_N` remains a compile-time bucket, avoiding a separate kernel variant for every crop size.
- `q`/`k`/`v`/`o` and the aligned bias tiles all move through Hopper TMA descriptors. The rank-4 `q`/`k`/`v`/`o` boxes (`[1, 1, BLOCK_*, DIM]` over the natural `[bh, n, n, dim]` layout) keep every tile inside one `(h, i)` slice so rows beyond `n` are clipped by the tensormap instead of wrapping into the neighbouring slice. Descriptor box shapes are rewritten per autotune config by a pre-hook (`BLOCK_K = 64` only pays off once TMA amortizes its per-iteration barrier/addressing cost). Traced/fake tensor execution (`torch.compile`, opcheck) falls back to the pointer kernel.
- The `[batch, n, n]` bool mask also moves through a TMA descriptor, as a bf16 copy. The pointer load looked harmless but was the kernel's single most expensive input: a rank-1 `[BLOCK_K]` tensor feeding a `[j, k]` broadcast gets the replicated `slice<dim=0, parent=#mma>` layout, so it issued **67 M `LDG` / 2.15 GB of L2 sectors to read a 1 MB tensor** (32x amplification) — 10x more requests than q/k/v/o/bias combined. Three details are load bearing and each gives back the whole gain if changed: bf16 rather than a 4-byte type (halves the TMA traffic), a rank-2 box over the flat `[batch·n, padded_n]` view rather than rank-3 over `[batch, i, k]`, and reusing `N` for the row index rather than passing the mask's row count. The box must also be >= 128 bytes or the pipelined loop faults, so `BLOCK_K = 32` configs take a 2-row box. `can_use_tma_mask` additionally requires the mask to really be `n x n`.
- The k loop is **peeled**: `_fwd_kv_block` takes a `K_MASKED` constexpr, and `_fwd` runs `N // BLOCK_K` full blocks with it `False` plus at most one ragged tail block with it `True`. Full blocks provably cannot read past `N`, so they skip the `[k]` range compare and three `[j,k]`-sized selects (`in_range`, and the `mask_k` guards on the scores and on `exp_scores`) that are no-ops there. `n_full` is a runtime value, so this stays one kernel variant per `CLOSEST_N` bucket. Worth **+8.7%** at n=1024 -- the single largest forward win, and it also makes the mask load unpredicated for free.
- The stable online-softmax path computes LSE directly in `_fwd`.

Together these took n=1024 from 92.99 to 103.46 TFLOP/s (**+11.3%**), and the BF16 tensor pipe from 64.9% to 72.2% of peak. `n=800` gains least (+7.4%) because `800 % 64 == 32` leaves a ragged tail block that still pays the masking. The kernel now issues **zero** non-TMA global loads, down from 67 M. See `FWD_NCU_ANALYSIS.md` for the profile and for the alternatives that were measured and rejected.

### Backward fusion

The backward ran three kernels that each recomputed `scores = q·kᵀ * sm_scale + bias` and `dp = do·vᵀ`, rebuilt `p` from `(mx, dn)` and rebuilt `ds` — nine matmuls and three softmax epilogues where five and one would do. `_bwd_b` existed only because `db` is a reduction over the triangle axis `i`, which the other two carry in their grid. `src/trifast/triton_bwd.py` replaces all three with `_bwd_fused` plus two memory-bound passes (a `delta` preprocess and a dq cast), selected by `USE_FUSED_BWD` in `torch.py`:

- **Every score tile is computed transposed, as `[k, j]`.** dV needs `pᵀ` and dK needs `dsᵀ`; in the `[j, k]` orientation those are `tl.trans` of a *computed fp32 accumulator*, which lowers to a shared-memory round trip (four `stmatrix`, four `ldmatrix`, two extra barriers) and is what put `_bwd_kv` at 196 registers and 12.45% occupancy. Transposing a *loaded* tile is free — it becomes `ttg.memdesc_trans`, metadata on the wgmma shared-memory operand descriptor. This alone takes the dk/dv work from 30.6 to **26.0 ms**.
- **`db` and `dq` are accumulated with fp32 atomics**, `db` into a transposed `[bh, k, j]` buffer because `ds` is produced as `[k, j]`. `sem="relaxed"` is load bearing: Triton's default `acq_rel` is **2.4x slower** (388 vs 941 G updates/s). `dk`/`dv` stay register accumulators and come out bit-identical to the old kernels.
- **The j loop is peeled** (`J_MASKED`), and whether this CTA's k tile is ragged is a uniform per-CTA branch rather than a host flag — so only the ragged CTA pays. Worth 2.6 ms.
- `BLOCK_K >= 64` and `num_warps=4` are cliffs, not preferences: `BLOCK_K=32` costs **3x** (it is the M dimension of the dV/dK dots, and M < 64 leaves wgmma) and `num_warps=8` costs 2.1x.

Both columns of the table below are counted against the same five matmuls -- the work a backward *has* to do, and the 2.5x-of-forward convention the attention benchmarks use -- so they are comparable to each other and to the Forward column above. In wall clock that is **62.5 -> 40.7 ms at n=1024**. The three-kernel path actually performs nine matmuls, so its column understates its raw matmul rate on purpose; what is being compared is how fast each path delivers the same gradients.

dq and db are no longer bitwise reproducible run to run — that is the cost of the atomics, at ~1e-7 relative, far below one bf16 ulp. `BWD_FUSION.md` has the profile, the per-piece costs, and the six things that looked right and were not — including that autotuning a kernel which *accumulates* silently produced gradients 400x too large under `torch.compile`.

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

### Kernel throughput

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┬──────────────┬───────────────┬────────────────┐
│  N   │ Forward │ Backward Q │ Backward K/V │ Backward Bias │ Backward Fused │
├──────┼─────────┼────────────┼──────────────┼───────────────┼────────────────┤
│  512 │   94.66 │      94.12 │        68.37 │         67.40 │          64.87 │
│  640 │   98.59 │      95.91 │        69.91 │         70.10 │          66.90 │
│  768 │  100.85 │      96.83 │        70.40 │         71.62 │          67.64 │
│  800 │   91.03 │      93.54 │        65.20 │         68.39 │          61.80 │
│ 1024 │  103.47 │      98.45 │        71.80 │         72.83 │          69.12 │
└──────┴─────────┴────────────┴──────────────┴───────────────┴────────────────┘
```

**These columns are not comparable to each other.** Throughput is counted over each
kernel's own matmuls, and the fused kernel performs 5 where Backward Q/K/V/Bias perform
3 + 4 + 2 = 9 between them — its lower TFLOP/s is 1.5x less work, not slower work. The
wall-clock table above is the one that answers the question.
