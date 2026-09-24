# TriFast

This is TriFast fork, which is optimized for Hopper GPU.

`src/trifast/triton.py`: the fwd and bwd kernels.
`src/trifast/torch.py`: `triangle_attention` torch op.
`src/trifast/autotune_helpers.py`: triton config autotune candidates.
`src/trifast/bio/`: external sm_90a CUDA forward kernel (`cuda_b`) kept as a reference to beat, see below.
`python scripts/bench_kernels.py` can be used to run benchmark (`-k fwd cuda_b` to pick kernels).

## Perf

H20 GPU, H = 8, D = 32.

### Forward optimizations

The forward kernel is tuned for the crop-size-varying training workload:

- `N` is a runtime value, while `CLOSEST_N` remains a compile-time bucket, avoiding a separate kernel variant for every crop size.
- `q`/`k`/`v`/`o` and the aligned bias tiles all move through Hopper TMA descriptors. The rank-4 `q`/`k`/`v`/`o` boxes (`[1, 1, BLOCK_*, DIM]` over the natural `[bh, n, n, dim]` layout) keep every tile inside one `(h, i)` slice so rows beyond `n` are clipped by the tensormap instead of wrapping into the neighbouring slice. Descriptor box shapes are rewritten per autotune config by a pre-hook (`BLOCK_K = 64` only pays off once TMA amortizes its per-iteration barrier/addressing cost). Traced/fake tensor execution (`torch.compile`, opcheck) falls back to the pointer kernel.
- The `[batch, n, n]` bool mask also moves through a TMA descriptor, as a bf16 copy. The pointer load looked harmless but was the kernel's single most expensive input: a rank-1 `[BLOCK_K]` tensor feeding a `[j, k]` broadcast gets the replicated `slice<dim=0, parent=#mma>` layout, so it issued **67 M `LDG` / 2.15 GB of L2 sectors to read a 1 MB tensor** (32x amplification) — 10x more requests than q/k/v/o/bias combined. Three details are load bearing and each gives back the whole gain if changed: bf16 rather than a 4-byte type (halves the TMA traffic), a rank-2 box over the flat `[batch·n, padded_n]` view rather than rank-3 over `[batch, i, k]`, and reusing `N` for the row index rather than passing the mask's row count. The box must also be >= 128 bytes or the pipelined loop faults, so `BLOCK_K = 32` configs take a 2-row box. `can_use_tma_mask` additionally requires the mask to really be `n x n`.
- The k loop is **peeled**: `_fwd_kv_block` takes a `K_MASKED` constexpr, and `_fwd` runs `N // BLOCK_K` full blocks with it `False` plus at most one ragged tail block with it `True`. Full blocks provably cannot read past `N`, so they skip the `[k]` range compare and three `[j,k]`-sized selects (`in_range`, and the `mask_k` guards on the scores and on `exp_scores`) that are no-ops there. `n_full` is a runtime value, so this stays one kernel variant per `CLOSEST_N` bucket. Worth **+8.7%** at n=1024 -- the single largest forward win, and it also makes the mask load unpredicated for free.
- The stable online-softmax path computes LSE directly in `_fwd`.

The forward column was remeasured after these changes. The backward columns retain the previous benchmark results.

Together these took n=1024 from 92.99 to 103.46 TFLOP/s (**+11.3%**), and the BF16 tensor pipe from 64.9% to 72.2% of peak. `n=800` gains least (+7.4%) because `800 % 64 == 32` leaves a ragged tail block that still pays the masking. The kernel now issues **zero** non-TMA global loads, down from 67 M. See `FWD_NCU_ANALYSIS.md` for the profile and for the alternatives that were measured and rejected.

### Kernel throughput

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┬──────────────┬───────────────┐
│  N   │ Forward │ Backward Q │ Backward K/V │ Backward Bias │
├──────┼─────────┼────────────┼──────────────┼───────────────┤
│  512 │   95.53 │      94.16 │        68.46 │         67.38 │
│  640 │   98.82 │      95.84 │        70.02 │         70.13 │
│  768 │  100.83 │      97.10 │        70.50 │         71.58 │
│  800 │   91.21 │      93.59 │        65.34 │         68.47 │
│ 1024 │  103.46 │      98.36 │        71.92 │         72.86 │
└──────┴─────────┴────────────┴──────────────┴───────────────┘
```

## Reference kernel: bio `cuda_b`

`src/trifast/bio/cuda_b` is the M1 sm_90a triangle-attention forward from
`uplifting-biomolecular-modeling` (`triattn_pkg/cuda_b`, CUDA + CUTLASS/CuTe): three consumer warpgroups (one pair
row each), max-free streaming of 64x32 score chunks, pair bias staged once per call in fp32 MMA-fragment order, and an
exact "SAFE" pass that recomputes any CTA tile the max-free pass cannot finish. Forward only, bf16, `D = 32`, Hopper only.

```python
from trifast.bio import cuda_b
out, lse, mx, dn = cuda_b.triangle_attention(q, k, v, bias, mask=None, scale=None)
```

- Layouts differ from `trifast.triangle_attention`: `q`/`k`/`v` are `[B, N, H, S, D]` (a `transpose(1, 2)` view of
  ours is TMA-legal, so no copy), `bias` is `[B, 1, H, S, S]`, `mask` is `[B, N, 1, 1, S]`. `out` is a contiguous
  `[B, N, H, S, D]` and `lse`/`mx`/`dn` are `[B, N, H, S]` fp32.
- Changed from upstream: the mask uses our convention (True = masked), and the kernel returns our softmax statistics,
  `P = exp2(x - mx) / dn` with `x = (scale * q.k + bias) * log2(e)`, `lse = (mx + log2 dn) * ln 2`. Fully-masked rows
  get our values (`mx = MASK_FILL * log2(e)`, `dn = S`) and output `mean(v)`.
- `mx`/`dn` are a consistent pair but not our values: the max-free pass never tracks the row max, so `mx` sits ~56-64
  above it and `dn` is ~2^-64 (tiles recomputed by the SAFE pass hold the exact max). `dn` sums the bf16-rounded `P`
  (the row sum comes from a `[V | 1]` MMA), so `lse` carries unbiased bf16 noise up to ~2e-3. Our backward fed cuda_b's
  `(o, mx, dn)` is as accurate as with ours (grad relative L2 error vs fp64: 2.50e-3 vs 2.46e-3).
- JIT-built with `torch.utils.cpp_extension` on first use (~80 s). CUTLASS headers come from `$CUTLASS_PATH`, else
  `/opt/cutlass`, else flashinfer's bundled copy; upstream seals with CUTLASS v4.7.1. Upstream's prebuilt binaries
  do not match the modified source and are not shipped.

Forward vs `cuda_b` (both write output plus lse/mx/dn; cuda_b's timing includes its per-call bias/mask staging):

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┐
│  N   │ Forward │ Bio cuda_b │
├──────┼─────────┼────────────┤
│  256 │   78.81 │      64.30 │
│  300 │   61.97 │      50.22 │
│  351 │   63.72 │      68.73 │
│  410 │   67.12 │      59.10 │
│  490 │   75.54 │      84.13 │
│  512 │   95.60 │      92.19 │
│  640 │   98.74 │      98.60 │
│  768 │  100.84 │     102.20 │
│  800 │   91.23 │      84.97 │
│ 1024 │  103.26 │     108.59 │
└──────┴─────────┴────────────┘
```

Their `cuda/` kernel (`triattn_sm90.cuh`) was also tried and was ~10% slower than our forward at every N, so it is not
included.
