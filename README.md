# TriFast

This is TriFast fork, which is optimized for Hopper GPU.

`src/trifast/triton.py`: the fwd and bwd kernels.
`src/trifast/torch.py`: `triangle_attention` torch op.
`src/trifast/autotune_helpers.py`: triton config autotune candidates.
`python scripts/bench_kernels.py` can be used to run benchmark.

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

## Opt-in fused backward

An optional fused implementation shares backward score/probability recomputation.
The original `triangle_attention` remains unchanged. The opt-in
`triangle_attention_fused` defaults to low-memory chunking (`chunk_i=128`),
trading additional launches for bounded FP32 dQ scratch. Pass `chunk_i=None`
for the full-workspace fused mode. This is a fixed default, not automatic
selection based on free GPU memory.

```python
from trifast import triangle_attention_fused
out = triangle_attention_fused(q, k, v, bias, mask)  # low-memory, chunk_i=128
# out = triangle_attention_fused(q, k, v, bias, mask, chunk_i=None)
```

The explicit `triangle_attention_fused_low_memory` helper remains available
for compatibility; it requires a positive integer chunk size and rejects `None`.

See [implementation, usage and validation](docs/fused_attention.md),
[中文说明](docs/fused_attention_zh.md), and the
[H20 measurements](docs/benchmarks/h20_fused.md) against the current baseline.
