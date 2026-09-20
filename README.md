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
- The stable online-softmax path computes LSE directly in `_fwd`.

The forward column was remeasured after these changes. The backward columns retain the previous benchmark results.

### Kernel throughput

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┬──────────────┬───────────────┐
│  N   │ Forward │ Backward Q │ Backward K/V │ Backward Bias │
├──────┼─────────┼────────────┼──────────────┼───────────────┤
│  512 │   87.70 │      94.16 │        68.46 │         67.38 │
│  640 │   89.89 │      95.84 │        70.02 │         70.13 │
│  768 │   91.32 │      97.10 │        70.50 │         71.58 │
│  800 │   84.96 │      93.59 │        65.34 │         68.47 │
│ 1024 │   92.99 │      98.36 │        71.92 │         72.86 │
└──────┴─────────┴────────────┴──────────────┴───────────────┘
```
