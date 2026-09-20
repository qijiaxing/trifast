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
- Aligned bias tiles use a padded 2D Hopper TMA descriptor whose tile shape is updated by an autotune config pre-hook to match `BLOCK_J` and `BLOCK_K`; this reduces compiler-generated shared-memory bank conflicts, while traced/fake tensor execution falls back to pointer loads.
- With the BF16 fixed-offset fast path disabled by default, the stable online-softmax path computes LSE directly in `_fwd` and skips the separate `_fwd_finalize` launch.
- The optional BF16 fixed-offset path is still available through `USE_FAST_PATH`, but the results below use `USE_FAST_PATH=False`.

The forward column was remeasured after these changes. The backward columns retain the previous benchmark results.

### Kernel throughput

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┬──────────────┬───────────────┐
│  N   │ Forward │ Backward Q │ Backward K/V │ Backward Bias │
├──────┼─────────┼────────────┼──────────────┼───────────────┤
│  512 │   79.46 │      94.16 │        68.46 │         67.38 │
│  640 │   81.08 │      95.84 │        70.02 │         70.13 │
│  768 │   82.29 │      97.10 │        70.50 │         71.58 │
│  800 │   79.38 │      93.59 │        65.34 │         68.47 │
│ 1024 │   83.64 │      98.36 │        71.92 │         72.86 │
└──────┴─────────┴────────────┴──────────────┴───────────────┘
```
