# TriFast

This is TriFast fork, which is optimized for Hopper GPU.

`python scripts/bench_kernels.py` can be used to run benchmark.

## Perf

H20 GPU, H = 8, D = 32.

```
TriFast individual-kernel throughput — BF16 (TFLOP/s)
┌──────┬─────────┬────────────┬──────────────┬───────────────┐
│  N   │ Forward │ Backward Q │ Backward K/V │ Backward Bias │
├──────┼─────────┼────────────┼──────────────┼───────────────┤
│  512 │   76.51 │      94.16 │        68.46 │         67.38 │
│  640 │   78.13 │      95.84 │        70.02 │         70.13 │
│  768 │   79.17 │      97.10 │        70.50 │         71.58 │
│  800 │   76.37 │      93.59 │        65.34 │         68.47 │
│ 1024 │   80.83 │      98.36 │        71.92 │         72.86 │
└──────┴─────────┴────────────┴──────────────┴───────────────┘
```
