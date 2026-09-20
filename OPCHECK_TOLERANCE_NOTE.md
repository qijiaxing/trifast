# Opcheck tolerance follow-up

`tests/unit/test_trifast_opcheck.py` currently uses:

```python
{"atol": 2e-2, "rtol": 1e-2}
```

This is intentionally temporary. Eager execution uses the Q/K/V/O TMA forward kernel, while fake-tensor/AOT execution falls back to the pointer kernel. The two autotuners can select different tile sizes, changing FP32 accumulation order before values are rounded to BF16. On an H20 with Torch 2.13 and Triton 3.7.1, observed backward-bias differences were approximately `0.012`–`0.016`, which exceeded the previous `atol=1e-2`.

The concern is that a global `2e-2` absolute tolerance may hide a real registration, compilation, or gradient bug. Future investigation should:

1. Add a deterministic seed and preserve a minimal failing input.
2. Compare every forward result and input gradient separately to identify which tensor needs relaxation.
3. Force TMA and pointer paths to use the same `BLOCK_J`/`BLOCK_K` configuration, separating accumulation-order differences from descriptor-path differences.
4. Prefer dtype/ULP-aware or per-output tolerances over one global tolerance.
5. Recheck with the supported Torch/Triton versions from `uv.lock`.

Once the source and expected bound are established, tighten `OPCHECK_TOL` accordingly.
