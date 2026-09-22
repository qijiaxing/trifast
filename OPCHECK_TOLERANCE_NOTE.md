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

## Measured, 2026-09-22 — the test is flaky, and `2e-2` is already too tight

Partial answers to items 2 and 3, found while profiling the fused backward. Eager vs
`torch.compile` gradients, worst absolute difference over seeds 0-5, bf16:

| shape | dq | dk | dv | **db** |
| --- | --- | --- | --- | --- |
| n=256, d=32, h=4 | 0.0078 | 0.0156 | 0.0156 | **0.0312** |
| n=128, d=64, h=1 | 0.0156 | 0.0312 | 0.0156 | **0.1250** |

**`db` at d=64 is 6x the `2e-2` tolerance.** The test passes only because nothing seeds
the RNG — `test_opcheck` calls `gen_tensors` off the global generator, so its inputs
depend on how many random draws the preceding tests consumed. It passes when run alone and
fails in the full suite, on one element out of 262,144 (`db`, n=256):

```
Mismatched elements: 1 / 262144 (0.0%)
Greatest absolute difference: 0.025390625 at index (0, 0, 226, 33) (up to 0.02 allowed)
```

That is item 1 (seed it) promoted from cleanup to prerequisite: right now the suite's
colour is a coin flip.

Towards item 3 — **the dominant source is the forward path, not the backward.** Two
independent pieces of evidence:

- The n=128/d=64 row above is *bit-identical* between the bf16-bias and fp32-bias versions
  of `_bwd_fused` (0.1250 both ways), and those differ substantially in the backward.
- The in-suite failure reproduces at the *same element with the same value*
  (`0.025390625` at index `(0, 0, 226, 33)`) across three different backward
  implementations: bf16 bias, fp32 bias with a pointer load, and fp32 bias through TMA.

So what survives is the eager TMA forward vs the fake-tensor pointer forward: a different
forward tile gives different `(mx, dn)`, hence a different `p`, hence a different `db`.
Forcing both paths to one `BLOCK_J`/`BLOCK_K` -- item 3 -- should therefore collapse most
of this, and is worth more than relaxing the tolerance further.

`db` being worst in every row is expected and is not evidence of a bug: `db[j,k]` sums `n`
signed terms over the triangle axis, so it cancels heavily, and the absolute error is set
by the largest intermediate rather than by the result. Per-output, cancellation-aware
bounds (item 4) are the right fix; a single global `atol` cannot express this.
