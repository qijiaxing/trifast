# Opt-in fused triangle attention

This implementation adds an opt-in fused backward whose default uses bounded-memory chunking (`chunk_i=128`). The existing `trifast.triangle_attention` remains unchanged. On the PR base (`b4ecec4`), that existing implementation already includes its own TMA forward; the new APIs use a separate pointer-based online-softmax forward and do not replace the upstream TMA configuration or hooks.

See the [H20 performance report](benchmarks/h20_fused.md) for measurements against this upstream baseline, raw samples, and validation evidence.

## Usage

```python
from trifast import triangle_attention_fused

# q, k, v: [B, H, N, N, D]
# bias:    [B, H, N, N]
# mask:    [B, N, N], torch.bool
output = triangle_attention_fused(q, k, v, bias, mask)
output.backward(grad_output)
```

All inputs must be on the same CUDA device. Query, key, value, and bias must have the same dtype: `torch.bfloat16`, `torch.float16`, or `torch.float32`. Supported head dimensions are 16, 32, 64, and 128. Batch size, head count, and sequence length must be positive, and the two attention axes must be square. Noncontiguous inputs and grad outputs are accepted; layout normalization may allocate copies.

`triangle_attention_fused(..., chunk_i=128)` is the default low-memory path.
Pass `chunk_i=None` to request the full-workspace fused backward explicitly:

```python
output = triangle_attention_fused(q, k, v, bias, mask, chunk_i=None)
```

The default is fixed; it does not inspect free GPU memory or automatically switch algorithms. The separate low-memory helper remains available for compatibility:

```python
from trifast.fused_low_memory_api import triangle_attention_fused_low_memory

output = triangle_attention_fused_low_memory(
    q, k, v, bias, mask, chunk_i=128,
)
output.backward(grad_output)
```

`triangle_attention_fused` accepts a positive integer or `None`; booleans are rejected. The compatibility helper accepts positive integers only and rejects `None`. Values greater than `N` use one chunk. Choose a different chunk size or full-workspace mode explicitly using measurements for the intended workload.

## Computation and numerical semantics

The original backward independently computes query, key/value, and bias gradients. Across those passes it performs nine matrix dot products. The fused main kernel shares reconstructed probabilities and score gradients to compute all four gradients with five dot products: QK, dO·Vᵀ, dQ, dK, and dV. A preprocessing kernel computes normalization-gradient terms and mask-support information. “Fused” therefore does not mean the complete operation has only one GPU launch: allocation initialization, preprocessing, and output casts also contribute to its cost.

Each main program owns a key tile. It accumulates dK and dV over all queries in FP32 and writes each output element once. dQ and dBias receive contributions from multiple programs, so they accumulate using FP32 atomics and are cast to the input dtype after reduction. No full attention probability matrix is materialized.

The boolean mask uses **finite score replacement**: `True` replaces a score with `-10000`, before softmax. An entirely masked row consequently produces the mean of V, not a zero output; its dQ, dK, and dBias are zero while dV remains nonzero. Padding outside the logical sequence is excluded separately and must not act as another finite-masked key.

A row with one unmasked key needs particular care. Subtracting independently rounded versions of the same inner product can introduce a spurious gradient. Preprocessing supplies a stable difference-based expression for this case, retaining contributions from finite-sentinel keys when their probability is nonzero. It does not assume all masked probabilities always underflow.

For FP32 inputs, forward subtracts a per-query bias offset before combining small score differences. The saved maximum and denominator are in the matching centered domain, and backward reconstructs scores in that same domain. This prevents loss of relevant differences near the finite sentinel. FP16/BF16 keep the uncentered score domain. Internal saved statistics are implementation details and must not be interchanged between incompatible forwards.

Forward tuning uses independent pointer-kernel configurations, without the upstream TMA descriptor hooks. Its tuning key includes the actual `N`, not only the next-power-of-two bucket. Shape and stride specialization permits removal of padding checks for complete tiles, while retaining tail checks otherwise. Backward uses a smaller query tile for D128; FP32 with D64/D128 also uses conservative tile/stage settings to limit shared-memory requirements.

## Memory tradeoff

Let `X = B * H * N * N * D` and `Y = B * H * N * N`. The full-workspace mode (`chunk_i=None`) uses FP32 dQ and dBias workspaces of `4X` and `4Y` bytes, in addition to outputs and preprocessing buffers. For B1/H8/N1024/D32, dQ scratch alone is 1 GiB. Input copies and saved forward state may increase peak memory further.

The default low-memory implementation processes independent `i` rows in chunks. Its FP32 dQ scratch is bounded by

```text
4 * B * H * min(chunk_i, N) * N * D bytes
```

All key-tile contributions for a query are still accumulated in FP32 before one final cast. dBias remains a global FP32 reduction across chunks. Each chunk adds scratch initialization and a cast/copy launch, so lower memory is not guaranteed to mean lower latency. Python chunk iteration can also increase compiled graph size or specialization count. These are explicit tradeoffs, not an automatic fallback.

## Validation

The unit tests compare output and all four gradients with an independent FP64 PyTorch finite-mask softmax/autograd reference. Fixed relative L2 limits are 0.012 for BF16, 0.002 for FP16, and 2e-5 for FP32. Reference-zero elements additionally require absolute error at most 2e-5, and all results must be finite. Baseline parity in the benchmark is only a diagnostic; it does not replace these independent gates.

From a checkout with the test dependencies installed and a CUDA device available:

```bash
python -m pytest -q tests/unit/test_fused.py tests/unit/test_fused_contracts.py -m 'not compile'
python -m pytest -q tests/unit/test_fused.py tests/unit/test_fused_contracts.py -m compile

compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest -q tests/unit/test_fused.py -m sanitizer
compute-sanitizer --tool initcheck --error-exitcode 99 \
  python -m pytest -q tests/unit/test_fused.py -m sanitizer
```

Coverage includes finite-sentinel and singleton cases, full masking, sequence tails, supported dtypes/head dimensions, low-memory chunk tails, shared views, noncontiguous or broadcast grad outputs, deterministic-mode rejection, and first-derivative-only behavior. Sanitizer cases use a CPU FP64 reference to avoid instrumenting reference cuBLAS operations. CUDA tests may skip when CUDA is absent; a skipped run is not evidence of GPU correctness.

## Benchmarking

```bash
python scripts/bench_fused.py --shapes 512,800,1024 \
  --baseline original --candidate low-memory --noncontiguous-do

python scripts/bench_fused.py --shapes 512,800,1024 \
  --baseline full-workspace --candidate low-memory --chunk-i 128 \
  --noncontiguous-do
```

The benchmark defaults to `--baseline original --candidate low-memory`.
`--baseline` accepts `original` or `full-workspace`; `--candidate` accepts
`full-workspace` or `low-memory`.

The standalone validator likewise defaults to the low-memory path:

```bash
python scripts/validate_fused.py --api low-memory
```

Its `--api` choices are `original`, `full-workspace`, and `low-memory`.

`--modes forward,backward,forward_backward` selects the measured boundaries. Forward uses grad-enabled inputs; backward reuses an existing graph; forward/backward rebuilds it each iteration. Compilation/autotuning and warmup precede timing. Interleaved ABBA CUDA-event samples include auxiliary launches, casts, copies, and host submission gaps. Peak allocation is measured separately, excluding preallocated inputs and including retained output, gradients, and temporary workspace. JSONL records the revision, source hashes, input settings, diagnostic parity, samples, and peak allocation.

Raw development caches, temporary paths, and the entire exploratory experiment archive are not part of the PR. Only selected reproducible benchmark evidence is intended for `docs/benchmarks`.

## Limitations and tested environment

The verified environment is NVIDIA H20 with PyTorch 2.13 development build and Triton 3.7.1. This does **not** establish support for every GPU or every version allowed by the repository's minimum dependency declarations. Resource choices and performance tuning were measured on H20; other architectures require their own validation.

Only first derivatives are supported. `once_differentiable` rejects second derivatives. Atomics are nondeterministic, so both forward and backward reject `torch.use_deterministic_algorithms(True)`, including warn-only mode. There is no claim of JVP, vmap, arbitrary nonfinite-input, or unlimited-shape support.

`torch.compile(..., fullgraph=True)` correctness has been exercised on the tested stack, but compilation is not a promise of improved latency. In observed compiled-autograd runs, a noncontiguous grad output can be copied to a contiguous tangent before entering generated backward code, although the eager kernel can read that strided grad output directly. Include this copy in end-to-end comparisons. Changing sequence lengths, strides, or chunk sizes may also trigger specialization or recompilation.

The implementation lives in the private `_fused_forward`, `_fused_backward`, and `_fused_chunked` modules, with public opt-in wrappers in `fused_api` and `fused_low_memory_api`. Experimental mask-distribution shortcuts and unrelated tuning candidates are outside this change.
