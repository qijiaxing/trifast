# Runtime sequence length — experimental branch

> Historical frozen experiment `db7285d`: these results apply to the strict cross-length reuse implementation. See the [bucketed report](bucketed_n.md) for the current implementation.

**This experimental branch has not met the performance target. Original PR #3 remains at `249f`; its shape-specialized performance results do not describe this prototype.**

The fused path passes sequence length `N` at runtime. For a fixed device, dtype, head count, channel dimension and launch configuration, changing `N` reuses the attention kernels instead of selecting a new per-length Triton specialization. This is a kernel reuse contract, not a promise that every PyTorch graph compiles only once.

The public shape remains `[B,H,N,N,D]`, with positive dimensions, D in {16,32,64,128}, matching floating-point inputs, and a boolean mask. The finite score-replacement value remains −10000. FP32 forward statistics retain the centered-domain convention used by backward. Atomic gradient reductions remain nondeterministic and only first derivatives are supported. Other devices, dtypes, D/H values or launch configurations can require distinct binaries; the observed reuse results below do not establish every possible configuration.

“Across lengths” means supported shapes that fit device memory and the launch grid, not unbounded sizes. The present mapping uses grid.y=N (forward/preprocessing) and grid.z=B×H, each limited to 65,535 on CUDA. The tests below do not establish correctness near those limits.

## Implementation

`_fused_forward.py` removes per-N autotuning and the rounded-N compilation parameter. The frozen forward uses one full-key-block loop plus a separate masked tail block via `_fwd_kv_block(K_MASKED=...)`. BF16/FP16 use fixed 64×32 tiles, four warps and three stages; FP32 uses 32×32, four warps and one stage. Query masks remain where needed. N uses an explicit `tl.int64` scalar ABI and `do_not_specialize`, avoiding value-one/alignment specialization and scalar-type changes across lengths.

Backward preprocessing scans fixed-width mask blocks with runtime loops instead of `arange(next_power_of_2(N))`. Chunk start, capacity and flush length remain runtime values. The current prototype **canonicalizes dO with `reshape(B*H,N,N,D).contiguous()`** and shares Q's dense address map; it does not directly pass arbitrary dO strides through to the attention kernel. Noncontiguous and wide-stride input views remain accepted through this materialization, whose copy cost and allocation belong to the call. Dense global address bases retain wide arithmetic. The singleton stable-gradient formula and N=1 behavior remain unchanged. The main backward selects three paths inside the same binary: full-tile-aligned N, N divisible by eight, and generic tails. These are runtime branches, not separate N-specialized launches. The obsolete `DO_TRANSPOSED` argument has been removed.

### Comparison with upstream

At upstream commit `b4ecec4`, `_fwd` already declares `N` as a runtime argument rather than `tl.constexpr`. However, the wrapper also computes `CLOSEST_N = 2**ceil(log2(N))`; this value is constexpr and part of the autotune key (`H`, `DIM`, `CLOSEST_N`). Therefore “upstream recompiles for every N” is inaccurate: it has length buckets and may reuse a kernel within a bucket, while crossing bucket boundaries can select another specialization. Other argument guards and alignment specialization also matter.

This prototype removes that bucket dependency and targets reuse across bucket boundaries for a fixed configuration. This is a stronger cross-length kernel-reuse objective, not the first introduction of a runtime N argument. It does not establish a performance advantage over upstream or over the shape-specialized fused PR.

A main backward CTA owns `(batch, head, i, key tile)` and loops over query tiles. It computes five matrix products: QKᵀ to reconstruct probabilities, dO Vᵀ, dScore K, dScoreᵀ Q and Pᵀ dO. P and dScore are shared across the gradient computations. dK/dV have a unique writer; dQ uses FP32 atomic reduction across key tiles and dBias uses FP32 atomics across i. No cubic probability tensor is stored. Preprocessing computes the softmax delta and mask metadata separately, so the complete backward is not a single launch.

The default low-memory path uses `chunk_i=128`. Its FP32 dQ scratch holds `B × H × min(128,N) × N × D` elements, or **4 × B × H × min(128,N) × N × D bytes**. This formula covers only dQ scratch, not all peak memory. The final dQ, dK/dV, FP32 dBias and normalization/metadata buffers are additional allocations. Chunks run sequentially; each clears scratch, launches the fused main kernel and flushes its dQ slice into the final dtype. All key-tile contributions for a slice accumulate before that conversion.

Opaque `torch.library.custom_op` dispatcher boundaries contain forward launch selection and the Python backward chunk loop. Their fake implementations describe output shapes. Consequently, `torch.compile` does not expand the loop into a separate graph per number of chunks. The public autograd function provides the backward and retains the one-derivative contract. This boundary does not suppress Dynamo's own shape guards or singleton-dimension graph specialization.

## Frozen-source validation

All links below refer to the frozen evidence snapshot, including source hashes. The historical `eager-fourth`/`compiled-first` prototype logs are superseded and do not establish final-source results.

| Gate | Completed evidence |
|---|---|
| [Eager tests](benchmarks/evidence/dynamic_n/pytest-frozen-eager.log) | 374 passed, 18 compile tests deselected |
| [Compile tests](benchmarks/evidence/dynamic_n/pytest-frozen-compile.log) | 18 passed, 331 deselected; two invocations per test, with independent variants isolated |
| [Large FP64](benchmarks/evidence/dynamic_n/large-frozen.log) | BF16 B1H8D32, N512/800/1024: 3 independent FP64 cases, zero failures |
| [Eager kernel reuse](benchmarks/evidence/dynamic_n/proof-frozen-eager.log) / [compiled reuse](benchmarks/evidence/dynamic_n/proof-frozen-compiled.log) | 12 target calls each, 4 target PTX events total, none after the first; zero numerical failures |
| [Full-workspace eager](benchmarks/evidence/dynamic_n/proof-full.log) / [compiled](benchmarks/evidence/dynamic_n/proof-full-compiled.log) | 12 target calls each, 3 target PTX events total per run, none after the first; N=1 permits a second Dynamo graph |
| [FP16](benchmarks/evidence/dynamic_n/proof-fp16.log) / [FP32](benchmarks/evidence/dynamic_n/proof-fp32.log) / [D128](benchmarks/evidence/dynamic_n/proof-d128.log) | 8 target calls each, 4 target PTX events each, none after the first; zero numerical failures |
| [memcheck](benchmarks/evidence/dynamic_n/memcheck-frozen.log) / [initcheck](benchmarks/evidence/dynamic_n/initcheck-frozen.log) | Five tests each; zero sanitizer errors |

The BF16 low-memory reuse sequence is `65,64,129,128,17,1,500,512,513,800,65`, plus a valid-key singleton check. Its independent FP64 checks cover N≤129; the separate large-shape gate above supplies the listed larger-shape accuracy evidence. Process-wide PTX counters also include unrelated kernels and must not be substituted for target counts. In the compiled reuse run, N=1 triggers a second Dynamo graph while target binaries remain unchanged: **kernel reuse is not a one-graph guarantee**. Compile tests reset Dynamo between independent parameter variants, while the two calls within a test reuse their compiled callable.

## Frozen performance: regression versus original upstream

[Raw benchmark](benchmarks/evidence/dynamic_n/bench-frozen.log) completed all 9 shape/mode summaries and exited zero. H20-3e, BF16 B1H8D32, contiguous dO, default `chunk_i=128`, mask probability .2. Three-round warm ABBA, 60 samples per label/shape/mode, CUDA events, **no CUDA graphs**; initialization, copies and autograd costs are included. This table uses complete forward+backward, with A=original upstream `b4ecec4`, B=runtime-N low-memory candidate.

| N | Original A ms | Dynamic B ms | A/B ratio | Latency increase B/A−1 |
|---|---:|---:|---:|---:|
| 512 | 9.052 | 10.136 | 0.893× | +12.0% |
| 800 | 35.113 | 67.505 | 0.520× | +92.3% |
| 1024 | 67.910 | 76.276 | 0.890× | +12.3% |

Ratios below one mean slower execution. These results are not speedups and are not measurements of original PR #3. Binary reuse reduces repeated specialization; it does not establish training-step improvements or performance parity.

Potential next experiments include TMA-based data movement and canonical padded strides to recover vectorization. They are not implemented or validated here, and no future speedup is promised.

```bash
PYTHONPATH=src python scripts/check_dynamic_n.py --shapes 65,64,129,128,17,1,500,512,513,800,65
PYTHONPATH=src python scripts/check_dynamic_n.py --shapes 65,64,129,128,17,1,500,512,513,800,65 --torch-compile
```
