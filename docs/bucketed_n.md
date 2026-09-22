# Fused attention with length buckets: frozen pointer baseline

This document records implementation, validation, and performance for frozen pointer candidate `64148ed`. Its forward regression remains the subject of subsequent TMA optimization. This is a control baseline, not a final submission conclusion.

## Bucket rule

The candidate follows upstream `b4ecec4`: for positive integer N, `CLOSEST_N = 2**ceil(log2(N))`, implemented as `1 << (N - 1).bit_length()`. Actual N is a runtime argument; CLOSEST_N is a compile-time constant and part of the tuning key.

| Actual N | CLOSEST_N |
|---|---:|
| 129–256 | 256 |
| 257–512 | 512 |
| 513–1024 | 1024 |
| 1025–2048 | 2048 |

There is no fixed four-bucket catalog. Training lengths in the inclusive range 256–2048 touch four buckets; allowing 1–2048 can touch 12 buckets: 1, 2, 4, …, 2048. Configurations are created for buckets actually encountered.

For example, 500 and 512 share bucket 512; 513 and 800 share bucket 1024. Computation and output semantics still use actual N. Bucket count is not compilation count: the first autotune for a bucket compiles and measures multiple candidates; forward and backward have separate kernels, and dtype, H, D, device, and other configuration changes can introduce additional versions. The target is to avoid specialization for each new exact N within a bucket at fixed configuration. Actual JIT keys and PTX generation must establish this; tuning keys alone are insufficient evidence.

## Candidate implementation

Forward and fused backward declare N as runtime `tl.int64` and include it in `do_not_specialize`. The forward tuning key contains H, DIM, CLOSEST_N, and DTYPE_ID; backward uses H, D, BJ, BK, CENTERED, and CLOSEST_N. Exact N is absent. Forward currently has six candidates: tiles 32×32, 64×32, and 64×64, each with stages=1/3. For FP32 with D≥64, pruning keeps only 32×32 with stage=1. Backward has stages=1/2/3 candidates, also restricted to stage=1 for FP32 with D≥64. Forward register-limit candidates were removed after providing no benefit. These configuration choices do not change the bucket rule.

To recover alignment information for bias rows, only the physical last dimension of bias is padded to CLOSEST_N, producing `[B,H,N,CLOSEST_N]`. Backward uses the same physical row width for FP32 dBias accumulation, then slices the first N columns and returns contiguous gradients. Q, K, V, and logical outputs retain actual N; the full attention problem is not expanded to the bucket size. Padded columns are not valid keys: tail probabilities must be zero and excluded from the softmax denominator. Bias padding, dBias slicing, and materializing contiguous dO count toward complete call cost.

Forward loops over full key tiles and handles the masked tail separately. Each fused backward CTA owns `(batch, head, i, key tile)` and traverses all query tiles. Full query tiles use a loop without query-tail masks; an incomplete final tile is handled separately. Runtime branches give CTAs owning full key tiles a fast path while retaining necessary masks for the last partial key tile. These paths live inside a bucket kernel rather than creating an exact-N specialization.

Backward shares intermediates across five matrix products: QKᵀ to reconstruct probabilities, dO Vᵀ, dScore K, dScoreᵀ Q, and Pᵀ dO. Each owner accumulates dK/dV in FP32 and writes them uniquely. FP32 atomics reduce dQ across key tiles and dBias across i. The full probability tensor is not materialized. Delta and mask metadata still require a separate preprocessing kernel.

The default `chunk_i=128` processes slices of i. FP32 dQ scratch requires `4*B*H*min(128,N)*N*D` bytes. FP32 dBias additionally requires `4*B*H*N*CLOSEST_N` bytes; final gradients, statistics, and other buffers also consume memory. Scratch size is not total peak memory. An explicit full-workspace mode remains available. Each chunk clears scratch, launches the fused kernel, and flushes dQ to its final dtype.

The public autograd integration and opaque `torch.library.custom_op` dispatch boundaries remain, preventing `torch.compile` from directly unrolling the Python chunk loop according to N. Kernel reuse within a bucket and Dynamo graph reuse are separate metrics: singleton dimensions and other guards can still introduce graphs.

Finite masked-score replacement at −10000, fully masked rows, stable singleton gradients, and FP32 centered statistics preserve their semantics. Only first-order gradients are supported; atomic reductions are nondeterministic.

## Historical versions

| Version | Length strategy | Scope of performance evidence |
|---|---|---|
| Original fused PR `249f3e2` | Exact-N specialization | Original performance numbers describe that frozen version only |
| Strict dynamic experiment `db7285d` | No length buckets; fixed configuration reused across N | See the [dynamic experiment report](dynamic_n.md); performance target was not met |
| Current candidate | Runtime actual N with upward power-of-two tuning buckets | See the frozen pointer baseline results below |

## Completed compilation-reuse checks

See the [frozen source inventory](benchmarks/evidence/bucketed_n/frozen_sources.json). All six commands below exited 0 on NVIDIA H20-3e, PyTorch 2.13.0a0+9186a08b2c.nv26.07, and Triton 3.7.1, with B=1 and H=2.

| Evidence | Configuration and length coverage | Total target-kernel PTX compilations | New compilations in previously seen buckets |
|---|---|---:|---:|
| [eager](benchmarks/evidence/bucketed_n/proof-eager.log) | BF16 D32, 15 steps, buckets 512/1024/256/1, chunk128 | 38 | 0 |
| [compiled](benchmarks/evidence/bucketed_n/proof-compiled.log) | Same sequence with dynamic/fullgraph compile | 38 | 0 |
| [full workspace](benchmarks/evidence/bucketed_n/proof-full.log) | BF16 D32, same 15 steps | 37 | 0 |
| [small reference](benchmarks/evidence/bucketed_n/proof-small-reference.log) | BF16 D32, 11 steps, buckets 128/256/1, independent reference for all N≤256 | 29 | 0 |
| [FP16 D128](benchmarks/evidence/bucketed_n/proof-fp16-d128.log) | 7 steps: N=65/80/127/128/65/1/1, transposed dO | 20 | 0 |
| [FP32 D128](benchmarks/evidence/bucketed_n/proof-fp32-d128.log) | Same 7 steps, contiguous dO | 6 | 0 |

The main 15-step sequence is `257,300,500,512,257,513,800,1024,513,129,200,256,129,1,1`. The default path compiles 11 target versions for its first bucket (six forward candidates, three backward candidates, preprocessing, and flush), then nine candidate versions for each new bucket. Revisiting a bucket adds zero JIT keys and zero PTX compilations. Full workspace starts with ten because it has no flush. After pruning, FP32 D128 compiles four for the first bucket and two for the next. Process-wide PTX counts also include reference code; the table counts target kernels only.

All six runs reported zero numerical failures and each included two extra singleton-valid-key checks. Independent reference evaluation in the main sequence was limited to N≤129; compilation reuse at larger N is not complete independent numerical validation. The small-reference and two D128 runs checked their entire main sequences against the reference. In the compiled run, Dynamo recorded one unique graph for N>1 and a second on the first N=1; repeating N=1 added none.

## Incomplete run and retry

The first cold-configuration complete eager matrix on 2026-09-22 reached the 1800-second command timeout and exited **124** (`experiments/bucketed-n-20260922/pytest-eager.log` and its `.command.json`). About 100 test dots were logged without a numerical failure, but no completion summary was produced: this is **not a complete-suite pass**. The matrix spans 12 dtype/D combinations and multiple length buckets. Its 30-minute timeout must not be described as startup time for a single training configuration.

GPU job 73115 subsequently reached its one-hour allocation limit; it had expired by the extension attempt. A new two-hour GPU session subsequently completed `pytest-eager-warm` serially, reusing saved workspace autotune selections and rebuilding its `/tmp` Triton compilation cache. Completed retry, compile matrix, and `bench-final` results appear below. The initial timeout record remains separate from the retry.

## Measurement and validation scope

The final benchmark measures the eager public API after warmup, samples in ABBA order, and reports the ratio of sample medians. Combined forward/backward latency is measured directly rather than constructed by adding separate columns. Backward-only timing uses `retain_graph`; forward timing has gradients enabled. These results describe warm API-call costs, not isolated kernel latency, a complete training step, or `torch.compile` performance.

Complete-call timing includes bias padding, casts, dQ flushing, and host submission gaps within the CUDA-event measurement interval. The first N encountered and existing caches can affect the tuning choice shared by a bucket. Final results must specify the actual shape order, warmup, and cache conditions, rather than listing an unordered set of N values.

Memory is the peak allocated increment beyond preallocated inputs and includes outputs, gradients, and temporary allocations. It is not total process GPU memory and excludes the already allocated inputs. Both sides must use the same measurement boundary.

The large-N independent FP64 reference traverses all i slices and checks complete tensors rather than sampled rows. Coverage is still limited to the actual command's B=1, H=8, specified dtype/D, and one random seed. The reference script's `candidate_seconds` records validation-flow elapsed time and must not be used as a performance benchmark. The warm eager retry is complete; pass counts and elapsed times appear below.

## Frozen pointer candidate: performance and completed gates

This section describes only **`64148ed4a1825f98d3175ba0267a11101fcc965e`**, the pointer-forward candidate used as the control before subsequent TMA forward work. The user requested investigation and optimization of its forward regression before publishing a PR. These are neither measurements of subsequent code nor a final submission conclusion.

[Raw bench-final evidence](benchmarks/evidence/bucketed_n/bench-final.log): upstream `b4ecec4` versus the candidate on H20-3e, BF16, B=1, H=8, D=32, default chunk128, contiguous dO, mask probability 0.2. Actual shape order: **500,512,513,640,768,800,1024**. Each mode used three warmups and three ABBA rounds, yielding 60 samples per side. Saved workspace tuning selections were reused; this is not independent cold tuning for every N. Read exact cache conditions and configuration choices together with raw logs and tuning records.

| N | Forward upstream → candidate (ms) | Backward upstream → candidate (ms) | F+B upstream → candidate (ms) | F+B speedup |
|---:|---:|---:|---:|---:|
| 500 | 1.710 → 2.032 | 12.658 → 6.496 | 14.378 → 8.514 | 1.689× |
| 512 | 1.493 → 1.987 | 7.557 → 5.557 | 9.050 → 7.543 | 1.200× |
| 513 | 2.130 → 2.410 | 15.608 → 7.677 | 17.744 → 10.088 | 1.759× |
| 640 | 2.820 → 3.821 | 14.376 → 10.595 | 17.199 → 14.432 | 1.192× |
| 768 | 4.734 → 6.491 | 24.500 → 17.870 | 29.233 → 24.395 | 1.198× |
| 800 | 5.947 → 7.587 | 29.256 → 23.859 | 35.200 → 31.468 | 1.119× |
| 1024 | 10.715 → 14.796 | 56.645 → 40.989 | 67.427 → 55.818 | 1.208× |

Forward regresses at all seven lengths: upstream/candidate speed ratios are 0.724–0.884×. Backward improves by 1.226–2.033×, and directly measured combined F+B improves by 1.119–1.759×. Backward gains do not resolve the forward regression; subsequent optimization uses this record as its control. The earlier exact-N implementation's approximately 1.5× figures do not replace this table.

Peak allocated increment for the same call (MiB=2²⁰ bytes):

| N | Upstream (MiB) | Candidate (MiB) | Increase (MiB) |
|---:|---:|---:|---:|
| 500 | 539.96 | 591.56 | 51.60 |
| 512 | 564.25 | 612.52 | 48.27 |
| 513 | 566.46 | 630.80 | 64.34 |
| 640 | 881.64 | 954.39 | 72.75 |
| 768 | 1269.56 | 1347.77 | 78.21 |
| 800 | 1380.84 | 1459.54 | 78.70 |
| 1024 | 2257.00 | 2321.03 | 64.03 |

“Low memory” means reduced dQ scratch relative to fused full workspace, not reduced memory relative to upstream. The candidate uses 48.27–78.70 MiB more than upstream for these shapes. A full-workspace comparison will be recorded separately.

| Completed check | Result |
|---|---|
| [Complete eager retry](benchmarks/evidence/bucketed_n/pytest-eager-warm.log) | 374 passed, 18 deselected; pytest 2024.81 seconds, total command 2026.39 seconds including startup; exit 0 |
| [Compile matrix](benchmarks/evidence/bucketed_n/pytest-compile.log) | 18 passed, 331 deselected; exit 0 |
| [Large-N independent FP64 reference](benchmarks/evidence/bucketed_n/large-reference.log) | N=500/512/513/800/1024, B1 H8 BF16 D32; complete output and four gradients over all i slices; exit 0 |
| [Final pointer benchmark](benchmarks/evidence/bucketed_n/bench-final.log) | All seven lengths measured; exit 0 |
| Six bucket proofs above | All exited 0, zero new compilation in previously seen buckets |

Large-N validation uses one seed per length (`20260921+n`), not a multiple-seed robustness study. No new sanitizer pass is claimed here; historical checks do not automatically cover this candidate. The original cold-matrix timeout remains part of the record. Subsequent TMA code requires its own source freeze and validation; these results cannot be transferred to it.
