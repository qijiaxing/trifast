# Forward regression investigation and optimized results

Subsequent tuning is documented in [the current report](forward_tuning.md). This report retains the historical `4b832de` source and measurements.

Frozen pointer candidate `64148ed` has slower forward execution than upstream, despite its fused backward improving combined latency. The [baseline report](bucketed_n.md) records its performance and validation. This report records corrected frozen implementation `4b832de`, its complete validation, and final performance.

## Measured contributors

[ablation.log](benchmarks/evidence/forward_recovery/ablation.log) uses warm interleaved ABBA measurements on H20-3e with BF16, B1/H8/D32, 60 samples per side, including forward wrapper costs.

| N | Our runtime-N / exact-N latency, matched configuration | Upstream latency increase with TMA disabled |
|---:|---:|---:|
| 512 | 1.080× | 12.1% |
| 800 | 1.062× | 7.4% |
| 1024 | 1.070× | 15.4% |

The exact-N ablation uses the configuration selected by the runtime candidate, without independent retuning. It isolates length specialization within that implementation and configuration; it is **not a reconstruction of historical `249f`**. Upstream's TMA toggle examines the memory path in a different implementation. These effects cannot be added or multiplied to claim a complete explanation of the regression: other implementation and configuration differences between the pointer candidate and upstream still matter. The evidence supports retaining runtime N within buckets while testing a restored TMA path.

## Final implementation

The new BF16/FP16 forward retains TMA reads for Q/K/V, bias, and mask, and uses ordinary vector stores for output. It produces O, LSE, mx, and dn for the existing fused backward. FP32 retains the centered-statistics forward to preserve its numerical contract. Forward bias/mask padding only provides 16-byte row alignment for descriptors; backward bias and FP32 dBias use CLOSEST_N bucket-width rows. Q/K/V are not padded to the entire bucket. Lengths still use upward power-of-two buckets with actual N at runtime; this does not return to exact-N specialization.

Padded tail keys receive negative-infinity scores before softmax and explicitly zero probabilities, excluding them from normalization over real keys. Real masked keys retain finite −10000 score semantics. Padding and masked real keys must not be conflated, particularly for fully masked rows.

N and strides use runtime int64; descriptor coordinates are explicitly converted to int32 at the descriptor interface. This does not make arbitrary coordinates valid or narrow all global pointer arithmetic to int32. Invalid query-tail rows do not participate in the independent softmax reductions of valid rows, and stores remain bounded. Additional softmax selects that cannot affect valid outputs are therefore removed.

## Output-store investigation and change

The earlier `5f1` candidate with TMA output stores produced initcheck reports of `_preprocess` reading uninitialized O. The original [initcheck.log](benchmarks/evidence/forward_recovery/initcheck.log) and [initcheck-interruption.json](benchmarks/evidence/forward_recovery/initcheck-interruption.json) are retained: the parent driver was interrupted, so still-running command metadata must not be interpreted as success or assigned an invented exit code.

All 12 coverage checks that warmed the kernel, allocated fresh output, and filled it with NaNs passed. Six inspected PTX variants contained output store, commit, and `wait_group.read` instructions. [NVIDIA Compute Sanitizer known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html) document unsupported tensormap/tcgen05 global instructions that may cause initcheck false positives. The observations are consistent with that limitation but do not establish that this particular report was a false positive.

The selected implementation uses ordinary vector output stores while retaining TMA input reads. Four targeted initcheck cases (BF16 D32 and FP16 D128, each in full/chunk modes) exited 0 with zero errors. Preliminary [bench-pointer-store](benchmarks/evidence/forward_recovery/bench-pointer-store.log) forward speed ratios for N=512/800/1024 were 1.033/1.002/1.014×, with directly measured F+B ratios of 1.293/1.182/1.311×. These diagnostics informed the implementation choice; they are not the final submission performance table. Frozen `4b832de` completed full reuse, numerical, eager/compile, memcheck/initcheck, workspace, and performance checks.

## Frozen implementation and final validation

Final code is frozen at **`4b832dee1e34b5303d5b39f9435cc83117064136`**. The following are its own complete validation results; historical pointer `64148ed` remains a control only. Performance measurements are complete below.

| Item | Result |
|---|---|
| Frozen source hashes / commit | `4b832de`; see [frozen_sources.json](benchmarks/evidence/forward_recovery/final/frozen_sources.json) |
| Final performance and memory | See tables below; `64148ed` is historical pointer control only |
| Within-bucket JIT/PTX reuse | Eager/compiled: each four buckets, 17 calls, 46 target PTX compilations; zero new compilation in reused buckets |
| Complete eager / compile | 386 passed (220.10 seconds) / 18 passed |
| Large-N independent FP64 reference | N=500/512/513/800/1024 passed, B1 H8 BF16 D32, one seed per N |
| memcheck / initcheck | Each five passed, zero errors, command exit 0 |


Final conclusions match this candidate's own frozen source and logs; the pointer baseline's passing gates do not transfer. Measurement scope follows the baseline report: warm eager public API including wrapper costs, not isolated kernels, complete training steps, or compilation performance.

Dynamo reports one graph for N>1 and two after the first N=1; repeated N=1 adds no graph. Kernel reuse is not a one-graph guarantee. The reuse proof independently checks only N≤129 in its main sequence; separate FP64 validation provides the large-N numerical evidence.

Hardware validation covers H20 (SM90). The low-precision dispatcher falls back to the pointer forward on SM<90, but other architectures were not tested in this run.

## Final performance and low-memory tradeoff

Both [bench-final](benchmarks/evidence/forward_recovery/final/bench-final.log) and the [workspace comparison](benchmarks/evidence/forward_recovery/final/bench-workspace.log) completed with command exit 0. Environment: H20-3e, PyTorch 2.13.0a0+9186a08b2c.nv26.07, Triton 3.7.1, CUDA 13.3, Compute Sanitizer 2026.2.1.0; BF16, B1 H8 D32, contiguous dO, mask probability 0.2. The upstream control is `b4ecec4`; fused uses default chunk128.

| N | Fwd upstream → fused (ms) | Bwd upstream → fused (ms) | F+B upstream → fused (ms) | Fwd speedup | F+B speedup |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.711 → 1.594 | 12.670 → 6.495 | 14.384 → 8.092 | 1.073× | 1.777× |
| 512 | 1.494 → 1.445 | 7.563 → 5.532 | 9.066 → 6.979 | 1.034× | 1.299× |
| 513 | 2.134 → 1.999 | 15.617 → 7.688 | 17.761 → 9.678 | 1.067× | 1.835× |
| 640 | 2.828 → 2.733 | 14.377 → 10.599 | 17.201 → 13.316 | 1.035× | 1.292× |
| 768 | 4.742 → 4.608 | 24.516 → 17.940 | 29.231 → 22.523 | 1.029× | 1.298× |
| 800 | 5.946 → 5.919 | 29.251 → 23.870 | 35.207 → 29.842 | 1.005× | 1.180× |
| 1024 | 10.790 → 10.637 | 57.144 → 41.290 | 67.936 → 51.961 | 1.014× | 1.307× |

Forward now matches or exceeds upstream on these lengths. Combined gains come from fused backward; these results do not establish gains on unmeasured dtype/D combinations, devices, or complete training steps. Ratios are upstream latency divided by fused latency; raw records also include backward-only speed ratios.

| N | Full → chunk128 F+B (ms) | Chunk latency change | Full → chunk128 peak increment (MiB) | Saved (MiB) |
|---:|---:|---:|---:|---:|
| 512 | 6.923 → 6.956 | +0.48% | 804.52 → 612.52 | 192.00 |
| 800 | 29.528 → 29.784 | +0.87% | 1985.54 → 1459.54 | 526.00 |
| 1024 | 52.360 → 51.970 | -0.74% | 3217.03 → 2321.03 | 896.00 |

Compared with fused full workspace, the default saves 192/526/896 MiB of peak increment, with F+B latency changes from −0.74% to +0.87%. This is a tradeoff against full workspace, not a claim of lower memory than upstream:

| N | Upstream → fused peak increment (MiB) |
|---:|---:|
| 500 | 539.96 → 591.29 |
| 512 | 564.25 → 612.52 |
| 513 | 566.46 → 630.80 |
| 640 | 881.64 → 954.39 |
| 768 | 1269.56 → 1347.77 |
| 800 | 1380.84 → 1459.54 |
| 1024 | 2257.00 → 2321.03 |

MiB=2²⁰ bytes. Peak allocated increment excludes preallocated inputs and includes outputs, gradients, and temporaries; it is not total GPU memory.

Timing uses the warm eager public API: three warmups, three ABBA rounds, 60 samples per side, and ratios of medians. Forward enables gradients, backward-only uses retain_graph, and F+B is measured directly rather than adding columns. Padding/casts/flush and host submission gaps within the event interval are included. These are not isolated-kernel or compiled-performance measurements.

**Cache order:** workspace ran first with `512,800,1024`. In the new H8 v3 forward tuning namespace, 512/800 first triggered buckets 512/1024. The final sequence `500,512,513,640,768,800,1024` reused those selections. Upstream and backward reused existing persistent configurations; [final configuration JSON files](benchmarks/evidence/forward_recovery/final/tuning-configs/) are archived. Therefore 500 did not first determine the final bucket winner, and warm timings are not cold-start costs.

The [final evidence directory](benchmarks/evidence/forward_recovery/final/) contains source inventories, command status, and logs. Large-N independent reference checks cover complete tensors over all i slices, limited to the stated configuration and one seed per N; candidate_seconds is not benchmark timing.

Validation logs: [proof eager](benchmarks/evidence/forward_recovery/final/proof-eager.log) · [proof compiled](benchmarks/evidence/forward_recovery/final/proof-compiled.log) · [eager](benchmarks/evidence/forward_recovery/final/pytest-eager.log) · [compile](benchmarks/evidence/forward_recovery/final/pytest-compile.log) · [FP64](benchmarks/evidence/forward_recovery/final/large-reference.log) · [memcheck](benchmarks/evidence/forward_recovery/final/memcheck.log) · [initcheck](benchmarks/evidence/forward_recovery/final/initcheck.log) · [SHA256SUMS.json](benchmarks/evidence/forward_recovery/final/SHA256SUMS.json)
