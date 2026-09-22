# Fused attention: final forward tuning

This report covers frozen implementation **`61da9e0`**, adding modest forward improvements to the [validated TMA-forward/fused-backward implementation](forward_recovery.md). Final public-API measurements and this validation round are complete: versus the previous fused version, forward speed improves 1.68–2.74% and directly measured F+B improves 0.35–0.52%. Earlier f21 measurements remain diagnostic only.

## Implementation

The fused TMA path gets its own configuration pool, cloned from upstream with descriptor setup hooks preserved. For D=32 only, it adds a 64×64 tile, four warps, two stages, and maxnreg=96 candidate. Other D values prune that candidate; upstream and the pointer fallback retain their existing pools. This is a tuning option, not a forced configuration for every input.

The wrapper skips padding when no additional columns are needed. Mask widening is still performed; an already suitable widened mask avoids the extra no-op pad. Bias requires both a 16-byte-aligned row pitch and base address. A contiguous bias view can have a misaligned storage offset, so it still gets a fresh buffer when `b.data_ptr() % 16 != 0`, even when its width is aligned. Four tests cover this protection.

Actual N remains runtime, with upward power-of-two buckets and no exact-N tuning key. Buckets share a selected configuration, and initial autotuning compiles multiple candidates. Backward, default chunk128, FP32 centered statistics, ordinary output stores, and the API's first-order/nondeterministic-gradient contract remain as documented in the preceding report. Forward bias/mask alignment padding is distinct from backward bias/dBias bucket-width storage; Q/K/V are not padded to bucket size.

## Evidence and rejected alternatives

[Archived diagnostics](benchmarks/evidence/forward_tuning/README.md) retain completed logs and command status. A 40-configuration sweep did not find a stable improvement without the register-cap candidate. Unified-loop and tail-first variants regressed and were rejected; skipping no-op padding alone showed at most about 0.5% and no material consistent gain. Nsight Compute failed with performance-counter permission error `ERR_NVGPUCTRPERM`; no counter-based bottleneck claim is made.

The capped candidate repeatedly improved the internal forward wrapper by 1.014–1.042× on the tested H20 BF16 B1/H8/D32 shapes. Preliminary f21 public-API comparisons showed smaller 1.78–2.72% forward speed gains and 0.28–0.53% directly measured F+B gains. These ranges must not be substituted for final `61da9e0` results. Without counters, a specific occupancy explanation remains unproven.

Compiler metadata records 96 registers, 14 spills, and 28,856 bytes of shared memory for the capped candidate, versus 111 registers, zero spills, and 45,408 shared bytes for the uncapped three-stage baseline. The candidate is retained because measurements improve despite spills; hardware counters have not established an occupancy explanation.

## Final validation

Final-source [eager proof](benchmarks/evidence/forward_tuning/proof-eager.log) and [compiled proof](benchmarks/evidence/forward_tuning/proof-compiled.log) both exited 0: each covered four buckets, 17 target calls, and 50 target PTX compilations, with zero new JIT keys/PTX compilations when reusing a previously seen bucket. Compiled execution used one graph for N>1 and two after the first N=1; repeating N=1 added none. Graph-break records were empty.

[memcheck](benchmarks/evidence/forward_tuning/memcheck.log) and [initcheck](benchmarks/evidence/forward_tuning/initcheck.log) both exited 0 with zero errors. Each covered two N65 BF16 cases, mixed/all-mask, forcing cap96. This is sanitizer coverage of those cases, not every input combination.

[Final forced-configuration correctness](benchmarks/evidence/forward_tuning/forced-correctness-final.log) passed 42/42, exit 0: BF16/FP16, D32 default N1/65/128/129 with mixed/all/singleton/sentinel masks, full N65/129 mixed/all, plus one D128 N65 mixed smoke case per dtype. Output and four gradients are checked against independent FP64. D32 forces cap96; D128 checks the unaffected path. [This round’s TMA tests](benchmarks/evidence/forward_tuning/pytest-tma.log) passed 16 cases including misaligned bias; the [previous-version comparison](benchmarks/evidence/forward_tuning/compare-previous-final.log) completed with exit 0. This source was checked with the 42+16 tests above plus proof/sanitizer runs. The earlier `4b832de` 386 eager and 18 compile passes are [historical validation](forward_recovery.md), not suites rerun in this round. Unsuffixed `forced-correctness.log` and `compare-previous.log` belong to f21; only the final-source runs establish this version's results.

## Public API: previous fused versus current

| N | Previous → current forward (ms) | Forward speed ratio | Previous → current F+B (ms) | F+B speed ratio |
|---:|---:|---:|---:|---:|
| 500 | 1.591 → 1.549 | 1.0274× | 8.079 → 8.041 | 1.0047× |
| 512 | 1.444 → 1.414 | 1.0212× | 7.010 → 6.974 | 1.0052× |
| 513 | 1.999 → 1.949 | 1.0255× | 9.671 → 9.622 | 1.0051× |
| 640 | 2.728 → 2.677 | 1.0192× | 13.327 → 13.270 | 1.0043× |
| 768 | 4.620 → 4.529 | 1.0202× | 22.534 → 22.444 | 1.0040× |
| 800 | 5.888 → 5.753 | 1.0233× | 29.546 → 29.431 | 1.0039× |
| 1024 | 10.639 → 10.463 | 1.0168× | 51.563 → 51.384 | 1.0035× |

The comparison holds fused backward constant and swaps only the previous `4b832de` versus current forward. The [final same-process comparison](benchmarks/evidence/forward_tuning/compare-previous-final.log) uses a private tuning cache and shape order 500/512/513/640/768/800/1024: 500/513 first trigger their buckets, reused thereafter. Old and new forwards tune separately; new forward selects cap96. Backward is unchanged, with measured backward-only fluctuations close to 1×. Both sides use three warmups, three ABBA rounds, and 60 samples per side.

## Public API: upstream versus current

| N | Upstream → current forward (ms) | Upstream → current backward (ms) | Forward speed ratio | Upstream → current F+B (ms) | F+B speed ratio |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.709 → 1.547 | 12.662 → 6.494 | 1.105× | 14.370 → 8.037 | 1.788× |
| 512 | 1.490 → 1.413 | 7.556 → 5.557 | 1.054× | 9.053 → 6.977 | 1.298× |
| 513 | 2.129 → 1.945 | 15.625 → 7.675 | 1.095× | 17.757 → 9.622 | 1.846× |
| 640 | 2.820 → 2.676 | 14.377 → 10.593 | 1.054× | 17.198 → 13.276 | 1.295× |
| 768 | 4.734 → 4.537 | 24.498 → 17.885 | 1.043× | 29.229 → 22.438 | 1.303× |
| 800 | 5.942 → 5.802 | 28.994 → 23.691 | 1.024× | 34.903 → 29.426 | 1.186× |
| 1024 | 10.811 → 10.490 | 56.705 → 40.910 | 1.031× | 67.422 → 51.276 | 1.315× |

Measurements use warm eager public APIs, ABBA sampling, and ratios of median CUDA-event times, including wrapper allocations, padding, casts, and host submission gaps within the interval. Forward enables gradients, backward-only uses retain_graph, and F+B is directly measured, not a sum of separate columns. They are not isolated-kernel, compiled, or complete training-step performance. Environment: H20-3e, PyTorch 2.13.0a0+9186a08b2c.nv26.07, Triton 3.7.1, BF16 B1 H8 D32, default chunk128, contiguous dO, mask probability 0.2. Final logs retain source hashes and complete raw samples.


[Final upstream comparison](benchmarks/evidence/forward_tuning/bench-upstream.log) exited 0: H20-3e, BF16 B1 H8 D32, chunk128, contiguous dO, mask probability 0.2; three warmups, three ABBA rounds, 60 samples per side. Shape order was 500/512/513/640/768/800/1024. In the new v4 forward namespace, 500/513 first triggered the two buckets and selected cap96; upstream/backward inherited persistent configurations, see [cache snapshots](benchmarks/evidence/forward_tuning/tuning-configs/). Forward speed ratios are 1.024–1.105×; directly measured F+B ratios are 1.186–1.846×.

This GPU allocation lasted 33 minutes 06 seconds, including compilation, host work and waiting; it is not GPU busy time. See [source/run provenance](benchmarks/evidence/forward_tuning/SOURCE_AND_RUN.json) and the [evidence hash manifest](benchmarks/evidence/forward_tuning/SHA256SUMS.json). The full/chunk memory comparison was not rerun in this forward-only update; see the [previous report](forward_recovery.md), including the distinction that low-memory is relative to full fused workspace, not upstream.
