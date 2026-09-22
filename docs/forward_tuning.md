# Fused triangle attention

This change adds `triangle_attention_fused`, with fused backward and a tuned TMA forward. Compared with TriFast’s mainline (`master`, `b4ecec4`), measured forward speed is **1.024–1.105×** and forward + backward speed is **1.186–1.846×** on the configurations below.

## Changes

- Share backward probability and score-gradient computation across dQ/dK/dV/dBias, without materializing the full probability tensor.
- Default to `chunk_i=128` to bound FP32 dQ scratch; `chunk_i=None` selects full workspace. This is a fixed default, not automatic selection by available memory.
- Retain TMA input reads for BF16/FP16, use ordinary vector output stores, add a D32 tuning candidate, and skip redundant padding copies while preserving TMA alignment.
- Keep actual N as a runtime parameter and use the same power-of-two tuning buckets as mainline. Q/K/V retain their actual shape. Initial bucket tuning may compile multiple candidates; subsequent lengths reuse the selected configuration.

The existing `triangle_attention` API remains unchanged. The fused API supports first-order gradients; FP32 atomic reductions make backward nondeterministic. Real masked keys preserve finite −10000 semantics; padded keys are excluded from normalization.

## Performance against TriFast mainline

| N | Upstream → current forward (ms) | Upstream → current backward (ms) | Forward speed ratio | Upstream → current F+B (ms) | F+B speed ratio |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.709 → 1.547 | 12.662 → 6.494 | 1.105× | 14.370 → 8.037 | 1.788× |
| 512 | 1.490 → 1.413 | 7.556 → 5.557 | 1.054× | 9.053 → 6.977 | 1.298× |
| 513 | 2.129 → 1.945 | 15.625 → 7.675 | 1.095× | 17.757 → 9.622 | 1.846× |
| 640 | 2.820 → 2.676 | 14.377 → 10.593 | 1.054× | 17.198 → 13.276 | 1.295× |
| 768 | 4.734 → 4.537 | 24.498 → 17.885 | 1.043× | 29.229 → 22.438 | 1.303× |
| 800 | 5.942 → 5.802 | 28.994 → 23.691 | 1.024× | 34.903 → 29.426 | 1.186× |
| 1024 | 10.811 → 10.490 | 56.705 → 40.910 | 1.031× | 67.422 → 51.276 | 1.315× |

H20-3e, BF16, B1/H8/D32, default chunk128, contiguous dO. Warm eager public API, three warmups and three ABBA rounds, 60 samples per side; times are medians. Forward includes saved statistics, backward-only uses `retain_graph`, and F+B is directly measured. Results include wrapper costs and do not represent a complete training step or untested hardware/shapes.

Lengths run in table order. N500/513 first select the two forward buckets; later lengths reuse those selections. Mainline and backward use warmed persistent configurations. [Raw samples](benchmarks/evidence/forward_tuning/bench-upstream.log) · [Tuning configurations](benchmarks/evidence/forward_tuning/tuning-configs/).

“Low-memory” is relative to full fused workspace, not mainline; the default fused implementation can use more memory than mainline.

## Validation

Validated implementation: `61da9e0`.

- [42 independent FP64 checks](benchmarks/evidence/forward_tuning/forced-correctness-final.log) for output and all four gradients, covering BF16/FP16, masks, length tails and full/chunked workspace; D32 forces the new configuration, with D128 smoke coverage.
- [16 TMA regression tests](benchmarks/evidence/forward_tuning/pytest-tma.log), including contiguous bias views with misaligned storage offsets.
- [Eager](benchmarks/evidence/forward_tuning/proof-eager.log) and [torch.compile](benchmarks/evidence/forward_tuning/proof-compiled.log): four buckets, 17 calls each, no new JIT/PTX compilation in reused buckets. Dynamo uses one graph for N>1 and another for N=1.
- [memcheck](benchmarks/evidence/forward_tuning/memcheck.log) and [initcheck](benchmarks/evidence/forward_tuning/initcheck.log): zero errors on two forced-configuration N65 BF16 cases each.

[Source and environment](benchmarks/evidence/forward_tuning/SOURCE_AND_RUN.json) · [Evidence checksums](benchmarks/evidence/forward_tuning/SHA256SUMS.json).
