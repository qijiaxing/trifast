# H20 fused attention measurements

These measurements compare the opt-in fused implementation against upstream master **b4ecec4c8ac599bf5aa16ae5aeb5ebdc02b7addb**. Low-memory backward with `chunk_i=128` is the primary reported configuration. The original `triangle_attention` API remains available; these tables do not imply changing its dispatch.

For N512–1024, low-memory backward is **1.528–1.640×** faster and complete forward+backward is **1.385–1.494×** faster. Forward is **1.022–1.050×** faster on aligned shapes and **0.943–0.944× at N800**: approximately 6% slower there. The forward regression is included in the end-to-end results.

## Measurement boundary

NVIDIA H20-3e, compute capability 9.0; PyTorch `2.13.0a0+9186a08b2c.nv26.07`; Triton `3.7.1`. B=1, H=8, D=32, BF16. Inputs use a seeded random mask with 20% replacement probability and finite sentinel −10000. These are synthetic dense attention cases, not all-masked-row shortcut benchmarks. dO is either contiguous or transposed over the two spatial axes.

Each pair runs in one process with three ABBA rounds, ten CUDA-event samples per block, and three untimed warmup calls per implementation: 60 samples per label and mode. Reported time is the median; speedup is baseline median divided by candidate median. Forward includes its wrapper and allocations. Backward reuses a constructed graph (`retain_graph=True`); F+B rebuilds it every call. Gradient clearing, autograd dispatch, helper kernels, workspace initialization, copies and casts are included. There are no CUDA graphs. CUDA-event timing includes device-visible submission gaps, but is not application wall-clock latency. Compilation/autotuning is warmed out. Separate F/B medians need not sum to the independently measured F+B median.

## Primary: low-memory versus master

Each cell is **master → candidate milliseconds (speedup)**.

| dO | N | Forward ms (×) | Backward ms (×) | F+B ms (×) |
|---|---:|---:|---:|---:|
| contiguous | 512 | 1.494 → 1.424 (1.049×) | 7.570 → 4.801 (1.577×) | 9.067 → 6.219 (1.458×) |
| contiguous | 640 | 2.791 → 2.721 (1.025×) | 14.386 → 9.129 (1.576×) | 17.165 → 11.853 (1.448×) |
| contiguous | 768 | 4.691 → 4.591 (1.022×) | 24.293 → 15.197 (1.599×) | 28.964 → 19.744 (1.467×) |
| contiguous | 800 | 5.813 → 6.162 (0.943×) | 29.040 → 19.004 (1.528×) | 34.854 → 25.170 (1.385×) |
| contiguous | 1024 | 10.769 → 10.498 (1.026×) | 56.705 → 34.575 (1.640×) | 67.395 → 45.102 (1.494×) |
| transposed | 512 | 1.496 → 1.424 (1.050×) | 7.610 → 4.899 (1.553×) | 9.108 → 6.319 (1.441×) |
| transposed | 640 | 2.792 → 2.722 (1.026×) | 14.456 → 9.190 (1.573×) | 17.241 → 11.820 (1.459×) |
| transposed | 768 | 4.658 → 4.558 (1.022×) | 24.444 → 15.356 (1.592×) | 29.100 → 19.925 (1.460×) |
| transposed | 800 | 5.815 → 6.161 (0.944×) | 29.214 → 19.077 (1.531×) | 35.026 → 25.242 (1.388×) |
| transposed | 1024 | 10.771 → 10.499 (1.026×) | 56.967 → 35.207 (1.618×) | 67.655 → 45.713 (1.480×) |

## Larger shapes: low-memory versus master

Contiguous dO, same BF16 B=1/H=8/D=32 and chunk128 settings. Only complete F+B was measured in this run; no standalone F/B or transposed-layout claim is made for these shapes.

| N | Master F+B ms | Low-memory F+B ms | Speedup | Master peak MiB | Low-memory peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1536 | 229.118 | 151.339 | 1.514× | 5078.25 | 5125.55 |
| 2048 | 532.966 | 348.557 | 1.529× | 9028.00 | 9026.06 |

## Auxiliary: full workspace versus master

Same reporting boundary, separately paired measurements.

| dO | N | Forward ms (×) | Backward ms (×) | F+B ms (×) |
|---|---:|---:|---:|---:|
| contiguous | 512 | 1.505 → 1.424 (1.057×) | 7.557 → 4.754 (1.589×) | 9.054 → 6.179 (1.465×) |
| contiguous | 640 | 2.792 → 2.719 (1.027×) | 14.366 → 8.953 (1.605×) | 17.165 → 11.683 (1.469×) |
| contiguous | 768 | 4.704 → 4.592 (1.024×) | 24.505 → 15.113 (1.621×) | 29.190 → 19.708 (1.481×) |
| contiguous | 800 | 5.870 → 6.208 (0.946×) | 29.266 → 18.956 (1.544×) | 35.123 → 25.163 (1.396×) |
| contiguous | 1024 | 10.770 → 10.578 (1.018×) | 56.709 → 34.558 (1.641×) | 67.387 → 45.057 (1.496×) |
| transposed | 512 | 1.493 → 1.423 (1.049×) | 7.614 → 4.806 (1.584×) | 9.110 → 6.234 (1.461×) |
| transposed | 640 | 2.792 → 2.722 (1.026×) | 14.343 → 8.952 (1.602×) | 17.109 → 11.651 (1.468×) |
| transposed | 768 | 4.658 → 4.567 (1.020×) | 24.443 → 15.085 (1.620×) | 29.097 → 19.630 (1.482×) |
| transposed | 800 | 5.814 → 6.172 (0.942×) | 29.206 → 18.869 (1.548×) | 35.025 → 25.042 (1.399×) |
| transposed | 1024 | 10.769 → 10.511 (1.025×) | 56.973 → 34.785 (1.638×) | 67.658 → 45.286 (1.494×) |

## Paired workspace tradeoff

The following uses direct full-workspace versus low-memory runs, not subtraction across unrelated benchmark sessions.

| dO | N | Full → low-memory F+B ms | Low-memory time increase | Peak full → low-memory MiB | Peak reduction |
|---|---:|---:|---:|---:|---:|
| contiguous | 512 | 6.181 → 6.221 | 0.64% | 804.52 → 612.52 | 23.87% |
| contiguous | 800 | 24.907 → 25.151 | 0.98% | 1967.04 → 1441.04 | 26.74% |
| contiguous | 1024 | 44.988 → 45.115 | 0.28% | 3217.03 → 2321.03 | 27.85% |
| transposed | 512 | 6.227 → 6.316 | 1.43% | 804.52 → 612.52 | 23.87% |
| transposed | 800 | 25.163 → 25.375 | 0.84% | 1967.04 → 1441.04 | 26.74% |
| transposed | 1024 | 45.315 → 45.740 | 0.94% | 3217.03 → 2321.03 | 27.85% |

Peak memory is the increment in PyTorch allocated bytes above preallocated inputs. It includes the retained output and four gradients; it excludes allocator-reserved memory and is not total device/process memory. MiB means 2²⁰ bytes. Both candidates retain identical output bytes. Saving memory versus the full-workspace candidate does **not** mean using less memory than master:

| N | Master peak MiB | Low-memory peak MiB |
|---:|---:|---:|
| 512 | 564.25 | 612.52 |
| 640 | 881.64 | 936.89 |
| 768 | 1269.56 | 1329.77 |
| 800 | 1380.84 | 1441.04 |
| 1024 | 2257.00 | 2321.03 |

## Evidence, reproduction and limits

Final gates passed on the recorded stack: 374 eager/input-contract cases; 18 compile configurations in each of warm and fresh-autotuning-cache runs (two calls per configuration); memcheck and initcheck each with 5 passing cases and 0 errors; 7 selected upstream regressions; and 13 independent FP64 large/target-shape cases. The upstream selection is not the complete upstream suite. Compile configurations reset Dynamo between tests; this is not an unlimited-specialization claim.

Reviewable artifacts: [benchmark summary](evidence/benchmark_summary.json), [all timing samples](evidence/benchmark_samples.jsonl.gz), [validation summary](evidence/validation_summary.json), [source manifest](evidence/source_manifest.json), and [artifact guide](evidence/README.md).

Run identifiers within these artifacts: `v4-benchmark-master-{low-memory,full-workspace}-{contiguous,transposed}.log`, `v4-benchmark-lowmemory-{contiguous,transposed}.log`, and `v4-benchmark-master-large.log`. Each contains environment information, repository-relative source SHA256 values, per-sample timings, parity diagnostics and a terminal completion record. The master revision identifies the base checkout; source hashes identify the added candidate files. Benchmark parity against master is not an independent correctness proof; use the separate FP64 validation script and test evidence.

Run from the repository root in the recorded CUDA/PyTorch/Triton environment:

```bash
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate low-memory --chunk-i 128 --shapes 512,640,768,800,1024 --modes forward,backward,forward_backward --rounds 3 --samples 10 --warmup 3
# Repeat with --noncontiguous-do for the transposed gradient layout.
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate low-memory --chunk-i 128 --shapes 1536,2048 --modes forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate full-workspace --shapes 512,640,768,800,1024 --modes forward,backward,forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/bench_fused.py --baseline full-workspace --candidate low-memory --chunk-i 128 --shapes 512,800,1024 --modes forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/validate_fused.py --api low-memory --chunk-i 128 --shapes 512,800,1024 --dim 32 --dtype bf16 --reference-chunk 8
```

Results cover this GPU, dtype, dimensions and warmed workload only. They do not establish gains on other GPUs, D values, full models or distributed training. FP32 atomic reductions are nondeterministic and the API supports first derivatives only. This work shares local GPU computation and reduces intermediate traffic; no NCCL or inter-GPU communication result is reported. No hardware speed-of-light claim follows from these timings.
