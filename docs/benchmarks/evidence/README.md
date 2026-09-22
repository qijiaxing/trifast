# Selected H20 evidence (2026-09-21)

- `benchmark_summary.json`: environment, arguments, diagnostic parity, separately measured memory, and all timing medians.
- `benchmark_samples.jsonl.gz`: complete benchmark JSONL including every timed sample; each record includes its run name. Decompress with `gzip -dc benchmark_samples.jsonl.gz`.
- `validation_summary.json`: final successful commands, actual exit status and command wall time, pytest/sanitizer summaries, and independent FP64 numerical results. Wall time is not GPU busy time.
- `source_manifest.json`: exact source hashes and base commit. Measurements used a working tree on `b4ecec4`; the source hashes, rather than the uncommitted revision alone, identify the candidate.

The performance report explains boundaries and limitations. All GPU commands ran serially on one NVIDIA H20-3e. Compilation/autotuning precedes timing. These are measurements of triangle attention, not a full model or distributed communication.

Two test-harness issues were corrected during preparation; their failed runs are not counted as successful gates:

1. The initial benchmark treated a zero in the rounded BF16 upstream result as an exact reference zero. At N800, this incorrectly rejected small nonzero candidate values. The benchmark now reports baseline-zero differences diagnostically and checks relative L2/finite parity. The original, full-workspace, and default low-memory APIs were separately checked against independent FP64 with the exact N800 benchmark inputs. The independent reference tolerances and true-reference-zero absolute checks were unchanged.
2. The initial parametrized compile test exhausted Dynamo's process-wide recompilation limit across different configurations (14 passed, 4 failed). Each independent configuration now resets Dynamo before creating its compiled callable. Its two calls still reuse that callable without resetting. Both warm and fresh-autotuning-cache final runs are recorded. This does not establish unlimited dynamic-shape compilation support.

Final evidence was regenerated after lint cleanup (imports, explicit benchmark helper arguments, dictionary literals and a redundant integer cast); the Triton JIT kernel bodies were unchanged.

The full exploratory archive, earlier failures, cluster commands, and caches remain in the development workspace; this directory contains selected reproducible final evidence.
