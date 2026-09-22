# Forward tuning evidence

Diagnostic logs and their command sidecars are retained together, including failed Nsight Compute profiling (`ERR_NVGPUCTRPERM`, exit 1). Scripts, generated candidates, prior source snapshots, and configuration inputs allow review of the experiment setup. Installation logs, Python caches, and mutable tuning-cache directories are excluded.

`compare-previous.log` and `forced-correctness.log` describe preliminary f21 source, not final `61da9e0` validation. The former began before committing, so its revision field is older; recorded source hashes identify the tested code. Final `61da9e0` logs are archived here: proof-eager/compiled, memcheck/initcheck, bench-upstream, forced-correctness-final, pytest-tma, and compare-previous-final, together with their command sidecars. The archive manifest records hashes; tuning-configs contains the relevant final cache snapshots.

Historical paths beginning `/workspace/` refer to the GPU container mount of the workspace. Temporary `/tmp/` directories and historical absolute paths are provenance, not links expected to exist in a checkout. Reproduction requires adapting script paths to the local checkout and providing compatible CUDA/PyTorch/Triton and GPU resources. Relative links in the reports resolve to this archive.
