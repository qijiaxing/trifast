"""CPU-configured, GPU-executed TMA forward sweep with independent warm retests.

Never writes project tuner caches. Initial ranking is exploratory; only the
interleaved ABBA retests should be used for performance conclusions. Every call
includes the complete forward wrapper. Run separately with --shapes
513,640,768,800,1024 to check within-bucket robustness.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import tempfile


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes', default='512,800,1024')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--configs-json', help='Optional JSON list of config_to_dict objects; sweep only these')
    args = parser.parse_args()
    shapes = [int(s) for s in args.shapes.split(',')]
    if min(*shapes, args.samples, args.warmup, args.rounds, args.top) < 1:
        parser.error('counts must be positive')
    private = Path(tempfile.mkdtemp(prefix='forward-sweep-'))
    os.environ['XDG_CONFIG_HOME'] = str(private / 'config')
    os.environ.pop('TRIFAST_FORCE_TUNE', None)
    import torch
    import triton
    module = importlib.import_module('trifast._fused_forward_tma')
    from trifast.autotune_helpers import config_to_dict, dict_to_config, _fwd_descriptor_pre_hook

    tuner = module._fused_tma
    original_configs = list(tuner.configs)
    tuner.cache_file = None
    module._fused_tma_pointer.cache_file = None
    # Observe the actual returned compiled kernel without modifying its source.
    original_run = tuner.fn.run
    last_kernel = [None]

    def capture_run(*launch_args, **launch_kwargs):
        result = original_run(*launch_args, **launch_kwargs)
        last_kernel[0] = result
        return result

    tuner.fn.run = capture_run

    def metadata():
        kernel = last_kernel[0]
        if kernel is None:
            return {'available': False}
        result = {name: getattr(kernel, name, None) for name in ('n_regs', 'n_spills')}
        meta = getattr(kernel, 'metadata', None)
        for name in ('shared', 'num_warps', 'num_stages', 'num_ctas', 'maxnreg'):
            result[name] = getattr(meta, name, None)
        result['available'] = True
        return result

    configs = []
    if args.configs_json:
        configs = [dict_to_config(c) for c in json.loads(Path(args.configs_json).read_text())]
        for config in configs:
            config.pre_hook = _fwd_descriptor_pre_hook
    else:
        # Forty candidates. Keep tiny M tiles at four warps and avoid the
        # largest MxK tile with four warps or deep shared-memory staging.
        for bj in (32, 64, 128):
            for bk in (32, 64, 128):
                for warps in (4, 8):
                    for stages in (1, 2, 3, 4):
                        allowed = (
                            (bj == 32 and bk <= 64 and warps == 4)
                            or (bj == 64 and warps == 4)
                            or (bj == 64 and warps == 8 and stages in (2, 3))
                            or (bj == 128 and bk <= 64 and stages <= 3)
                            or (bj == 128 and bk == 128 and warps == 8 and stages <= 2)
                        )
                        if allowed:
                            configs.append(triton.Config(
                                {'BLOCK_J': bj, 'BLOCK_K': bk}, num_warps=warps,
                                num_stages=stages, pre_hook=_fwd_descriptor_pre_hook))
    files = [Path(__file__), Path(module.__file__),
             Path(importlib.import_module('trifast.autotune').__file__),
             Path(importlib.import_module('trifast.autotune_helpers').__file__)]
    emit(event='environment', args=vars(args), torch=torch.__version__,
         triton=triton.__version__, gpu=torch.cuda.get_device_name(),
         candidate_count=len(configs), private_config_directory=str(private),
         source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         scope='B1 H8 D32 BF16; full forward wrapper; warm CUDA events; no graphs',
         baseline_configs=[config_to_dict(c) for c in original_configs])

    def select(config):
        tuner.configs = [config]
        tuner.cache.clear()

    def samples(fn, count):
        events = []
        for _ in range(count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            del output
            events.append((start, end))
        torch.cuda.synchronize()
        return [start.elapsed_time(end) for start, end in events]

    try:
        for n in shapes:
            torch.manual_seed(734 + n)
            shape = (1, 8, n, n, 32)
            values = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
            values.append(torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16))
            mask = torch.rand((1, n, n), device='cuda') < .2
            fn = lambda: module.fused_forward_tma(*values, mask)
            # Tune the original candidate set independently at each N for a
            # strong baseline, rather than inheriting a preceding shape's winner.
            tuner.configs = original_configs
            tuner.cache.clear()
            reference = fn()[0].float()
            torch.cuda.synchronize()
            baseline = tuner.best_config
            emit(event='baseline', n=n, config=config_to_dict(baseline), metadata=metadata())
            order = list(configs)
            random.Random(811 + n).shuffle(order)
            ranking = []
            for index, config in enumerate(order):
                select(config)
                try:
                    output = fn()[0].float()
                    torch.cuda.synchronize()
                except Exception as exc:
                    # Compilation/resource errors are recoverable. A device
                    # illegal access is not: don't continue in a poisoned context.
                    category = type(exc).__name__
                    emit(event='candidate_failure', n=n, index=index,
                         config=config_to_dict(config), error_type=category, error=str(exc))
                    text = str(exc).lower()
                    if any(s in text for s in ('illegal memory', 'device-side assert', 'misaligned address')):
                        raise
                    continue
                relative = ((output - reference).norm() / reference.norm().clamp_min(1e-12)).item()
                passed = bool(torch.isfinite(output).all()) and relative <= .012
                del output
                if not passed:
                    emit(event='candidate_failure', n=n, config=config_to_dict(config),
                         error_type='numerical', relative_l2=relative)
                    continue
                kernel_metadata = metadata()
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                times = samples(fn, args.samples)
                median = statistics.median(times)
                ranking.append((median, config))
                emit(event='sweep', n=n, index=index, config=config_to_dict(config),
                     relative_l2=relative, metadata=kernel_metadata,
                     samples_ms=times, median_ms=median)
            ranking.sort(key=lambda pair: pair[0])
            if not ranking:
                raise RuntimeError(f'no valid candidates at N={n}')
            for rank, (_, candidate) in enumerate(ranking[:args.top], 1):
                pair = {'A': baseline, 'B': candidate}
                for config in pair.values():
                    select(config)
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                recorded = {'A': [], 'B': []}
                for round_index in range(args.rounds):
                    for block, label in enumerate('ABBA'):
                        select(pair[label])
                        times = samples(fn, args.samples)
                        recorded[label].extend(times)
                        emit(event='retest_samples', n=n, rank=rank, round=round_index,
                             block=block, label=label, config=config_to_dict(pair[label]),
                             samples_ms=times)
                medians = {key: statistics.median(ts) for key, ts in recorded.items()}
                emit(event='retest_summary', n=n, rank=rank,
                     baseline=config_to_dict(baseline), candidate=config_to_dict(candidate),
                     median_ms=medians, speedup=medians['A'] / medians['B'],
                     samples_per_label=len(recorded['A']))
            emit(event='shape_complete', n=n)
            del values, mask, reference, fn
            torch.cuda.empty_cache()
        emit(event='complete')
    finally:
        tuner.fn.run = original_run
        tuner.configs = original_configs


if __name__ == '__main__':
    main()
