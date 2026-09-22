"""Forward causal ablation; complete eager wrappers, warm CUDA events, no graphs.

The exact-N variant uses the runtime variant's chosen tile/stages. Its only
kernel change is constexpr N (plus equivalent i64 cast syntax). Thus this pair
isolates specialization without changing layouts, math, or tuning candidates.
Upstream TMA-vs-pointer compares optimized routes, including their own tuners;
it is not a claim about a single instruction's isolated cost.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile


def emit(**data):
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes', default='512,800,1024')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    shapes = [int(x) for x in args.shapes.split(',')]
    if min(*shapes, args.rounds, args.samples, args.warmup, args.heads) < 1:
        parser.error('all counts must be positive')
    # Fresh autotuning avoids stale config files, but compile cache can be shared.
    scratch = Path(tempfile.mkdtemp(prefix='forward-ablation-'))
    os.environ['XDG_CONFIG_HOME'] = str(scratch / 'config')
    os.environ.pop('TRIFAST_FORCE_TUNE', None)
    import torch
    import triton
    upstream = importlib.import_module('trifast.torch')
    upstream_kernels = importlib.import_module('trifast.triton')
    from trifast.autotune_helpers import config_to_dict

    source_path = Path(__file__).with_name('frozen_pointer_forward.py')
    source = source_path.read_text()
    if source.count('N: tl.int64') != 1:
        raise RuntimeError('unexpected frozen source, refuse uncontrolled ablation')

    def module_for(label, exact):
        text = source.replace('trifast::fused_attention_forward_padded_bucket',
                              f'trifast::ablation_forward_{label}')
        text = text.replace('cache_name="padded_bucket_v1"',
                            f'cache_name="ablation_{label}"')
        if exact:
            text = text.replace('@triton.jit(do_not_specialize=["N"])', '@triton.jit')
            text = text.replace('N: tl.int64', 'N: tl.constexpr')
            text = text.replace('N.to(tl.int64)', 'tl.cast(N, tl.int64)')
        path = scratch / f'{label}.py'
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(f'ablate_{label}', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, path

    runtime, runtime_path = module_for('runtime', False)
    exact, exact_path = module_for('exact', True)
    paths = [source_path, runtime_path, exact_path, Path(__file__),
             Path(upstream.__file__), Path(upstream_kernels.__file__)]
    emit(event='environment', args=vars(args), torch=torch.__version__,
         triton=triton.__version__, gpu=torch.cuda.get_device_name(),
         sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         generated_source_directory=str(scratch),
         scope='B=1, BF16, D=32; forward returns O/LSE/mx/dn; full wrapper costs',
         timing='warm interleaved ABBA CUDA events; per pair; no CUDA graphs',
         exact_control='exact N uses runtime winner; no independent retuning')

    def upstream_call(values, mask, tma):
        upstream.USE_TMA = upstream.USE_TMA_BIAS = upstream.USE_TMA_MASK = tma
        return upstream._triangle_attention(*values, mask)

    for n in shapes:
        torch.manual_seed(734 + n)
        shape = (1, args.heads, n, n, 32)
        values = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        values.append(torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16))
        mask = torch.rand((1, n, n), device='cuda') < .2
        functions = {
            'upstream_tma': lambda: upstream_call(values, mask, True),
            'upstream_pointer': lambda: upstream_call(values, mask, False),
            'ours_runtime': lambda: runtime.fused_forward_optimized(*values, mask),
            'ours_exact_matched': lambda: exact.fused_forward_optimized(*values, mask),
        }
        # Tune runtime first, then hold the exact-N variant to that same config.
        runtime.fused_forward_optimized(*values, mask)
        winner = runtime._fwd_fused_optimized.best_config
        exact._fwd_fused_optimized.configs = [winner]
        exact._fwd_fused_optimized.cache.clear()
        outputs = {name: fn() for name, fn in functions.items()}
        reference = outputs['upstream_tma'][0].float()
        for name, output in outputs.items():
            result = output[0].float()
            rel = ((result - reference).norm() / reference.norm().clamp_min(1e-12)).item()
            passed = bool(torch.isfinite(result).all()) and rel <= .012
            emit(event='parity', n=n, label=name, relative_l2=rel,
                 passed=passed, independent_reference=False)
            if not passed:
                raise RuntimeError(f'forward diagnostic parity failed: {name}, N={n}')
        del outputs, reference, result, output
        emit(event='configs', n=n, configs={
            'upstream_tma': config_to_dict(upstream_kernels._fwd.best_config),
            'upstream_pointer': config_to_dict(upstream_kernels._fwd_pointer.best_config),
            'ours_runtime': config_to_dict(winner),
            'ours_exact_matched': config_to_dict(exact._fwd_fused_optimized.best_config),
        })
        pairs = [('upstream_tma', 'upstream_pointer'),
                 ('upstream_tma', 'ours_runtime'),
                 ('ours_runtime', 'ours_exact_matched')]
        for a, b in pairs:
            selected = {'A': functions[a], 'B': functions[b]}
            for fn in selected.values():
                for _ in range(args.warmup):
                    fn()
            torch.cuda.synchronize()
            samples = {'A': [], 'B': []}
            for round_index in range(args.rounds):
                for block, label in enumerate('ABBA'):
                    events = []
                    for _ in range(args.samples):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        result = selected[label]()
                        end.record()
                        del result
                        events.append((start, end))
                    torch.cuda.synchronize()
                    for index, (start, end) in enumerate(events):
                        ms = start.elapsed_time(end)
                        samples[label].append(ms)
                        emit(event='sample', n=n, pair=[a, b], label=label,
                             round=round_index, block=block, sample=index, ms=ms)
            medians = {label: statistics.median(times) for label, times in samples.items()}
            emit(event='summary', n=n, pair=[a, b],
                 median_ms={a: medians['A'], b: medians['B']},
                 a_over_b=medians['A'] / medians['B'],
                 samples_per_label=len(samples['A']))
        del values, mask, functions, selected
        torch.cuda.empty_cache()
    upstream.USE_TMA = upstream.USE_TMA_BIAS = upstream.USE_TMA_MASK = True
    emit(event='complete')


if __name__ == '__main__':
    main()
