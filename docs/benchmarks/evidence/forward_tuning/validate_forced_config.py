"""Independent FP64 O/dQ/dK/dV/dBias gate for the new D32 TMA config.

--sanitizer limits to N65 BF16 mixed/all with a CPU reference, suitable for
compute-sanitizer memcheck/initcheck. No project persistent caches are written.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def emit(**row):
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitizer', action='store_true')
    args = parser.parse_args()
    private = Path(tempfile.mkdtemp(prefix='forced-tma-config-'))
    os.environ['XDG_CONFIG_HOME'] = str(private / 'config')
    os.environ.pop('TRIFAST_FORCE_TUNE', None)
    import torch
    import triton
    import trifast
    root = Path(inspect.getfile(trifast)).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tests.unit.test_fused import entry, make_case, reference, run, verify
    from trifast.autotune_helpers import config_to_dict, _fwd_descriptor_pre_hook
    module = importlib.import_module('trifast._fused_forward_tma')
    tuner = module._fused_tma
    original_configs = list(tuner.configs)
    tuner.cache_file = None
    module._fused_tma_pointer.cache_file = None
    forced = triton.Config({'BLOCK_J': 64, 'BLOCK_K': 64}, num_warps=4,
                           num_stages=2, maxnreg=96,
                           pre_hook=_fwd_descriptor_pre_hook)
    # Record actual launch arguments; a stale best_config alone is insufficient
    # evidence that this case reached the intended forward kernel.
    original_run = tuner.fn.run
    launches = []

    def capture_run(*values, **kwargs):
        result = original_run(*values, **kwargs)
        launches.append({key: kwargs.get(key) for key in
                         ('DIM', 'BLOCK_J', 'BLOCK_K', 'num_warps',
                          'num_stages', 'maxnreg', 'USE_TMA', 'USE_TMA_BIAS', 'USE_TMA_MASK')})
        return result

    tuner.fn.run = capture_run
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    files = list((root / 'src/trifast').glob('*.py')) + [
        root / 'tests/unit/test_fused.py', root / 'tests/fused_reference.py', Path(__file__)]
    emit(event='environment', args=vars(args), torch=torch.__version__,
         triton=triton.__version__, gpu=torch.cuda.get_device_name(),
         forced_config=config_to_dict(forced), private_cache=str(private),
         source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    cases = []
    if args.sanitizer:
        cases = [('default', torch.bfloat16, 32, 65, mode) for mode in ('mixed', 'all')]
    else:
        for dtype in (torch.bfloat16, torch.float16):
            cases.extend(('default', dtype, 32, n, mode)
                         for n in (1, 65, 128, 129)
                         for mode in ('mixed', 'all', 'singleton', 'sentinel'))
            cases.extend(('full', dtype, 32, n, mode)
                         for n in (65, 129) for mode in ('mixed', 'all'))
            # The new register cap is D32 only; smoke the unaffected wide-D path.
            cases.append(('default', dtype, 128, 65, 'mixed'))
    passed = 0
    try:
        for index, (api, dtype, d, n, mode) in enumerate(cases):
            started = time.perf_counter()
            tuner.configs = [forced] if d == 32 else original_configs
            tuner.cache.clear()
            launches.clear()
            label = dict(index=index, api=api, dtype=str(dtype), d=d, n=n, mode=mode)
            emit(event='case_start', **label)
            try:
                values, mask, do = make_case(n, d, dtype, mode, seed=1337 + n + d,
                                            heads=1 if args.sanitizer else 2)
                result = run(entry(api), values, mask, do)
                torch.cuda.synchronize()
                if not launches:
                    raise AssertionError('case did not invoke the TMA kernel')
                if d == 32:
                    for launch in launches:
                        assert launch['DIM'] == 32 and launch['BLOCK_J'] == 64
                        assert launch['BLOCK_K'] == 64 and launch['num_warps'] == 4
                        assert launch['num_stages'] == 2 and launch['maxnreg'] == 96
                        assert launch['USE_TMA'] is True
                    assert config_to_dict(tuner.best_config) == config_to_dict(forced)
                expected = reference(values, mask, do, device='cpu' if args.sanitizer else None)
                verify(result, expected, dtype)
                torch.cuda.synchronize()
            except Exception as exc:
                emit(event='case', **label, passed=False, error_type=type(exc).__name__,
                     error=str(exc), actual_launches=launches,
                     elapsed_seconds=time.perf_counter() - started)
                raise
            passed += 1
            emit(event='case', **label, passed=True,
                 actual_config=config_to_dict(tuner.best_config), actual_launches=launches,
                 reference='independent FP64', checked=['output', 'dq', 'dk', 'dv', 'db'],
                 elapsed_seconds=time.perf_counter() - started)
            del values, mask, do, result, expected
        emit(event='complete', passed=passed, total=len(cases))
    finally:
        tuner.fn.run = original_run
        tuner.configs = original_configs


if __name__ == '__main__':
    main()
