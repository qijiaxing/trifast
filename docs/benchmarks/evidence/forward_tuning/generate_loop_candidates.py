"""Build CPU-only forward-loop experiments from frozen production source."""
from pathlib import Path
import ast
import difflib
import hashlib
import re
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[1] / 'trifast-pr'
base = subprocess.check_output(
    ['git', 'show', '4b832de:src/trifast/_fused_forward_tma.py'], cwd=repo, text=True
)
start = base.index('    # Peel the k loop:')
end = base.index('    normalize = acc / sm_denom[:, None]', start)
segment = base[start:end]
loop_start = segment.index('    for start_k in tl.range')
tail_start = segment.index('    # The ragged tail,')
loop = segment[loop_start:tail_start].rstrip() + '\n'
tail = segment[segment.index('    if n_full < N:'):].rstrip() + '\n'

unified = '''    # Experiment: include the ragged block in the same runtime pipeline.
    # Every iteration masks key padding; the full-block operation order stays
    # the same, and invalid keys remain -inf / exactly zero probability.
''' + loop.replace('tl.range(0, n_full, BLOCK_K)', 'tl.range(0, N, BLOCK_K)').replace('K_MASKED=False', 'K_MASKED=True') + '\n\n'

# Tail loads use advanced pointers without mutating the base pointers consumed
# by the subsequent complete-block loop. TMA coordinates use n_full directly.
tail_first = tail.replace(
    '            kt_ptrs, b_ptrs, v_ptrs, mask_ptrs,',
    '''            kt_ptrs + n_full * stride_kn,
            b_ptrs + n_full * stride_bn,
            v_ptrs + n_full * stride_vn,
            mask_ptrs + n_full * stride_maskn,'''
)
tail_first = '''    # Experiment: seed online softmax with the ragged key block before
    # the pipelined full-block loop. This changes floating-point reduction order.
    n_full = (N // BLOCK_K) * BLOCK_K
''' + tail_first + '\n' + loop + '\n\n'

manifest = {'base_commit': '4b832de', 'base_sha256': hashlib.sha256(base.encode()).hexdigest(), 'candidates': {}}
for name, body in [('unified', unified), ('tail_first', tail_first)]:
    source = base[:start] + body + base[end:]
    for old, new in [('_tma_kv_block', f'_tma_{name}_kv_block'), ('_fused_tma_pointer', f'_fused_tma_{name}_pointer'), ('_fused_tma', f'_fused_tma_{name}')]:
        source = re.sub(r'\b' + old + r'\b', new, source)
    source = source.replace('trifast::fused_forward_tma_bucket', f'trifast_experiment::fused_forward_tma_{name}')
    source = source.replace('fused_tma_runtime_bucket_vector_store_v3', f'forward_experiment_{name}_tma_v1')
    source = source.replace('fused_tma_pointer_runtime_bucket_vector_store_v3', f'forward_experiment_{name}_pointer_v1')
    # Exactly the same eight ordinary TMA configs as the frozen baseline. Do
    # not expand the candidate list through TRIFAST_FORCE_TUNE's extra configs.
    source = source.replace('configs=_fwd_configs,', 'configs=_fwd_configs[:8],').replace('configs=_fwd_pointer_configs,', 'configs=_fwd_pointer_configs[:4],')
    ast.parse(source)
    compile(source, str(here / f'candidate_{name}.py'), 'exec')
    path = here / f'candidate_{name}.py'
    path.write_text(source)
    diff = ''.join(difflib.unified_diff(base.splitlines(keepends=True), source.splitlines(keepends=True), fromfile='4b832de/_fused_forward_tma.py', tofile=path.name))
    (here / f'candidate_{name}.diff').write_text(diff)
    manifest['candidates'][name] = {'file': path.name, 'sha256': hashlib.sha256(source.encode()).hexdigest(), 'kernel': f'_fused_tma_{name}', 'custom_op': f'trifast_experiment::fused_forward_tma_{name}'}
import json
(here / 'loop_candidates_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
