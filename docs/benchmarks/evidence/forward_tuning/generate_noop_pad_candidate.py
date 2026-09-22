"""CPU-only generation of a wrapper-copy ablation; kernel math unchanged."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import re
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[1] / 'trifast-pr'
base = subprocess.check_output(['git', 'show', '4b832de:src/trifast/_fused_forward_tma.py'], cwd=repo, text=True)
s = base.replace('''        wide_mask = torch.nn.functional.pad(
            mask.to(MASK_TMA_DTYPE), (0, padded_mask_n - n)
        )''', '''        wide_mask = mask.to(MASK_TMA_DTYPE)
        if padded_mask_n != n:
            wide_mask = torch.nn.functional.pad(wide_mask, (0, padded_mask_n - n))''')
s = s.replace('        padded_b = torch.nn.functional.pad(b, (0, padded_n - n))', '''        padded_b = b
        if padded_n != n:
            padded_b = torch.nn.functional.pad(b, (0, padded_n - n))''')
for old, new in [('_tma_kv_block', '_tma_noop_pad_kv_block'), ('_fused_tma_pointer', '_fused_tma_noop_pad_pointer'), ('_fused_tma', '_fused_tma_noop_pad')]:
    s = re.sub(r'\b' + old + r'\b', new, s)
s = s.replace('trifast::fused_forward_tma_bucket', 'trifast_experiment::fused_forward_tma_noop_pad')
s = s.replace('fused_tma_runtime_bucket_vector_store_v3', 'forward_experiment_noop_pad_tma_v1')
s = s.replace('fused_tma_pointer_runtime_bucket_vector_store_v3', 'forward_experiment_noop_pad_pointer_v1')
s = s.replace('configs=_fwd_configs,', 'configs=_fwd_configs[:8],').replace('configs=_fwd_pointer_configs,', 'configs=_fwd_pointer_configs[:4],')
path = here / 'candidate_noop_pad.py'
ast.parse(s)
compile(s, str(path), 'exec')
path.write_text(s)
(here / 'candidate_noop_pad.diff').write_text(''.join(difflib.unified_diff(base.splitlines(keepends=True), s.splitlines(keepends=True), fromfile='4b832de/_fused_forward_tma.py', tofile=path.name)))
manifest = {'base_commit': '4b832de', 'base_sha256': hashlib.sha256(base.encode()).hexdigest(), 'candidate_sha256': hashlib.sha256(s.encode()).hexdigest(), 'file': path.name, 'kernel': '_fused_tma_noop_pad', 'custom_op': 'trifast_experiment::fused_forward_tma_noop_pad'}
(here / 'noop_pad_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
