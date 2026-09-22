import importlib.util,sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd()))
import torch
import trifast._fused_forward_tma as original
from tests.fused_reference import make_case,assert_close,reference
root=Path(__file__).parent
text=Path(original.__file__).read_text()
old='''    if USE_TMA:
        desc_o.store(
            [start_h.to(tl.int32), start_i.to(tl.int32), start_j.to(tl.int32), 0],
            final_output.reshape(1, 1, BLOCK_J, DIM),
        )
    else:
        tl.store(o_ptrs, final_output, mask=mask_j[:, None])'''
assert old in text
text=text.replace(old,'    tl.store(o_ptrs, final_output, mask=mask_j[:, None])')
text=text.replace('base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)','base_o_ptr = o_ptr + ((start_h * N + start_i) * N * DIM)')
text=text.replace('fused_forward_tma_bucket','forward_store_probe').replace('_fused_tma','_store_probe_tma').replace('runtime_bucket_v2','runtime_bucket_store_probe')
p=root/'pointer_store_candidate.py';p.write_text(text)
spec=importlib.util.spec_from_file_location('pointer_store_candidate',p);candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
if len(sys.argv)>1:
 import trifast._fused_dispatch as dispatch
 import runpy
 dispatch.fused_forward_tma=candidate.fused_forward_tma
 script=sys.argv.pop(1)
 runpy.run_path(script,run_name='__main__')
else:
 for n,d,dtype in [(65,32,torch.bfloat16),(65,128,torch.float16),(1,32,torch.bfloat16)]:
  values,mask,do=make_case(n,d,dtype)
  expected=reference(values,mask,do)[0]
  for label,module in [('tma_store',original),('pointer_store',candidate)]:
   module.fused_forward_tma(*values,mask)
   allocate=torch.empty_like
   def poisoned(t,*a,**kw):
    out=allocate(t,*a,**kw)
    if out.ndim==4:out.fill_(float('nan'))
    return out
   for synchronize in (False,True):
    with patch('torch.empty_like',poisoned):o,*_=module.fused_forward_tma(*values,mask)
    if synchronize:torch.cuda.synchronize()
    assert_close(o,expected,dtype)
    print(json.dumps(dict(event='poison_pass',n=n,d=d,dtype=str(dtype),store=label,explicit_sync=synchronize)),flush=True)
  for device,cache in original._fused_tma.fn.device_caches.items():
   for index,binary in enumerate(cache[0].values()):
    (root/f'store-{n}-{d}-{index}.ptx').write_text(binary.asm['ptx'])
