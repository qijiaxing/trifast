import argparse,hashlib,json,statistics
from pathlib import Path
import torch
import trifast._fused_forward_tma as baseline
import candidate_unified, candidate_tail_first
p=argparse.ArgumentParser();p.add_argument('--shapes',default='512,800,1024');a=p.parse_args()
mods={'baseline':baseline,'unified':candidate_unified,'tail_first':candidate_tail_first}
def emit(**x):print(json.dumps(x),flush=True)
emit(event='environment',shapes=a.shapes,sources={k:hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest() for k,v in mods.items()},gpu=torch.cuda.get_device_name(),scope='forward wrapper B1H8D32BF16, ABBA 60 samples per side')
for n in map(int,a.shapes.split(',')):
 torch.manual_seed(734+n);shape=(1,8,n,n,32)
 values=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 values.append(torch.randn(shape[:-1],device='cuda',dtype=torch.bfloat16));mask=torch.rand((1,n,n),device='cuda')<.2
 reference=baseline.fused_forward_tma(*values,mask)[0].float()
 for label in ['unified','tail_first']:
  output=mods[label].fused_forward_tma(*values,mask)[0].float()
  relative=((output-reference).norm()/reference.norm()).item()
  assert bool(torch.isfinite(output).all()) and relative<.012,(n,label,relative)
  emit(event='parity',n=n,label=label,relative_l2=relative,independent_reference=False)
  selected={'A':baseline.fused_forward_tma,'B':mods[label].fused_forward_tma}
  for fn in selected.values():
   for _ in range(3):result=fn(*values,mask)
  torch.cuda.synchronize();times={'A':[],'B':[]}
  for round in range(3):
   for side in 'ABBA':
    events=[]
    for _ in range(10):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     start.record();result=selected[side](*values,mask);end.record();events.append((start,end))
    torch.cuda.synchronize();times[side].extend(x.elapsed_time(y) for x,y in events)
  med={k:statistics.median(v) for k,v in times.items()}
  emit(event='summary',n=n,label=label,median_ms=med,speedup=med['A']/med['B'],samples=times)
 del values,mask,result,reference,output
 torch.cuda.empty_cache()
emit(event='complete')
