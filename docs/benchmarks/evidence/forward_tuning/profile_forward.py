import argparse,json,torch
from trifast._fused_forward_tma import fused_forward_tma,_fused_tma
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=800);a=p.parse_args()
n=a.n;torch.manual_seed(734+n)
shape=(1,8,n,n,32)
values=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
values.append(torch.randn(shape[:-1],device='cuda',dtype=torch.bfloat16))
mask=torch.rand((1,n,n),device='cuda')<.2
for _ in range(3):o=fused_forward_tma(*values,mask)
torch.cuda.synchronize()
print(json.dumps({'n':n,'config':str(_fused_tma.best_config)}),flush=True)
torch.cuda.cudart().cudaProfilerStart()
o=fused_forward_tma(*values,mask)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print('complete',flush=True)
