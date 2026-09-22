import json,torch
from trifast import triangle_attention_fused
from tests.fused_reference import make_case,reference,assert_close
for n,d,dtype in [(65,32,torch.bfloat16),(65,128,torch.float16)]:
 values,mask,do=make_case(n,d,dtype)
 expected=reference(values,mask,do)
 for chunk in [None,128]:
  leaves=[v.detach().requires_grad_() for v in values]
  o=triangle_attention_fused(*leaves,mask,chunk_i=chunk)
  grads=torch.autograd.grad(o,leaves,do)
  for x,y in zip((o,*grads),expected):assert_close(x,y,dtype)
  print(json.dumps(dict(event='passed',n=n,d=d,chunk=chunk)),flush=True)
