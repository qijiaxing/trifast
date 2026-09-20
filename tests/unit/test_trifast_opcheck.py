import torch
from trifast.torch import _triangle_attention
from trifast.utils import gen_tensors
from torch.library import opcheck

# See OPCHECK_TOLERANCE_NOTE.md for follow-up work on this temporary tolerance.
# Eager execution takes the TMA forward while fake-tensor execution falls back
# to the pointer kernel, and the two paths legitimately pick different tile
# sizes -- so eager and compiled forwards/grads can differ by a few bf16 ulps
# (different fp32 accumulation order), never more. The default opcheck tolerance
# (atol 1e-5) is stricter than that; relax it to bf16-tile-rounding scale while
# still catching real registration/compile bugs, which produce O(1) errors.
OPCHECK_TOL = {"atol": 2e-2, "rtol": 1e-2}


def test_opcheck():
    for n in [16, 128, 256]:
        for d in [16, 32, 64]:
            for h in [1, 4]:
                q, k, v, b, m = gen_tensors(n, d, h, True, "cuda", dtype=torch.bfloat16)
                # opcheck raises an exception on failure
                opcheck(
                    _triangle_attention,
                    (q, k, v, b, m),
                    raise_exception=True,
                    **OPCHECK_TOL,
                )
