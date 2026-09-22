"""Independent finite-mask FP64 reference and fixed fused-attention gates."""

import torch

MASK_FILL = -10000.0
RELATIVE_LIMIT = {torch.bfloat16: 0.012, torch.float16: 0.002, torch.float32: 2e-5}
ZERO_ATOL = 2e-5
NAMES = ("output", "dq", "dk", "dv", "db")


def assert_close(actual, reference, dtype):
    x, y = actual.detach().double(), reference.detach().double()
    assert x.shape == y.shape
    assert bool(torch.isfinite(x).all()), "nonfinite candidate"
    assert bool(torch.isfinite(y).all()), "nonfinite reference"
    error = x - y
    zero = y.abs() <= 1e-12
    zero_error = error[zero].abs().max().item() if zero.any() else 0.0
    assert zero_error <= ZERO_ATOL, f"zero-reference absolute error {zero_error}"
    norm = y.norm().item()
    if norm > 1e-10:
        relative = error.norm().item() / max(norm, 1e-12)
        assert relative <= RELATIVE_LIMIT[dtype], f"relative L2 {relative}"
    else:
        assert error.abs().max().item() <= ZERO_ATOL


def reference(values, mask, do, device=None):
    device = values[0].device if device is None else device
    leaves = [
        x.detach().to(device=device, dtype=torch.float64).requires_grad_()
        for x in values
    ]
    q, k, v, bias = leaves
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5 + bias[:, :, None]
    output = (
        scores.masked_fill(mask.to(device)[:, None, :, None, :], MASK_FILL).softmax(-1)
        @ v
    )
    gradients = torch.autograd.grad(
        output, leaves, do.to(device=device, dtype=torch.float64)
    )
    return (output.detach(), *gradients)


def make_case(n, d, dtype, mode="mixed", seed=1337, batch=1, heads=2, layout="spatial"):
    torch.manual_seed(seed)
    shape = (batch, heads, n, n, d)
    values = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    values.append(torch.randn(shape[:-1], device="cuda", dtype=dtype))
    mask = torch.rand((batch, n, n), device="cuda") < 0.2
    if mode == "all":
        mask.fill_(True)
    elif mode in ("singleton", "sentinel"):
        mask.fill_(True)
        mask[:, :, -1] = False
        if mode == "sentinel":
            values[0].mul_(0.125)
            values[1].mul_(0.125)
            values[3].fill_(-10000.0)
    elif mode == "mixed":
        mask[:, 0] = True
        if n > 1:
            mask[:, 1] = True
            mask[:, 1, -1] = False
    else:
        raise ValueError(mode)
    if layout == "channel":
        do = torch.randn((*shape[:-1], d * 2), device="cuda", dtype=dtype)[..., ::2]
    elif layout == "sum":
        do = torch.ones((), device="cuda", dtype=dtype).expand(shape)
    elif layout == "batch_head":
        do = torch.randn((heads, batch, n, n, d), device="cuda", dtype=dtype).transpose(
            0, 1
        )
    else:
        do = torch.randn(shape, device="cuda", dtype=dtype).transpose(2, 3)
    return values, mask, do
