"""Opt-in fused APIs: independent FP64, autograd contracts, compile and sanitizer.

pytest -q tests/unit/test_fused.py -m 'not compile'
pytest -q tests/unit/test_fused.py -m compile
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest -q tests/unit/test_fused.py -m sanitizer
"""

import functools

import pytest
import torch

from tests.fused_reference import MASK_FILL, assert_close, make_case, reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
APIS = ["full", "chunk32", "default"]


@pytest.fixture(autouse=True)
def restore_flags():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)


def entry(name):
    # Import after skip evaluation: package imports can query the CUDA device.
    from trifast.fused_api import triangle_attention_fused

    if name == "full":
        return functools.partial(triangle_attention_fused, chunk_i=None)
    if name == "default":
        return triangle_attention_fused
    return functools.partial(triangle_attention_fused, chunk_i=32)


def run(fn, values, mask, do):
    leaves = [x.detach().requires_grad_() for x in values]
    output = fn(*leaves, mask)
    gradients = torch.autograd.grad(output, leaves, do)
    return (output.detach(), *gradients)


def verify(result, expected, dtype):
    assert len(result) == len(expected) == 5
    for x, y in zip(result, expected):
        assert x.dtype == dtype and x.device.type == "cuda"
        assert_close(x.to(y.device), y, dtype)


# 288 cases: full dtype/D contract and tails without a costly all-N cross product.
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "n,mode",
    [
        (1, "mixed"),
        (3, "sentinel"),
        (17, "mixed"),
        (17, "all"),
        (65, "mixed"),
        (65, "all"),
        (65, "singleton"),
        (129, "mixed"),
    ],
)
def test_fp64(api, dtype, d, n, mode):
    values, mask, do = make_case(n, d, dtype, mode, seed=1337 + n + d)
    verify(run(entry(api), values, mask, do), reference(values, mask, do), dtype)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_shared_view_base_gradients(api, dtype):
    n, d = 17, 32
    base = torch.randn(
        (1, 2, n, n + 2, d * 2), device="cuda", dtype=dtype
    ).requires_grad_()
    bias_base = torch.randn((1, 2, n, n), device="cuda", dtype=dtype).requires_grad_()

    def views(x, b):
        return (
            *[x[:, :, :, start : start + n, ::2].transpose(2, 3) for start in range(3)],
            b.transpose(2, 3),
        )

    values = views(base, bias_base)
    mask = (torch.rand((1, n, n), device="cuda") < 0.2).transpose(1, 2)
    do = torch.randn((1, 2, n, n, d * 2), device="cuda", dtype=dtype)[..., ::2]
    output = entry(api)(*values, mask)
    actual = torch.autograd.grad(output, (base, bias_base), do)
    rb, bb = [x.detach().double().requires_grad_() for x in (base, bias_base)]
    q, k, v, bias = views(rb, bb)
    scores = q @ k.transpose(-1, -2) * d**-0.5 + bias[:, :, None]
    expected = scores.masked_fill(mask[:, None, :, None, :], MASK_FILL).softmax(-1) @ v
    gradients = torch.autograd.grad(expected, (rb, bb), do.double())
    for x, y in zip((output, *actual), (expected, *gradients)):
        assert_close(x, y, dtype)


@pytest.mark.parametrize("api", APIS)
def test_sum_backward(api):
    values, mask, do = make_case(65, 32, torch.bfloat16, layout="sum")
    leaves = [x.detach().requires_grad_() for x in values]
    output = entry(api)(*leaves, mask)
    output.sum().backward()
    verify(
        (output, *(x.grad for x in leaves)), reference(values, mask, do), torch.bfloat16
    )


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("warn_only", [False, True])
def test_determinism_after_forward(api, warn_only):
    values, mask, do = make_case(3, 32, torch.bfloat16)
    leaves = [x.requires_grad_() for x in values]
    output = entry(api)(*leaves, mask)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with pytest.raises(RuntimeError, match="nondeterministic atomic"):
        output.backward(do)


@pytest.mark.parametrize("api", APIS)
def test_second_derivative_rejected(api):
    values, mask, do = make_case(3, 32, torch.bfloat16)
    leaves = [x.requires_grad_() for x in values]
    output = entry(api)(*leaves, mask)
    grads = torch.autograd.grad(output, leaves, do.requires_grad_(), create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        sum(x.float().sum() for x in grads).backward()


@pytest.mark.parametrize(
    "chunk,error",
    [
        (True, TypeError),
        (False, TypeError),
        (1.5, TypeError),
        ("32", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
@pytest.mark.parametrize("api", ["default", "explicit_low_memory"])
def test_invalid_chunk(chunk, error, api):
    from trifast.fused_api import triangle_attention_fused
    from trifast.fused_low_memory_api import triangle_attention_fused_low_memory

    values, mask, do = make_case(3, 32, torch.bfloat16)
    fn = (
        triangle_attention_fused
        if api == "default"
        else triangle_attention_fused_low_memory
    )
    if chunk is None and api == "default":
        # Main entry accepts None to request the full workspace implementation.
        verify(
            run(functools.partial(fn, chunk_i=None), values, mask, do),
            reference(values, mask, do),
            torch.bfloat16,
        )
        return
    with pytest.raises(error):
        fn(*values, mask, chunk_i=chunk)


@pytest.mark.compile
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [32, 128])
def test_compile_reuse(api, dtype, d):
    # Each parametrized case is an independent compile configuration. Do not
    # consume another case's process-global Dynamo specialization quota; the
    # two calls below still reuse one compiled callable without a reset.
    torch._dynamo.reset()
    compiled = torch.compile(entry(api), fullgraph=True)
    for repeat in (0, 1):  # 18 parametrized tests, 36 actual compiled calls.
        values, mask, do = make_case(65, d, dtype, seed=9231 + repeat)
        if repeat:
            do.zero_()
        verify(run(compiled, values, mask, do), reference(values, mask, do), dtype)


@pytest.mark.sanitizer
@pytest.mark.parametrize(
    "api,n,d,dtype,layout,batch,heads",
    [
        ("full", 65, 32, torch.bfloat16, "spatial", 1, 1),
        ("chunk32", 65, 32, torch.bfloat16, "channel", 1, 1),
        ("chunk32", 65, 128, torch.float16, "batch_head", 2, 2),
        ("full", 3, 32, torch.float32, "channel", 2, 2),
        ("default", 1, 32, torch.float32, "batch_head", 2, 2),
    ],
)
def test_sanitizer_smoke(api, n, d, dtype, layout, batch, heads):
    values, mask, do = make_case(
        n,
        d,
        dtype,
        "sentinel" if n == 3 else "mixed",
        layout=layout,
        batch=batch,
        heads=heads,
    )
    verify(
        run(entry(api), values, mask, do),
        reference(values, mask, do, device="cpu"),
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_default_matches_explicit_low_memory(dtype):
    from trifast.fused_low_memory_api import triangle_attention_fused_low_memory

    values, mask, do = make_case(129, 32, dtype, seed=20261004)
    default = run(entry("default"), values, mask, do)
    explicit = run(triangle_attention_fused_low_memory, values, mask, do)
    expected = reference(values, mask, do)
    verify(default, expected, dtype)
    verify(explicit, expected, dtype)
    for x, y in zip(default, explicit):
        assert_close(x, y, dtype)
