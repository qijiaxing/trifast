"""Additional public input contracts, separate from the numerical GPU suite."""

import functools

import pytest
import torch

from tests.fused_reference import make_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=["full", "chunk32", "default"])
def attention(request):
    from trifast.fused_api import triangle_attention_fused

    if request.param == "full":
        return functools.partial(triangle_attention_fused, chunk_i=None)
    if request.param == "default":
        return triangle_attention_fused
    return functools.partial(triangle_attention_fused, chunk_i=32)


def test_finite_mask_contract():
    from trifast.torch import MASK_FILL

    assert MASK_FILL == -10000.0


@pytest.mark.parametrize(
    "problem,error",
    [
        ("q_rank", ValueError),
        ("q_nonsquare", ValueError),
        ("empty_attention", ValueError),
        ("unsupported_dim", ValueError),
        ("k_shape", ValueError),
        ("bias_shape", ValueError),
        ("mask_shape", ValueError),
        ("mixed_dtype", TypeError),
        ("integer_input", TypeError),
        ("mask_dtype", TypeError),
        ("cpu_input", ValueError),
        ("mixed_device", ValueError),
    ],
)
def test_invalid_inputs(attention, problem, error):
    values, mask, _ = make_case(3, 32, torch.bfloat16)
    q, k, v, bias = values
    if problem == "q_rank":
        q = q[0]
    elif problem == "q_nonsquare":
        q = q[:, :, :, :2]
    elif problem == "empty_attention":
        q, k, v = [x[:, :, :0, :0] for x in (q, k, v)]
        bias = bias[:, :, :0, :0]
        mask = mask[:, :0, :0]
    elif problem == "unsupported_dim":
        q, k, v = [x[..., :24] for x in (q, k, v)]
    elif problem == "k_shape":
        k = k[:, :, :, :2]
    elif problem == "bias_shape":
        bias = bias[:, :, :2]
    elif problem == "mask_shape":
        mask = mask[:, :2]
    elif problem == "mixed_dtype":
        k = k.float()
    elif problem == "integer_input":
        q, k, v, bias = [x.to(torch.int32) for x in (q, k, v, bias)]
    elif problem == "mask_dtype":
        mask = mask.float()
    elif problem == "cpu_input":
        q, k, v, bias = [x.cpu() for x in (q, k, v, bias)]
        mask = mask.cpu()
    elif problem == "mixed_device":
        bias = bias.cpu()
    with pytest.raises(error):
        attention(q, k, v, bias, mask)


@pytest.mark.parametrize("warn_only", [False, True])
def test_forward_determinism(attention, warn_only):
    values, mask, _ = make_case(3, 32, torch.bfloat16)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with pytest.raises(RuntimeError, match="nondeterministic atomic"):
            attention(*values, mask)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
