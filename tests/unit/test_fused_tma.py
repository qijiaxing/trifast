"""Regression: padded keys cannot dominate finite real scores below MASK_FILL."""

import pytest
import torch
from tests.fused_reference import make_case, reference
from tests.unit.test_fused import entry, run, verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("api", ["default", "full"])
@pytest.mark.parametrize("n", [3, 65, 129])
def test_padding_below_mask_sentinel(dtype, api, n):
    values, mask, do = make_case(n, 32, dtype)
    values[0].zero_()
    values[1].zero_()
    values[3].fill_(-20000)
    mask.zero_()
    verify(run(entry(api), values, mask, do), reference(values, mask, do), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [64, 65])
def test_contiguous_bias_with_misaligned_storage_offset(dtype, n):
    values, mask, do = make_case(n, 32, dtype)
    bias = values[3]
    storage = torch.empty(bias.numel() + 1, device=bias.device, dtype=bias.dtype)
    offset_bias = storage[1:].view(bias.shape)
    offset_bias.copy_(bias)
    assert offset_bias.is_contiguous()
    assert offset_bias.data_ptr() % 16 != 0
    values[3] = offset_bias
    verify(run(entry("default"), values, mask, do), reference(values, mask, do), dtype)
