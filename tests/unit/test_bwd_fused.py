"""The fused backward against the three kernels it replaces.

`triangle_attention_bwd` can run either path, selected by `trifast.torch.USE_FUSED_BWD`.
The rest of the suite only ever exercises whichever one is the default and compares it to
a PyTorch reference, which is the right test for the maths but a weak one for a swap
between two kernels that should agree to a rounding. These tests pin them against each
other directly.

They also cover the one failure mode that belongs to the fused kernel alone: it
*accumulates* db and dq with atomics, so anything that launches it more than once per
zeroed buffer multiplies those two gradients instead of perturbing them. That bug is
invisible to a tolerance-based comparison against PyTorch -- it produced gradients ~400x
too large -- so it gets its own test.
"""

import contextlib

import pytest
import torch

import trifast.torch as trifast_torch
from trifast.torch import triangle_attention
from trifast.triton_bwd import _bwd_fused_tuned
from trifast.utils import clone_and_clear_grad, gen_tensors

# The two paths do the same arithmetic in a different order: the fused kernel reduces dq
# over k with atomics in whatever order the CTAs land, and its score tile is transposed,
# so the fp32 accumulation order differs. dk and dv usually come out bit-identical.
# Measured worst case across this matrix is 2.5e-3 for bf16 dq and 3.3e-7 for fp32, so
# these bounds leave room for a rounding but catch anything structural.
PATH_AGREEMENT_TOL = {
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}


@contextlib.contextmanager
def backward_path(fused: bool):
    previous = trifast_torch.USE_FUSED_BWD
    trifast_torch.USE_FUSED_BWD = fused
    try:
        yield
    finally:
        trifast_torch.USE_FUSED_BWD = previous


def backward_grads(q, k, v, b, mask):
    out = triangle_attention(q, k, v, b, mask)
    out.sum().backward()
    return clone_and_clear_grad(q, k, v, b)


def assert_paths_agree(fused, three_kernel, dtype, context=""):
    names = ("dq", "dk", "dv", "db")
    tol = PATH_AGREEMENT_TOL[dtype]
    for name, got, want in zip(names, fused, three_kernel):
        assert torch.isfinite(got).all(), f"{context}{name} has non-finite values"
        scale = want.float().abs().max().item() or 1.0
        error = (got.float() - want.float()).abs().max().item() / scale
        assert error <= tol, (
            f"{context}{name} disagrees with the three-kernel path: "
            f"{error:.3e} relative, tolerance {tol:.0e}"
        )


def fully_masked_row_mask(bs: int, n: int, device) -> torch.Tensor:
    """A random mask with a few fully-masked rows.

    Mirrors `test_trifast.py::make_mask_with_fully_masked_rows`. Kept local rather than
    imported so this module does not depend on another test module's layout.
    """
    m = torch.randint(0, 2, (bs, n, n), device=device, dtype=torch.bool)
    m[:, :, 0] = False  # keep one key visible so no *other* row is fully masked
    for bi in range(bs):
        for i in (0, n // 2, n - 1):
            m[bi, i, :] = True
    return m


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_mode", ["random", "none", "fully_masked_rows"])
@pytest.mark.parametrize(
    ("n", "h", "d"),
    [
        (128, 2, 32),  # several whole tiles, the shape the kernel is tuned for
        (100, 2, 32),  # ragged in j *and* k, so both peeled paths run
        (17, 2, 32),  # n < BLOCK_K: one k tile, almost all of it masked off
        (64, 1, 64),  # DIM = 64
        (16, 4, 128),  # DIM = 128, the register-tightest shape
    ],
)
def test_matches_three_kernel_path(n, h, d, mask_mode, dtype):
    device = torch.device("cuda")
    torch.manual_seed(1337)
    q, k, v, b, mask = gen_tensors(
        n=n, d=d, h=h, use_mask=(mask_mode == "random"), device=device, dtype=dtype
    )
    if mask_mode == "fully_masked_rows":
        mask = fully_masked_row_mask(1, n, device)

    with backward_path(False):
        three_kernel = backward_grads(q, k, v, b, mask)
    with backward_path(True):
        fused = backward_grads(q, k, v, b, mask)

    assert_paths_agree(fused, three_kernel, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_autotuning_does_not_corrupt_the_accumulators(dtype):
    """Re-benchmarking every config must leave db and dq alone.

    The fused kernel adds into the db and dq accumulators, so each of the autotuner's
    trial launches contributes another copy. `reset_to_zero` on the autotuner is what
    keeps that out of the result. Dropping the in-memory config cache forces the sweep to
    run again on the next launch, which is the situation that first exposed this:
    gradients came out a large multiple of the truth rather than slightly off.
    """
    device = torch.device("cuda")
    torch.manual_seed(1337)
    q, k, v, b, mask = gen_tensors(
        n=96, d=32, h=2, use_mask=True, device=device, dtype=dtype
    )

    with backward_path(False):
        three_kernel = backward_grads(q, k, v, b, mask)

    # Force a fresh sweep. The on-disk cache is rewritten with the same answer, so this
    # costs tuning time and nothing else.
    _bwd_fused_tuned.cache.clear()
    assert len(_bwd_fused_tuned.configs) > 1, "a single config would not be benchmarked"

    with backward_path(True):
        fused = backward_grads(q, k, v, b, mask)

    assert_paths_agree(fused, three_kernel, dtype, context="after re-tuning, ")
