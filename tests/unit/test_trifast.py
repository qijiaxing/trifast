import torch
import pytest
from einops import rearrange
from trifast.torch import triangle_attention
from trifast.equiv import attention_reference
from trifast.utils import gen_tensors, clone_and_clear_grad, disable_tf32, enable_tf32

from tests.utils import (
    set_seed,
    compare_directions,
    compare_relative_direction,
    dot,
    compare_values,
)

set_seed(1337)


dtype_eps = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-4,
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("std", [1.0, 2.0])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
        (32, 1, 32),
        (64, 1, 64),
        (16, 4, 128),
        *[(n, 4, 32) for n in range(17, 200, 1)],
        (191, 4, 32),
    ],
)
def test_values(
    n: int, h: int, d: int, mask: bool, bs: int, dtype: torch.dtype, std: float
):
    device = torch.device("cuda")
    q, k, v, b, m = gen_tensors(
        n, d, h, use_mask=mask, device=device, dtype=torch.float32, batch=bs, std=std
    )
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()

    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_kernel.sum().backward()
    dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    compare_values(o_kernel, o_pt, o_ref, "o failed", eps=dtype_eps[dtype])
    compare_values(dq_kernel, dq_pt, dq_ref, "dq failed", eps=dtype_eps[dtype])
    compare_values(dk_kernel, dk_pt, dk_ref, "dk failed", eps=dtype_eps[dtype])
    compare_values(dv_kernel, dv_pt, dv_ref, "dv failed", eps=dtype_eps[dtype])
    compare_values(db_kernel, db_pt, db_ref, "db failed", eps=dtype_eps[dtype])
    torch.cuda.synchronize()


# (rtol, atol) for the closed-form fully-masked-row output, o[..., i, j, :] == mean_k(v).
# The kernel accumulates in fp32 and only rounds the final result, so the slack is
# essentially one rounding of the output dtype.
fully_masked_tol = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (5e-2, 5e-2),
    torch.float32: (1e-4, 1e-5),
}


def make_mask_with_fully_masked_rows(
    bs: int, n: int, rows_per_batch: list[tuple[int, ...]], device: torch.device
) -> torch.Tensor:
    """A random mask where `rows_per_batch[bi]` are *fully* masked.

    `mask` is indexed (batch, i, k) and True means "masked out", so a fully masked
    row is `mask[bi, i, :] = True`: query row `i` can see no key at all. Every other
    row keeps at least one visible key, so a single tensor exercises both the
    degenerate and the ordinary partially-masked path.
    """
    m = torch.randint(0, 2, (bs, n, n), device=device, dtype=torch.bool)
    # Keep key 0 visible everywhere so no *other* row is accidentally fully masked.
    m[:, :, 0] = False
    for bi, rows in enumerate(rows_per_batch):
        for i in rows:
            m[bi, i, :] = True
    return m


def fully_masked_rows_for(n: int) -> list[tuple[int, ...]]:
    """One row set per batch item: first, interior, and last row.

    Different rows per batch item so the test also pins down that the kernel indexes
    the mask by batch and not by (batch x head).
    """
    return [(0, n // 2), (n // 3, n - 1)]


def zero_rows(t: torch.Tensor, keep_row: torch.Tensor) -> torch.Tensor:
    """Zero the query rows (dim 2 of [b, h, i, j, d]) where `keep_row` [b, i] is False."""
    return t * keep_row[:, None, :, None, None].to(t.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        # n=128 spans several whole BLOCK_K tiles; n=100 additionally exercises the
        # ragged tail (it is not a multiple of 64).
        (128, 2, 32),
        (100, 2, 32),
    ],
)
def test_fully_masked_row_values(n: int, h: int, d: int, dtype: torch.dtype):
    """A row where every key is masked must still agree with the reference.

    Both the kernel and `attention_reference` degenerate such a row to uniform
    weights, so the forward output (and dV) must match. dQ/dK/dB deliberately do
    *not*: the kernel treats the mask as a hard select, whose gradient is zero, while
    the reference adds `finfo.min` and so leaks the gradient of a merely-very-negative
    bias. See `test_fully_masked_row_semantics` for that part.
    """
    device = torch.device("cuda")
    bs = 2

    q, k, v, b, _ = gen_tensors(
        n, d, h, use_mask=False, device=device, dtype=torch.float32, batch=bs
    )
    m = make_mask_with_fully_masked_rows(
        bs, n, fully_masked_rows_for(n), device=device
    )
    # [bs, n], True for rows that still see at least one key.
    keep_row = ~m.all(dim=-1)
    assert keep_row.sum() < bs * n, "expected at least one fully masked row"
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()
    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_kernel.sum().backward()
    dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    eps = dtype_eps[dtype]
    compare_values(o_kernel, o_pt, o_ref, "o failed", eps=eps)
    compare_values(dv_kernel, dv_pt, dv_ref, "dv failed", eps=eps)
    # dQ/dK are compared on the rows that are not fully masked.
    compare_values(
        zero_rows(dq_kernel, keep_row),
        zero_rows(dq_pt, keep_row),
        zero_rows(dq_ref, keep_row),
        "dq failed",
        eps=eps,
    )
    compare_values(
        zero_rows(dk_kernel, keep_row),
        zero_rows(dk_pt, keep_row),
        zero_rows(dk_ref, keep_row),
        "dk failed",
        eps=eps,
    )

    # dB is shared across i (bias is [b, h, j, k]), so the fully-masked rows cannot be
    # excluded from it; only require that it stays finite.
    for name, g in [
        ("dq", dq_kernel),
        ("dk", dk_kernel),
        ("dv", dv_kernel),
        ("db", db_kernel),
    ]:
        assert torch.isfinite(g).all(), f"{name} has non-finite values"

    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (128, 2, 32),
        (100, 2, 32),
    ],
)
def test_fully_masked_row_semantics(n: int, h: int, d: int, dtype: torch.dtype):
    """Pin the documented behaviour of a fully masked row (see MASK_FILL in torch.py).

    Uniform weights, hence `o[b, h, i, j, :] == mean_k(v[b, h, i, k, :])` for every j,
    and never NaN/Inf. This is what an *additive* mask sentinel silently breaks: the
    online softmax's max subtraction cancels the sentinel back out and the row behaves
    as if it were not masked at all.
    """
    device = torch.device("cuda")
    bs = 2
    rows_per_batch = fully_masked_rows_for(n)

    q, k, v, b, _ = gen_tensors(
        n, d, h, use_mask=False, device=device, dtype=torch.float32, batch=bs
    )
    m = make_mask_with_fully_masked_rows(bs, n, rows_per_batch, device=device)
    torch.cuda.synchronize()

    qd, kd, vd, bd = q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype)
    o = triangle_attention(qd, kd, vd, bd, m)

    assert torch.isfinite(o).all(), "output has NaN/Inf"

    rtol, atol = fully_masked_tol[dtype]
    for bi, rows in enumerate(rows_per_batch):
        for i in rows:
            # Uniform over the k axis -> plain mean of V, broadcast over j.
            expected = vd[bi, :, i].float().mean(dim=-2, keepdim=True)
            torch.testing.assert_close(
                o[bi, :, i].float(),
                expected.expand_as(o[bi, :, i]),
                rtol=rtol,
                atol=atol,
                msg=lambda s, bi=bi, i=i: f"batch {bi} row {i} is not mean(V): {s}",
            )

    o.sum().backward()
    for name, t in [("dq", q), ("dk", k), ("dv", v), ("db", b)]:
        assert torch.isfinite(t.grad).all(), f"{name} has non-finite values"
        # The kernel substitutes the score for a masked key instead of biasing it, so
        # there is no gradient path from a fully masked row back to q/k. (The reference
        # disagrees here: `finfo.min` saturates the score but autograd still routes a
        # gradient through the add.)
        if name in ("dq", "dk"):
            for bi, rows in enumerate(rows_per_batch):
                for i in rows:
                    assert (
                        t.grad[bi, :, i] == 0
                    ).all(), f"{name} batch {bi} row {i} should have no gradient"
    clone_and_clear_grad(q, k, v, b)

    torch.cuda.synchronize()


def compare_dot(kernel_output, pytorch_output, ref_output, msg="", threshold=0.05):
    # threshold is how much worse tri can be than the pytorch version.

    # magnitude of the ref vector
    ref_magnitude = dot(ref_output, ref_output)

    # dot product of tri and pt, normed by ref magnitude
    # These are 1.0 if perfect.
    kernel_score = dot(kernel_output, ref_output) / ref_magnitude
    pt_score = dot(pytorch_output, ref_output) / ref_magnitude

    # If kernel is better than pt, that is fine (hence the negative threshold)
    error = kernel_score - pt_score

    assert (
        error >= (-1 * threshold)
    ), f"{msg} dot product mismatch: {error:.3f} tri: {kernel_score:.3f}, pt: {pt_score:.3f}"


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("std", [1.0, 2.0])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
        (32, 1, 32),
        (64, 1, 64),
        (16, 4, 128),
        *[(n, 4, 32) for n in range(17, 200, 1)],
    ],
)
def test_vectors(
    n: int, h: int, d: int, mask: bool, bs: int, dtype: torch.dtype, std: float
):
    device = torch.device("cuda")

    q, k, v, b, m = gen_tensors(
        n, d, h, mask, device, dtype=torch.float32, std=std, batch=bs
    )
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()

    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_kernel.sum().backward()
    dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    compare_relative_direction(o_kernel, o_pt, o_ref, "Output")
    compare_relative_direction(dq_kernel, dq_pt, dq_ref, "dQ")
    compare_relative_direction(dk_kernel, dk_pt, dk_ref, "dK")
    compare_relative_direction(dv_kernel, dv_pt, dv_ref, "dV")
    compare_relative_direction(db_kernel, db_pt, db_ref, "dB")

    compare_directions(o_kernel, o_pt, o_ref, "Output")
    compare_directions(dq_kernel, dq_pt, dq_ref, "dQ")
    compare_directions(dk_kernel, dk_pt, dk_ref, "dK")
    compare_directions(dv_kernel, dv_pt, dv_ref, "dV")
    compare_directions(db_kernel, db_pt, db_ref, "dB")

    compare_dot(o_kernel, o_pt, o_ref, "Output", threshold=0.01)
    compare_dot(dq_kernel, dq_pt, dq_ref, "dQ", threshold=0.01)
    compare_dot(dk_kernel, dk_pt, dk_ref, "dK", threshold=0.01)
    compare_dot(dv_kernel, dv_pt, dv_ref, "dV", threshold=0.01)
    compare_dot(db_kernel, db_pt, db_ref, "dB", threshold=0.01)

    torch.cuda.synchronize()


class FakeModule(torch.nn.Module):
    def __init__(self, h: int, d: int):
        super().__init__()

        self.h = h

        self.q = torch.nn.Linear(d, h * d)
        self.k = torch.nn.Linear(d, h * d)
        self.v = torch.nn.Linear(d, h * d)
        self.b = torch.nn.Linear(d, h)

    def forward(self, x, mask):
        # triangle_attention's contract is q/k/v [b, h, n, n, d] and bias [b, h, n, n]
        # (see its jaxtyping annotations). x is [b, n, n, d], so the head axis the
        # Linear introduces has to move in *front* of both n axes. Leaving it last
        # gave [b, n, n, h, d], which still unpacks as five dims, so
        # `bs, h, _, n, dim = q.shape` silently read h=n and n=1 and the kernel
        # computed a single i slice.
        q = rearrange(self.q(x), "b i j (h d) -> b h i j d", h=self.h)
        k = rearrange(self.k(x), "b i j (h d) -> b h i j d", h=self.h)
        v = rearrange(self.v(x), "b i j (h d) -> b h i j d", h=self.h)
        b = rearrange(self.b(x), "b i j h -> b h i j")

        return triangle_attention(q, k, v, b, mask)


@pytest.mark.parametrize(
    ("n, h, d, do_mask, bs, dtype"),
    [
        (16, 1, 16, False, 1, torch.float16),
        (32, 1, 32, False, 1, torch.bfloat16),
        (64, 1, 64, False, 4, torch.float32),
    ],
)
def test_weight_updates(
    n: int, h: int, d: int, do_mask: bool, bs: int, dtype: torch.dtype
):
    x = torch.randn((bs, n, n, d), device="cuda", dtype=dtype, requires_grad=True)

    model = FakeModule(h, d).to("cuda", dtype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mask = (
        torch.randint(0, 2, (bs, n, n), device="cuda", dtype=torch.bool)
        if do_mask
        else torch.zeros((bs, n, n), device="cuda", dtype=torch.bool)
    )

    orig_params = {k: v.clone() for k, v in model.named_parameters()}

    for _ in range(5):
        out = model(x, mask)

        lbl = torch.randn_like(out) * 4

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(out, lbl)
        loss.backward()
        opt.step()

    updated_params = dict(model.named_parameters())

    for k in orig_params.keys():
        assert not torch.all(
            orig_params[k] == updated_params[k]
        ), f"Parameter {k} did not update."
