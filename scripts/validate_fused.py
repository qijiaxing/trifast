"""Independent FP64 target-shape gate, chunked over the independent i axis.

The reference uses PyTorch matmul/softmax/autograd with finite MASK_FILL.
Only one i chunk has FP64 activations/gradients live at once. Global relative
L2 metrics accumulate sums of squares across chunks; dBias is summed in FP64.
"""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch

import trifast.fused_api
from trifast.fused_api import triangle_attention_fused
from trifast.torch import triangle_attention

MASK_FILL = -10000.0

REL_LIMITS = {torch.bfloat16: 0.012, torch.float16: 0.002, torch.float32: 2e-5}
ZERO_ATOL = 2.0e-5
NAMES = ("output", "dq", "dk", "dv", "db")


def emit(**row):
    print(json.dumps(row), flush=True)


class Metric:
    def __init__(self, dtype):
        self.relative_limit = REL_LIMITS[dtype]
        self.error2 = 0.0
        self.reference2 = 0.0
        self.max_abs = 0.0
        self.zero_max_abs = 0.0
        self.finite = True
        self.rounded_zero_reference_max_abs = 0.0

    def add(self, actual, reference):
        # One slice only: no full-target double clones or zero masks.
        x = actual.double()
        difference = x - reference
        self.rounded_zero_reference_max_abs = max(
            self.rounded_zero_reference_max_abs,
            torch.where(x == 0, reference.abs(), 0.0).max().item(),
        )
        self.error2 += difference.square().sum().item()
        self.reference2 += reference.square().sum().item()
        self.max_abs = max(self.max_abs, difference.abs().max().item())
        self.zero_max_abs = max(
            self.zero_max_abs,
            torch.where(reference.abs() <= 1.0e-12, difference.abs(), 0.0).max().item(),
        )
        self.finite = self.finite and bool(
            torch.isfinite(x).all() and torch.isfinite(reference).all()
        )

    def result(self):
        relative = self.error2**0.5 / max(self.reference2**0.5, 1.0e-12)
        passed = (
            self.finite
            and self.zero_max_abs <= ZERO_ATOL
            and (
                relative <= self.relative_limit
                if self.reference2**0.5 > 1.0e-10
                else self.max_abs <= ZERO_ATOL
            )
        )
        return {
            "relative_l2": relative,
            "max_abs": self.max_abs,
            "zero_max_abs": self.zero_max_abs,
            "rounded_zero_reference_max_abs": self.rounded_zero_reference_max_abs,
            "finite": self.finite,
            "passed": passed,
        }


def validate(n, chunk, dim, dtype, api, chunk_i, benchmark_inputs=False):
    torch.manual_seed((734 if benchmark_inputs else 20260921) + n)
    shape = (1, 8, n, n, dim)
    values = [
        torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(3)
    ]
    bias = torch.randn(shape[:-1], device="cuda", dtype=dtype).requires_grad_()
    mask = torch.rand((1, n, n), device="cuda") < 0.2
    if not benchmark_inputs:
        mask[:, 0] = True
        if n > 1:
            mask[:, 1] = True
            mask[:, 1, -1] = False
    do = torch.randn(shape, device="cuda", dtype=dtype).transpose(2, 3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if api == "original":
        output = triangle_attention(*values, bias, mask)
    else:
        output = triangle_attention_fused(
            *values, bias, mask, chunk_i=chunk_i if api == "low-memory" else None
        )
    gradients = torch.autograd.grad(output, (*values, bias), do)
    output = output.detach()
    gradients = tuple(x.detach() for x in gradients)
    torch.cuda.synchronize()
    candidate_seconds = time.perf_counter() - started
    candidate_peak = torch.cuda.max_memory_allocated()
    metrics = {name: Metric(dtype) for name in NAMES}
    db = torch.zeros_like(bias, dtype=torch.float64)
    reference_started = time.perf_counter()
    for first in range(0, n, chunk):
        last = min(n, first + chunk)
        refs = [x[:, :, first:last].detach().double().requires_grad_() for x in values]
        rb = bias.detach().double().requires_grad_()
        rq, rk, rv = refs
        scores = (rq @ rk.transpose(-1, -2)) * dim**-0.5 + rb[:, :, None]
        scores = scores.masked_fill(mask[:, None, first:last, None, :], MASK_FILL)
        expected = scores.softmax(-1) @ rv
        expected_g = torch.autograd.grad(
            expected, (*refs, rb), do[:, :, first:last].double()
        )
        metrics["output"].add(output[:, :, first:last], expected.detach())
        for name, actual, reference in zip(
            ("dq", "dk", "dv"), gradients[:3], expected_g[:3]
        ):
            metrics[name].add(actual[:, :, first:last], reference)
        db.add_(expected_g[3])
        emit(
            event="chunk",
            n=n,
            first=first,
            last=last,
            elapsed_seconds=time.perf_counter() - reference_started,
            allocated_bytes=torch.cuda.memory_allocated(),
        )
        del refs, rb, rq, rk, rv, scores, expected, expected_g
    # dBias has a reduction over i, so compare only after every FP64 chunk sums.
    metrics["db"].add(gradients[3], db)
    torch.cuda.synchronize()
    checks = {name: m.result() for name, m in metrics.items()}
    emit(
        event="case",
        n=n,
        dtype=str(dtype),
        dim=dim,
        checks=checks,
        passed=all(x["passed"] for x in checks.values()),
        candidate_seconds=candidate_seconds,
        reference_seconds=time.perf_counter() - reference_started,
        candidate_peak_allocated_bytes=candidate_peak,
        total_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    return all(x["passed"] for x in checks.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", default="512,800,1024")
    p.add_argument("--dim", type=int, choices=(16, 32, 64, 128), default=32)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument(
        "--api",
        choices=("original", "full-workspace", "low-memory"),
        default="low-memory",
    )
    p.add_argument("--chunk-i", type=int, default=128)
    p.add_argument("--reference-chunk", type=int, default=8)
    p.add_argument(
        "--benchmark-inputs",
        action="store_true",
        help="Use benchmark seed and random mask without forced edge rows",
    )
    a = p.parse_args()
    try:
        shapes = [int(x) for x in a.shapes.split(",")]
    except ValueError:
        p.error("--shapes must contain comma-separated integers")
    if not shapes or min(shapes) < 1 or min(a.chunk_i, a.reference_chunk) < 1:
        p.error("all shape and chunk sizes must be positive")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        a.dtype
    ]
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve()]
    paths.extend(sorted((root / "src" / "trifast").glob("*.py")))
    # Refuse an unrelated installed candidate while hashing this repository.
    imported = Path(inspect.getfile(trifast.fused_api)).resolve()
    if imported.parent != (root / "src" / "trifast").resolve():
        p.error(
            "imported trifast is not this checkout; install it editable or set PYTHONPATH=src"
        )
    emit(
        event="start",
        args=vars(a),
        seed="734+n" if a.benchmark_inputs else "20260921+n",
        dtype=str(dtype),
        batch=1,
        heads=8,
        dim=a.dim,
        relative_limit=REL_LIMITS[dtype],
        zero_atol=ZERO_ATOL,
        mask_fill=MASK_FILL,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        reference="independent FP64 matmul/finite-mask/softmax/autograd, chunked over i",
        sources={
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
    )
    failures = completed = 0
    for n in shapes:
        try:
            passed = validate(
                n,
                a.reference_chunk,
                a.dim,
                dtype,
                a.api,
                a.chunk_i,
                a.benchmark_inputs,
            )
            failures += not passed
            completed += 1
        except Exception as error:  # noqa: BLE001 - emit failure evidence, then exit nonzero
            failures += 1
            emit(
                event="error",
                n=n,
                error_type=type(error).__name__,
                message=str(error).replace(str(root), "<repo>"),
            )
            # A device failure may poison CUDA state; preserve a final failure
            # summary instead of attempting additional cases on that context.
            break
        torch.cuda.empty_cache()
    success = failures == 0 and completed == len(shapes)
    emit(
        event="summary",
        requested_cases=len(shapes),
        completed_cases=completed,
        failures=failures,
        passed=success,
        api=a.api,
        dtype=a.dtype,
        dim=a.dim,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
