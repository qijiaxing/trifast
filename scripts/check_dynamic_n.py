"""Verify runtime-N reuse with cache-key inventories and PTX-generation counts.

Run eagerly first; use --torch-compile in a separate process to distinguish
Dynamo graph specialization from Triton specialization. Every process uses
fresh disk caches by default; never infer compilation reuse from timing alone.
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--shapes", default="65,64,129,128,17,1,500,512,513,800,65")
p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
p.add_argument("--dim", type=int, default=32)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--torch-compile", action="store_true")
p.add_argument(
    "--chunk-i",
    type=int,
    default=128,
    help="positive chunk size; 0 selects full workspace",
)
p.add_argument("--contiguous-do", action="store_true")
p.add_argument("--reference-max-n", type=int, default=129)
p.add_argument(
    "--allow-new-specializations",
    action="store_true",
    help="diagnostic exit0 despite new compilation; numerical failures still fail",
)
a = p.parse_args()
if a.chunk_i < 0:
    p.error("--chunk-i must be nonnegative")
cache = Path(tempfile.mkdtemp(prefix="dynamic-n-"))
os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
os.environ["XDG_CONFIG_HOME"] = str(cache / "config")

import torch
import triton
from torch._dynamo.utils import (
    counters,
)
from triton.backends.nvidia.compiler import (
    CUDABackend,
)
from triton.runtime.jit import (
    JITFunction,
)

import trifast._fused_backward as backward_module
import trifast._fused_chunked as chunk_module
import trifast._fused_forward as forward_module
from trifast import (
    triangle_attention_fused,
)


def emit(**row):
    print(json.dumps(row), flush=True)


# Observe actual backend lowering, including Inductor-generated kernels. This
# counter is distinct from loading a disk-cached binary into a fresh JIT key.
ptx_events = []
original_descriptor = inspect.getattr_static(CUDABackend, "make_ptx")
original_make_ptx = CUDABackend.make_ptx


def observed_make_ptx(*args, **kwargs):
    result = original_make_ptx(*args, **kwargs)
    ptx_events.append(
        {
            "ordinal": len(ptx_events) + 1,
            "entry_names": re.findall(r"\.entry\s+([A-Za-z0-9_$]+)", str(result)),
            "ptx_sha256": hashlib.sha256(str(result).encode()).hexdigest(),
        }
    )
    return result


if isinstance(original_descriptor, staticmethod):
    CUDABackend.make_ptx = staticmethod(observed_make_ptx)
elif inspect.isfunction(original_descriptor):
    # Triton 3.7 exposes make_ptx as an instance method; args includes self.
    CUDABackend.make_ptx = observed_make_ptx
else:
    raise RuntimeError(
        "Unreviewed make_ptx descriptor: instrumentation must be adapted explicitly"
    )


def inventory():
    result = {}
    for module in (forward_module, backward_module, chunk_module):
        for name, obj in vars(module).items():
            # Some versions expose an autotuner around the actual JITFunction.
            seen = set()
            while (
                not isinstance(obj, JITFunction)
                and hasattr(obj, "fn")
                and id(obj) not in seen
            ):
                seen.add(id(obj))
                obj = obj.fn
            if not isinstance(obj, JITFunction):
                continue
            kernel_name = obj.__module__ + "." + obj.__name__
            if kernel_name in result:
                continue
            entries = []
            for device, device_cache in obj.device_caches.items():
                for key, binary in device_cache[0].items():
                    key_text = repr(key)
                    entries.append(
                        {
                            "device": str(device),
                            "key": key_text,
                            "key_sha256": hashlib.sha256(key_text.encode()).hexdigest(),
                            "binary_hash": str(getattr(binary, "hash", None)),
                            "kernel": getattr(binary, "name", name),
                        }
                    )
            result[kernel_name] = entries
    return result


def keys(inv):
    return {
        (name, x["device"], x["key_sha256"])
        for name, entries in inv.items()
        for x in entries
    }


def run(fn, values, mask, do):
    leaves = [x.detach().requires_grad_() for x in values]
    o = fn(*leaves, mask, chunk_i=a.chunk_i or None)
    gs = torch.autograd.grad(o, leaves, do)
    return (o.detach(), *(x.detach() for x in gs))


def reference(values, mask, do):
    leaves = [x.double().detach().requires_grad_() for x in values]
    q, k, v, b = leaves
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5 + b[:, :, None]
    o = scores.masked_fill(mask[:, None, :, None, :], -10000.0).softmax(-1) @ v
    return (o.detach(), *torch.autograd.grad(o, leaves, do.double()))


def check(x, y, dtype):
    x = x.double()
    diff = x - y
    norm = y.norm().item()
    maximum = diff.abs().max().item()
    rel = diff.norm().item() / max(norm, 1e-12)
    zero = torch.where(y.abs() <= 1e-12, diff.abs(), 0.0).max().item()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    limit = {torch.bfloat16: 0.012, torch.float16: 0.002, torch.float32: 2e-5}[dtype]
    return {
        "relative_l2": rel,
        "max_abs": maximum,
        "zero_max_abs": zero,
        "finite": finite,
        "passed": finite
        and zero <= 2e-5
        and (rel <= limit if norm > 1e-10 else maximum <= 2e-5),
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        a.dtype
    ]
    ns = list(map(int, a.shapes.split(",")))
    if not ns or min(ns) < 1:
        raise ValueError("positive sequence lengths required")
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "src/trifast").glob("*.py")), Path(__file__).resolve()]
    emit(
        event="start",
        args=vars(a),
        cache=str(cache),
        torch=torch.__version__,
        triton=triton.__version__,
        device=torch.cuda.get_device_name(),
        sources={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        counter_meaning="JIT keys track in-memory specializations; make_ptx observes actual backend code generation",
    )
    counters.clear()
    fn = (
        torch.compile(triangle_attention_fused, fullgraph=True, dynamic=True)
        if a.torch_compile
        else triangle_attention_fused
    )
    previous = inventory()
    compile_violations = failures = target_total = singleton_checks = 0
    for step, n in enumerate(ns):
        torch.manual_seed(20261010 + n)
        shape = (a.batch, a.heads, n, n, a.dim)
        values = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
        values.append(torch.randn(shape[:-1], device="cuda", dtype=dtype))
        mask = torch.rand((a.batch, n, n), device="cuda") < 0.2
        mask[:, 0] = True
        if n > 1:
            mask[:, 1] = True
            mask[:, 1, -1] = False
        do = torch.randn(shape, device="cuda", dtype=dtype).transpose(2, 3)
        if a.contiguous_do:
            do = do.contiguous()
        before_ptx = len(ptx_events)
        result = run(fn, values, mask, do)
        torch.cuda.synchronize()
        current = inventory()
        added = keys(current) - keys(previous)
        generated = ptx_events[before_ptx:]
        target_total += len(generated)
        if step == 0:
            expected_kernels = 3 if a.chunk_i == 0 else 4
            roles = (
                "_fwd_fused_optimized",
                "_preprocess",
                "_fused_bwd_k_owned" if a.chunk_i == 0 else "_chunk_bwd_k_owned",
                "_flush_dq",
            )[:expected_kernels]
            observed_roles = {
                role
                for role in roles
                for event in generated
                for entry in event["entry_names"]
                if role in entry
            }
            # Eager invokes the owned JITFunctions; Inductor can instead create
            # new JIT objects, so verify actual PTX target entry names there.
            instrumented = (
                len(added) >= expected_kernels
                if not a.torch_compile
                else len(observed_roles) >= expected_kernels
            )
            emit(
                event="instrumentation_check",
                expected_minimum_target_kernels=expected_kernels,
                new_jit_keys=len(added),
                observed_ptx_roles=sorted(observed_roles),
                backend_ptx_events=len(generated),
                passed=instrumented,
            )
            if not instrumented or not generated:
                raise RuntimeError(
                    "Target compilation instrumentation was empty or incomplete"
                )
        # Snapshot before the independent reference so reference matmul cannot
        # contaminate measured target compilation activity.
        checks = None
        if n <= a.reference_max_n:
            expected = reference(values, mask, do)
            checks = {
                name: check(x, y, dtype)
                for name, x, y in zip(
                    ("output", "dq", "dk", "dv", "db"), result, expected
                )
            }
            failures += not all(x["passed"] for x in checks.values())
            del expected
        new_after_first = step > 0 and bool(added or generated)
        compile_violations += new_after_first
        emit(
            event="step",
            step=step,
            n=n,
            new_n=n not in ns[:step],
            checks=checks,
            reference_tested=checks is not None,
            new_specialization_keys=sorted(added),
            new_backend_ptx_events=generated,
            jit_inventory=current,
            dynamo_counters={group: dict(values) for group, values in counters.items()},
            new_compilation_after_first=new_after_first,
        )
        previous = current
        if n == 1:
            # Exercise the runtime singleton branch with a valid key as well
            # as the all-masked first call. Keep dtype/D/B/H and all layouts.
            mask.zero_()
            before_ptx = len(ptx_events)
            singleton_result = run(fn, values, mask, do)
            torch.cuda.synchronize()
            singleton_inventory = inventory()
            singleton_added = keys(singleton_inventory) - keys(previous)
            singleton_generated = ptx_events[before_ptx:]
            target_total += len(singleton_generated)
            singleton_expected = reference(values, mask, do)
            singleton_metrics = {
                name: check(x, y, dtype)
                for name, x, y in zip(
                    ("output", "dq", "dk", "dv", "db"),
                    singleton_result,
                    singleton_expected,
                )
            }
            singleton_passed = all(
                metric["passed"] for metric in singleton_metrics.values()
            )
            failures += not singleton_passed
            singleton_checks += 1
            singleton_new_compile = bool(singleton_added or singleton_generated)
            compile_violations += singleton_new_compile
            emit(
                event="singleton_valid_key",
                step=step,
                n=n,
                checks=singleton_metrics,
                passed=singleton_passed,
                reference_tested=True,
                new_specialization_keys=sorted(singleton_added),
                new_backend_ptx_events=singleton_generated,
                jit_inventory=singleton_inventory,
                dynamo_counters={group: dict(v) for group, v in counters.items()},
                new_compilation_after_first=singleton_new_compile,
            )
            previous = singleton_inventory
            del singleton_result, singleton_expected
        del values, mask, do, result
    emit(
        event="summary",
        steps=len(ns),
        extra_singleton_valid_key_checks=singleton_checks,
        target_calls=len(ns) + singleton_checks,
        numerical_failures=failures,
        steps_with_new_compilation_after_first=compile_violations,
        target_total_backend_ptx_events=target_total,
        process_total_backend_ptx_events=len(ptx_events),
        no_new_compilation_after_first=compile_violations == 0,
        note="Dynamo unique_graphs/recompilations are reported separately; graph reuse is not inferred from Triton reuse",
    )
    raise SystemExit(
        1 if failures or (compile_violations and not a.allow_new_specializations) else 0
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        CUDABackend.make_ptx = original_descriptor
