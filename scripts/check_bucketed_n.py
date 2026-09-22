"""Verify power-of-two bucket reuse with JIT inventories and PTX counts.

Run eagerly first; use --torch-compile in a separate process to distinguish
Dynamo graph specialization from Triton specialization. Every process uses
fresh disk caches by default; never infer reuse from timing alone. The first
call of each power-of-two bucket may tune multiple configurations. Subsequent
calls in that bucket must add no target JIT keys or backend PTX generation.
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
p.add_argument(
    "--shapes", default="257,300,500,512,257,513,800,1024,513,129,200,256,129,1,1"
)
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
import trifast._fused_forward_tma as tma_module
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
    for module in (forward_module, tma_module, backward_module, chunk_module):
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


TARGET_ROLES = {
    "_fwd_fused_optimized"
    if a.dtype == "fp32" or torch.cuda.get_device_capability()[0] < 9
    else "_fused_tma",
    "_preprocess",
    "_fused_bwd_k_owned" if a.chunk_i == 0 else "_chunk_bwd_k_owned",
}
if a.chunk_i != 0:
    TARGET_ROLES.add("_flush_dq")


def ptx_roles(events):
    return {
        role
        for role in TARGET_ROLES
        for event in events
        for entry in event["entry_names"]
        if entry == role or entry.startswith(role + "_")
    }


def target_ptx_events(events):
    return [event for event in events if ptx_roles([event])]


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
        counter_meaning="JIT keys track owned in-memory specializations; target PTX counts filter entry names by expected kernel roles; process PTX counts include every backend event",
    )
    counters.clear()
    fn = (
        torch.compile(triangle_attention_fused, fullgraph=True, dynamic=True)
        if a.torch_compile
        else triangle_attention_fused
    )
    previous = inventory()
    compile_violations = failures = target_total = singleton_checks = 0
    buckets = {}
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
        bucket = 1 << (n - 1).bit_length()
        first_in_bucket = bucket not in buckets
        if first_in_bucket:
            buckets[bucket] = {"steps": [], "new_keys": 0, "ptx_events": 0}
        before_ptx = len(ptx_events)
        result = run(fn, values, mask, do)
        torch.cuda.synchronize()
        current = inventory()
        added = keys(current) - keys(previous)
        generated = target_ptx_events(ptx_events[before_ptx:])
        target_total += len(generated)
        if step == 0:
            expected_kernels = len(TARGET_ROLES)
            observed_roles = ptx_roles(generated)
            inventory_roles = {
                name.rsplit(".", 1)[-1] for name, _, _ in added
            } & TARGET_ROLES
            # Multiple configurations of one role must not hide missing roles.
            # Eager requires both owned JIT inventories and actual backend
            # lowering for every role. Inductor may create different JIT objects,
            # so its complete target coverage is checked by PTX entry names.
            instrumented = TARGET_ROLES <= observed_roles and (
                a.torch_compile or TARGET_ROLES <= inventory_roles
            )
            emit(
                event="instrumentation_check",
                expected_minimum_target_kernels=expected_kernels,
                new_jit_keys=len(added),
                expected_roles=sorted(TARGET_ROLES),
                observed_inventory_roles=sorted(inventory_roles),
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
        buckets[bucket]["steps"].append({"step": step, "n": n})
        buckets[bucket]["new_keys"] += len(added)
        buckets[bucket]["ptx_events"] += len(generated)
        new_after_first = not first_in_bucket and bool(added or generated)
        compile_violations += new_after_first
        emit(
            event="step",
            step=step,
            n=n,
            new_n=n not in ns[:step],
            bucket=bucket,
            first_in_bucket=first_in_bucket,
            tuning_and_new_specializations_allowed=first_in_bucket,
            checks=checks,
            reference_tested=checks is not None,
            new_specialization_keys=sorted(added),
            new_backend_ptx_events=generated,
            jit_inventory=current,
            dynamo_counters={group: dict(values) for group, values in counters.items()},
            new_compilation_in_reused_bucket=new_after_first,
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
            singleton_generated = target_ptx_events(ptx_events[before_ptx:])
            target_total += len(singleton_generated)
            buckets[bucket]["new_keys"] += len(singleton_added)
            buckets[bucket]["ptx_events"] += len(singleton_generated)
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
                bucket=bucket,
                step=step,
                n=n,
                checks=singleton_metrics,
                passed=singleton_passed,
                reference_tested=True,
                new_specialization_keys=sorted(singleton_added),
                new_backend_ptx_events=singleton_generated,
                jit_inventory=singleton_inventory,
                dynamo_counters={group: dict(v) for group, v in counters.items()},
                new_compilation_in_reused_bucket=singleton_new_compile,
            )
            previous = singleton_inventory
            del singleton_result, singleton_expected
        del values, mask, do, result
    emit(
        event="summary",
        steps=len(ns),
        bucket_count=len(buckets),
        buckets=buckets,
        final_jit_inventory=previous,
        extra_singleton_valid_key_checks=singleton_checks,
        target_calls=len(ns) + singleton_checks,
        numerical_failures=failures,
        calls_with_new_compilation_in_reused_bucket=compile_violations,
        target_total_backend_ptx_events=target_total,
        process_total_backend_ptx_events=len(ptx_events),
        no_new_compilation_in_reused_buckets=compile_violations == 0,
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
