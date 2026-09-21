"""Compare complete public attention calls using interleaved CUDA-event samples.

Backward retains an existing graph; forward_backward rebuilds it each time.
Forward uses grad-enabled inputs, including saved statistics. Timings include
auxiliary launches, casts, copies, and host submission gaps. Compilation and
autotuning are warmed before timing. JSONL contains every sample and source hash.
"""

import argparse
import functools
import hashlib
import inspect
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton

from trifast import triangle_attention, triangle_attention_fused


def emit(**row):
    print(json.dumps(row), flush=True)


def clear(leaves):
    for group in leaves.values():
        for leaf in group:
            leaf.grad = None


def run(label, leaves, graphs, mode, selected, mask, do):
    # Explicit arguments avoid retaining a previous shape through closures.
    for leaf in leaves[label]:
        leaf.grad = None
    output = (
        graphs[label] if mode == "backward" else selected[label](*leaves[label], mask)
    )
    if mode != "forward":
        output.backward(do, retain_graph=(mode == "backward"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="512,640,768,800,1024")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--baseline", choices=("original", "full-workspace"), default="original"
    )
    parser.add_argument(
        "--candidate", choices=("full-workspace", "low-memory"), default="low-memory"
    )
    parser.add_argument("--chunk-i", type=int, default=128)
    parser.add_argument("--noncontiguous-do", action="store_true")
    parser.add_argument("--modes", default="forward,backward,forward_backward")
    args = parser.parse_args()
    shapes = [int(n) for n in args.shapes.split(",")]
    if (
        min(
            *shapes,
            args.batch,
            args.heads,
            args.rounds,
            args.samples,
            args.warmup,
            args.chunk_i,
        )
        < 1
    ):
        parser.error("shapes and counts must be positive")
    modes = args.modes.split(",")
    if any(mode not in ("forward", "backward", "forward_backward") for mode in modes):
        parser.error("unknown mode")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    torch.backends.cuda.matmul.allow_tf32 = False
    functions = {
        "original": triangle_attention,
        "full-workspace": functools.partial(triangle_attention_fused, chunk_i=None),
        "low-memory": functools.partial(triangle_attention_fused, chunk_i=args.chunk_i),
    }
    selected = {"A": functions[args.baseline], "B": functions[args.candidate]}
    root = Path(__file__).resolve().parents[1]
    if (
        Path(inspect.getfile(triangle_attention_fused)).resolve().parent
        != root / "src/trifast"
    ):
        parser.error(
            "imported trifast is not this checkout; install editable or set PYTHONPATH=src"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    source_files = list((root / "src/trifast").glob("*.py")) + [
        Path(__file__).resolve()
    ]
    emit(
        event="environment",
        args=vars(args),
        torch=torch.__version__,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        revision=revision.stdout.strip() or "unavailable",
        sources={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_files)
        },
        labels={"A": args.baseline, "B": args.candidate},
        mask_probability=0.2,
        timing="CUDA event per iteration; warm ABBA; no CUDA graphs",
    )
    for n in shapes:
        torch.manual_seed(734 + n)
        shape = (args.batch, args.heads, n, n, args.dim)
        values = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
        values.append(torch.randn(shape[:-1], device="cuda", dtype=dtype))
        mask = torch.rand((args.batch, n, n), device="cuda") < 0.2
        do = torch.randn(shape, device="cuda", dtype=dtype)
        if args.noncontiguous_do:
            do = do.transpose(2, 3)
        leaves = {
            label: [x.detach().requires_grad_() for x in values] for label in selected
        }

        diagnostic = {}
        for label, fn in selected.items():
            output = fn(*leaves[label], mask)
            gradients = torch.autograd.grad(output, leaves[label], do)
            diagnostic[label] = (output.detach(), *(g.detach() for g in gradients))
        checks = {}
        limit = {torch.bfloat16: 0.012, torch.float16: 0.002, torch.float32: 2e-5}[
            dtype
        ]
        for name, x, y in zip(
            ("output", "dq", "dk", "dv", "db"), diagnostic["B"], diagnostic["A"]
        ):
            x, y = x.float(), y.float()
            difference = x - y
            relative = (difference.norm() / y.norm().clamp_min(1e-12)).item()
            zero_error = torch.where(y == 0, difference.abs(), 0).max().item()
            passed = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            # A rounded baseline zero is not an exact FP64 reference zero.
            # Keep this difference as a diagnostic, not an absolute-error gate.
            # Independent FP64 tests/validate_fused.py enforce the zero bound.
            passed = passed and relative <= limit
            checks[name] = {
                "relative_l2": relative,
                "baseline_zero_max_abs_diagnostic": zero_error,
                "passed": passed,
            }
        emit(event="parity", n=n, checks=checks, independent_reference=False)
        if not all(row["passed"] for row in checks.values()):
            raise RuntimeError(
                "Diagnostic parity failed; refuse to benchmark this pair"
            )
        del diagnostic, output, gradients, x, y, difference
        clear(leaves)

        # Peak increment excludes preallocated inputs, includes retained O and
        # four leaf gradients, and is not process-total or allocator-reserved memory.
        peaks = {}
        for label, fn in selected.items():
            clear(leaves)
            output = fn(*leaves[label], mask)
            output.backward(do)
            del output
            clear(leaves)
            torch.cuda.synchronize()
            initial = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn(*leaves[label], mask)
            output.backward(do)
            torch.cuda.synchronize()
            peaks[label] = {
                "peak_increment_bytes": torch.cuda.max_memory_allocated() - initial,
                "retained_increment_bytes": torch.cuda.memory_allocated() - initial,
            }
            del output
            clear(leaves)
        emit(event="memory", n=n, measurements=peaks)
        for mode in modes:
            graphs = (
                {label: fn(*leaves[label], mask) for label, fn in selected.items()}
                if mode == "backward"
                else {}
            )

            for label in selected:
                for _ in range(args.warmup):
                    run(label, leaves, graphs, mode, selected, mask, do)
            torch.cuda.synchronize()
            samples = {label: [] for label in selected}
            for round_index in range(args.rounds):
                for block, label in enumerate("ABBA"):
                    events = []
                    for _ in range(args.samples):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        run(label, leaves, graphs, mode, selected, mask, do)
                        end.record()
                        events.append((start, end))
                    torch.cuda.synchronize()
                    for sample, (start, end) in enumerate(events):
                        ms = start.elapsed_time(end)
                        samples[label].append(ms)
                        emit(
                            event="sample",
                            n=n,
                            mode=mode,
                            round=round_index,
                            block=block,
                            label=label,
                            sample=sample,
                            ms=ms,
                        )
            medians = {
                label: statistics.median(times) for label, times in samples.items()
            }
            emit(
                event="summary",
                n=n,
                mode=mode,
                median_ms=medians,
                speedup=medians["A"] / medians["B"],
                samples_per_label=len(samples["A"]),
            )
            del graphs
            clear(leaves)
        del leaves, values, mask, do
        torch.cuda.empty_cache()
    emit(event="complete")


if __name__ == "__main__":
    main()
