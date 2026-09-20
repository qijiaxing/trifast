"""Benchmark the individual Triton kernels used by TriFast.

The forward kernel produces the softmax statistics consumed by the backward
kernels.  The backward-q kernel additionally produces ``delta``, which is an
input to the backward-kv and backward-bias kernels.  Those prerequisite
launches happen before timing starts.
"""

import argparse
import math
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.testing
from triton.tools.tensor_descriptor import TensorDescriptor

from trifast.autotune_helpers import device_name
from trifast.torch import MASK_FILL, USE_TMA, USE_TMA_BIAS
from trifast.triton import _bwd_b, _bwd_kv, _bwd_q, _fwd
from trifast.utils import gen_tensors

N_VALUES = [512, 640, 768, 800, 1024]
DTYPES = [torch.bfloat16]
KERNELS = ("fwd", "bwd_q", "bwd_kv", "bwd_b")
KERNEL_NAMES = {
    "fwd": "Forward",
    "bwd_q": "Backward Q",
    "bwd_kv": "Backward K/V",
    "bwd_b": "Backward Bias",
}
KERNEL_STYLES = {
    "fwd": ("blue", "-"),
    "bwd_q": ("green", "-"),
    "bwd_kv": ("orange", "-"),
    "bwd_b": ("red", "-"),
}

# Number of matrix multiplications performed by each kernel.  One matrix
# multiplication costs 2 * batch * h * n^3 * d FLOPs.  This is the standard
# attention benchmark convention: pointwise softmax operations are not counted.
MATMULS_PER_KERNEL = {
    "fwd": 2,  # QK^T and PV
    "bwd_q": 3,  # QK^T recomputation, dO V^T, and dS K
    "bwd_kv": 4,  # QK^T recomputation, P^T dO, dO V^T, and dS^T Q
    "bwd_b": 2,  # QK^T recomputation and dO V^T
}


def _kernel_tflops(n: int, h: int, d: int, kernel: str, ms: float) -> float:
    """Return algorithmic matrix-multiply throughput in decimal TFLOP/s."""
    flops_per_matmul = 2 * h * n**3 * d  # batch is fixed at one
    total_flops = MATMULS_PER_KERNEL[kernel] * flops_per_matmul
    return total_flops * 1e-9 / ms


def _report(
    dtype: torch.dtype,
    kernels: tuple[str, ...],
) -> triton.testing.Benchmark:
    selection = "all" if kernels == KERNELS else "-".join(kernels)
    return triton.testing.Benchmark(
        x_names=["n"],
        x_vals=N_VALUES,
        line_arg="kernel",
        line_vals=list(kernels),
        line_names=[KERNEL_NAMES[kernel] for kernel in kernels],
        styles=[KERNEL_STYLES[kernel] for kernel in kernels],
        ylabel="TFLOP/s",
        plot_name=f"tri_attn_kernels_{selection}_{dtype}",
        args={"dtype": dtype},
    )


def _make_launchers(
    n: int,
    h: int,
    d: int,
    dtype: torch.dtype,
) -> dict[str, Callable[[], None]]:
    """Allocate kernel buffers and return one launcher per TriFast kernel."""
    q, k, v, bias, mask = gen_tensors(
        n=n,
        h=h,
        d=d,
        dtype=dtype,
        device=torch.device("cuda"),
        use_mask=True,
    )

    # The kernels combine the batch and head dimensions.  The mask deliberately
    # remains [batch, n, n], because kernels map a combined head back to its batch
    # with pid_h // H.
    q = q.flatten(0, 1).contiguous()
    k = k.flatten(0, 1).contiguous()
    v = v.flatten(0, 1).contiguous()
    bias = bias.flatten(0, 1).contiguous()
    mask = mask.contiguous()

    bh = q.shape[0]
    sm_scale = d**-0.5
    closest_n = 2 ** math.ceil(math.log2(n))

    o = torch.empty_like(q)
    # _fwd uses one set of strides for all three statistics tensors.
    lse = torch.empty((bh, n, n), device=q.device, dtype=torch.float32)
    mx = torch.zeros_like(lse)
    dn = torch.empty_like(lse)

    do = torch.randn_like(o)
    delta = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty_like(bias)
    use_tma_bias = USE_TMA_BIAS and d <= 64
    if use_tma_bias:
        bias_alignment = 16 // bias.element_size()
        padded_n = triton.cdiv(n, bias_alignment) * bias_alignment
        padded_bias = torch.nn.functional.pad(bias, (0, padded_n - n))
        desc_b = TensorDescriptor.from_tensor(
            padded_bias.reshape(bh * n, padded_n), block_shape=[64, 32]
        )
    else:
        desc_b = bias

    # Keep this in sync with trifast.torch._triangle_attention.
    use_tma = USE_TMA and d * q.element_size() % 16 == 0
    if use_tma:
        desc_q = TensorDescriptor.from_tensor(q, block_shape=[1, 1, 64, 32])
        desc_k = TensorDescriptor.from_tensor(k, block_shape=[1, 1, 64, 32])
        desc_v = TensorDescriptor.from_tensor(v, block_shape=[1, 1, 64, 32])
        desc_o = TensorDescriptor.from_tensor(o, block_shape=[1, 1, 64, 32])
    else:
        desc_q, desc_k, desc_v, desc_o = q, k, v, o

    def fwd_grid(meta):
        return (triton.cdiv(n, meta["BLOCK_J"]), n, bh)

    def bwd_q_grid(meta):
        return (triton.cdiv(n, meta["BLOCK_J"]), n, bh)

    def bwd_kv_grid(meta):
        return (triton.cdiv(n, meta["BLOCK_K"]), n, bh)

    def bwd_b_grid(meta):
        return (
            triton.cdiv(n, meta["BLOCK_J"]),
            triton.cdiv(n, meta["BLOCK_K"]),
            bh,
        )

    def run_fwd() -> None:
        # Keep this argument list in sync with trifast.torch._triangle_attention.
        _fwd[fwd_grid](
            o,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            lse,
            mx,
            dn,
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias,
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            mask,
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            desc_b,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            sm_scale=sm_scale,
            neg_inf=MASK_FILL,
            N=n,
            H=h,
            DIM=d,
            CLOSEST_N=closest_n,
            USE_TMA=use_tma,
            USE_TMA_BIAS=use_tma_bias,
        )

    def run_bwd_q() -> None:
        # Besides dq, this kernel produces delta for bwd_kv and bwd_b.
        _bwd_q[bwd_q_grid](
            delta,
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias,
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            mx,
            dn,
            mx.stride(0),
            mx.stride(1),
            mx.stride(2),
            mask,
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            o,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dq,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            sm_scale=sm_scale,
            neg_inf=MASK_FILL,
            H=h,
            N=n,
            DIM=d,
            CLOSEST_N=closest_n,
        )

    def run_bwd_kv() -> None:
        _bwd_kv[bwd_kv_grid](
            delta,
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias,
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            mx,
            dn,
            mx.stride(0),
            mx.stride(1),
            mx.stride(2),
            mask,
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            do,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dk,
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv,
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            sm_scale=sm_scale,
            neg_inf=MASK_FILL,
            H=h,
            N=n,
            DIM=d,
            CLOSEST_N=closest_n,
        )

    def run_bwd_b() -> None:
        _bwd_b[bwd_b_grid](
            delta,
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias,
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            mx,
            dn,
            mx.stride(0),
            mx.stride(1),
            mx.stride(2),
            mask,
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            do,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            db,
            db.stride(0),
            db.stride(1),
            db.stride(2),
            sm_scale=sm_scale,
            neg_inf=MASK_FILL,
            H=h,
            N=n,
            DIM=d,
            CLOSEST_N=closest_n,
        )

    return {
        "fwd": run_fwd,
        "bwd_q": run_bwd_q,
        "bwd_kv": run_bwd_kv,
        "bwd_b": run_bwd_b,
    }


def benchmark(n, dtype, kernel):
    """Measure only ``kernel``; prerequisite kernel launches are not timed."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel: {kernel}")

    # AlphaFold 3 uses d=32.  Keep h=8 to match the intended kernel benchmark.
    d = 32
    h = 8

    try:
        launchers = _make_launchers(n=n, h=h, d=d, dtype=dtype)

        # Populate o/mx/dn.  All backward kernels consume these values.
        launchers["fwd"]()
        if kernel != "fwd":
            # Populate delta.  This also compiles/tunes bwd_q before bwd_q itself
            # is timed and supplies the input needed by bwd_kv and bwd_b.
            launchers["bwd_q"]()
        torch.cuda.synchronize()

        ms, min_ms, max_ms = triton.testing.do_bench(
            launchers[kernel],
            warmup=5,
            rep=100,
            quantiles=[0.5, 0.1, 0.9],
        )

        # Throughput is inversely proportional to runtime, so the runtime bounds
        # must be swapped when converted to throughput bounds.
        tflops = _kernel_tflops(n, h, d, kernel, ms)
        min_tflops = _kernel_tflops(n, h, d, kernel, max_ms)
        max_tflops = _kernel_tflops(n, h, d, kernel, min_ms)
        return tflops, min_tflops, max_tflops
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 0.0, 0.0, 0.0


def _print_table(
    title: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> None:
    """Print a compact, right-aligned Unicode table."""
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def row(values: tuple[str, ...], centered: bool = False) -> str:
        alignment = "^" if centered else ">"
        cells = [
            f" {value:{alignment}{width}} " for value, width in zip(values, widths)
        ]
        return "│" + "│".join(cells) + "│"

    print(f"\n{title}")
    print(border("┌", "┬", "┐"))
    print(row(headers, centered=True))
    print(border("├", "┼", "┤"))
    for values in rows:
        print(row(values))
    print(border("└", "┴", "┘"))


def _print_perf_tables(result_dfs, kernels: tuple[str, ...]) -> None:
    """Print one readable throughput table for each dtype."""
    headers = ("N", *(KERNEL_NAMES[kernel] for kernel in kernels))

    for dtype, result_df in zip(DTYPES, result_dfs):
        rows = [
            (
                str(int(result_df.iloc[row_index, 0])),
                *(
                    f"{result_df.iloc[row_index, column_index]:.2f}"
                    for column_index in range(1, len(kernels) + 1)
                ),
            )
            for row_index in range(len(result_df))
        ]
        dtype_name = {
            torch.bfloat16: "BF16",
            torch.float16: "FP16",
            torch.float32: "FP32",
        }.get(dtype, str(dtype).removeprefix("torch.").upper())
        _print_table(
            f"TriFast individual-kernel throughput — {dtype_name} (TFLOP/s)",
            headers,
            rows,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark individual TriFast Triton kernels.",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=list(KERNELS),
        help="kernels to benchmark (default: all four)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Preserve the requested order while avoiding duplicate benchmark runs.
    kernels = tuple(dict.fromkeys(args.kernels))
    configs = [_report(dtype, kernels) for dtype in DTYPES]
    runner = triton.testing.perf_report(configs)(benchmark)

    out_dir = Path(__file__).parent.parent
    save_path = out_dir / "benchmark" / device_name / "flops"
    save_path.mkdir(parents=True, exist_ok=True)
    result_dfs = runner.run(
        print_data=False,
        show_plots=False,
        save_path=str(save_path),
        save_precision=2,
        return_df=True,
    )
    _print_perf_tables(result_dfs, kernels)


if __name__ == "__main__":
    main()
