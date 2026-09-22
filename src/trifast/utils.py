import subprocess
import sys
import time
from functools import wraps

import torch


def gpu_state(index: int = 0) -> tuple[int, int, int]:
    """(utilization %, memory used MiB, memory total MiB) for one device."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        text=True,
    )
    util, used, total = (int(x) for x in out.strip().split(", "))
    return util, used, total


def require_idle_gpu(max_util: int = 10, samples: int = 5, exit_on_busy: bool = True):
    """Refuse to benchmark on a busy GPU. Call this before any timing run.

    This device is shared with other users, and a concurrent process silently cost 28 %
    in one forward measurement round (see FWD_NCU_ANALYSIS.md) -- enough to make a real
    win read as a regression. Other users may be in a different container, so
    `--query-compute-apps` can come back empty while `memory.used` shows most of the card
    taken; utilization is the signal that matters for timing.

    Returns the samples so a caller can re-check afterwards and flag a run that had
    company partway through.
    """
    utils = []
    used = total = 0
    for _ in range(samples):
        util, used, total = gpu_state()
        utils.append(util)
        time.sleep(0.15)
    peak = max(utils)
    print(
        f"[gpu] utilization {utils} (peak {peak}%), memory {used}/{total} MiB used",
        flush=True,
    )
    if peak > max_util and exit_on_busy:
        sys.exit(
            f"[gpu] BUSY: peak utilization {peak}% > {max_util}%. Another job is using "
            "this GPU; timings would be meaningless. Re-run when it is idle."
        )
    return utils


def confirm_gpu_stayed_idle(label: str, max_util: int = 10) -> bool:
    """Did anyone else start a job while we were measuring?

    Sleeps first: utilization is sampled over a trailing window, so reading it straight
    after our own last launch reports our own work.
    """
    time.sleep(2.0)
    utils = [gpu_state()[0] for _ in range(4)]
    if max(utils) > max_util:
        print(
            f"[gpu] WARNING after {label}: utilization {utils} -- another job is "
            "running, so these numbers may be contaminated. Re-measure."
        )
        return False
    print(f"[gpu] still idle after {label} {utils} -- timings are clean.")
    return True


def gen_tensors(
    n: int,
    d: int,
    h: int,
    use_mask: bool,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 1,
    std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    k = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    v = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    b = torch.normal(
        0, std, (batch, h, n, n), device=device, dtype=dtype, requires_grad=True
    )
    m = (
        torch.randint(0, 2, (batch, n, n), device=device, dtype=torch.bool)
        if use_mask
        else torch.zeros((batch, n, n), device=device, dtype=torch.bool)
    )

    return q, k, v, b, m


def clone_and_clear_grad(*tensors):
    """
    Clone gradients of tensors and clear them.
    Returns a tuple of cloned gradients.
    """
    grads = tuple(t.grad.clone() if t.grad is not None else None for t in tensors)
    for t in tensors:
        t.grad = None
    return grads


def disable_tf32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        cuda, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            False,
            False,
        )
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                cuda,
                cudnn,
            )

    return wrapped


def enable_tf32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        cuda, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            True,
            True,
        )
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                cuda,
                cudnn,
            )

    return wrapped
