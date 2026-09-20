import os
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from pathlib import Path
import platformdirs
from importlib.metadata import version

FORCE_TUNE = os.getenv("TRIFAST_FORCE_TUNE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

device_capability = torch.cuda.get_device_capability()
device_capability = f"{device_capability[0]}-{device_capability[1]}"

device_name = torch.cuda.get_device_name().replace(" ", "-")


def get_config_dir() -> Path:
    config_dir = Path(
        platformdirs.user_config_dir(appname="trifast", version=version("trifast")),
        ensure_exists=False,
    )

    if config_dir.exists():
        return config_dir

    # If it doesn't exist, this is a fresh install.
    config_dir.mkdir(parents=True, exist_ok=True)

    return config_dir


config_dir = get_config_dir()


def config_to_dict(config: triton.Config) -> dict:
    # This assume we are not making use of `pre_hook` in the `triton.Config`
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def dict_to_config(d: dict) -> triton.Config:
    return triton.Config(
        kwargs=d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg"),
    )


def _fwd_descriptor_pre_hook(nargs):
    """Match the host TMA tiles to the selected forward configuration.

    Mutating ``block_shape`` changes the kernel's specialization key, so the
    launch rebuilds the device tensormap with the matching box. The values
    set here are placeholders; this hook rewrites them per config, both
    during tuning (before each candidate's timed run) and on every launch.
    """
    block_j = nargs["BLOCK_J"]
    block_k = nargs["BLOCK_K"]
    dim = nargs["DIM"]
    tiles = {
        # [bh, n, n, dim] tensors: a box of rows (j or k) inside one (h, i)
        # slice, so the hardware clips rows >= n instead of wrapping into
        # the neighbouring slice.
        "desc_q": [1, 1, block_j, dim],
        "desc_k": [1, 1, block_k, dim],
        "desc_v": [1, 1, block_k, dim],
        "desc_o": [1, 1, block_j, dim],
        # Bias is the padded [bh, n, padded_n] tensor reshaped 2D.
        "desc_b": [block_j, block_k],
    }
    for name, block_shape in tiles.items():
        desc = nargs.get(name)
        if isinstance(desc, TensorDescriptor) and desc.block_shape != block_shape:
            desc.block_shape = block_shape


# Base configs targeting H20 and shared by the TMA and pointer kernels.
_fwd_common_configs = [
    triton.Config(
        kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3, maxnreg=80
    ),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=8, num_stages=1),
]

# TMA q/k/v/o pays a fixed per-iteration cost in barrier and multi-buffer
# shared-memory addressing, which only amortizes over wider K tiles. Keep
# these candidates out of the traced/fake-tensor pointer fallback.
_fwd_tma_configs = [
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
]

_fwd_configs = [*_fwd_common_configs, *_fwd_tma_configs]


def prune_fwd_configs(configs, named_args, **kwargs):
    """The H20 register cap cannot compile DIM=128."""
    if kwargs["DIM"] <= 64:
        return configs
    return [config for config in configs if config.maxnreg is None]


_fwd_force_tune_configs = []
if FORCE_TUNE:
    _fwd_force_tune_configs = [
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=5),
        triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=5),
        triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    ]
    _fwd_configs.extend(_fwd_force_tune_configs)

for config in _fwd_configs:
    config.pre_hook = _fwd_descriptor_pre_hook

# torch.compile currently rejects autotuners carrying config hooks. This hook-free
# clone shares the same kernel body and is used by traced/fake-tensor execution.
# The exhaustive FORCE_TUNE set predates TMA and remains available to both paths.
_fwd_pointer_configs = [
    triton.Config(
        kwargs=dict(config.kwargs),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        maxnreg=config.maxnreg,
    )
    for config in [*_fwd_common_configs, *_fwd_force_tune_configs]
]


_bwd_q_configs = [
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]
if FORCE_TUNE:
    _bwd_q_configs.extend(
        [
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        ]
    )


_bwd_kv_configs = [
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
]

if FORCE_TUNE:
    _bwd_kv_configs.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        ]
    )

_bwd_b_configs = [
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
]

if FORCE_TUNE:
    _bwd_b_configs.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=4, num_stages=4),
        ]
    )
