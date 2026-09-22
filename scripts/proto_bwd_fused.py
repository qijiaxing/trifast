"""Prototype harness for the fused backward in `trifast.triton_bwd`.

Throwaway: pinned configs, no autotune, no disk cache, no autograd wiring. It exists to
answer two questions before the kernel is wired into `trifast.torch`:

  1. Is every gradient right, across the masked / ragged / dtype matrix the test suite
     pins?
  2. Which (BLOCK_J, BLOCK_K, num_warps, num_stages) config wins, and does the fused
     kernel actually beat the three-kernel baseline (62.4 ms at n=1024, h=8, d=32, bf16)?

The GPU is shared, so every timing run is gated on `require_idle_gpu`.

    python scripts/proto_bwd_fused.py check      # correctness matrix only
    python scripts/proto_bwd_fused.py bench      # config sweep at n=1024
    python scripts/proto_bwd_fused.py ablate     # price each fused piece
    python scripts/proto_bwd_fused.py regs       # registers/spills/smem, compile only
"""

import argparse
import math
import subprocess
import sys
import time

import torch
import triton
import triton.testing
from einops import rearrange

from triton.tools.tensor_descriptor import TensorDescriptor

import trifast.torch as trifast_torch
from trifast.torch import MASK_FILL, USE_TMA_BWD_BIAS
from trifast.torch import _triangle_attention, triangle_attention_bwd
from trifast.triton_bwd import (
    _bwd_bias_prep,
    _bwd_fused,
    _bwd_preprocess,
    _bwd_scale_cast,
)
from trifast.utils import gen_tensors

# ---------------------------------------------------------------- GPU guard


def gpu_state() -> tuple[int, int, int]:
    """(utilization %, memory used MiB, memory total MiB) for device 0."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        text=True,
    )
    util, used, total = (int(x) for x in out.strip().split(", "))
    return util, used, total


def require_idle_gpu(max_util: int = 10, samples: int = 5) -> None:
    """Refuse to benchmark on a busy GPU.

    This box is shared. A concurrent process silently cost 28 % in one forward
    measurement round (FWD_NCU_ANALYSIS.md), so a number taken next to someone else's
    job is worse than no number at all.
    """
    utils = []
    for _ in range(samples):
        util, used, total = gpu_state()
        utils.append(util)
        time.sleep(0.15)
    peak = max(utils)
    print(
        f"[gpu] utilization samples {utils} (peak {peak}%), "
        f"memory {used}/{total} MiB used",
        flush=True,
    )
    if peak > max_util:
        sys.exit(
            f"[gpu] BUSY: peak utilization {peak}% > {max_util}%. Someone else is using "
            "this GPU; timings would be meaningless. Re-run when it is idle."
        )


def report_gpu_after(label: str) -> None:
    """Did somebody else start a job while we were measuring?

    Sleep first: utilization is a sampled metric over a trailing window, so reading it
    immediately after our own last launch reports our own work.
    """
    time.sleep(2.0)
    utils = []
    for _ in range(4):
        utils.append(gpu_state()[0])
        time.sleep(0.15)
    if max(utils) > 10:
        print(f"[gpu] WARNING after {label}: utilization {utils} -- another job is "
              "running; the later rows above may be contaminated. Re-measure.")
    else:
        print(f"[gpu] still idle after {label} {utils} -- timings are clean.")


# ---------------------------------------------------------------- inputs


def fully_masked_mask(bs: int, n: int, device) -> torch.Tensor:
    """Random mask with a few fully-masked rows, as tests/unit/test_trifast.py builds."""
    m = torch.randint(0, 2, (bs, n, n), device=device, dtype=torch.bool)
    m[:, :, 0] = False  # keep one visible key so no *other* row is fully masked
    for bi in range(bs):
        for i in (0, n // 2, n - 1):
            m[bi, i, :] = True
    return m


def make_inputs(n, h, d, dtype, bs=1, mask_mode="random", seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q, k, v, b, mask = gen_tensors(
        n=n, d=d, h=h, use_mask=(mask_mode != "none"), device=device, dtype=dtype, batch=bs
    )
    if mask_mode == "fullrow":
        mask = fully_masked_mask(bs, n, device)
    elif mask_mode == "none":
        mask = torch.zeros((bs, n, n), device=device, dtype=torch.bool)
    o, _lse, mx, dn = _triangle_attention(q, k, v, b, mask)
    do = torch.randn_like(o)
    return dict(q=q, k=k, v=v, b=b, mask=mask, o=o, mx=mx, dn=dn, do=do,
                n=n, h=h, d=d, bs=bs, dtype=dtype)


def flatten(t):
    return rearrange(t, "b h ... -> (b h) ...").contiguous()


# ---------------------------------------------------------------- fused launch

DEFAULT_CFG = dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=3, maxnreg=None)


def launch_fused(inp, cfg, need_db=True, need_dq=True, stages=None, no_tma_bias=False):
    """Run preprocess -> fused -> cast. Returns (outputs, per-stage closures).

    `no_tma_bias` forces the pointer bias load so `bench` can A/B the TMA path.
    """
    n, h, d, bs, dtype = inp["n"], inp["h"], inp["d"], inp["bs"], inp["dtype"]
    q, k, v, b = (flatten(inp[x]) for x in ("q", "k", "v", "b"))
    o, mx, dn, do = (flatten(inp[x]) for x in ("o", "mx", "dn", "do"))
    mask = inp["mask"].contiguous()
    bh = q.shape[0]
    sm_scale = d**-0.5
    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))
    BJ, BK = cfg["BLOCK_J"], cfg["BLOCK_K"]

    delta = torch.empty((bh, n, n), dtype=torch.float32, device=q.device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # Atomic accumulators must start at zero: unlike the three-kernel path, not every
    # element is overwritten by a plain store. `reset_to_zero` does not cover this --
    # it only fires during autotune trials, never on a cached-config launch.
    dbt = torch.zeros((bh, n, n), dtype=torch.float32, device=q.device)
    dq_acc = torch.zeros((bh, n, n, d), dtype=torch.float32, device=q.device)
    s_dq_j, s_dq_d = dq_acc.stride(2), dq_acc.stride(3)
    dq = torch.empty_like(q)
    # The fp32, inv_ln2-scaled, [bh, k, j] bias, padded for 16-byte row alignment.
    padded_n = triton.cdiv(n, 16) * 16
    b2t = torch.zeros((bh, n, padded_n), dtype=torch.float32, device=q.device)
    # This harness launches the bare `_bwd_fused` with a pinned config, so no config
    # pre_hook runs -- unlike the autotuned path, the box has to be set here, to match
    # what _bwd_descriptor_pre_hook would have written.
    use_tma_bias = USE_TMA_BWD_BIAS and not no_tma_bias
    if use_tma_bias:
        desc_b2t = TensorDescriptor.from_tensor(
            b2t.reshape(bh * n, padded_n), block_shape=[BK, BJ]
        )
    else:
        desc_b2t = b2t

    def run_bias_prep():
        _bwd_bias_prep[(triton.cdiv(n, 32), triton.cdiv(n, 32), bh)](
            b, b.stride(0), b.stride(1), b.stride(2),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            n, BLOCK=32, num_warps=4,
        )

    def run_pre():
        _bwd_preprocess[(triton.cdiv(n, 64), n, bh)](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta, delta.stride(0), delta.stride(1), delta.stride(2),
            n, DIM=d, BLOCK_J=64, num_warps=4,
        )

    def run_fused():
        _bwd_fused[(triton.cdiv(n, BK), n, bh)](
            delta, delta.stride(0), delta.stride(1), delta.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
            mask, mask.stride(0), mask.stride(1), mask.stride(2),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            desc_b2t,
            dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dbt, dbt.stride(0), dbt.stride(1), dbt.stride(2),
            dq_acc, dq_acc.stride(0), dq_acc.stride(1), s_dq_j, s_dq_d,
            sm_scale=sm_scale, neg_inf=MASK_FILL,
            N=n, H=h, DIM=d, CLOSEST_N=CLOSEST_N,
            BLOCK_J=BJ, BLOCK_K=BK,
            NEED_DB=need_db, NEED_DQ=need_dq,
            USE_TMA_BIAS=use_tma_bias,
            num_warps=cfg["num_warps"],
            num_stages=stages if stages is not None else cfg["num_stages"],
            maxnreg=cfg.get("maxnreg"),
        )

    def run_cast():
        numel = dq_acc.numel()
        _bwd_scale_cast[(triton.cdiv(numel, 4096),)](
            dq_acc, dq, sm_scale, numel, BLOCK=4096, num_warps=4
        )

    def run_db_cast():
        return dbt.transpose(1, 2).contiguous().to(dtype)

    def run_all():
        run_bias_prep()
        run_pre()
        run_fused()
        run_cast()
        run_db_cast()

    outs = dict(dq=dq, dk=dk, dv=dv, dbt=dbt, dq_acc=dq_acc, delta=delta)
    stages_d = dict(bias_prep=run_bias_prep, pre=run_pre, fused=run_fused,
                    cast=run_cast, db_cast=run_db_cast, all=run_all)
    return outs, stages_d


def fused_grads(inp, cfg, **kw):
    """Materialize the four gradients through the fused path, in [b, h, ...] layout."""
    outs, st = launch_fused(inp, cfg, **kw)
    st["bias_prep"]()
    st["pre"]()
    st["fused"]()
    st["cast"]()
    db = st["db_cast"]()
    n, h, bs = inp["n"], inp["h"], inp["bs"]
    unflat = lambda t: rearrange(t, "(b h) ... -> b h ...", h=h, b=bs)
    return unflat(outs["dq"]), unflat(outs["dk"]), unflat(outs["dv"]), unflat(db)


def ref_grads(inp):
    """The three-kernel path, forced.

    `USE_FUSED_BWD` defaults to True, so without this override the "reference" was the
    fused path and `check` was comparing it against itself.
    """
    previous = trifast_torch.USE_FUSED_BWD
    trifast_torch.USE_FUSED_BWD = False
    try:
        dq, dk, dv, db, _ = triangle_attention_bwd(
            inp["do"], inp["q"], inp["k"], inp["v"], inp["b"],
            inp["o"], inp["mx"], inp["dn"], inp["mask"],
        )
    finally:
        trifast_torch.USE_FUSED_BWD = previous
    return dq, dk, dv, db


# ---------------------------------------------------------------- checks


def rel(new, ref):
    new, ref = new.float(), ref.float()
    denom = ref.abs().max().item()
    return (new - ref).abs().max().item() / (denom if denom else 1.0)


def cmd_check(args):
    cases = []
    for n, d in [(128, 32), (100, 32), (17, 32), (65, 32), (16, 16), (16, 64), (16, 128)]:
        for dtype in (torch.bfloat16, torch.float32):
            for mode in ("random", "fullrow"):
                cases.append((n, 2, d, dtype, mode))
    cases.append((800, 8, 32, torch.bfloat16, "random"))  # ragged at BLOCK_J=64
    cases.append((256, 4, 32, torch.bfloat16, "none"))

    cfgs = [dict(DEFAULT_CFG), dict(DEFAULT_CFG, BLOCK_J=32, BLOCK_K=32)]
    bad = 0
    print(f"{'n':>5}{'h':>3}{'d':>5}{'dtype':>10}{'mask':>9}{'BJ/BK':>8}"
          f"{'dq':>10}{'dk':>10}{'dv':>10}{'db':>10}")
    for (n, h, d, dtype, mode) in cases:
        bs = 2 if mode == "fullrow" else 1
        inp = make_inputs(n, h, d, dtype, bs=bs, mask_mode=mode)
        r = ref_grads(inp)
        for cfg in cfgs:
            try:
                g = fused_grads(inp, cfg)
            except triton.runtime.errors.OutOfResources as e:
                # A legitimate outcome, not a bug: autotune prunes such configs. fp32
                # takes the non-tensor-core `ieee` path, which stages both dot operands
                # through shared memory, so DIM=128 at 64x64 needs 255 KB of 232 KB.
                print(f"{n:>5}{h:>3}{d:>5}{str(dtype).split('.')[-1]:>10}{mode:>9}"
                      f"{cfg['BLOCK_J']:>4}/{cfg['BLOCK_K']:<3}   out of resources: "
                      f"{e.required} > {e.limit} B smem")
                continue
            errs = [rel(a, b) for a, b in zip(g, r)]
            flag = "" if max(errs) < 2e-2 else "  <-- FAIL"
            if max(errs) >= 2e-2:
                bad += 1
            print(f"{n:>5}{h:>3}{d:>5}{str(dtype).split('.')[-1]:>10}{mode:>9}"
                  f"{cfg['BLOCK_J']:>4}/{cfg['BLOCK_K']:<3}"
                  + "".join(f"{e:>10.2e}" for e in errs) + flag)
            for name, t in zip(("dq", "dk", "dv", "db"), g):
                assert torch.isfinite(t).all(), f"{name} has non-finite values"
    print("FAILURES:", bad)
    return bad


def cmd_bench(args):
    require_idle_gpu()
    n, h, d = args.n, 8, 32
    inp = make_inputs(n, h, d, torch.bfloat16, mask_mode="random")
    # Force the three-kernel path: USE_FUSED_BWD defaults to True, so this used to time
    # the fused path and label it "reference".
    def ref():
        previous = trifast_torch.USE_FUSED_BWD
        trifast_torch.USE_FUSED_BWD = False
        try:
            return triangle_attention_bwd(
                inp["do"], inp["q"], inp["k"], inp["v"], inp["b"],
                inp["o"], inp["mx"], inp["dn"], inp["mask"])
        finally:
            trifast_torch.USE_FUSED_BWD = previous

    ref(); torch.cuda.synchronize()
    ref_ms = triton.testing.do_bench(ref, warmup=100, rep=500, quantiles=[0.5])
    print(f"reference three-kernel path (incl. rearrange): {ref_ms:.2f} ms\n")

    # BLOCK_K < 64 is excluded on purpose: BLOCK_K is the M dimension of the dv/dk dots
    # in this orientation, and M=32 drops off wgmma onto mma.sync -- measured 119 ms at
    # 64x32 against 40 ms at 64x64. num_warps=8 (85 ms) and maxnreg caps (spills) were
    # measured and lose too.
    sweep = [
        dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=2, maxnreg=None),
        dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=3, maxnreg=None),
        dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=1, maxnreg=None),
        dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=4, maxnreg=None),
        dict(BLOCK_J=32, BLOCK_K=64, num_warps=4, num_stages=2, maxnreg=None),
        dict(BLOCK_J=128, BLOCK_K=64, num_warps=4, num_stages=2, maxnreg=None),
        dict(BLOCK_J=64, BLOCK_K=128, num_warps=4, num_stages=2, maxnreg=None),
    ]
    hdr = (f"{'BJ':>4}{'BK':>4}{'w':>3}{'s':>3}{'nreg':>6}{'fused':>9}{'pre':>7}"
           f"{'bias':>7}{'cast':>7}{'total':>8}{'vs ref':>8}{'regs':>6}{'spill':>6}"
           f"{'smem':>8}")
    print(hdr)
    best = None
    for cfg in sweep + [dict(c, _no_tma_bias=True) for c in sweep[:2]]:
        no_tma = cfg.pop("_no_tma_bias", False)
        try:
            outs, st = launch_fused(inp, cfg, no_tma_bias=no_tma)
            st["all"](); torch.cuda.synchronize()
            ms_f = triton.testing.do_bench(st["fused"], warmup=100, rep=400, quantiles=[0.5])
            ms_p = triton.testing.do_bench(st["pre"], warmup=20, rep=100, quantiles=[0.5])
            ms_c = triton.testing.do_bench(st["cast"], warmup=20, rep=100, quantiles=[0.5])
            ms_d = triton.testing.do_bench(st["db_cast"], warmup=20, rep=100, quantiles=[0.5])
            ms_b = triton.testing.do_bench(st["bias_prep"], warmup=20, rep=100, quantiles=[0.5])
            meta = last_meta(_bwd_fused)
            total = ms_f + ms_p + ms_c + ms_d + ms_b
            print(f"{cfg['BLOCK_J']:>4}{cfg['BLOCK_K']:>4}{cfg['num_warps']:>3}"
                  f"{cfg['num_stages']:>3}{str(cfg['maxnreg']):>6}{ms_f:>9.2f}{ms_p:>7.2f}"
                  f"{ms_b:>7.2f}{ms_c + ms_d:>7.2f}{total:>8.2f}{ref_ms / total:>7.2f}x"
                  f"{meta.n_regs:>6}{meta.n_spills:>6}{meta.metadata.shared:>8}"
                  f"{'  pointer-bias' if no_tma else ''}")
            # The pointer-bias rows are an A/B, not candidates -- they would otherwise win
            # the `best` line without the label that says what they are.
            if not no_tma and (best is None or total < best[0]):
                best = (total, cfg)
        except Exception as e:  # noqa: BLE001 - a throwaway harness
            print(f"{cfg['BLOCK_J']:>4}{cfg['BLOCK_K']:>4}{cfg['num_warps']:>3}"
                  f"{cfg['num_stages']:>3} FAILED: {str(e)[:60]}")
    report_gpu_after("bench")
    print(f"\nbest: {best[1]} -> {best[0]:.2f} ms ({ref_ms / best[0]:.2f}x)")


def cmd_ablate(args):
    require_idle_gpu()
    n, h, d = args.n, 8, 32
    inp = make_inputs(n, h, d, torch.bfloat16, mask_mode="random")
    variants = [
        ("dk+dv only", dict(need_db=False, need_dq=False)),
        ("dk+dv+db", dict(need_db=True, need_dq=False)),
        ("dk+dv+dq", dict(need_db=False, need_dq=True)),
        ("everything", dict(need_db=True, need_dq=True)),
    ]
    for cfg in (dict(DEFAULT_CFG, num_stages=2), dict(DEFAULT_CFG, BLOCK_J=32, num_stages=2)):
        print(f"\nBLOCK_J={cfg['BLOCK_J']} BLOCK_K={cfg['BLOCK_K']} "
              f"w{cfg['num_warps']} s{cfg['num_stages']}")
        print(f"  {'variant':<26}{'ms':>9}{'regs':>6}{'spill':>6}{'smem':>8}")
        for name, kw in variants:
            outs, st = launch_fused(inp, cfg, **kw)
            st["bias_prep"](); st["fused"](); torch.cuda.synchronize()
            ms = triton.testing.do_bench(st["fused"], warmup=100, rep=400, quantiles=[0.5])
            meta = last_meta(_bwd_fused)
            print(f"  {name:<26}{ms:>9.2f}{meta.n_regs:>6}{meta.n_spills:>6}"
                  f"{meta.metadata.shared:>8}")
    report_gpu_after("ablate")


def cmd_regs(args):
    """Compile-only resource report: registers, spills, shared memory, CTAs/SM.

    Register pressure is a function of the constexprs, not of N, so this compiles at a
    tiny shape and takes seconds rather than the minutes `bench` needs. It is the gate for
    anything that wants to spend registers -- BLOCK_I > 1 costs 32 reg/thread per extra i
    (a second dk and dv accumulator), and 2 CTAs/SM at 128 threads allows only
    65536 / (2 * 128) = 256.
    """
    inp = make_inputs(64, 2, 32, torch.bfloat16, mask_mode="random")
    variants = [
        ("everything", {}),
        ("no db atomic", dict(need_db=False)),
        ("no dq atomic", dict(need_dq=False)),
        ("dk/dv only", dict(need_db=False, need_dq=False)),
    ]
    print(f"{'BJ':>4}{'BK':>4}{'w':>3}{'s':>3}  {'variant':<16}"
          f"{'regs':>6}{'spill':>6}{'smem':>8}{'CTA/SM':>8}{'headroom':>9}")
    for cfg in (dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=2, maxnreg=None),
                dict(BLOCK_J=64, BLOCK_K=64, num_warps=4, num_stages=3, maxnreg=None),
                dict(BLOCK_J=32, BLOCK_K=64, num_warps=4, num_stages=2, maxnreg=None)):
        for name, kw in variants:
            _outs, st = launch_fused(inp, cfg, **kw)
            st["fused"]()
            torch.cuda.synchronize()
            m = last_meta(_bwd_fused)
            threads = cfg["num_warps"] * 32
            by_reg = 65536 // (m.n_regs * threads)
            by_smem = 233472 // max(m.metadata.shared, 1)
            # Registers spare before dropping below the CTAs/SM we have today.
            cap = 65536 // (max(by_reg, 1) * threads)
            print(f"{cfg['BLOCK_J']:>4}{cfg['BLOCK_K']:>4}{cfg['num_warps']:>3}"
                  f"{cfg['num_stages']:>3}  {name:<16}{m.n_regs:>6}{m.n_spills:>6}"
                  f"{m.metadata.shared:>8}{min(by_reg, by_smem):>8}{cap - m.n_regs:>9}")


def last_meta(kern):
    """Resource footprint of the most recently compiled variant of `kern`."""
    cache = list(kern.device_caches.values())[0][0]
    return list(cache.values())[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "bench", "ablate", "regs"])
    p.add_argument("-n", type=int, default=1024)
    args = p.parse_args()
    {"check": cmd_check, "bench": cmd_bench, "ablate": cmd_ablate,
     "regs": cmd_regs}[args.cmd](args)


if __name__ == "__main__":
    main()
