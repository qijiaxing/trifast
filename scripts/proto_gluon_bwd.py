"""Check and time `trifast.gluon_bwd._gl_bwd_fused` against the shipping Triton kernel.

Throwaway, same spirit as `proto_bwd_fused.py`: pinned config, no autotune, no autograd
wiring. It answers one question -- does hand-scheduling the wgmma waits in Gluon beat what
Triton's wait-insertion pass generates?

    python scripts/proto_gluon_bwd.py check      # gradients vs the three-kernel path
    python scripts/proto_gluon_bwd.py bench      # bias dtype x stage depth sweep vs triton
    python scripts/proto_gluon_bwd.py sass       # DEPBAR / HGMMA counts, compile only
    python scripts/proto_gluon_bwd.py ncu        # launch under ncu (prereqs then 3 reps)

`bench` sweeps the staged bias dtype against the stage depth, which is the axis that moved
the kernel from 0.89x to 0.95x. The defaults here (bf16 bias, unscaled, STAGES=5) are the
winning point; `stages=4` is a reproducible scheduling cliff and is in the table to keep it
from being rediscovered.
"""

import argparse
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from scripts.proto_bwd_fused import (  # noqa: E402
    DEFAULT_CFG,
    flatten,
    launch_fused,
    make_inputs,
    ref_grads,
    report_gpu_after,
    require_idle_gpu,
)
from trifast.gluon_bwd import _gl_bwd_fused, gluon_bwd_stages  # noqa: E402
from trifast.torch import MASK_FILL  # noqa: E402
from trifast.triton_bwd import _bwd_bias_prep, _bwd_preprocess, _bwd_scale_cast  # noqa: E402

BJ, BK = 64, 64


# Device-side `tma.make_tensor_descriptor` builds its tensormap in global scratch, so Triton
# needs an allocator. This is global process state, which is a real wart for library code --
# a host-built `TensorDescriptor` avoids it at the cost of having to keep the descriptor's
# swizzle in agreement with the kernel's shared layout by hand.
def _scratch(size: int, align: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_scratch)


def launch_gluon(inp, need_db=True, need_dq=True, num_warps=4, stages=5, maxnreg=None,
                 bias_dtype=torch.bfloat16, bias_scaled=False, **flags):
    """Allocate and return (outputs, per-stage closures) for the Gluon kernel."""
    n, h, d = inp["n"], inp["h"], inp["d"]
    q, k, v, b = (flatten(inp[x]) for x in ("q", "k", "v", "b"))
    o, mx, dn, do = (flatten(inp[x]) for x in ("o", "mx", "dn", "do"))
    mask = inp["mask"].contiguous()
    bh = q.shape[0]
    sm_scale = d**-0.5

    delta = torch.empty((bh, n, n), dtype=torch.float32, device=q.device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbt = torch.zeros((bh, n, n), dtype=torch.float32, device=q.device)
    dq_acc = torch.zeros((bh, n, n, d), dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    padded_n = triton.cdiv(n, 16) * 16
    # `_bwd_bias_prep` casts on store, so the kernel's staged bias dtype is chosen here.
    b2t = torch.zeros((bh, n, padded_n), dtype=bias_dtype, device=q.device)

    def run_bias_prep():
        _bwd_bias_prep[(triton.cdiv(n, 32), triton.cdiv(n, 32), bh)](
            b, b.stride(0), b.stride(1), b.stride(2),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            n, BLOCK=32, SCALE=(1.4426950408889634 if bias_scaled else 1.0), num_warps=4,
        )

    def run_pre():
        _bwd_preprocess[(triton.cdiv(n, 64), n, bh)](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta, delta.stride(0), delta.stride(1), delta.stride(2),
            n, DIM=d, BLOCK_J=64, num_warps=4,
        )

    # A precondition of the kernel, not a tuning choice -- see `gluon_bwd_stages`.
    eff_stages = gluon_bwd_stages(n, BJ, stages)

    def run_fused():
        # fmt: off
        return _gl_bwd_fused[(n // BK, n, bh)](
            delta, delta.stride(0), delta.stride(1), delta.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            mx, dn, mx.stride(0), mx.stride(1), mx.stride(2),
            mask, mask.stride(0), mask.stride(1), mask.stride(2),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dbt, dbt.stride(0), dbt.stride(1), dbt.stride(2),
            dq_acc, dq_acc.stride(0), dq_acc.stride(1), dq_acc.stride(2), dq_acc.stride(3),
            sm_scale, MASK_FILL,
            n, h,
            DIM=d, BLOCK_J=BJ, BLOCK_K=BK,
            NEED_DB=need_db, NEED_DQ=need_dq, STAGES=eff_stages,
            BIAS_SCALED=bias_scaled,
            num_warps=num_warps, num_stages=1, maxnreg=maxnreg,
            **flags,
        )
        # fmt: on

    def run_cast():
        numel = dq_acc.numel()
        _bwd_scale_cast[(triton.cdiv(numel, 4096),)](
            dq_acc, dq, sm_scale, numel, BLOCK=4096, num_warps=4
        )

    def run_db_cast():
        return dbt.transpose(1, 2).contiguous().to(inp["dtype"])

    def run_all():
        run_bias_prep(); run_pre(); run_fused(); run_cast(); run_db_cast()

    outs = dict(dq=dq, dk=dk, dv=dv, dbt=dbt, dq_acc=dq_acc, delta=delta)
    return outs, dict(bias_prep=run_bias_prep, pre=run_pre, fused=run_fused,
                      cast=run_cast, db_cast=run_db_cast, all=run_all)


def gluon_grads(inp, **kw):
    from einops import rearrange
    outs, st = launch_gluon(inp, **kw)
    st["bias_prep"](); st["pre"](); st["fused"](); st["cast"]()
    db = st["db_cast"]()
    n, h, bs = inp["n"], inp["h"], inp["bs"]
    un = lambda t: rearrange(t, "(b h) ... -> b h ...", h=h, b=bs)
    return un(outs["dq"]), un(outs["dk"]), un(outs["dv"]), un(db)


def rel(got, want):
    scale = want.float().abs().max().item() or 1.0
    return (got.float() - want.float()).abs().max().item() / scale


def cmd_check(args):
    cases = [(128, 2, 32), (256, 2, 32), (64, 1, 64), (512, 4, 32), (1024, 8, 32)]
    bad = 0
    print(f"{'n':>5}{'h':>3}{'d':>5}{'mode':>9}   "
          + "".join(f"{x:>11}" for x in ("dq", "dk", "dv", "db")))
    for n, h, d in cases:
        for mode in ("random", "fullrow", "none"):
            bs = 2 if mode == "fullrow" else 1
            inp = make_inputs(n, h, d, torch.bfloat16, bs=bs, mask_mode=mode)
            try:
                g = gluon_grads(inp)
            except Exception as e:  # noqa: BLE001 - throwaway harness
                print(f"{n:>5}{h:>3}{d:>5}{mode:>9}   FAILED {type(e).__name__}: "
                      f"{str(e)[:90]}")
                bad += 1
                continue
            r = ref_grads(inp)
            errs = [rel(a, b) for a, b in zip(g, r)]
            flag = "" if max(errs) < 2e-2 else "   <-- FAIL"
            if flag:
                bad += 1
            if not all(torch.isfinite(t).all() for t in g):
                flag += " NONFINITE"
                bad += 1
            print(f"{n:>5}{h:>3}{d:>5}{mode:>9}   "
                  + "".join(f"{e:>11.2e}" for e in errs) + flag)
    print("FAILURES:", bad)
    return bad


def cmd_bench(args):
    require_idle_gpu()
    n, h, d = args.n, 8, 32
    inp = make_inputs(n, h, d, torch.bfloat16, mask_mode="random")

    _, st_t = launch_fused(inp, dict(DEFAULT_CFG))
    st_t["all"](); torch.cuda.synchronize()
    t_tri = triton.testing.do_bench(st_t["fused"], warmup=100, rep=400, quantiles=[0.5])

    # (label, stages, launch overrides). `fp32 bias` at 3 stages is where the kernel
    # started, 39.03 ms; the rest is the bf16 staged bias at every depth that fits.
    variants = [
        # label, stages, kernel flags / launch overrides
        # The fp32 pre-scaled bias at 3 stages is where this kernel started.
        ("fp32 bias", 3, dict(bias_dtype=torch.float32, bias_scaled=True)),
        ("bf16 bias", 2, {}),
        ("bf16 bias", 3, {}),
        ("bf16 bias", 4, {}),
        ("bf16 bias", 5, {}),
    ]
    if args.variants:
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in keep]

    best = None
    print(f"{'variant':>16}{'stages':>7}{'fused ms':>10}{'vs triton':>11}"
          f"{'regs':>6}{'spill':>6}{'smem':>8}")
    print(f"{'triton _bwd_fused':>16}{3:>7}{t_tri:>10.2f}{'1.00x':>11}")
    for label, stages, flags in variants:
        try:
            _, st_g = launch_gluon(inp, num_warps=4, stages=stages, **flags)
            st_g["all"](); torch.cuda.synchronize()
            ms = triton.testing.do_bench(st_g["fused"], warmup=100, rep=400,
                                         quantiles=[0.5])
            kern = st_g["fused"]()
            torch.cuda.synchronize()
            print(f"{label:>16}{stages:>7}{ms:>10.2f}"
                  f"{t_tri / ms:>10.2f}x{kern.n_regs:>6}{kern.n_spills:>6}"
                  f"{kern.metadata.shared:>8}")
            if best is None or ms < best[0]:
                best = (ms, label, stages)
        except Exception as e:  # noqa: BLE001
            print(f"{label:>16}{stages:>7}  FAILED {type(e).__name__}: {str(e)[:70]}")
    report_gpu_after("bench")
    if best:
        print(f"\nbest gluon: {best[1]} s{best[2]} -> {best[0]:.2f} ms "
              f"({t_tri / best[0]:.2f}x triton's {t_tri:.2f} ms)")


def cmd_sass(args):
    """Compile only, and count the wgmma sync the whole exercise is about."""
    import pathlib
    import subprocess
    inp = make_inputs(256, 2, 32, torch.bfloat16, mask_mode="random")
    _, st = launch_gluon(inp)
    st["bias_prep"](); st["pre"]()
    kern = st["fused"]()
    torch.cuda.synchronize()
    p = pathlib.Path("/tmp/gl_bwd.cubin")
    p.write_bytes(kern.asm["cubin"])
    sass = subprocess.run(["nvdisasm", "-c", str(p)], capture_output=True,
                          text=True).stdout
    print(f"regs={kern.n_regs} spills={kern.n_spills} smem={kern.metadata.shared}")
    for pat in ("WARPGROUP.ARRIVE", "WARPGROUP.DEPBAR.LE gsb0, 0x0",
                "WARPGROUP.DEPBAR.LE gsb0, 0x1", "HGMMA", "REDG", "STS", "LDS",
                "STSM", "LDSM", "PRMT", "BAR.SYNC"):
        print(f"  {pat:32s} {sass.count(pat)}")


def cmd_ncu(args):
    """Body of an `ncu` run: warm the JIT, then launch exactly `--iters` profilable reps.

    The prerequisite stages run first so the profiled kernel sees real inputs, and the
    fused kernel is called once before the measured reps so that compilation is not
    inside the profiled launches. Pair with `--launch-skip 0 -k regex:_gl_bwd_fused`.
    """
    which = args.which
    inp = make_inputs(args.n, 8, 32, torch.bfloat16, mask_mode="random")
    if which == "gluon":
        extra = (dict(bias_dtype=torch.bfloat16, bias_scaled=False) if args.bf16_bias
                 else {})
        _, st = launch_gluon(inp, stages=args.stages, **extra)
    else:
        _, st = launch_fused(inp, dict(DEFAULT_CFG))
    st["bias_prep"](); st["pre"]()
    st["fused"]()          # warm: compile + autotune outside the profiled reps
    torch.cuda.synchronize()
    for _ in range(args.iters):
        st["fused"]()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    b = sub.add_parser("bench"); b.add_argument("-n", type=int, default=1024)
    b.add_argument("--variants", default=None,
                   help="comma-separated subset of the variant labels to run")
    b.set_defaults(fn=cmd_bench)
    sub.add_parser("sass").set_defaults(fn=cmd_sass)
    p = sub.add_parser("ncu")
    p.add_argument("-n", type=int, default=1024)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--which", choices=("gluon", "triton"), default="gluon")
    p.add_argument("--bf16-bias", action="store_true")
    p.set_defaults(fn=cmd_ncu)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
