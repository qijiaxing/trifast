"""Check and time `trifast.gluon_bwd_ws._gl_bwd_ws` -- the warp-specialized fused backward.

Same throwaway spirit as `proto_gluon_bwd.py`, and it reuses that module's launch plumbing
for the prerequisite stages. It answers one question: does moving the loads into their own
warpgroup beat the single-warpgroup schedule's 36.48 ms?

    python scripts/proto_gluon_ws.py smoke      # does warp_specialize work on sm90 at all
    python scripts/proto_gluon_ws.py check      # gradients vs the three-kernel path
    python scripts/proto_gluon_ws.py bench      # ws vs gluon_bwd vs triton, n=1024
    python scripts/proto_gluon_ws.py ncu        # launch under ncu (prereqs then 3 reps)
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
from scripts.proto_gluon_bwd import launch_gluon, rel  # noqa: E402
from trifast.gluon_bwd_ws import WS_MAXNREG, _gl_bwd_ws  # noqa: E402
from trifast.torch import MASK_FILL  # noqa: E402
from trifast.triton_bwd import _bwd_bias_prep, _bwd_preprocess, _bwd_scale_cast  # noqa: E402

BJ, BK = 64, 64
INV_LN2 = 1.4426950408889634


def launch_ws(inp, need_db=True, need_dq=True, stages=3, bias_dtype=torch.bfloat16,
              bias_scaled=False, prod_warps=4, prod_regs=40, maxnreg=WS_MAXNREG,
              cons_warps=4):
    """Allocate and return (outputs, per-stage closures) for the warp-specialized kernel."""
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
    b2t = torch.zeros((bh, n, padded_n), dtype=bias_dtype, device=q.device)

    def run_bias_prep():
        _bwd_bias_prep[(triton.cdiv(n, 32), triton.cdiv(n, 32), bh)](
            b, b.stride(0), b.stride(1), b.stride(2),
            b2t, b2t.stride(0), b2t.stride(1), b2t.stride(2),
            n, BLOCK=32, SCALE=(INV_LN2 if bias_scaled else 1.0), num_warps=4,
        )

    def run_pre():
        _bwd_preprocess[(triton.cdiv(n, 64), n, bh)](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta, delta.stride(0), delta.stride(1), delta.stride(2),
            n, DIM=d, BLOCK_J=64, num_warps=4,
        )

    def run_fused():
        # fmt: off
        return _gl_bwd_ws[(n // BK, n, bh)](
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
            NEED_DB=need_db, NEED_DQ=need_dq, STAGES=stages,
            BIAS_SCALED=bias_scaled, PROD_WARPS=prod_warps, PROD_REGS=prod_regs,
            CONS_WARPS=cons_warps,
            num_warps=cons_warps, num_stages=1, maxnreg=maxnreg,
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


def ws_grads(inp, **kw):
    from einops import rearrange
    outs, st = launch_ws(inp, **kw)
    st["bias_prep"](); st["pre"](); st["fused"](); st["cast"]()
    db = st["db_cast"]()
    h, bs = inp["h"], inp["bs"]
    un = lambda t: rearrange(t, "(b h) ... -> b h ...", h=h, b=bs)  # noqa: E731
    return un(outs["dq"]), un(outs["dk"]), un(outs["dv"]), un(db)


def cmd_smoke(args):
    """The minimal warp_specialize question, kept because it is the thing to re-run first
    whenever a Triton upgrade breaks this module."""
    inp = make_inputs(256, 2, 32, torch.bfloat16, mask_mode="random")
    print(f"{'pwarps':>7}{'pregs':>7}{'stages':>7}  {'compiles':>10}{'regs':>6}{'spill':>6}"
          f"{'smem':>8}")
    for pw in (1, 2, 4):
        for pr in (24, 40, 64):
            try:
                _, st = launch_ws(inp, stages=3, prod_warps=pw, prod_regs=pr)
                st["bias_prep"](); st["pre"]()
                kern = st["fused"]()
                torch.cuda.synchronize()
                print(f"{pw:>7}{pr:>7}{3:>7}  {'OK':>10}{kern.n_regs:>6}"
                      f"{kern.n_spills:>6}{kern.metadata.shared:>8}")
            except Exception as e:  # noqa: BLE001
                print(f"{pw:>7}{pr:>7}{3:>7}  FAILED {type(e).__name__}: {str(e)[:70]}")


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
                g = ws_grads(inp, stages=args.stages, maxnreg=args.maxnreg)
            except Exception as e:  # noqa: BLE001
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

    # The single-warpgroup Gluon kernel at its best known config, as the thing to beat.
    _, st_g = launch_gluon(inp, stages=5)
    st_g["all"](); torch.cuda.synchronize()
    t_gl = triton.testing.do_bench(st_g["fused"], warmup=100, rep=400, quantiles=[0.5])

    print(f"{'variant':>20}{'stages':>7}{'fused ms':>10}{'vs triton':>11}{'vs gluon':>10}"
          f"{'regs':>6}{'spill':>6}{'smem':>8}")
    print(f"{'triton _bwd_fused':>20}{3:>7}{t_tri:>10.2f}{'1.00x':>11}")
    print(f"{'gluon _gl_bwd_fused':>20}{5:>7}{t_gl:>10.2f}{t_tri / t_gl:>10.2f}x"
          f"{'1.00x':>10}")

    # The maxnreg reasoning lives in `gluon_bwd_ws.WS_MAXNREG`; this sweep is what it is
    # based on, kept narrow so a re-run is cheap. The dropped-`setmaxnreg` A/B (37.49 ms) is
    # no longer expressible: `worker_num_regs` has a hardware floor of 24.
    variants = ([(f"ws m{mx} s{st}", st, dict(maxnreg=mx))
                 for st in (3, 5) for mx in (112, 116)]
                + [("ws m116 s3 p1", 3, dict(maxnreg=116, prod_warps=1))])
    best = None
    for label, stages, flags in variants:
        try:
            _, st = launch_ws(inp, stages=stages, **flags)
            st["all"](); torch.cuda.synchronize()
            ms = triton.testing.do_bench(st["fused"], warmup=100, rep=400, quantiles=[0.5])
            kern = st["fused"]()
            torch.cuda.synchronize()
            print(f"{label:>20}{stages:>7}{ms:>10.2f}{t_tri / ms:>10.2f}x"
                  f"{t_gl / ms:>9.2f}x{kern.n_regs:>6}{kern.n_spills:>6}"
                  f"{kern.metadata.shared:>8}")
            if best is None or ms < best[0]:
                best = (ms, label, stages)
        except Exception as e:  # noqa: BLE001
            print(f"{label:>20}{stages:>7}  FAILED {type(e).__name__}: {str(e)[:60]}")
    report_gpu_after("bench")
    if best:
        print(f"\nbest ws: {best[1]} s{best[2]} -> {best[0]:.2f} ms "
              f"({t_tri / best[0]:.2f}x triton, {t_gl / best[0]:.2f}x gluon)")


def cmd_ncu(args):
    inp = make_inputs(args.n, 8, 32, torch.bfloat16, mask_mode="random")
    _, st = launch_ws(inp, stages=args.stages, prod_warps=args.prod_warps,
                      prod_regs=args.prod_regs, maxnreg=args.maxnreg,
                      cons_warps=args.cons_warps)
    st["bias_prep"](); st["pre"]()
    st["fused"]()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        st["fused"]()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    c = sub.add_parser("check"); c.add_argument("--stages", type=int, default=3)
    c.add_argument("--maxnreg", type=int, default=116)
    c.set_defaults(fn=cmd_check)
    b = sub.add_parser("bench"); b.add_argument("-n", type=int, default=1024)
    b.set_defaults(fn=cmd_bench)
    p = sub.add_parser("ncu")
    p.add_argument("-n", type=int, default=512)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--prod-warps", type=int, default=4)
    p.add_argument("--prod-regs", type=int, default=40)
    p.add_argument("--maxnreg", type=int, default=WS_MAXNREG)
    p.add_argument("--cons-warps", type=int, default=4)
    p.set_defaults(fn=cmd_ncu)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
