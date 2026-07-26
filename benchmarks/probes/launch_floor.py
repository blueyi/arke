"""Generic launch-floor probe: is an op's residual gap launch-overhead bound?

Usage:
    python -m benchmarks.probes.launch_floor --op softmax
    python -m benchmarks.probes.launch_floor --op reduce_sum --tier 2

If Arke's CPU dispatch cost (AK cpu, measured with NO per-call sync) is FLAT
across shapes whose FLOP/bytes differ by orders of magnitude, the cost is
host-side dispatch, not the kernel. See skill
arke-benchmark-harness/references/launch-overhead-op-optimization.md.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from benchmarks.baselines.arke_runner import ArkeRunner
from benchmarks.baselines.flaggems import FlagGemsRunner
from benchmarks.shapes import get_shapes


def timeit(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts)


def cpu_launch_cost(fn, iters=300, warmup=50):
    """Wall time with NO sync -> measures CPU-side dispatch cost per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--tier", type=int, default=2)
    args = ap.parse_args()

    ak = ArkeRunner()
    fg = FlagGemsRunner()

    print(f"op={args.op}  (AK cpu FLAT across FLOP range => launch-bound)")
    print(f"{'shape':16s} {'M':>6s} {'N':>7s} {'K':>6s} "
          f"{'FG gpu':>8s} {'AK gpu':>8s} {'FG cpu':>8s} {'AK cpu':>8s}")
    for shape in get_shapes(args.op, tier=args.tier):
        M = getattr(shape, "M", 0)
        N = getattr(shape, "N", 0)
        K = getattr(shape, "K", 0)
        fn_fg = fg.get_fn(args.op, M, N, K, torch.float16)
        fn_ak = ak.get_fn(args.op, M, N, K, torch.float16)
        if fn_fg is None or fn_ak is None:
            print(f"{shape.tag:16s} SKIP (fg={fn_fg is not None} ak={fn_ak is not None})")
            continue
        fg_gpu = timeit(fn_fg)
        ak_gpu = timeit(fn_ak)
        fg_cpu = cpu_launch_cost(fn_fg)
        ak_cpu = cpu_launch_cost(fn_ak)
        print(f"{shape.tag:16s} {M:6d} {N:7d} {K:6d} "
              f"{fg_gpu:8.1f} {ak_gpu:8.1f} {fg_cpu:8.1f} {ak_cpu:8.1f}")


if __name__ == "__main__":
    main()
