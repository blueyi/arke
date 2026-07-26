"""Interleaved FG-vs-Arke probe: same loop, same clock state, per-call medians."""
import statistics

import torch
import flag_gems

from benchmarks.baselines.arke_runner import ArkeRunner
from benchmarks.baselines._runtime_ctx import set_current_shape
from benchmarks.shapes import BATCH_MATMUL_SHAPES


def interleaved(fn_a, fn_b, iters=150, warmup=30):
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(iters):
        sa = torch.cuda.Event(enable_timing=True)
        ea = torch.cuda.Event(enable_timing=True)
        sb = torch.cuda.Event(enable_timing=True)
        eb = torch.cuda.Event(enable_timing=True)
        sa.record(); fn_a(); ea.record()
        sb.record(); fn_b(); eb.record()
        torch.cuda.synchronize()
        ta.append(sa.elapsed_time(ea) * 1000)
        tb.append(sb.elapsed_time(eb) * 1000)
    return statistics.median(ta), statistics.median(tb), min(ta), min(tb)


runner = ArkeRunner()
print(f"{'shape':16s} {'FG med':>8s} {'Arke med':>9s} {'FG min':>8s} {'Arke min':>9s} {'FG/Arke':>8s}")
import math
speedups = []
for shape in BATCH_MATMUL_SHAPES:
    if shape.tier > 2:
        continue
    set_current_shape(shape)
    A = torch.randn(shape.B, shape.M, shape.K, device="cuda", dtype=torch.float16)
    Bm = torch.randn(shape.B, shape.K, shape.N, device="cuda", dtype=torch.float16)
    fn_arke = runner.get_fn("batch_matmul", shape.M, shape.N, shape.K, torch.float16)
    set_current_shape(None)
    if fn_arke is None:
        print(f"{shape.tag}: no arke fn")
        continue
    fg_med, ak_med, fg_min, ak_min = interleaved(
        lambda: flag_gems.bmm(A, Bm), fn_arke)
    s = fg_med / ak_med
    speedups.append(s)
    print(f"{shape.tag:16s} {fg_med:8.1f} {ak_med:9.1f} {fg_min:8.1f} {ak_min:9.1f} {s:8.3f}")

g = math.exp(sum(math.log(x) for x in speedups) / len(speedups))
print(f"interleaved geomean: {g:.3f}")
