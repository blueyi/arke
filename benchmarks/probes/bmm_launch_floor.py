"""Prove/disprove: is the residual batch_matmul gap launch-overhead bound?

If Arke's latency is FLAT (~same us) across shapes whose compute differs by
10x, the cost is dispatch/launch, not the kernel.
"""
import statistics
import torch
import flag_gems

from benchmarks.baselines.arke_runner import ArkeRunner
from benchmarks.baselines._runtime_ctx import set_current_shape
from benchmarks.shapes import BATCH_MATMUL_SHAPES


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
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


runner = ArkeRunner()
print(f"{'shape':16s} {'GFLOP':>8s} {'FG gpu':>8s} {'AK gpu':>8s} "
      f"{'FG cpu':>8s} {'AK cpu':>8s}")
for shape in BATCH_MATMUL_SHAPES:
    if shape.tier > 2:
        continue
    set_current_shape(shape)
    A = torch.randn(shape.B, shape.M, shape.K, device="cuda", dtype=torch.float16)
    Bm = torch.randn(shape.B, shape.K, shape.N, device="cuda", dtype=torch.float16)
    fn_arke = runner.get_fn("batch_matmul", shape.M, shape.N, shape.K, torch.float16)
    set_current_shape(None)
    if fn_arke is None:
        continue
    gflop = 2 * shape.B * shape.M * shape.N * shape.K / 1e9
    fg_gpu = timeit(lambda: flag_gems.bmm(A, Bm))
    ak_gpu = timeit(fn_arke)
    fg_cpu = cpu_launch_cost(lambda: flag_gems.bmm(A, Bm))
    ak_cpu = cpu_launch_cost(fn_arke)
    print(f"{shape.tag:16s} {gflop:8.3f} {fg_gpu:8.1f} {ak_gpu:8.1f} "
          f"{fg_cpu:8.1f} {ak_cpu:8.1f}")
