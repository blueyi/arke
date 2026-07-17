#!/usr/bin/env python3
"""Quick kernel-only benchmark for Arke LLVM backend vs PyTorch.

Tests representative ops across multiple shapes.
Reports kernel-only latency (no H2D/D2H, no module load).

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    python benchmarks/llvm_quick_bench.py
"""

import gc
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from arke.backend.llvm_backend import LLVMBackend
from arke.ir.graph import IRGraph, IRNode


WARMUP = 30
TRIALS = 100
DEVICE = "cuda"


def _bench_kernel_only(backend, op_name, shape, graph_fn, pt_fn):
    """Benchmark kernel-only latency."""
    gc.collect()
    torch.cuda.empty_cache()

    g = graph_fn()
    art = backend.lower(g)
    kern = backend.compile(art)
    if not kern.success:
        return None, None, None

    emitted = kern.metadata["emitted"]
    cached = backend.prepare(kern)

    # Build inputs
    from arke.backend.cuda_c_backend import _ir_dtype_to_numpy
    inputs = {}
    for name in emitted.param_names:
        if name == emitted.output_name:
            continue
        s = emitted.shapes[name]
        dt = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
        inputs[name] = np.random.randn(*s).astype(dt)

    backend.run_fast(cached, inputs)
    for _ in range(WARMUP):
        backend.run_fast_no_copy(cached)

    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        backend.run_fast_no_copy(cached)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    llvm_med = statistics.median(times)
    backend.release(cached)

    # PyTorch
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        pt_fn()
    torch.cuda.synchronize()

    pt_times = []
    for _ in range(TRIALS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pt_fn()
        e.record()
        torch.cuda.synchronize()
        pt_times.append(s.elapsed_time(e) * 1000)
    pt_med = statistics.median(pt_times)

    return llvm_med, pt_med, llvm_med / pt_med


def main():
    assert torch.cuda.is_available()
    backend = LLVMBackend(chip="sm_86")
    torch.cuda.init()

    print(f"Arke LLVM Backend — Kernel-Only Quick Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"warmup={WARMUP}, trials={TRIALS}")
    print()
    print(f"{'op':<17} {'shape':<16} {'LLVM(µs)':>10} {'PT(µs)':>10} {'ratio':>8}")
    print("-" * 64)

    results = []

    def _row(op, shape_str, l, p, r):
        marker = " 🏆" if r and r < 1.0 else ""
        print(f"{op:<17} {shape_str:<16} {l:>10.1f} {p:>10.1f} {r:>7.2f}x{marker}")
        results.append((op, shape_str, l, p, r))

    # --- Elementwise ---
    for M, N in [(4096, 4096), (128, 4096), (32, 4096)]:
        x = torch.randn(M, N, device=DEVICE)
        for op, pt_fn in [
            ("relu", lambda: torch.relu(x)),
            ("silu", lambda: F.silu(x)),
            ("gelu", lambda: F.gelu(x)),
        ]:
            def gf(o=op, m=M, n=N):
                g = IRGraph(name="b")
                g.add_input("X", dtype="float32", shape=[m, n])
                g.add_node(IRNode(id="n0", op=o, inputs={"X": "X"}, outputs=["out"]))
                g.set_outputs(["out"])
                return g
            l, p, r = _bench_kernel_only(backend, op, f"{M}x{N}", gf, pt_fn)
            if l:
                _row(op, f"{M}x{N}", l, p, r)
        del x

    # --- Fused ---
    for M, N in [(2048, 4096), (128, 4096)]:
        a = torch.randn(M, N, device=DEVICE)
        b = torch.randn(M, N, device=DEVICE)
        for op, pt_fn in [
            ("silu_and_mul", lambda: F.silu(a) * b),
            ("gelu_and_mul", lambda: F.gelu(a) * b),
        ]:
            def gf(o=op, m=M, n=N):
                g = IRGraph(name="b")
                g.add_input("A", dtype="float32", shape=[m, n])
                g.add_input("B", dtype="float32", shape=[m, n])
                g.add_node(IRNode(id="n0", op=o, inputs={"A": "A", "B": "B"}, outputs=["out"]))
                g.set_outputs(["out"])
                return g
            l, p, r = _bench_kernel_only(backend, op, f"{M}x{N}", gf, pt_fn)
            if l:
                _row(op, f"{M}x{N}", l, p, r)
        del a, b

    # --- Reduction ---
    for M, N in [(1024, 4096), (32, 4096)]:
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)
        b_ln = torch.zeros(N, device=DEVICE)

        # softmax
        def gf_sm(m=M, n=N):
            g = IRGraph(name="b")
            g.add_input("X", dtype="float32", shape=[m, n])
            g.add_node(IRNode(id="n0", op="softmax", inputs={"X": "X"}, outputs=["out"]))
            g.set_outputs(["out"])
            return g
        l, p, r = _bench_kernel_only(backend, "softmax", f"{M}x{N}", gf_sm, lambda: F.softmax(x, dim=-1))
        if l:
            _row("softmax", f"{M}x{N}", l, p, r)

        # layernorm
        def gf_ln(m=M, n=N):
            g = IRGraph(name="b")
            g.add_input("X", dtype="float32", shape=[m, n])
            g.add_input("W", dtype="float32", shape=[1, n])
            g.add_input("Bias", dtype="float32", shape=[1, n])
            g.add_node(IRNode(id="n0", op="layernorm", inputs={"X": "X", "W": "W", "Bias": "Bias"}, outputs=["out"]))
            g.set_outputs(["out"])
            return g
        l, p, r = _bench_kernel_only(backend, "layernorm", f"{M}x{N}", gf_ln, lambda: F.layer_norm(x, [N], weight=w, bias=b_ln))
        if l:
            _row("layernorm", f"{M}x{N}", l, p, r)

        # rmsnorm
        def gf_rms(m=M, n=N):
            g = IRGraph(name="b")
            g.add_input("X", dtype="float32", shape=[m, n])
            g.add_input("W", dtype="float32", shape=[1, n])
            g.add_node(IRNode(id="n0", op="rmsnorm", inputs={"X": "X", "W": "W"}, outputs=["out"]))
            g.set_outputs(["out"])
            return g
        def pt_rms():
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
            return (x / rms) * w
        l, p, r = _bench_kernel_only(backend, "rmsnorm", f"{M}x{N}", gf_rms, pt_rms)
        if l:
            _row("rmsnorm", f"{M}x{N}", l, p, r)

        del x, w, b_ln

    # --- Matmul ---
    for sz in [512, 1024]:
        a = torch.randn(sz, sz, device=DEVICE)
        b = torch.randn(sz, sz, device=DEVICE)
        def gf_mm(s=sz):
            g = IRGraph(name="b")
            g.add_input("A", dtype="float32", shape=[s, s])
            g.add_input("B", dtype="float32", shape=[s, s])
            g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["out"]))
            g.set_outputs(["out"])
            return g
        l, p, r = _bench_kernel_only(backend, "matmul", f"{sz}³", gf_mm, lambda: torch.mm(a, b))
        if l:
            _row("matmul", f"{sz}³", l, p, r)
        del a, b

    # --- Summary ---
    print("-" * 64)
    from math import exp, log
    if results:
        beats = [r for r in results if r[4] < 1.0]
        total = len(results)
        print(f"\n{len(beats)}/{total} ops beat PyTorch")
        if beats:
            print("Best:", ", ".join(f"{r[0]}@{r[1]}={r[4]:.2f}x" for r in sorted(beats, key=lambda x: x[4])))
        geo = exp(statistics.mean([log(r[4]) for r in results]))
        print(f"Geomean ratio: {geo:.2f}x")

    backend.release_all()


if __name__ == "__main__":
    main()
