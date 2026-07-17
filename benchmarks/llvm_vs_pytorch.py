#!/usr/bin/env python3
"""Benchmark: Arke LLVM backend vs PyTorch/cuBLAS baselines.

Measures latency for representative ops at realistic shapes.
Reports:
  - LLVM e2e legacy (module load + alloc + H2D + kernel + D2H + free each call)
  - LLVM cached (H2D + kernel + D2H, module pre-loaded, buffers pre-allocated)
  - LLVM kernel-only (just cuLaunchKernel + cuCtxSynchronize)
  - PyTorch (CUDA events on GPU-resident tensors)
  - Ratios: cached/PT, kern/PT

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    python benchmarks/llvm_vs_pytorch.py
"""

import statistics
import time
import numpy as np
import torch
import torch.nn.functional as F

from arke.backend.llvm_backend import LLVMBackend, llvm_toolchain_available
from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_numpy
from arke.ir.graph import IRGraph, IRNode


# ─── Config ────────────────────────────────────────────────────
WARMUP = 10
TRIALS = 50
DEVICE = "cuda"


# ─── IRGraph builders ──────────────────────────────────────────

def _graph_unary(op: str, M: int, N: int) -> IRGraph:
    g = IRGraph(name=f"{op}_{M}x{N}")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"X": "X"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_binary(op: str, M: int, N: int) -> IRGraph:
    g = IRGraph(name=f"{op}_{M}x{N}")
    g.add_input("A", dtype="float32", shape=[M, N])
    g.add_input("B", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"A": "A", "B": "B"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_matmul(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name=f"matmul_{M}x{K}x{N}")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_layernorm(M: int, N: int) -> IRGraph:
    g = IRGraph(name=f"layernorm_{M}x{N}")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_input("W", dtype="float32", shape=[1, N])
    g.add_input("Bias", dtype="float32", shape=[1, N])
    g.add_node(IRNode(id="n0", op="layernorm",
                      inputs={"X": "X", "W": "W", "Bias": "Bias"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_rmsnorm(M: int, N: int) -> IRGraph:
    g = IRGraph(name=f"rmsnorm_{M}x{N}")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_input("W", dtype="float32", shape=[1, N])
    g.add_node(IRNode(id="n0", op="rmsnorm",
                      inputs={"X": "X", "W": "W"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


# ─── Timing helpers ────────────────────────────────────────────

def _time_pytorch(fn, warmup=WARMUP, trials=TRIALS):
    """Time a PyTorch callable using CUDA events. Returns median µs."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> µs
    return statistics.median(times)


def _time_llvm_e2e(backend, kernel, inputs_fn, warmup=WARMUP, trials=TRIALS):
    """Time b.run() end-to-end (module load + alloc + H2D + kernel + D2H + free each call)."""
    for _ in range(warmup):
        backend.run(kernel, inputs_fn())

    times = []
    for _ in range(trials):
        inp = inputs_fn()
        t0 = time.perf_counter()
        backend.run(kernel, inp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def _time_llvm_cached(backend, kernel, inputs_fn, warmup=WARMUP, trials=TRIALS):
    """Time using cached execution: H2D + kernel + sync + D2H.
    Module loaded once, GPU buffers pre-allocated.
    Returns median µs."""
    cached = backend.prepare(kernel)

    # Warmup
    for _ in range(warmup):
        backend.run_fast(cached, inputs_fn())

    times = []
    for _ in range(trials):
        inp = inputs_fn()
        t0 = time.perf_counter()
        backend.run_fast(cached, inp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    backend.release(cached)
    return statistics.median(times)


def _time_llvm_kernel_only(backend, kernel, inputs_fn, warmup=WARMUP, trials=TRIALS):
    """Time ONLY kernel launch + sync (no H2D/D2H, no module load, no alloc).
    Uses cached API with run_fast_no_copy.
    Returns median µs."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    cached = backend.prepare(kernel)

    # Do one H2D to fill buffers with valid data
    backend.run_fast(cached, inputs_fn())

    # Warmup
    for _ in range(warmup):
        backend.run_fast_no_copy(cached)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        backend.run_fast_no_copy(cached)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    backend.release(cached)
    return statistics.median(times)


# ─── Benchmark wrappers ────────────────────────────────────────

def _bench_op(op_name, graph, pt_fn_setup, backend):
    """Generic benchmark for an op. Returns (e2e, cached, kern, pytorch) in µs."""
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for {op_name}: {kern.error}"

    emitted = kern.metadata["emitted"]

    # Build input factory for LLVM (numpy)
    def llvm_inputs():
        result = {}
        for name in emitted.param_names:
            if name == emitted.output_name:
                continue
            shape = emitted.shapes[name]
            np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
            result[name] = np.random.randn(*shape).astype(np_dtype)
        return result

    t_pt = _time_pytorch(pt_fn_setup)
    t_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_cached = _time_llvm_cached(backend, kern, llvm_inputs)
    t_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_e2e, t_cached, t_kern, t_pt


# ─── Main ──────────────────────────────────────────────────────

def main():
    assert llvm_toolchain_available(), "LLVM toolchain not available!"
    assert torch.cuda.is_available(), "CUDA not available!"

    # Pre-warm PyTorch CUDA runtime
    print("Warming up PyTorch CUDA runtime...")
    _warmup = torch.randn(4096, 4096, device=DEVICE)
    for fn in [torch.relu, F.gelu, F.silu, torch.exp, torch.sigmoid]:
        for _ in range(3):
            fn(_warmup)
    F.softmax(_warmup[:1024], dim=-1)
    F.layer_norm(_warmup[:1024], [4096])
    torch.mm(_warmup[:1024, :1024], _warmup[:1024, :1024])
    _ = F.silu(_warmup[:2048]) * _warmup[:2048]
    _rms = torch.sqrt(torch.mean(_warmup[:1024] ** 2, dim=-1, keepdim=True) + 1e-5)
    _ = (_warmup[:1024] / _rms)
    del _warmup, _rms
    torch.cuda.synchronize()
    print("Warmup done.\n")

    backend = LLVMBackend(chip="sm_86")

    # Header
    hdr = (f"{'op':<15} | {'shape':<14} | {'e2e(µs)':>10} | {'cached(µs)':>11} | "
           f"{'kern(µs)':>10} | {'PT(µs)':>10} | "
           f"{'e2e/PT':>7} | {'cch/PT':>7} | {'krn/PT':>7}")
    sep = "-" * len(hdr)
    print(f"\nArke LLVM Backend vs PyTorch/cuBLAS  (warmup={WARMUP}, trials={TRIALS})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    print(hdr)
    print(sep)

    results = []

    def _print_row(name, shape_str, t_e2e, t_cached, t_kern, t_pt):
        r_e2e = t_e2e / t_pt if t_pt > 0 else float('inf')
        r_cch = t_cached / t_pt if t_pt > 0 else float('inf')
        r_krn = t_kern / t_pt if t_pt > 0 else float('inf')
        print(f"{name:<15} | {shape_str:<14} | {t_e2e:>10.1f} | {t_cached:>11.1f} | "
              f"{t_kern:>10.1f} | {t_pt:>10.1f} | "
              f"{r_e2e:>6.1f}x | {r_cch:>6.1f}x | {r_krn:>6.1f}x")
        results.append((name, shape_str, t_e2e, t_cached, t_kern, t_pt, r_e2e, r_cch, r_krn))

    # --- Elementwise ops [4096, 4096] ---
    ew_ops = [
        ("relu",    lambda x: torch.relu(x)),
        ("gelu",    lambda x: F.gelu(x)),
        ("silu",    lambda x: F.silu(x)),
        ("exp",     lambda x: torch.exp(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
    ]
    x_gpu = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float32)
    for op_name, pt_fn in ew_ops:
        try:
            graph = _graph_unary(op_name, 4096, 4096)
            t_e2e, t_cached, t_kern, t_pt = _bench_op(
                op_name, graph, lambda: pt_fn(x_gpu), backend)
            _print_row(op_name, "4096x4096", t_e2e, t_cached, t_kern, t_pt)
        except Exception as e:
            print(f"{op_name:<15} | {'4096x4096':<14} | FAILED: {e}")

    # --- Reduction ops [1024, 4096] ---
    x_red = torch.randn(1024, 4096, device=DEVICE, dtype=torch.float32)

    # softmax
    try:
        graph = _graph_unary("softmax", 1024, 4096)
        t_e2e, t_cached, t_kern, t_pt = _bench_op(
            "softmax", graph, lambda: F.softmax(x_red, dim=-1), backend)
        _print_row("softmax", "1024x4096", t_e2e, t_cached, t_kern, t_pt)
    except Exception as e:
        print(f"{'softmax':<15} | {'1024x4096':<14} | FAILED: {e}")

    # layernorm
    try:
        graph = _graph_layernorm(1024, 4096)
        w_ln = torch.ones(4096, device=DEVICE)
        b_ln = torch.zeros(4096, device=DEVICE)
        t_e2e, t_cached, t_kern, t_pt = _bench_op(
            "layernorm", graph, lambda: F.layer_norm(x_red, [4096], weight=w_ln, bias=b_ln), backend)
        _print_row("layernorm", "1024x4096", t_e2e, t_cached, t_kern, t_pt)
    except Exception as e:
        print(f"{'layernorm':<15} | {'1024x4096':<14} | FAILED: {e}")

    # rmsnorm
    try:
        graph = _graph_rmsnorm(1024, 4096)
        w_rms = torch.ones(4096, device=DEVICE)
        def _pt_rmsnorm():
            rms = torch.sqrt(torch.mean(x_red ** 2, dim=-1, keepdim=True) + 1e-5)
            return (x_red / rms) * w_rms
        t_e2e, t_cached, t_kern, t_pt = _bench_op(
            "rmsnorm", graph, _pt_rmsnorm, backend)
        _print_row("rmsnorm", "1024x4096", t_e2e, t_cached, t_kern, t_pt)
    except Exception as e:
        print(f"{'rmsnorm':<15} | {'1024x4096':<14} | FAILED: {e}")

    # --- Matmul [1024, 1024, 1024] ---
    try:
        graph = _graph_matmul(1024, 1024, 1024)
        a_mm = torch.randn(1024, 1024, device=DEVICE)
        b_mm = torch.randn(1024, 1024, device=DEVICE)
        t_e2e, t_cached, t_kern, t_pt = _bench_op(
            "matmul", graph, lambda: torch.mm(a_mm, b_mm), backend)
        _print_row("matmul", "1024³", t_e2e, t_cached, t_kern, t_pt)
    except Exception as e:
        print(f"{'matmul':<15} | {'1024³':<14} | FAILED: {e}")

    # --- Fused: silu_and_mul [2048, 4096] ---
    try:
        graph = _graph_binary("silu_and_mul", 2048, 4096)
        a_fused = torch.randn(2048, 4096, device=DEVICE)
        b_fused = torch.randn(2048, 4096, device=DEVICE)
        t_e2e, t_cached, t_kern, t_pt = _bench_op(
            "silu_and_mul", graph, lambda: F.silu(a_fused) * b_fused, backend)
        _print_row("silu_and_mul", "2048x4096", t_e2e, t_cached, t_kern, t_pt)
    except Exception as e:
        print(f"{'silu_and_mul':<15} | {'2048x4096':<14} | FAILED: {e}")

    # --- Summary ---
    print(sep)
    if results:
        from math import exp, log
        geo_e2e = exp(statistics.mean([log(r[6]) for r in results]))
        geo_cch = exp(statistics.mean([log(r[7]) for r in results]))
        geo_krn = exp(statistics.mean([log(r[8]) for r in results]))
        print(f"\nGeometric mean ratios (vs PyTorch):")
        print(f"  Legacy e2e: {geo_e2e:.2f}x")
        print(f"  Cached:     {geo_cch:.2f}x  ← P5-S3 target: close to kernel-only")
        print(f"  Kernel-only:{geo_krn:.2f}x  ← pure compute gap (no transfer)")
        print()
        print("Legend:")
        print("  e2e    = legacy run() — module load + alloc + H2D + kernel + D2H + free per call")
        print("  cached = prepare() once, run_fast() — H2D + kernel + sync + D2H only")
        print("  kern   = kernel launch + sync only (GPU-resident data)")
        print("  PT     = PyTorch on GPU-resident tensors")
        print("  Ratio < 1.0 means LLVM is faster")
    print()

    backend.release_all()


if __name__ == "__main__":
    main()
