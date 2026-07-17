#!/usr/bin/env python3
"""Benchmark: Arke LLVM backend vs PyTorch/cuBLAS baselines.

Measures latency for representative ops at realistic shapes.
Reports:
  - LLVM end-to-end (b.run(): H2D + kernel + D2H + module load/unload each call)
  - LLVM kernel-only (load module once, alloc+H2D once, time only launch+sync)
  - PyTorch (CUDA events on GPU-resident tensors)
  - Ratios: LLVM_e2e/PT, LLVM_kern/PT

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
WARMUP = 5
TRIALS = 20
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
    # Warmup
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
    """Time b.run() end-to-end (H2D + kernel + D2H + module load/unload each call).
    Uses time.perf_counter since b.run() calls cuCtxSynchronize internally.
    Returns median µs."""
    # Warmup
    for _ in range(warmup):
        backend.run(kernel, inputs_fn())
    
    times = []
    for _ in range(trials):
        inp = inputs_fn()
        t0 = time.perf_counter()
        backend.run(kernel, inp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # s -> µs
    return statistics.median(times)


def _time_llvm_kernel_only(backend, kernel, inputs_fn, warmup=WARMUP, trials=TRIALS):
    """Time ONLY the kernel launch + sync, excluding module load, alloc, H2D, D2H.
    Loads the module once, allocates GPU buffers once, copies H2D once,
    then times only cuLaunchKernel + cuCtxSynchronize.
    Returns median µs."""
    from cuda.bindings import driver

    emitted: CudaCKernel = kernel.metadata["emitted"]
    cubin: bytes = kernel.metadata["cubin"]

    # Load module once
    mod = backend._chk(driver, driver.cuModuleLoadData(cubin))
    func = backend._chk(driver, driver.cuModuleGetFunction(
        mod, emitted.kernel_name.encode()
    ))

    # Prepare inputs
    sample_inputs = inputs_fn()
    np_inputs = {}
    for name in emitted.param_names:
        if name == emitted.output_name:
            continue
        val = sample_inputs[name]
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
        np_inputs[name] = np.ascontiguousarray(val, dtype=np_dtype)

    # Allocate GPU memory + H2D (once)
    gpu_ptrs = {}
    allocs = []
    for name in emitted.param_names:
        if name == emitted.output_name:
            out_shape = emitted.shapes[name]
            np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
            nbytes = int(np.prod(out_shape)) * np_dtype.itemsize
            dptr = backend._chk(driver, driver.cuMemAlloc(nbytes))
            gpu_ptrs[name] = int(dptr)
            allocs.append(int(dptr))
        else:
            arr = np_inputs[name]
            dptr = backend._chk(driver, driver.cuMemAlloc(arr.nbytes))
            backend._chk(driver, driver.cuMemcpyHtoD(dptr, arr.ctypes.data, arr.nbytes))
            gpu_ptrs[name] = int(dptr)
            allocs.append(int(dptr))

    # Build kernel args (once)
    arg_buffers = []
    for arg_type, arg_val in emitted.kernel_args:
        if arg_type == "ptr":
            arg_buffers.append(np.array([gpu_ptrs[arg_val]], dtype=np.uint64))
        elif arg_type == "int":
            arg_buffers.append(np.array([arg_val], dtype=np.int32))
        elif arg_type == "float":
            arg_buffers.append(np.array([arg_val], dtype=np.float32))

    arg_ptrs = np.array([a.ctypes.data for a in arg_buffers], dtype=np.uint64)
    gx, gy, gz = emitted.grid
    bx, by, bz = emitted.block

    # Warmup
    for _ in range(warmup):
        backend._chk(driver, driver.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz,
            emitted.shared_mem, 0, arg_ptrs.ctypes.data, 0,
        ))
        backend._chk(driver, driver.cuCtxSynchronize())

    # Timed runs
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        backend._chk(driver, driver.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz,
            emitted.shared_mem, 0, arg_ptrs.ctypes.data, 0,
        ))
        backend._chk(driver, driver.cuCtxSynchronize())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    # Cleanup
    for dptr in allocs:
        driver.cuMemFree(dptr)
    driver.cuModuleUnload(mod)

    return statistics.median(times)


# ─── Benchmark definitions ─────────────────────────────────────

def bench_unary_elementwise(op_name, M, N, pt_fn, backend):
    """Benchmark a unary elementwise op."""
    graph = _graph_unary(op_name, M, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for {op_name}: {kern.error}"

    # LLVM inputs: numpy
    def llvm_inputs():
        return {"X": np.random.randn(M, N).astype(np.float32)}

    # PyTorch
    x_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    def pt_call():
        return pt_fn(x_gpu)

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


def bench_softmax(M, N, backend):
    graph = _graph_unary("softmax", M, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for softmax: {kern.error}"

    def llvm_inputs():
        return {"X": np.random.randn(M, N).astype(np.float32)}

    x_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    def pt_call():
        return F.softmax(x_gpu, dim=-1)

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


def bench_layernorm(M, N, backend):
    graph = _graph_layernorm(M, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for layernorm: {kern.error}"

    def llvm_inputs():
        return {
            "X": np.random.randn(M, N).astype(np.float32),
            "W": np.ones((1, N), dtype=np.float32),
            "Bias": np.zeros((1, N), dtype=np.float32),
        }

    x_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    w_gpu = torch.ones(N, device=DEVICE, dtype=torch.float32)
    b_gpu = torch.zeros(N, device=DEVICE, dtype=torch.float32)
    def pt_call():
        return F.layer_norm(x_gpu, [N], weight=w_gpu, bias=b_gpu)

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


def bench_rmsnorm(M, N, backend):
    graph = _graph_rmsnorm(M, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for rmsnorm: {kern.error}"

    def llvm_inputs():
        return {
            "X": np.random.randn(M, N).astype(np.float32),
            "W": np.ones((1, N), dtype=np.float32),
        }

    x_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    w_gpu = torch.ones(N, device=DEVICE, dtype=torch.float32)
    eps = 1e-5
    def pt_call():
        rms = torch.sqrt(torch.mean(x_gpu ** 2, dim=-1, keepdim=True) + eps)
        return (x_gpu / rms) * w_gpu

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


def bench_matmul(M, K, N, backend):
    graph = _graph_matmul(M, K, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for matmul: {kern.error}"

    def llvm_inputs():
        return {
            "A": np.random.randn(M, K).astype(np.float32),
            "B": np.random.randn(K, N).astype(np.float32),
        }

    a_gpu = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    b_gpu = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    def pt_call():
        return torch.mm(a_gpu, b_gpu)

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


def bench_silu_and_mul(M, N, backend):
    graph = _graph_binary("silu_and_mul", M, N)
    art = backend.lower(graph)
    kern = backend.compile(art)
    assert kern.success, f"LLVM compile failed for silu_and_mul: {kern.error}"

    def llvm_inputs():
        return {
            "A": np.random.randn(M, N).astype(np.float32),
            "B": np.random.randn(M, N).astype(np.float32),
        }

    a_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    b_gpu = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    def pt_call():
        return F.silu(a_gpu) * b_gpu

    t_pt = _time_pytorch(pt_call)
    t_llvm_e2e = _time_llvm_e2e(backend, kern, llvm_inputs)
    t_llvm_kern = _time_llvm_kernel_only(backend, kern, llvm_inputs)

    return t_llvm_e2e, t_llvm_kern, t_pt


# ─── Main ──────────────────────────────────────────────────────

def main():
    assert llvm_toolchain_available(), "LLVM toolchain not available!"
    assert torch.cuda.is_available(), "CUDA not available!"

    backend = LLVMBackend(chip="sm_86")

    # Header
    hdr = (f"{'op':<15} | {'shape':<14} | {'LLVM e2e(µs)':>12} | "
           f"{'LLVM kern(µs)':>13} | {'PyTorch(µs)':>11} | "
           f"{'e2e/PT':>8} | {'kern/PT':>8}")
    sep = "-" * len(hdr)
    print(f"\nArke LLVM Backend vs PyTorch/cuBLAS  (warmup={WARMUP}, trials={TRIALS})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    print(hdr)
    print(sep)

    results = []

    # --- Elementwise ops [4096, 4096] ---
    ew_ops = [
        ("relu",    lambda x: torch.relu(x)),
        ("gelu",    lambda x: F.gelu(x)),
        ("silu",    lambda x: F.silu(x)),
        ("exp",     lambda x: torch.exp(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
    ]
    for op_name, pt_fn in ew_ops:
        try:
            t_e2e, t_kern, t_pt = bench_unary_elementwise(op_name, 4096, 4096, pt_fn, backend)
            ratio_e2e = t_e2e / t_pt if t_pt > 0 else float('inf')
            ratio_kern = t_kern / t_pt if t_pt > 0 else float('inf')
            shape_str = "4096x4096"
            print(f"{op_name:<15} | {shape_str:<14} | {t_e2e:>12.2f} | {t_kern:>13.2f} | "
                  f"{t_pt:>11.2f} | {ratio_e2e:>7.2f}x | {ratio_kern:>7.2f}x")
            results.append((op_name, shape_str, t_e2e, t_kern, t_pt, ratio_e2e, ratio_kern))
        except Exception as e:
            print(f"{op_name:<15} | {'4096x4096':<14} | {'FAILED':>12} | {'':>13} | {'':>11} | {'':>8} | {'':>8}  ({e})")

    # --- Reduction ops [1024, 4096] ---
    red_benchmarks = [
        ("softmax",   lambda: bench_softmax(1024, 4096, backend)),
        ("layernorm", lambda: bench_layernorm(1024, 4096, backend)),
        ("rmsnorm",   lambda: bench_rmsnorm(1024, 4096, backend)),
    ]
    for op_name, bench_fn in red_benchmarks:
        try:
            t_e2e, t_kern, t_pt = bench_fn()
            ratio_e2e = t_e2e / t_pt if t_pt > 0 else float('inf')
            ratio_kern = t_kern / t_pt if t_pt > 0 else float('inf')
            shape_str = "1024x4096"
            print(f"{op_name:<15} | {shape_str:<14} | {t_e2e:>12.2f} | {t_kern:>13.2f} | "
                  f"{t_pt:>11.2f} | {ratio_e2e:>7.2f}x | {ratio_kern:>7.2f}x")
            results.append((op_name, shape_str, t_e2e, t_kern, t_pt, ratio_e2e, ratio_kern))
        except Exception as e:
            print(f"{op_name:<15} | {'1024x4096':<14} | {'FAILED':>12} | {'':>13} | {'':>11} | {'':>8} | {'':>8}  ({e})")

    # --- Dense: matmul [1024, 1024, 1024] ---
    try:
        t_e2e, t_kern, t_pt = bench_matmul(1024, 1024, 1024, backend)
        ratio_e2e = t_e2e / t_pt if t_pt > 0 else float('inf')
        ratio_kern = t_kern / t_pt if t_pt > 0 else float('inf')
        shape_str = "1024x1024x1024"
        print(f"{'matmul':<15} | {shape_str:<14} | {t_e2e:>12.2f} | {t_kern:>13.2f} | "
              f"{t_pt:>11.2f} | {ratio_e2e:>7.2f}x | {ratio_kern:>7.2f}x")
        results.append(("matmul", shape_str, t_e2e, t_kern, t_pt, ratio_e2e, ratio_kern))
    except Exception as e:
        print(f"{'matmul':<15} | {'1024x1024x1024':<14} | {'FAILED':>12} | {'':>13} | {'':>11} | {'':>8} | {'':>8}  ({e})")

    # --- Fused: silu_and_mul [2048, 4096] ---
    try:
        t_e2e, t_kern, t_pt = bench_silu_and_mul(2048, 4096, backend)
        ratio_e2e = t_e2e / t_pt if t_pt > 0 else float('inf')
        ratio_kern = t_kern / t_pt if t_pt > 0 else float('inf')
        shape_str = "2048x4096"
        print(f"{'silu_and_mul':<15} | {shape_str:<14} | {t_e2e:>12.2f} | {t_kern:>13.2f} | "
              f"{t_pt:>11.2f} | {ratio_e2e:>7.2f}x | {ratio_kern:>7.2f}x")
        results.append(("silu_and_mul", shape_str, t_e2e, t_kern, t_pt, ratio_e2e, ratio_kern))
    except Exception as e:
        print(f"{'silu_and_mul':<15} | {'2048x4096':<14} | {'FAILED':>12} | {'':>13} | {'':>11} | {'':>8} | {'':>8}  ({e})")

    # --- Summary ---
    print(sep)
    if results:
        avg_e2e = statistics.mean([r[5] for r in results])
        avg_kern = statistics.mean([r[6] for r in results])
        print(f"\nGeometric mean ratios:")
        from math import exp, log
        geo_e2e = exp(statistics.mean([log(r[5]) for r in results]))
        geo_kern = exp(statistics.mean([log(r[6]) for r in results]))
        print(f"  LLVM e2e / PyTorch:    {geo_e2e:.2f}x  (geomean)")
        print(f"  LLVM kernel / PyTorch: {geo_kern:.2f}x  (geomean)")
        print(f"\nNote: LLVM e2e includes module load + H2D + kernel + D2H + module unload per call.")
        print(f"      LLVM kernel-only times just cuLaunchKernel + cuCtxSynchronize.")
        print(f"      PyTorch times just the kernel on GPU-resident data.")
        print(f"      Ratio < 1.0 means LLVM is faster.")
    print()


if __name__ == "__main__":
    main()
