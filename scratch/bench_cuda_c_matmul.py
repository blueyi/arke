#!/usr/bin/env python3
"""Kernel-only benchmark: Arke CUDA-C matmul (float4+double-buf) vs torch.mm.

Measures per-iteration latency with CUDA events, reports median and speedup.
Shapes: N=128, 256, 512, 768, 1024 (square MxNxK = NxNxN).
Warmup: 10 iterations; Timed: 50 iterations (per-iteration events).

Usage:
    cd ~/workspace/repos/arke
    source ~/.venvs/arke/bin/activate
    python scratch/bench_cuda_c_matmul.py
"""

import statistics
import sys
import os

import numpy as np
import torch

# Ensure arke is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arke.ir.graph import IRGraph, IRNode, IRValue
from arke.backend.cuda_c_backend import CudaCBackend, CudaCKernel, _ir_dtype_to_numpy, _as_tuple


def make_matmul_graph(N: int) -> IRGraph:
    """Construct a minimal IRGraph for square matmul [N,N] @ [N,N] -> [N,N]."""
    g = IRGraph(name=f"matmul_{N}")
    g.add_input("A", dtype="float32", shape=[N, N])
    g.add_input("B", dtype="float32", shape=[N, N])
    # Output value
    g.values["C"] = IRValue(name="C", dtype="float32", shape=[N, N])
    g.add_node(IRNode(
        id="mm0",
        op="matmul",
        inputs={"lhs": "A", "rhs": "B"},
        outputs=["C"],
    ))
    g.set_outputs(["C"])
    return g


def bench_torch_mm(N: int, warmup: int = 10, runs: int = 50) -> list[float]:
    """Per-iteration kernel-only latency of torch.mm (ms) via CUDA events."""
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    return latencies


def bench_arke_kernel(N: int, warmup: int = 10, runs: int = 50) -> tuple[list[float], str]:
    """Per-iteration kernel-only latency of Arke CUDA-C matmul (ms).

    Returns (latencies, kernel_variant_name).
    Uses cuda.bindings.driver for precise CUDA event timing per iteration.
    """
    from cuda.bindings import driver

    graph = make_matmul_graph(N)
    backend = CudaCBackend(chip="sm_86")

    # Lower + compile (no strategy → auto-selects f4+db when eligible)
    artifact = backend.lower(graph)
    kernel = backend.compile(artifact)
    if not kernel.success:
        raise RuntimeError(f"Compilation failed: {kernel.error}")

    emitted: CudaCKernel = kernel.metadata["emitted"]
    cubin: bytes = kernel.metadata["cubin"]
    variant = emitted.kernel_name

    # Init driver context
    _as_tuple(driver.cuInit(0))
    torch.cuda.init()  # ensure torch's CUDA context is live

    err, ctx = _as_tuple(driver.cuCtxGetCurrent())
    if err != driver.CUresult.CUDA_SUCCESS or int(ctx) == 0:
        dev = _chk(driver, driver.cuDeviceGet(0))
        ctx = _chk(driver, driver.cuCtxCreate(driver.CUctxCreateParams(), 0, dev))

    # Load module + get function
    mod = _chk(driver, driver.cuModuleLoadData(cubin))
    func = _chk(driver, driver.cuModuleGetFunction(mod, emitted.kernel_name.encode()))

    # Alloc GPU buffers + H2D
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)
    nbytes = N * N * 4  # float32

    allocs = []
    gpu_ptrs = {}

    for name, arr in [("A", A_np), ("B", B_np)]:
        dptr = _chk(driver, driver.cuMemAlloc(arr.nbytes))
        _chk(driver, driver.cuMemcpyHtoD(dptr, arr.ctypes.data, arr.nbytes))
        gpu_ptrs[name] = int(dptr)
        allocs.append(int(dptr))

    # Output buffer
    dptr = _chk(driver, driver.cuMemAlloc(nbytes))
    gpu_ptrs["C"] = int(dptr)
    allocs.append(int(dptr))

    # Build kernel args
    arg_buffers = []
    for arg_type, arg_val in emitted.kernel_args:
        if arg_type == "ptr":
            arg_buffers.append(np.array([gpu_ptrs[arg_val]], dtype=np.uint64))
        elif arg_type == "int":
            arg_buffers.append(np.array([arg_val], dtype=np.int32))
        elif arg_type == "float":
            arg_buffers.append(np.array([arg_val], dtype=np.float32))

    arg_ptrs = np.array([a.ctypes.data for a in arg_buffers], dtype=np.uint64)
    arg_data_ptr = arg_ptrs.ctypes.data

    gx, gy, gz = emitted.grid
    bx, by, bz = emitted.block
    smem = emitted.shared_mem

    # Warmup
    for _ in range(warmup):
        _chk(driver, driver.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, smem, 0, arg_data_ptr, 0))
    _chk(driver, driver.cuCtxSynchronize())

    # Per-iteration timed runs
    latencies = []
    for _ in range(runs):
        start_ev = _chk(driver, driver.cuEventCreate(
            driver.CUevent_flags.CU_EVENT_DEFAULT))
        stop_ev = _chk(driver, driver.cuEventCreate(
            driver.CUevent_flags.CU_EVENT_DEFAULT))
        _chk(driver, driver.cuEventRecord(start_ev, 0))
        _chk(driver, driver.cuLaunchKernel(
            func, gx, gy, gz, bx, by, bz, smem, 0, arg_data_ptr, 0))
        _chk(driver, driver.cuEventRecord(stop_ev, 0))
        _chk(driver, driver.cuEventSynchronize(stop_ev))
        ms = _chk(driver, driver.cuEventElapsedTime(start_ev, stop_ev))
        latencies.append(float(ms))
        driver.cuEventDestroy(start_ev)
        driver.cuEventDestroy(stop_ev)

    # Cleanup
    for dptr in allocs:
        driver.cuMemFree(dptr)
    driver.cuModuleUnload(mod)

    return latencies, variant


def _chk(driver, ret):
    t = _as_tuple(ret)
    err = t[0]
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    rest = t[1:]
    return rest[0] if len(rest) == 1 else rest


def main():
    shapes = [128, 256, 512, 768, 1024]
    warmup = 10
    runs = 50

    print("=" * 80)
    print("Arke CUDA-C Matmul Kernel Benchmark (float4+double-buf vs torch.mm)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {warmup}, Timed runs: {runs}, Metric: median latency (ms)")
    print("=" * 80)

    header = f"{'N':>5} | {'Arke kernel':>40} | {'Arke med(ms)':>12} | {'torch med(ms)':>13} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for N in shapes:
        # Bench torch.mm
        torch_lats = bench_torch_mm(N, warmup=warmup, runs=runs)
        torch_med = statistics.median(torch_lats)

        # Bench Arke
        arke_lats, variant = bench_arke_kernel(N, warmup=warmup, runs=runs)
        arke_med = statistics.median(arke_lats)

        # Speedup = torch_time / arke_time (>1 means Arke is faster)
        speedup = torch_med / arke_med if arke_med > 0 else float("inf")

        # Truncate variant name for display
        short_variant = variant if len(variant) <= 40 else variant[:37] + "..."
        print(f"{N:>5} | {short_variant:>40} | {arke_med:>12.4f} | {torch_med:>13.4f} | {speedup:>7.3f}x")

        results.append({
            "N": N,
            "variant": variant,
            "arke_median_ms": arke_med,
            "arke_p25_ms": sorted(arke_lats)[len(arke_lats) // 4],
            "arke_p75_ms": sorted(arke_lats)[3 * len(arke_lats) // 4],
            "torch_median_ms": torch_med,
            "torch_p25_ms": sorted(torch_lats)[len(torch_lats) // 4],
            "torch_p75_ms": sorted(torch_lats)[3 * len(torch_lats) // 4],
            "speedup": speedup,
        })

    print()
    print("Detailed per-shape results:")
    print("-" * 80)
    for r in results:
        is_f4db = "f4db" in r["variant"]
        tag = "float4+double-buf" if is_f4db else "scalar-tiled"
        print(f"  N={r['N']:>4}  [{tag}]")
        print(f"    Kernel: {r['variant']}")
        print(f"    Arke : median={r['arke_median_ms']:.4f}ms  p25={r['arke_p25_ms']:.4f}ms  p75={r['arke_p75_ms']:.4f}ms")
        print(f"    Torch: median={r['torch_median_ms']:.4f}ms  p25={r['torch_p25_ms']:.4f}ms  p75={r['torch_p75_ms']:.4f}ms")
        print(f"    Speedup (torch/arke): {r['speedup']:.3f}x  {'(Arke FASTER)' if r['speedup'] > 1 else '(torch FASTER)'}")
        print()

    # Check the +19-35% claim
    print("=" * 80)
    print("Claim verification: float4+double-buf is +19% to +35% faster than baseline")
    print("  (Claim is about f4db vs scalar-tiled Arke kernel, NOT vs torch.mm)")
    print("  This bench measures f4db (or scalar) vs torch.mm (cuBLAS).")
    print("  Speedup > 1 means Arke kernel is faster than torch.mm.")
    print()
    for r in results:
        is_f4db = "f4db" in r["variant"]
        ratio_vs_cublas = r["arke_median_ms"] / r["torch_median_ms"] if r["torch_median_ms"] > 0 else 0
        print(f"  N={r['N']:>4}: Arke/cuBLAS = {ratio_vs_cublas:.3f}x  "
              f"({'f4db' if is_f4db else 'scalar'})  "
              f"{'→ Arke faster' if ratio_vs_cublas < 1 else '→ cuBLAS faster'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
