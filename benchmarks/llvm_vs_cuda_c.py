#!/usr/bin/env python3
"""P5-S3 gate evidence: Arke LLVM backend vs Arke CUDA-C backend (same-backend fairness).

The P5-S3 exit criterion is "LLVM geomean >= C-like + 5% (Cat A+C+D)".
All prior benchmarks compared against PyTorch; this script produces the
LLVM-vs-CUDA-C comparison the gate actually requires.

Both backends consume identical IRGraphs. Timing is kernel-only:
  - LLVM: prepare() once, then run_fast_no_copy() in a loop (median of TRIALS).
  - CUDA-C: backend.benchmark() -> mean kernel-only ms via CUDA events.

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    export PATH=/usr/local/cuda-13.2/bin:$PATH
    python benchmarks/llvm_vs_cuda_c.py
"""

import gc
import statistics
import time
from math import exp, log

import numpy as np
import torch

from arke.backend.cuda_c_backend import CudaCBackend, _ir_dtype_to_numpy
from arke.backend.llvm_backend import LLVMBackend
from arke.ir.graph import IRGraph, IRNode

WARMUP = 30
TRIALS = 100
CUDA_C_ITERS = 100
CUDA_C_WARMUP = 30
DEVICE = "cuda"


def _make_inputs(emitted):
    """Build numpy inputs matching an emitted kernel's params (excl. output)."""
    inputs = {}
    for name in emitted.param_names:
        if name == emitted.output_name:
            continue
        s = emitted.shapes[name]
        dt = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
        inputs[name] = np.random.randn(*s).astype(dt)
    return inputs


def _bench_llvm(backend, graph_fn):
    """Kernel-only median latency (us) for the LLVM backend."""
    gc.collect()
    torch.cuda.empty_cache()
    g = graph_fn()
    kern = backend.compile(backend.lower(g))
    if not kern.success:
        return None
    emitted = kern.metadata["emitted"]
    cached = backend.prepare(kern)
    inputs = _make_inputs(emitted)
    backend.run_fast(cached, inputs)  # H2D so GPU data is resident
    # CUDA-event batched timing — apples-to-apples with CudaCBackend.benchmark
    ms = backend.benchmark_cached(cached, iters=TRIALS, warmup=WARMUP)
    backend.release(cached)
    return ms * 1e3  # ms -> us


def _bench_cuda_c(backend, graph_fn):
    """Kernel-only mean latency (us) for the CUDA-C backend."""
    gc.collect()
    torch.cuda.empty_cache()
    g = graph_fn()
    kern = backend.compile(backend.lower(g))
    if not kern.success:
        return None
    emitted = kern.metadata["emitted"]
    inputs = _make_inputs(emitted)
    ms = backend.benchmark(kern, inputs, iters=CUDA_C_ITERS, warmup=CUDA_C_WARMUP)
    return ms * 1e3  # ms -> us


# --- Graph builders (Cat A elementwise, Cat C reduction, Cat D matmul/fused) ---

def _g_unary(op, m, n):
    def f():
        g = IRGraph(name="b")
        g.add_input("X", dtype="float32", shape=[m, n])
        g.add_node(IRNode(id="n0", op=op, inputs={"X": "X"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def _g_binary(op, m, n):
    def f():
        g = IRGraph(name="b")
        g.add_input("A", dtype="float32", shape=[m, n])
        g.add_input("B", dtype="float32", shape=[m, n])
        g.add_node(IRNode(id="n0", op=op, inputs={"A": "A", "B": "B"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def _g_softmax(m, n):
    def f():
        g = IRGraph(name="b")
        g.add_input("X", dtype="float32", shape=[m, n])
        g.add_node(IRNode(id="n0", op="softmax", inputs={"X": "X"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def _g_layernorm(m, n):
    def f():
        g = IRGraph(name="b")
        g.add_input("X", dtype="float32", shape=[m, n])
        g.add_input("W", dtype="float32", shape=[1, n])
        g.add_input("Bias", dtype="float32", shape=[1, n])
        g.add_node(IRNode(id="n0", op="layernorm",
                          inputs={"X": "X", "W": "W", "Bias": "Bias"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def _g_rmsnorm(m, n):
    def f():
        g = IRGraph(name="b")
        g.add_input("X", dtype="float32", shape=[m, n])
        g.add_input("W", dtype="float32", shape=[1, n])
        g.add_node(IRNode(id="n0", op="rmsnorm", inputs={"X": "X", "W": "W"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def _g_matmul(s):
    def f():
        g = IRGraph(name="b")
        g.add_input("A", dtype="float32", shape=[s, s])
        g.add_input("B", dtype="float32", shape=[s, s])
        g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    return f


def main():
    assert torch.cuda.is_available()
    torch.cuda.init()
    llvm = LLVMBackend(chip="sm_86")
    cudac = CudaCBackend(chip="sm_86")

    print("Arke LLVM vs CUDA-C — Kernel-Only (same-backend fairness, P5-S3 gate)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LLVM: median of {TRIALS}; CUDA-C: mean of {CUDA_C_ITERS} (CUDA events)")
    print()
    print(f"{'cat':<4} {'op':<15} {'shape':<12} {'LLVM(us)':>10} {'CUDA-C(us)':>11} {'ratio':>8}")
    print("-" * 66)

    # (category, op_name, shape_label, shape_str, llvm_graph, cudac_graph)
    cases = []

    # Cat A — elementwise
    for M, N in [(4096, 4096), (128, 4096), (32, 4096)]:
        for op in ["relu", "silu", "gelu"]:
            cases.append(("A", op, f"{M}x{N}", _g_unary(op, M, N)))
        for op in ["silu_and_mul", "gelu_and_mul"]:
            cases.append(("A", op, f"{M}x{N}", _g_binary(op, M, N)))

    # Cat C — reduction
    for M, N in [(1024, 4096), (32, 4096)]:
        cases.append(("C", "softmax", f"{M}x{N}", _g_softmax(M, N)))
        cases.append(("C", "layernorm", f"{M}x{N}", _g_layernorm(M, N)))
        cases.append(("C", "rmsnorm", f"{M}x{N}", _g_rmsnorm(M, N)))

    # Cat D — matmul
    for s in [512, 1024]:
        cases.append(("D", "matmul", f"{s}x{s}", _g_matmul(s)))

    results = []
    for cat, op, shape_str, gf in cases:
        try:
            l = _bench_llvm(llvm, gf)
            c = _bench_cuda_c(cudac, gf)
        except Exception as e:
            print(f"{cat:<4} {op:<15} {shape_str:<12}  ERROR: {e}")
            continue
        if l is None or c is None:
            print(f"{cat:<4} {op:<15} {shape_str:<12}  (compile fail: llvm={l}, cudac={c})")
            continue
        ratio = l / c  # <1.0 means LLVM faster than CUDA-C
        marker = " <-LLVM wins" if ratio < 1.0 else ""
        print(f"{cat:<4} {op:<15} {shape_str:<12} {l:>10.1f} {c:>11.1f} {ratio:>7.2f}x{marker}")
        results.append((cat, op, shape_str, l, c, ratio))

    print("-" * 66)
    if results:
        wins = [r for r in results if r[5] < 1.0]
        geo_all = exp(statistics.mean([log(r[5]) for r in results]))
        print(f"\n{len(wins)}/{len(results)} ops: LLVM faster than CUDA-C")
        print(f"Geomean LLVM/CUDA-C ratio (all): {geo_all:.3f}x  (lower = LLVM faster)")
        for cat in ["A", "C", "D"]:
            sub = [r for r in results if r[0] == cat]
            if sub:
                g = exp(statistics.mean([log(r[5]) for r in sub]))
                print(f"  Cat {cat}: {g:.3f}x  ({len(sub)} ops)")
        # Gate: LLVM geomean >= C-like + 5%  =>  LLVM latency <= CUDA-C / 1.05
        acd = [r for r in results if r[0] in ("A", "C", "D")]
        g_acd = exp(statistics.mean([log(r[5]) for r in acd]))
        gate_pass = g_acd <= (1.0 / 1.05)
        print(f"\nP5-S3 gate (Cat A+C+D geomean <= 0.952x): "
              f"{g_acd:.3f}x -> {'PASS' if gate_pass else 'FAIL'}")

    llvm.release_all()


if __name__ == "__main__":
    main()
