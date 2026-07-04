# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU perf infra): kernel-only CUDA-event timing.

The correctness ``run()`` path rebuilds a CUDA context + reloads PTX + copies
H2D/D2H on every call, so its wall time is dominated by one-time overhead and is
useless for a perf comparison. ``MLIRGPUBackend.benchmark()`` (backed by
``CudaLauncher.time_kernel``) reuses one context, loads PTX once, copies inputs
once, and times ``iters`` back-to-back launches with CUDA events — the fair
kernel-only latency to compare against Triton/torch (both timed kernel-only).

This suite validates the timing *interface* (positive ms, correct-then-timed,
stable across iters) — NOT a perf threshold, which is the still-open P3-S2
perf-half baseline decision. Skips cleanly without the GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _matmul_graph(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


def _relu_graph(M: int, N: int) -> IRGraph:
    g = IRGraph(name="relu")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op="relu", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def test_benchmark_returns_positive_ms():
    be = MLIRGPUBackend()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 64)).astype(np.float32)
    B = rng.standard_normal((64, 64)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(64, 64, 64)))
    assert ker.success, ker.error
    ms = be.benchmark(ker, {"A": A, "B": B}, iters=10, warmup=3)
    assert ms > 0.0
    assert ms < 10_000.0  # sanity: a 64^3 kernel is well under 10s


def test_benchmark_matches_run_correctness():
    """benchmark() must exercise the same kernel run() validates as correct."""
    be = MLIRGPUBackend()
    rng = np.random.default_rng(1)
    A = rng.standard_normal((32, 16)).astype(np.float32)
    B = rng.standard_normal((16, 24)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(32, 16, 24)))
    out = be.run(ker, {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3)
    ms = be.benchmark(ker, {"A": A, "B": B}, iters=10, warmup=3)
    assert ms > 0.0


def test_benchmark_elementwise():
    be = MLIRGPUBackend()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((128, 128)).astype(np.float32)
    ker = be.compile(be.lower(_relu_graph(128, 128)))
    assert ker.success, ker.error
    ms = be.benchmark(ker, {"X": X}, iters=20, warmup=5)
    assert ms > 0.0


def test_benchmark_stable_across_iters():
    """More iters should not change the per-launch mean by orders of magnitude."""
    be = MLIRGPUBackend()
    rng = np.random.default_rng(3)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    B = rng.standard_normal((128, 128)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(128, 128, 128)))
    ms_short = be.benchmark(ker, {"A": A, "B": B}, iters=10, warmup=3)
    ms_long = be.benchmark(ker, {"A": A, "B": B}, iters=50, warmup=5)
    assert 0.2 * ms_short <= ms_long <= 5.0 * ms_short
