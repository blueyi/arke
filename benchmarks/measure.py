# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified measurement utilities for benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class BenchResult:
    """Result of a single kernel measurement."""

    latency_us: float  # mean latency in microseconds
    latency_min_us: float  # min latency
    latency_max_us: float  # max latency
    tflops: float | None = None  # achieved TFLOPS (compute-bound ops)
    gbps: float | None = None  # achieved GB/s (memory-bound ops)


def bench_fn(
    fn: Callable,
    warmup: int = 200,
    reps: int = 500,
) -> BenchResult:
    """Benchmark a callable using CUDA events.

    Args:
        fn: Zero-arg callable to benchmark (should call GPU kernel).
        warmup: Number of warmup iterations.
        reps: Number of timed iterations.

    Returns:
        BenchResult with latency statistics.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure overall
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    mean_us = start.elapsed_time(end) / reps * 1000

    # Measure individual runs for min/max (subsample to avoid overhead)
    n_individual = min(reps, 50)
    individual_us = []
    for _ in range(n_individual):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        individual_us.append(s.elapsed_time(e) * 1000)

    return BenchResult(
        latency_us=mean_us,
        latency_min_us=min(individual_us),
        latency_max_us=max(individual_us),
    )


def compute_matmul_tflops(M: int, N: int, K: int, latency_us: float) -> float:
    """Compute TFLOPS for a matmul operation."""
    flops = 2 * M * N * K  # multiply + add
    return flops / (latency_us * 1e-6) / 1e12


def compute_bandwidth_gbps(
    bytes_moved: int, latency_us: float
) -> float:
    """Compute effective bandwidth in GB/s."""
    return bytes_moved / (latency_us * 1e-6) / 1e9
