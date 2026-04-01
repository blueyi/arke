# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark task definitions for Arke evaluation.

Each task defines:
- A kernel (via KernelBuilder)
- Input shapes and dtypes
- Target hardware
- Expected baseline performance range
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arke.ir.builder import KernelBuilder
from arke.ir.semantic import SemanticIR


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    name: str
    description: str
    semantic_ir: SemanticIR
    target_hw: str = "nvidia_ampere"
    dtype: str = "f16"
    tags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        shapes = [f"{p.name}:{p.shape}" for p in self.semantic_ir.params]
        return f"Task({self.name}, {', '.join(shapes)})"


def _build_matmul(
    name: str, M: int, N: int, K: int, dtype: str = "f16"
) -> SemanticIR:
    b = KernelBuilder(name)
    b.param("A", [M, K], dtype)
    b.param("B", [K, N], dtype)
    m = b.op("matmul", A="A", B="B")
    b.returns(m, [M, N], dtype)
    return b.build()


def _build_matmul_relu(
    name: str, M: int, N: int, K: int, dtype: str = "f16"
) -> SemanticIR:
    b = KernelBuilder(name)
    b.param("A", [M, K], dtype)
    b.param("B", [K, N], dtype)
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [M, N], dtype)
    return b.build()


def _build_softmax(
    name: str, M: int, N: int, dtype: str = "f16"
) -> SemanticIR:
    b = KernelBuilder(name)
    b.param("X", [M, N], dtype)
    s = b.op("softmax", X="X")
    b.returns(s, [M, N], dtype)
    return b.build()


def _build_matmul_gelu(
    name: str, M: int, N: int, K: int, dtype: str = "f16"
) -> SemanticIR:
    b = KernelBuilder(name)
    b.param("A", [M, K], dtype)
    b.param("B", [K, N], dtype)
    m = b.op("matmul", A="A", B="B")
    g = b.op("gelu", X=m)
    b.returns(g, [M, N], dtype)
    return b.build()


# ============================================================
# Task Registry
# ============================================================

BENCHMARK_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        name="matmul_1024",
        description="Square matmul 1024×1024, the baseline GPU kernel",
        semantic_ir=_build_matmul("matmul_1024", 1024, 1024, 1024),
        tags=["matmul", "square", "core"],
    ),
    BenchmarkTask(
        name="matmul_2048",
        description="Large square matmul 2048×2048, higher compute intensity",
        semantic_ir=_build_matmul("matmul_2048", 2048, 2048, 2048),
        tags=["matmul", "square", "large"],
    ),
    BenchmarkTask(
        name="matmul_rect",
        description=(
            "Rectangular matmul 1024×512×2048, "
            "tests non-square tiling strategies"
        ),
        semantic_ir=_build_matmul("matmul_rect", 1024, 2048, 512),
        tags=["matmul", "rectangular"],
    ),
    BenchmarkTask(
        name="softmax_4096",
        description="Softmax on 4096×4096 matrix, memory-bound kernel",
        semantic_ir=_build_softmax("softmax_4096", 4096, 4096),
        tags=["softmax", "memory_bound"],
    ),
    BenchmarkTask(
        name="fused_matmul_relu",
        description=(
            "Matmul 1024×1024 + ReLU epilogue fusion, "
            "tests fusion decision quality"
        ),
        semantic_ir=_build_matmul_relu(
            "fused_matmul_relu", 1024, 1024, 1024
        ),
        tags=["matmul", "fusion", "relu"],
    ),
    BenchmarkTask(
        name="fused_matmul_gelu",
        description=(
            "Matmul 1024×2048 + GELU epilogue fusion, "
            "GELU is more complex than ReLU"
        ),
        semantic_ir=_build_matmul_gelu(
            "fused_matmul_gelu", 1024, 2048, 1024
        ),
        tags=["matmul", "fusion", "gelu"],
    ),
]


def get_tasks(
    tags: list[str] | None = None,
) -> list[BenchmarkTask]:
    """Get benchmark tasks, optionally filtered by tags."""
    if tags is None:
        return BENCHMARK_TASKS
    return [
        t for t in BENCHMARK_TASKS
        if any(tag in t.tags for tag in tags)
    ]


def get_task(name: str) -> BenchmarkTask:
    """Get a single task by name."""
    for t in BENCHMARK_TASKS:
        if t.name == name:
            return t
    available = [t.name for t in BENCHMARK_TASKS]
    raise ValueError(
        f"Unknown task: {name}. Available: {available}"
    )
