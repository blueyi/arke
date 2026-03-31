# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke AST — Abstract Syntax Tree node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Type Nodes
# ============================================================

@dataclass(frozen=True)
class ScalarType:
    """Scalar data type: f16, f32, f64, bf16, i8, i16, i32, i64, etc."""
    name: str  # e.g., "f16", "f32", "i32"


@dataclass(frozen=True)
class TensorType:
    """Tensor type: Tensor<shape, dtype, layout>."""
    shape: list[int]
    dtype: ScalarType
    layout: str = "row_major"  # row_major | col_major | tiled | custom


@dataclass(frozen=True)
class MemoryLevel:
    """Memory hierarchy level."""
    level: str  # global | shared | local | register


# ============================================================
# Expression Nodes
# ============================================================

@dataclass
class Identifier:
    """Variable or function name."""
    name: str


@dataclass
class OpCall:
    """Operator invocation: matmul(A, B), relu(C), etc."""
    op: str
    args: list[Identifier | OpCall]
    kwargs: dict[str, str | int | float | list] = field(default_factory=dict)


@dataclass
class LetBinding:
    """let C = matmul(A, B);"""
    name: str
    value: OpCall


@dataclass
class ReturnStmt:
    """return Y;"""
    value: Identifier


# ============================================================
# Top-Level Nodes
# ============================================================

@dataclass
class Parameter:
    """Kernel parameter: A: Tensor<[1024, 512], f16>."""
    name: str
    type: TensorType


@dataclass
class KernelDef:
    """Top-level kernel definition."""
    name: str
    params: list[Parameter]
    return_type: TensorType
    body: list[LetBinding | ReturnStmt]


# ============================================================
# Schedule Nodes
# ============================================================

@dataclass
class Rationale:
    """@rationale annotation — natural language explanation."""
    text: str
    lang: str = "en"  # "en" | "zh" | ...


@dataclass
class ScheduleDirective:
    """A single schedule decision (tile, reorder, fuse, etc.)."""
    kind: str  # "tile" | "reorder" | "fuse" | "parallel" | "place"
    params: dict[str, str | int | float | list]
    rationale: Optional[Rationale] = None


@dataclass
class ScheduleDef:
    """Top-level schedule definition."""
    kernel_name: str
    target: str  # e.g., "nvidia_ampere", "ascend_a3"
    directives: list[ScheduleDirective]


# ============================================================
# Program
# ============================================================

@dataclass
class Program:
    """A complete Arke program (one or more kernels + schedules)."""
    kernels: list[KernelDef] = field(default_factory=list)
    schedules: list[ScheduleDef] = field(default_factory=list)
