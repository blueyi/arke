# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke AST — Abstract Syntax Tree node definitions.

Maps to the Arke Language Spec (docs/spec/arke-language-spec.md).
Terminology: uses 'Strategy' (not 'Schedule') per naming-system.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    layout: str = "row_major"  # row_major | col_major


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
# Strategy Nodes
# ============================================================

@dataclass
class Rationale:
    """@rationale annotation — natural language explanation."""
    text: str
    lang: str = "en"


@dataclass
class StrategyDirective:
    """A single strategy decision (tile, reorder, fuse, etc.)."""
    kind: str  # "tile" | "reorder" | "fuse" | "parallel" | "place" | "vectorize" | "unroll" | "algorithm"
    params: dict[str, str | int | float | list]
    rationale: Rationale | None = None


@dataclass
class StrategyDef:
    """Top-level strategy definition."""
    kernel_name: str
    target: str  # e.g., "nvidia_ampere"
    directives: list[StrategyDirective]


# ============================================================
# Program
# ============================================================

@dataclass
class Program:
    """A complete Arke program (one or more kernels + strategies)."""
    kernels: list[KernelDef] = field(default_factory=list)
    strategies: list[StrategyDef] = field(default_factory=list)


# ============================================================
# Backward compatibility aliases (deprecated)
# ============================================================

ScheduleDirective = StrategyDirective
ScheduleDef = StrategyDef
