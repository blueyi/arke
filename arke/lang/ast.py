# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke AST — Abstract Syntax Tree node definitions.

Maps to the Arke Language Spec v2.0 (docs/spec/arke-lang-spec-design.md).
Terminology: uses 'Strategy' (not 'Schedule') per naming-system.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

# ============================================================
# Type Nodes
# ============================================================

@dataclass(frozen=True)
class ScalarType:
    """Scalar data type: f16, f32, f64, bf16, i8, i16, i32, i64, etc."""
    name: str  # e.g., "f16", "f32", "i32"


@dataclass(frozen=True)
class TensorType:
    """Tensor type: Tensor<[shape], dtype, layout>.

    shape elements can be int (static dim) or str (symbolic dim).
    """
    shape: list[Union[int, str]]
    dtype: ScalarType
    layout: str = "row_major"  # row_major | col_major


@dataclass(frozen=True)
class InferType:
    """Infer type: _ (type inference placeholder)."""
    pass


@dataclass(frozen=True)
class TupleType:
    """Tuple of types: (T1, T2, ...).

    Used for multi-return kernels.
    """
    types: tuple[Union[TensorType, InferType], ...]


@dataclass(frozen=True)
class MemoryLevel:
    """Memory hierarchy level."""
    level: str  # global | shared | local | register


# ============================================================
# Import
# ============================================================

@dataclass
class ImportStmt:
    """import "path" as alias;"""
    path: str
    alias: str | None = None


# ============================================================
# Where Clause
# ============================================================

@dataclass(frozen=True)
class DimDecl:
    """Dimension declaration in where clause.

    kind: "static" or "dynamic"
    opts: dict with optional keys: max, min, multiple_of, default
    """
    name: str
    kind: str  # "static" | "dynamic"
    opts: dict[str, int] = field(default_factory=dict)


@dataclass
class WhereClause:
    """where M: dynamic(max=4096), K: static, N: static"""
    dims: list[DimDecl]


# ============================================================
# Annotation
# ============================================================

@dataclass
class Annotation:
    """@name(args)

    args can be positional strings or keyword arguments.
    """
    name: str
    args: list[Union[str, tuple[str, Any]]] = field(default_factory=list)


# ============================================================
# Expression Nodes
# ============================================================

@dataclass
class Identifier:
    """Variable or function name."""
    name: str


@dataclass
class OpCall:
    """Operator invocation with named arguments: matmul(A=X, B=W).

    args is a list of (name, value) tuples where value can be:
    - str (identifier reference)
    - int, float, bool
    - str starting with '"' (string literal)
    - list (array literal)
    """
    op: str
    args: list[tuple[str, Any]]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LetStmt:
    """let lhs = op_call;

    lhs: str for single binding, list[str] for tuple destructuring
    """
    lhs: Union[str, list[str]]
    op_call: OpCall


@dataclass
class ReturnStmt:
    """return expr;

    value: str (single ident) or list[str] (tuple return)
    """
    value: Union[str, Identifier, list[str]]


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
    return_type: Union[TensorType, InferType, TupleType]
    body: list[Union[LetStmt, ReturnStmt]]
    where_clause: WhereClause | None = None
    annotations: list[Annotation] = field(default_factory=list)


# ============================================================
# Strategy Nodes
# ============================================================

@dataclass
class Rationale:
    """@rationale annotation — natural language explanation."""
    text: str
    lang: str = "en"


@dataclass
class StrategyStmt:
    """Strategy statement: directive(kwargs) @annotation;

    directive: str — the directive name (tile, fuse, compute, etc.)
    kwargs: dict of keyword arguments
    annotations: list of annotations (e.g., @rationale)
    """
    directive: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class WhenBlock:
    """Conditional strategy block.

    arms: list of (condition, body) pairs from when clauses
    otherwise_body: optional body for the otherwise clause
    """
    arms: list[tuple[Any, list[Union[StrategyStmt, 'WhenBlock']]]]
    otherwise_body: list[Union[StrategyStmt, 'WhenBlock']] | None = None


@dataclass
class Condition:
    """Condition for when blocks."""
    pass


@dataclass
class CompareCondition(Condition):
    """Simple comparison: IDENT op INT"""
    ident: str
    op: str  # <=, <, >=, >, ==, !=
    value: int


@dataclass
class BoolCondition(Condition):
    """Boolean combination: cond and/or cond"""
    op: str  # "and" | "or"
    left: Condition
    right: Condition


@dataclass
class StrategyDef:
    """Top-level strategy definition."""
    name: str
    target: str  # e.g., "nvidia_ampere"
    body: list[Union[StrategyStmt, WhenBlock]]
    kernel_name: str | None = None


# ============================================================
# Program
# ============================================================

@dataclass
class Program:
    """A complete Arke program (imports + kernels + strategies)."""
    imports: list[ImportStmt] = field(default_factory=list)
    kernels: list[KernelDef] = field(default_factory=list)
    strategies: list[StrategyDef] = field(default_factory=list)

