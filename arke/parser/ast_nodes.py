# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""AST node definitions for the Arke language."""

from __future__ import annotations

from dataclasses import dataclass, field


# ─── Types ───

@dataclass
class TensorType:
    """Tensor type: Tensor<[dims], dtype, layout?>."""
    shape: list[int]
    dtype: str
    layout: str = "row_major"


# ─── Kernel AST ───

@dataclass
class Param:
    """Kernel parameter: name: TensorType."""
    name: str
    type: TensorType


@dataclass
class OpCall:
    """Operator call: op_name(args)."""
    op: str
    args: dict[str, str]  # named: {param_name: var_name}


@dataclass
class LetStmt:
    """let var = op_call;"""
    name: str
    value: OpCall


@dataclass
class ReturnStmt:
    """return var;"""
    name: str


@dataclass
class KernelDef:
    """kernel name(params) -> return_type { body }"""
    name: str
    params: list[Param]
    return_type: TensorType
    body: list[LetStmt | ReturnStmt]

    @property
    def return_var(self) -> str | None:
        """Get the return variable name."""
        for stmt in reversed(self.body):
            if isinstance(stmt, ReturnStmt):
                return stmt.name
        return None


# ─── Strategy AST ───

@dataclass
class Annotation:
    """@rationale("text") or @key("value")."""
    key: str
    value: str


@dataclass
class StrategyAction:
    """strategy action: tile(loop="i", factors=[64, 16])"""
    action: str
    params: dict[str, object]
    annotation: Annotation | None = None


@dataclass
class StrategyDef:
    """strategy name for target("hw") { actions }"""
    name: str
    target: str
    actions: list[StrategyAction]


# ─── Import ───

@dataclass
class ImportStmt:
    """import "path" as alias;"""
    path: str
    alias: str


# ─── Program ───

@dataclass
class Program:
    """Top-level AST node: list of definitions."""
    imports: list[ImportStmt] = field(default_factory=list)
    kernels: list[KernelDef] = field(default_factory=list)
    strategies: list[StrategyDef] = field(default_factory=list)

    def get_kernel(self, name: str) -> KernelDef | None:
        for k in self.kernels:
            if k.name == name:
                return k
        return None

    def get_strategy(self, name: str) -> StrategyDef | None:
        for s in self.strategies:
            if s.name == name:
                return s
        return None
