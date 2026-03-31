# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Semantic IR, Strategy IR, and supporting types."""

from arke.ir.semantic import (
    Edge,
    FusionGroup,
    InputRef,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticGraph,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.strategy import Decision, HardwareConstraints, Rationale, StrategyIR

__all__ = [
    "SemanticIR", "SemanticGraph",  # SemanticGraph is deprecated alias
    "Node", "TensorDesc", "Semantics", "Edge", "FusionGroup",
    "Param", "ParamRef", "NodeRef", "InputRef",
    "StrategyIR", "Decision", "Rationale", "HardwareConstraints",
]
