# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Extended Operator Schema (S6 OpRegistry).

Extends the existing OpDefinition with new fields for:
- Declarative shape inference (shape_rule)
- Template routing hints (template_hint)
- PyTorch reference implementations (reference_impl)
- Test input generation rules (input_gen)
- Runtime attributes (attrs)

Extended fields are optional so operators can declare only the metadata they need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ShapeRule:
    """Declarative shape inference rule.

    Supported kinds:
    - "same_as_input": output shape = input_key tensor shape
    - "matmul_rule": [M,K] × [K,N] → [M,N]
    - "batch_matmul_rule": [B,M,K] × [B,K,N] → [B,M,N]
    - "reduce_rule": drop axes from input shape
    - "topk_rule": replace last dim with k (from attrs[k_attr])
    - "concat_rule": join along axis_attr
    - "split_rule": split into n parts along axis_attr
    - "gather_rule": shape from index tensor
    - "embedding_rule": [vocab,dim] indexed by [seq] → [seq,dim]
    - "permute_rule": reorder dims per dims_attr
    - "gated_halve_rule": halve last dim (swiglu/geglu)
    - "attention_rule": [B,H,S,D] from Q shape
    - "custom": delegate to fn(input_shapes, attrs) → list[int]
    """

    kind: str
    input_key: str = "X"
    axes: list[str | int] = field(default_factory=list)
    k_attr: str = "k"
    axis_attr: str = "axis"
    dims_attr: str = "dims"
    fn: Callable[[dict[str, list[int]], dict], list[int]] | None = None


@dataclass(frozen=True)
class TemplateHint:
    """Routing hint for the Triton template engine.

    Attributes:
        template_name: Jinja2 template filename without .j2 extension
        primary_op: anchor op for fused kernels; defaults to op.name
        extra_ctx: static key=value pairs injected into Jinja2 context
    """

    template_name: str
    primary_op: str = ""
    extra_ctx: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceImpl:
    """PyTorch eager reference for numerical validation.

    Attributes:
        fn: Callable with signature (inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor
        dtype_map: promote dtypes before running, e.g. {"bf16": "f32"}
    """

    fn: Callable[[dict[str, torch.Tensor], dict], torch.Tensor]
    dtype_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InputGen:
    """Rules for generating test inputs.

    Attributes:
        distributions: per-input "uniform"|"normal"|"ones"|"eye"|"randint"|"bool_mask"
        ranges: per-input (low, high) for uniform/randint
        dtype_override: force dtype for all inputs in tests
        constraints: informational strings for the test harness
    """

    distributions: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    dtype_override: str | None = None
    constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpSchema:
    """Extended operator definition — Single Source of Truth for OpRegistry.

    This extends the original catalog schema with S6 infrastructure fields.
    """

    # Original fields (from catalog.py OpDefinition)
    name: str
    category: str  # "compute" | "elementwise" | "reduce" | "move" | "attention" | "norm" | "gated"
    inputs: dict[str, str]  # {"A": "Tensor[M,K]", "B": "Tensor[K,N]"}
    output: str  # "Tensor[M,N]"
    computation: str  # "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str] = field(default_factory=list)
    reduction_axes: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    can_fuse_as: str | None = None  # "epilogue" | "prologue" | None
    numpy_ref: str = ""  # "np.matmul(A, B)"

    # New S6 fields (optional per operator)
    shape_rule: ShapeRule | None = None
    template_hint: TemplateHint | None = None
    reference_impl: ReferenceImpl | None = None
    input_gen: InputGen | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
