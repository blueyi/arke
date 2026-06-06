# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — input generator (D8-F1.2).

Schema-driven tensor generator. Reads `OpSchema.input_gen.distributions /
ranges / dtype_override / constraints` and produces a `dict[str, torch.Tensor]`
deterministically (via seed) for any op + shape map.

Reference-impl validation (`verify_correctness`) needs reproducible inputs
across the candidate and reference backends. This module is the single
source of truth for that.

Design ref: docs/architecture/arke-harness.md §6 (verify_correctness)
"""

from __future__ import annotations

from typing import Any

import torch

from arke.ir.ops.registry import REGISTRY


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "f16": torch.float16, "float16": torch.float16,
    "f32": torch.float32, "float32": torch.float32,
    "f64": torch.float64, "float64": torch.float64,
    "i32": torch.int32, "int32": torch.int32,
    "i64": torch.int64, "int64": torch.int64,
}


def _resolve_dtype(name: str | None, default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return _DTYPE_ALIASES.get(name, default)


def generate_inputs(
    op_name: str,
    shapes: dict[str, list[int]],
    *,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    attrs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Generate input tensors for an operator, schema-driven.

    Args:
        op_name: must be in REGISTRY.
        shapes: per-input shape map (e.g. {"X": [4,128], "W": [128]}).
                Must cover all of `op.inputs` keys.
        seed: deterministic seed (per-call torch.Generator).
        dtype: default float dtype for distributions that don't override.
        device: target device.
        attrs: optional op-level attrs (unused here, kept for symmetry).

    Returns:
        dict mapping input name → torch.Tensor on `device` with shape from `shapes`.

    Reads from OpSchema.input_gen:
      - distributions: per-input "normal" | "uniform" | "ones" | "eye" |
                       "randint" | "bool_mask" (default: "normal")
      - ranges: per-input (low, high) used for uniform/randint
      - dtype_override: forces dtype for all generated tensors (overrides `dtype` arg)

    Raises:
        KeyError: op not in REGISTRY, or shape missing for an op input.
    """
    if op_name not in REGISTRY:
        raise KeyError(f"Unknown op: {op_name!r}")
    op = REGISTRY.get(op_name)
    _ = attrs  # reserved for future per-attr-driven generation

    gen = op.input_gen
    distributions: dict[str, str] = dict(gen.distributions) if gen else {}
    ranges: dict[str, tuple[float, float]] = dict(gen.ranges) if gen else {}
    if gen and gen.dtype_override is not None:
        dtype = _resolve_dtype(gen.dtype_override, dtype)

    device = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # Iterate every declared op input — fail loudly if shape missing
    input_names = list(op.inputs.keys()) if isinstance(op.inputs, dict) else list(op.inputs)
    out: dict[str, torch.Tensor] = {}
    for name in input_names:
        if name not in shapes:
            raise KeyError(
                f"shape missing for input {name!r} of op {op_name!r}; provided keys: {list(shapes.keys())}"
            )
        shape = list(shapes[name])
        dist = distributions.get(name, "normal")
        rng = ranges.get(name)

        if dist == "randint":
            lo, hi = (int(rng[0]), int(rng[1])) if rng else (0, 8)
            hi = max(hi, lo + 1)
            t = torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)
        elif dist == "uniform":
            lo, hi = (float(rng[0]), float(rng[1])) if rng else (-1.0, 1.0)
            t = torch.empty(shape, dtype=dtype).uniform_(lo, hi, generator=g)
        elif dist == "ones":
            t = torch.ones(shape, dtype=dtype)
        elif dist == "eye":
            # eye only sensible for 2-D square; fall back to ones otherwise
            if len(shape) == 2 and shape[0] == shape[1]:
                t = torch.eye(shape[0], dtype=dtype)
            else:
                t = torch.ones(shape, dtype=dtype)
        elif dist == "bool_mask":
            t = torch.randint(0, 2, shape, generator=g, dtype=torch.bool)
        else:  # "normal" (default)
            t = torch.empty(shape, dtype=dtype).normal_(0.0, 1.0, generator=g)

        out[name] = t.to(device)

    return out
