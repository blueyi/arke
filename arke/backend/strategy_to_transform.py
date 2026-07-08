# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""StrategyIR L2 → MLIR transform dialect lowering (P3-S5).

Converts StrategyIR optimization decisions into MLIR ``transform`` dialect
schedules. This is the key P3-S5 deliverable: proving that the same
StrategyIR decisions that drive Triton codegen in Phase 1 can also drive
MLIR-level transformations in Phase 3.

Supported decision kinds:
  - ``tile``     → ``transform.structured.tile_using_for``
  - ``reorder``  → ``transform.structured.interchange``
  - ``vectorize`` → ``transform.structured.vectorize``

Usage::

    from arke.ir.strategy import StrategyIR, Decision
    from arke.backend.strategy_to_transform import lower_strategy_to_transform

    sir = StrategyIR(op="matmul", decisions=[
        Decision(kind="tile", params={"loop": "all", "factors": [64, 64, 16]}, level=2),
    ])
    schedule_text = lower_strategy_to_transform(sir, linalg_op="linalg.matmul")
    # → MLIR transform.named_sequence text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arke.ir.strategy import StrategyIR, Decision


# ── linalg op metadata ─────────────────────────────────────────────
# Maps op name to (linalg_op_name, iteration_rank).

_LINALG_OPS: dict[str, tuple[str, int]] = {
    "matmul":    ("linalg.matmul",    3),   # M, N, K
    "matvec":    ("linalg.matvec",    2),   # M, K
    "batch_matmul": ("linalg.batch_matmul", 4),  # B, M, N, K
    # Generic linalg ops (via linalg.generic) — used for elementwise/reduction
    "reduce_sum":  ("linalg.generic", 2),   # M, K (reduce last dim)
    "layernorm":   ("linalg.generic", 2),   # M, K
    "softmax":     ("linalg.generic", 2),   # M, K
}


@dataclass
class TransformSchedule:
    """A compiled MLIR transform schedule ready for mlir-opt."""
    text: str                     # Full transform module text
    decisions_applied: list[str]  # List of decision summaries


def _emit_tile(linalg_op: str, rank: int, decision: Decision) -> list[str]:
    """Emit transform.structured.tile_using_for from a 'tile' decision."""
    factors = decision.params.get("factors", [])
    if not factors:
        return []
    # Pad or truncate to match rank
    tile_sizes = list(factors[:rank])
    while len(tile_sizes) < rank:
        tile_sizes.append(0)

    sizes_str = ", ".join(str(int(t)) for t in tile_sizes)
    n_nonzero = sum(1 for t in tile_sizes if t != 0)

    lines = []
    if n_nonzero > 0:
        loop_results = ", %loops:" + str(n_nonzero)
        loop_types = ", " + ", ".join(
            "!transform.any_op" for _ in range(n_nonzero)
        )
    else:
        loop_results = ""
        loop_types = ""

    lines.append(
        f'    %target_tile = transform.structured.match ops{{["{linalg_op}"]}} '
        "in %arg0 : (!transform.any_op) -> !transform.any_op"
    )
    lines.append(
        f"    %tiled{loop_results} = transform.structured.tile_using_for "
        f"%target_tile tile_sizes [{sizes_str}] : (!transform.any_op) -> "
        f"(!transform.any_op{loop_types})"
    )
    return lines


def _emit_interchange(linalg_op: str, decision: Decision) -> list[str]:
    """Emit transform.structured.interchange from a 'reorder' decision."""
    order = decision.params.get("order", [])
    if not order:
        return []
    order_str = ", ".join(str(int(i)) for i in order)
    lines = [
        f'    %target_ic = transform.structured.match ops{{["{linalg_op}"]}} '
        "in %arg0 : (!transform.any_op) -> !transform.any_op",
        f"    %ic = transform.structured.interchange %target_ic "
        f"iterator_interchange = [{order_str}] : (!transform.any_op) "
        "-> !transform.any_op",
    ]
    return lines


def _emit_vectorize(linalg_op: str, decision: Decision) -> list[str]:
    """Emit transform.structured.vectorize from a 'vectorize' decision."""
    lines = [
        f'    %target_vec = transform.structured.match ops{{["{linalg_op}"]}} '
        "in %arg0 : (!transform.any_op) -> !transform.any_op",
        "    transform.structured.vectorize %target_vec : !transform.any_op",
    ]
    return lines


def lower_strategy_to_transform(
    strategy: StrategyIR,
    linalg_op: str | None = None,
) -> TransformSchedule:
    """Lower a StrategyIR into an MLIR transform dialect schedule.

    Args:
        strategy: The StrategyIR containing optimization decisions.
        linalg_op: Override for the linalg op name (e.g. "linalg.matmul").
            If None, inferred from strategy.op via ``_LINALG_OPS``.

    Returns:
        TransformSchedule with the full MLIR text and applied decision list.
    """
    op_name = strategy.kernel_id
    if linalg_op is None:
        if op_name not in _LINALG_OPS:
            raise ValueError(
                f"Unknown op {op_name!r} — provide linalg_op explicitly. "
                f"Known: {sorted(_LINALG_OPS)}"
            )
        linalg_op, rank = _LINALG_OPS[op_name]
    else:
        # Infer rank from known ops or default to 3
        rank = _LINALG_OPS.get(op_name, (linalg_op, 3))[1]

    body_lines: list[str] = []
    summaries: list[str] = []

    for d in strategy.decisions:
        if not hasattr(d, 'kind'):
            continue  # Skip ConditionalDecision or other non-standard types
        if d.kind == "tile":
            lines = _emit_tile(linalg_op, rank, d)  # type: ignore[arg-type]
            if lines:
                body_lines.extend(lines)
                summaries.append(
                    f"tile({d.params.get('factors', [])})"
                )
        elif d.kind == "reorder":
            lines = _emit_interchange(linalg_op, d)  # type: ignore[arg-type]
            if lines:
                body_lines.extend(lines)
                summaries.append(
                    f"reorder({d.params.get('order', [])})"  # type: ignore[union-attr]
                )
        elif d.kind == "vectorize":
            lines = _emit_vectorize(linalg_op, d)  # type: ignore[arg-type]
            if lines:
                body_lines.extend(lines)
                summaries.append("vectorize")
        # Skip unknown kinds silently (L1 decisions don't map to transform ops)

    if not body_lines:
        # No transform decisions — emit a no-op schedule
        body_lines = ["    // no applicable transform decisions"]

    body_lines.append(
        "    transform.yield"
    )

    text = "\n".join([
        "module attributes {transform.with_named_sequence} {",
        "  transform.named_sequence @__transform_main("
        "%arg0: !transform.any_op {transform.readonly}) {",
        *body_lines,
        "  }",
        "}",
    ])

    return TransformSchedule(text=text, decisions_applied=summaries)
