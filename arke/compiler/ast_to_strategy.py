# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke compiler — AST to Strategy IR converter.

Converts a parsed ``StrategyDef`` AST node (from a ``.ak`` file) into a
``StrategyIR`` object that the rest of the pipeline can consume.

Also exposes ``program_to_strategy(program, kernel_name)`` which looks up
the matching strategy definition in a ``Program`` and converts it.

Design:
  - ``StrategyDef.name`` is usually ``<kernel_name>_strategy``
  - ``StrategyDef.target`` is the hardware target string (e.g. "nvidia_ampere")
  - ``StrategyAction.action`` maps 1:1 to ``Decision.kind``
  - ``StrategyAction.params`` maps 1:1 to ``Decision.params``
  - ``@rationale("text")`` maps to ``Decision.rationale``
"""

from __future__ import annotations

from arke.ir.strategy import Decision, Rationale, StrategyIR
from arke.parser.ast_nodes import Program, StrategyAction, StrategyDef


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ast_to_strategy(
    strategy_def: StrategyDef,
    kernel_id: str = "",
) -> StrategyIR:
    """Convert a ``StrategyDef`` AST node to a ``StrategyIR``.

    Args:
        strategy_def: The parsed strategy AST node.
        kernel_id:    The semantic IR kernel id to reference (optional;
                      defaults to ``strategy_def.name``).

    Returns:
        A fully populated ``StrategyIR``.
    """
    target = strategy_def.target

    # Strip surrounding quotes if present (parser may include them)
    if isinstance(target, str) and len(target) >= 2 and target[0] in ('"', "'"):
        target = target[1:-1]

    kid = kernel_id or strategy_def.name

    ir = StrategyIR(
        kernel_id=kid,
        target_hw=target,
    )

    for action in strategy_def.actions:
        decision = _action_to_decision(action)
        ir.add_decision(decision)

    return ir


def program_to_strategy(
    program: Program,
    kernel_name: str,
    kernel_id: str = "",
) -> StrategyIR | None:
    """Look up the strategy definition for *kernel_name* in *program* and convert it.

    Convention: the strategy is named ``<kernel_name>_strategy`` or
    exactly ``kernel_name``.

    Returns ``None`` if no matching strategy definition is found.
    """
    # Try exact match first, then <kernel_name>_strategy
    candidates = [
        kernel_name,
        f"{kernel_name}_strategy",
        f"{kernel_name}_strat",
    ]
    for candidate in candidates:
        sdef = program.get_strategy(candidate)
        if sdef is not None:
            return ast_to_strategy(sdef, kernel_id=kernel_id or kernel_name)

    # Fall back to the first strategy if only one exists
    if len(program.strategies) == 1:
        return ast_to_strategy(program.strategies[0], kernel_id=kernel_id or kernel_name)

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _action_to_decision(action: StrategyAction) -> Decision:
    """Convert a single ``StrategyAction`` to a ``Decision``."""
    kind = action.action
    params = _convert_params(action.params)
    rationale: Rationale | None = None

    if action.annotation is not None:
        ann = action.annotation
        if ann.key == "rationale":
            text = ann.value
            if isinstance(text, str) and len(text) >= 2 and text[0] in ('"', "'"):
                text = text[1:-1]
            rationale = Rationale(text=text)
        # Other annotations (e.g. @source) are stored in params for transparency
        else:
            params[f"@{ann.key}"] = ann.value

    return Decision(kind=kind, params=params, rationale=rationale)


def _convert_params(raw: dict) -> dict:
    """Recursively convert AST param values to Python native types."""
    result: dict = {}
    for key, value in raw.items():
        result[key] = _convert_value(value)
    return result


def _convert_value(value: object) -> object:
    """Convert a single AST value to a Python native type."""
    if isinstance(value, (bool, int, float, str)):
        # Strip string quotes if present
        if isinstance(value, str) and len(value) >= 2 and value[0] in ('"', "'"):
            return value[1:-1]
        return value
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {_convert_value(k): _convert_value(v) for k, v in value.items()}
    # Fallback: return as-is
    return value
