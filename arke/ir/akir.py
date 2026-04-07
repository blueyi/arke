# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

""".akir file format — Combined SemanticIR + StrategyIR serialization.

The .akir format bundles a SemanticIR and an optional StrategyIR into a single
JSON file with format metadata, enabling full round-trip persistence.

File format (JSON):
    {
        "format": "akir",
        "version": "1.0.0",
        "semantic_ir": { ... SemanticIR.to_dict() ... },
        "strategy_ir": { ... StrategyIR.to_dict() ... }  // or null
    }

Usage:
    from arke.ir.akir import save_akir, load_akir

    save_akir(semantic_ir, strategy_ir, "kernel.akir")
    semantic_ir, strategy_ir = load_akir("kernel.akir")
"""

from __future__ import annotations

import json
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

AKIR_FORMAT = "akir"
AKIR_VERSION = "1.0.0"


def akir_to_dict(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR | None,
) -> dict[str, Any]:
    """Convert SemanticIR + StrategyIR to a combined dict with format metadata.

    Args:
        semantic_ir: The semantic IR to serialize.
        strategy_ir: The strategy IR to serialize (may be None).

    Returns:
        Combined dict with format, version, semantic_ir, and strategy_ir keys.
    """
    return {
        "format": AKIR_FORMAT,
        "version": AKIR_VERSION,
        "semantic_ir": semantic_ir.to_dict(),
        "strategy_ir": strategy_ir.to_dict() if strategy_ir is not None else None,
    }


def akir_from_dict(
    data: dict[str, Any],
) -> tuple[SemanticIR, StrategyIR | None]:
    """Parse a combined .akir dict into SemanticIR + StrategyIR.

    Args:
        data: Dict with format metadata and IR data.

    Returns:
        Tuple of (SemanticIR, StrategyIR or None).

    Raises:
        ValueError: If the dict is not a valid .akir format.
    """
    # Validate format
    fmt = data.get("format")
    if fmt != AKIR_FORMAT:
        raise ValueError(
            f"Invalid .akir format: expected {AKIR_FORMAT!r}, got {fmt!r}"
        )

    version = data.get("version")
    if not version:
        raise ValueError("Missing 'version' field in .akir data")

    # Parse semantic IR (required)
    semantic_data = data.get("semantic_ir")
    if semantic_data is None:
        raise ValueError("Missing 'semantic_ir' field in .akir data")
    semantic_ir = SemanticIR.from_dict(semantic_data)

    # Parse strategy IR (optional)
    strategy_data = data.get("strategy_ir")
    strategy_ir = (
        StrategyIR.from_dict(strategy_data) if strategy_data is not None else None
    )

    return semantic_ir, strategy_ir


def save_akir(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR | None,
    path: str,
    indent: int = 2,
) -> None:
    """Save SemanticIR + StrategyIR to a .akir JSON file.

    Args:
        semantic_ir: The semantic IR to save.
        strategy_ir: The strategy IR to save (may be None).
        path: Output file path.
        indent: JSON indentation (default 2).
    """
    combined = akir_to_dict(semantic_ir, strategy_ir)
    with open(path, "w") as f:
        json.dump(combined, f, indent=indent)


def load_akir(path: str) -> tuple[SemanticIR, StrategyIR | None]:
    """Load SemanticIR + StrategyIR from a .akir JSON file.

    Args:
        path: Path to the .akir file.

    Returns:
        Tuple of (SemanticIR, StrategyIR or None).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not valid .akir format.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path) as f:
        data = json.load(f)
    return akir_from_dict(data)
