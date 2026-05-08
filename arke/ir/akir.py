# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

""".akir file format — Combined multi-layer Arke IR serialization.

The .akir format is the native persistence format for the active Arke IR stack:
SemanticIR (Layer 4), StrategyIR (Layer 3), ScheduleIR (Layer 2), and
InstructionIR (Layer 1).

File format (JSON):
    {
        "format": "akir",
        "version": "0.1.0",
        "semantic_ir": { ... SemanticIR.to_dict() ... },
        "strategy_ir": { ... StrategyIR.to_dict() ... },      // or null
        "schedule_ir": { ... ScheduleIR.to_dict() ... },      // or null
        "instruction_ir": { ... InstructionIR.to_dict() ... } // or null
    }
"""

from __future__ import annotations

import json
from typing import Any

from arke.ir.instruction import InstructionIR
from arke.ir.schedule import ScheduleIR
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR
from arke.version import IR_SCHEMA_VERSION

AKIR_FORMAT = "akir"
AKIR_VERSION = IR_SCHEMA_VERSION


def akir_to_dict(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR | None,
    schedule_ir: ScheduleIR | None = None,
    instruction_ir: InstructionIR | None = None,
) -> dict[str, Any]:
    """Convert the active multi-layer Arke IR stack to a .akir dict."""
    return {
        "format": AKIR_FORMAT,
        "version": AKIR_VERSION,
        "semantic_ir": semantic_ir.to_dict(),
        "strategy_ir": strategy_ir.to_dict() if strategy_ir is not None else None,
        "schedule_ir": schedule_ir.to_dict() if schedule_ir is not None else None,
        "instruction_ir": instruction_ir.to_dict() if instruction_ir is not None else None,
    }


def akir_from_dict(
    data: dict[str, Any],
) -> tuple[SemanticIR, StrategyIR | None, ScheduleIR | None, InstructionIR | None]:
    """Parse a .akir dict into the active multi-layer Arke IR stack."""
    fmt = data.get("format")
    if fmt != AKIR_FORMAT:
        raise ValueError(
            f"Invalid .akir format: expected {AKIR_FORMAT!r}, got {fmt!r}"
        )

    version = data.get("version")
    if version != AKIR_VERSION:
        raise ValueError(
            f"Unsupported .akir version: expected {AKIR_VERSION!r}, got {version!r}"
        )

    semantic_data = data.get("semantic_ir")
    if semantic_data is None:
        raise ValueError("Missing 'semantic_ir' field in .akir data")
    semantic_ir = SemanticIR.from_dict(semantic_data)

    strategy_data = data.get("strategy_ir")
    strategy_ir = (
        StrategyIR.from_dict(strategy_data) if strategy_data is not None else None
    )

    schedule_data = data.get("schedule_ir")
    schedule_ir = (
        ScheduleIR.from_dict(schedule_data) if schedule_data is not None else None
    )

    instruction_data = data.get("instruction_ir")
    instruction_ir = (
        InstructionIR.from_dict(instruction_data) if instruction_data is not None else None
    )

    return semantic_ir, strategy_ir, schedule_ir, instruction_ir


def save_akir(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR | None,
    path: str,
    indent: int = 2,
    schedule_ir: ScheduleIR | None = None,
    instruction_ir: InstructionIR | None = None,
) -> None:
    """Save the active multi-layer Arke IR stack to a .akir JSON file."""
    combined = akir_to_dict(
        semantic_ir,
        strategy_ir,
        schedule_ir=schedule_ir,
        instruction_ir=instruction_ir,
    )
    with open(path, "w") as f:
        json.dump(combined, f, indent=indent)


def load_akir(
    path: str,
) -> tuple[SemanticIR, StrategyIR | None, ScheduleIR | None, InstructionIR | None]:
    """Load the active multi-layer Arke IR stack from a .akir JSON file."""
    with open(path) as f:
        data = json.load(f)
    return akir_from_dict(data)
