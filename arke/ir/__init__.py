# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR package.

Active multi-layer modules:
- semantic.py      — SemanticIR (Layer 4)
- strategy.py      — StrategyIR (Layer 3)
- schedule.py      — ScheduleIR (Layer 2)
- instruction.py   — InstructionIR (Layer 1)
- akir.py          — combined serialization format
"""

from arke.ir.instruction import Instruction, InstructionBlock, InstructionIR
from arke.ir.schedule import ScheduleIR
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

__all__ = [
    "SemanticIR",
    "StrategyIR",
    "ScheduleIR",
    "InstructionIR",
    "Instruction",
    "InstructionBlock",
]
