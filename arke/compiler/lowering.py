# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lowering helpers for multi-layer Arke IR.

Initial Track 4 skeleton:
- StrategyIR -> ScheduleIR
- ScheduleIR -> InstructionIR

The current implementation is intentionally minimal but structured, so later
passes can refine each lowering stage without changing the public shape.
"""

from __future__ import annotations

from arke.ir.instruction import Instruction, InstructionIR
from arke.ir.schedule import ScheduleDecisionRecord, ScheduleIR
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import ConditionalDecision, Decision, StrategyIR


class LoweringError(RuntimeError):
    """Raised when a lowering stage cannot proceed."""


def strategy_to_schedule(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR,
) -> ScheduleIR:
    """Lower StrategyIR (Layer 3) into ScheduleIR (Layer 2)."""
    schedule = ScheduleIR(
        kernel_id=strategy_ir.kernel_id or semantic_ir.kernel_id,
        target_hw=strategy_ir.target_hw,
        constraints=strategy_ir.constraints,
    )

    for dec in strategy_ir.decisions:
        if isinstance(dec, ConditionalDecision):
            _apply_conditional(schedule, dec)
        elif isinstance(dec, Decision):
            schedule.apply_decision(dec)
        else:
            raise LoweringError(f"Unsupported strategy decision type: {type(dec)!r}")

    return schedule


def schedule_to_instruction(
    semantic_ir: SemanticIR,
    schedule_ir: ScheduleIR,
) -> InstructionIR:
    """Lower ScheduleIR (Layer 2) into InstructionIR (Layer 1)."""
    instr = InstructionIR(
        kernel_id=schedule_ir.kernel_id or semantic_ir.kernel_id,
        target_hw=schedule_ir.target_hw,
        metadata={
            "node_count": len(semantic_ir.nodes),
            "return_node": semantic_ir.return_node,
        },
    )

    entry = instr.add_block("entry")

    for loop in schedule_ir.loop_nests:
        entry.instructions.append(Instruction(
            opcode="loop.configure",
            operands=[loop.loop],
            attrs=loop.to_dict(),
            comment="derived from ScheduleIR loop nest",
        ))

    for placement in schedule_ir.placements:
        entry.instructions.append(Instruction(
            opcode="memory.place",
            operands=[placement.tensor, placement.memory],
            comment="derived from ScheduleIR placement",
        ))

    if schedule_ir.resources.to_dict():
        entry.instructions.append(Instruction(
            opcode="resource.bind",
            attrs=schedule_ir.resources.to_dict(),
            comment="derived from compute/resource schedule",
        ))

    for fusion in schedule_ir.fusion_groups:
        entry.instructions.append(Instruction(
            opcode="fusion.group",
            operands=list(fusion.ops),
            attrs={"type": fusion.fusion_type},
        ))

    if not entry.instructions:
        entry.instructions.append(Instruction(
            opcode="nop",
            comment="no schedule-derived instructions generated",
        ))

    return instr


def lower_full_stack(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR | None,
) -> tuple[ScheduleIR | None, InstructionIR | None]:
    """Convenience lowering through Layer 2 and Layer 1 when strategy exists."""
    if strategy_ir is None:
        return None, None
    schedule = strategy_to_schedule(semantic_ir, strategy_ir)
    instruction = schedule_to_instruction(semantic_ir, schedule)
    return schedule, instruction


def _apply_conditional(schedule: ScheduleIR, decision: ConditionalDecision) -> None:
    """Initial conditional lowering policy.

    For the skeleton stage we conservatively materialize both branches into
    provenance via the normal decision application path, but only the true
    branch affects concrete schedule state. A later shape-specialization pass
    can turn this into explicit dispatched ScheduleIR regions.
    """
    for dec in decision.true_decisions:
        schedule.apply_decision(dec)
    schedule.provenance.append(ScheduleDecisionRecord(
        source_kind="conditional",
        source_step=decision.step,
        effect=f"predicate:{decision.predicate}",
        rationale=decision.rationale,
    ))
