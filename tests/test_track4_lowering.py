# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.compiler.lowering import schedule_to_instruction, strategy_to_schedule
from arke.compiler.pipeline import ArkePipeline
from arke.ir.akir import akir_from_dict, akir_to_dict

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"


def test_strategy_to_schedule_lowering_on_example_relu():
    pipeline = ArkePipeline()
    result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))

    assert result.success, result.errors
    assert result.semantic_ir is not None
    assert result.strategy_ir is not None
    assert result.schedule_ir is not None
    assert result.instruction_ir is not None

    schedule = result.schedule_ir
    assert schedule.kernel_id == "relu_kernel_strategy"
    assert schedule.target_hw == "nvidia_ampere"
    assert len(schedule.loop_nests) == 1
    assert schedule.loop_nests[0].loop == "row"
    assert schedule.loop_nests[0].tile_factors == [4]
    assert schedule.resources.warps == 4
    assert schedule.resources.num_stages == 1
    assert schedule.resources.threads_per_block == 128


def test_schedule_to_instruction_materializes_schedule_effects():
    pipeline = ArkePipeline()
    result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))

    assert result.success, result.errors
    instr = result.instruction_ir
    assert instr is not None
    assert len(instr.blocks) == 1

    opcodes = [inst.opcode for inst in instr.blocks[0].instructions]
    assert "loop.configure" in opcodes
    assert "resource.bind" in opcodes


def test_akir_roundtrip_preserves_schedule_and_instruction_layers():
    pipeline = ArkePipeline()
    result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))

    assert result.success, result.errors
    payload = akir_to_dict(
        result.semantic_ir,
        result.strategy_ir,
        schedule_ir=result.schedule_ir,
        instruction_ir=result.instruction_ir,
    )

    semantic_rt, strategy_rt, schedule_rt, instruction_rt = akir_from_dict(payload)
    assert semantic_rt.kernel_id == result.semantic_ir.kernel_id
    assert strategy_rt is not None
    assert schedule_rt is not None
    assert instruction_rt is not None
    assert schedule_rt.loop_nests[0].tile_factors == [4]
    assert instruction_rt.blocks[0].instructions[0].opcode == "loop.configure"


def test_direct_lowering_helpers_match_pipeline_lowered_layers():
    pipeline = ArkePipeline()
    result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))

    assert result.success, result.errors
    schedule = strategy_to_schedule(result.semantic_ir, result.strategy_ir)
    instruction = schedule_to_instruction(result.semantic_ir, schedule)

    assert result.schedule_ir is not None
    assert result.instruction_ir is not None
    assert schedule.to_dict() == result.schedule_ir.to_dict()
    assert instruction.to_dict() == result.instruction_ir.to_dict()


def test_pipeline_load_akir_uses_persisted_lowered_layers(tmp_path):
    pipeline = ArkePipeline()
    result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
    assert result.success, result.errors

    akir_path = tmp_path / "relu.akir"
    result.save_akir(str(akir_path))
    loaded = ArkePipeline.load_akir(str(akir_path))

    assert loaded.success, loaded.errors
    assert loaded.schedule_ir is not None
    assert loaded.instruction_ir is not None
    assert loaded.schedule_ir.to_dict() == result.schedule_ir.to_dict()
    assert loaded.instruction_ir.to_dict() == result.instruction_ir.to_dict()
