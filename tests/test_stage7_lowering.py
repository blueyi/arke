# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.compiler.pipeline import ArkePipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS_DIR = REPO_ROOT / "examples" / "operators"


class TestStage7Lowering:
    def test_bl1_matmul_traverses_full_stage7_skeleton(self):
        pipeline = ArkePipeline()
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))

        assert result.success, result.errors
        assert result.semantic_ir is not None
        assert result.strategy_ir is not None
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None
        assert result.mlir_module is not None
        assert "module {" in result.mlir_module
        assert "func.func @matmul" in result.mlir_module
        assert "linalg.matmul" in result.mlir_module

    def test_strategy_kernel_materializes_schedule_and_instruction_ir(self):
        pipeline = ArkePipeline()
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))

        assert result.success, result.errors
        assert result.semantic_ir is not None
        assert result.strategy_ir is not None
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None

        schedule = result.schedule_ir
        instruction = result.instruction_ir

        assert schedule.kernel_id == result.semantic_ir.kernel_id
        assert instruction.kernel_id == result.semantic_ir.kernel_id
        assert len(schedule.loop_nests) >= 3
        assert any(loop.loop == "M" and loop.tile_factors for loop in schedule.loop_nests)
        assert any(loop.loop == "N" and loop.tile_factors for loop in schedule.loop_nests)
        assert any(loop.loop == "K" and loop.tile_factors for loop in schedule.loop_nests)
        assert schedule.fusion_groups
        assert schedule.resources.warps == 4
        assert schedule.resources.num_stages == 3
        assert len(schedule.provenance) >= 4

        assert instruction.blocks
        entry = instruction.blocks[0]
        opcodes = [inst.opcode for inst in entry.instructions]
        assert "loop.configure" in opcodes
        assert "resource.bind" in opcodes
        assert "fusion.group" in opcodes

        assert result.mlir_module is not None
        assert "func.func @matmul_gelu_kernel" in result.mlir_module
        assert '"arke.gelu"' in result.mlir_module or '"arke.matmul"' in result.mlir_module or "linalg.matmul" in result.mlir_module
        assert "instruction block: entry" in result.mlir_module

    def test_conditional_strategy_lowers_true_branch_and_provenance(self):
        pipeline = ArkePipeline()
        source = '''
kernel conditional_relu(
    X: Tensor<[B, D], f16>
) -> Tensor<[B, D], f16>
where B: dynamic(max=1024), D: static
{
    let Y = relu(X=X);
    return Y;
}

strategy conditional_relu_strategy for target("nvidia_ampere") {
    when dim("B") <= 128 {
        tile(dim="B", factors=[64])
            @rationale("small batch tile");
        compute(warps=4, num_stages=2)
            @rationale("small batch resources");
    } otherwise {
        tile(dim="B", factors=[128])
            @rationale("large batch tile");
    }
}
        '''
        result = pipeline.compile_string(source)

        assert result.success, result.errors
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None

        schedule = result.schedule_ir
        loop_b = next(loop for loop in schedule.loop_nests if loop.loop == "B")
        assert loop_b.tile_factors == [64]
        assert schedule.resources.warps == 4
        assert schedule.resources.num_stages == 2
        assert any(record.source_kind == "conditional" for record in schedule.provenance)

        entry = result.instruction_ir.blocks[0]
        assert any(inst.opcode == "loop.configure" for inst in entry.instructions)
        assert any(inst.opcode == "resource.bind" for inst in entry.instructions)
