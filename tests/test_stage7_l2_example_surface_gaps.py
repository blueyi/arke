# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.compiler.pipeline import ArkePipeline


EXAMPLES = {
    "matmul_relu": Path("examples/operators/l2/matmul_relu.ak"),
    "linear_ce": Path("examples/operators/l2/linear_ce.ak"),
    "qkv_fa": Path("examples/operators/l2/qkv_fa.ak"),
}


class TestStage7L2ExampleSurfaceGaps:
    def test_required_l2_examples_exist(self):
        missing = [name for name, path in EXAMPLES.items() if not path.exists()]
        assert missing == []

    def test_required_l2_examples_compile_with_expected_semantics(self):
        pipeline = ArkePipeline()
        result = pipeline.compile_file(str(EXAMPLES["matmul_relu"]))
        assert result.success, result.errors
        assert result.semantic_ir is not None
        ops = [node.op for node in result.semantic_ir.nodes]
        assert ops == ["matmul", "relu"]
        assert result.strategy_ir is not None
        assert any(getattr(decision, "kind", None) == "fuse" for decision in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.schedule_ir.fusion_groups
        assert result.schedule_ir.fusion_groups[0].ops == ["matmul", "relu"]
        assert result.schedule_ir.fusion_groups[0].fusion_type == "epilogue"

        result = pipeline.compile_file(str(EXAMPLES["linear_ce"]))
        assert result.success, result.errors
        assert result.semantic_ir is not None
        ops = [node.op for node in result.semantic_ir.nodes]
        assert ops == ["fused_linear_cross_entropy"]
        assert result.strategy_ir is not None
        assert any(getattr(decision, "kind", None) == "fuse" for decision in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.schedule_ir.fusion_groups
        assert result.schedule_ir.fusion_groups[0].ops == ["matmul", "cross_entropy"]
        assert result.schedule_ir.fusion_groups[0].fusion_type == "producer_consumer"

        result = pipeline.compile_file(str(EXAMPLES["qkv_fa"]))
        assert result.success, result.errors
        assert result.semantic_ir is not None
        ops = [node.op for node in result.semantic_ir.nodes]
        assert ops.count("matmul") == 3
        assert "flash_attention" in ops
        assert result.strategy_ir is not None
        assert any(getattr(decision, "kind", None) == "fuse" for decision in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.schedule_ir.fusion_groups
        assert result.schedule_ir.fusion_groups[0].ops == ["matmul", "matmul", "matmul", "flash_attention"]
        assert result.schedule_ir.fusion_groups[0].fusion_type == "producer_consumer"
