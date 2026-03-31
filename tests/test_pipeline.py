# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for Arke Pipeline."""

from arke.pipeline import ArkePipeline


def test_pipeline_matmul_basic():
    """E2E: Build matmul → apply decisions → validate numerically."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul(1024, 512, 2048, "f16")

    result = pipeline.run(
        ir,
        target_hw="nvidia_ampere",
        decisions=[
            ("tile", {"loop": "i", "factors": [64, 16]}, "L2 cache alignment"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "coalescing"),
        ],
        validate_numerical=True,
        codegen=False,
    )

    assert result.kernel_id == "matmul"
    assert result.decisions == 2
    assert len(result.errors) == 0
    assert result.numerical_validation is not None
    assert result.numerical_validation["passed"] is True


def test_pipeline_matmul_relu_fused():
    """E2E: Build matmul+relu → fuse → tile → validate."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul_relu(1024, 512, 2048, "f16")

    # Get the node IDs from the IR
    node_ids = [n.id for n in ir.nodes]
    assert len(node_ids) == 2  # matmul + relu

    result = pipeline.run(
        ir,
        target_hw="nvidia_ampere",
        decisions=[
            ("fuse", {"ops": node_ids, "type": "epilogue"}, "eliminate intermediate write"),
            ("tile", {"loop": "i", "factors": [64, 16]}, "cache alignment"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "coalescing"),
        ],
        validate_numerical=True,
    )

    assert result.decisions == 3
    assert len(result.errors) == 0
    assert result.numerical_validation["passed"] is True
    # Check strategy IR has the decisions
    assert len(result.strategy_ir["decisions"]) == 3
    assert result.strategy_ir["decisions"][0]["kind"] == "fuse"


def test_pipeline_softmax():
    """E2E: Build softmax → tile → validate."""
    pipeline = ArkePipeline()
    ir = pipeline.build_softmax(1024, 2048, "f32")

    result = pipeline.run(
        ir,
        target_hw="nvidia_ampere",
        decisions=[
            ("tile", {"loop": "i", "factors": [64, 16]}, "row parallelism"),
        ],
        validate_numerical=True,
    )

    assert result.decisions == 1
    assert result.numerical_validation["passed"] is True


def test_pipeline_invalid_decision_stops():
    """E2E: Invalid decision stops the pipeline."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul(1024, 512, 2048, "f16")

    result = pipeline.run(
        ir,
        target_hw="nvidia_ampere",
        decisions=[
            ("tile", {"loop": "i", "factors": [64, 16]}, "good"),
            ("tile", {"loop": "i", "factors": [32]}, "duplicate — should fail"),
            ("tile", {"loop": "j", "factors": [128]}, "should not reach"),
        ],
    )

    assert result.decisions == 1  # Only first succeeded
    assert len(result.errors) == 1
    assert "already tiled" in result.errors[0].lower()


def test_pipeline_empty_decisions():
    """E2E: Pipeline works with no decisions."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul(256, 256, 256, "f32")

    result = pipeline.run(ir, "nvidia_ampere", validate_numerical=True)
    assert result.decisions == 0
    assert result.numerical_validation["passed"] is True


def test_pipeline_strategy_ir_serialized():
    """E2E: Strategy IR is properly serialized in result."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul(512, 512, 512, "f16")

    result = pipeline.run(
        ir,
        "nvidia_ampere",
        decisions=[
            ("tile", {"loop": "i", "factors": [64, 16]}, "test rationale"),
        ],
    )

    strategy = result.strategy_ir
    assert strategy["kernel_id"] == "matmul"
    assert strategy["target_hw"] == "nvidia_ampere"
    assert len(strategy["decisions"]) == 1
    d = strategy["decisions"][0]
    assert d["kind"] == "tile"
    assert d["rationale"]["text"] == "test rationale"


def test_pipeline_semantic_ir_complete():
    """E2E: Semantic IR in result has all fields."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul_relu(256, 128, 512, "f16")

    result = pipeline.run(ir, "nvidia_ampere")

    sem = result.semantic_ir
    assert sem["kernel_id"] == "fused_matmul_relu"
    assert len(sem["params"]) == 2
    assert sem["return_type"] is not None
    assert len(sem["nodes"]) == 2
    assert sem["return_node"] != ""


def test_pipeline_timing():
    """E2E: Pipeline reports duration."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul(64, 64, 64, "f32")

    result = pipeline.run(ir, "nvidia_ampere", validate_numerical=True)
    assert result.duration_seconds >= 0
    assert result.duration_seconds < 10  # Should be fast without GPU


def test_pipeline_full_optimization_sequence():
    """E2E: Full optimization sequence mimicking LLM decisions."""
    pipeline = ArkePipeline()
    ir = pipeline.build_matmul_relu(1024, 512, 2048, "f16")

    node_ids = [n.id for n in ir.nodes]

    result = pipeline.run(
        ir,
        target_hw="nvidia_ampere",
        decisions=[
            # Step 1: Fuse matmul + relu
            ("fuse", {"ops": node_ids, "type": "epilogue"},
             "relu is elementwise; fusing eliminates 4MB intermediate global write"),
            # Step 2: Tile M dimension
            ("tile", {"loop": "i", "factors": [64, 16]},
             "L2 cache line = 64 bytes, 16 threads per warp half"),
            # Step 3: Tile N dimension
            ("tile", {"loop": "j", "factors": [128, 16]},
             "maximize memory coalescing along N"),
            # Step 4: Memory placement
            ("place", {"tensor": "A_tile", "memory": "shared"},
             "A is broadcast across j; reused 16x"),
            # Step 5: Parallelization
            ("parallel", {"loops": ["i_outer", "j_outer"],
                          "mapping": {"i_outer": "block.x", "j_outer": "block.y"}},
             "Map outer loops to 2D grid for good SM occupancy"),
        ],
        validate_numerical=True,
    )

    assert result.decisions == 5
    assert len(result.errors) == 0
    assert result.numerical_validation["passed"] is True

    # Verify all decisions recorded with rationale
    for d in result.strategy_ir["decisions"]:
        assert d["rationale"] is not None
        assert len(d["rationale"]["text"]) > 0
