# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""BL4×L1 Full Operator Verification (S6 Track 5, G6-BL4).

Tests ALL 45 operators through the full pipeline:
1. Shape inference correctness
2. SemanticInterpreter (reference_impl) execution
3. MockBackend E2E (pipeline → backend → validate)

This is the Gate verification test — all must pass.
"""

import pytest
import torch

from arke.backend.mock_backend import MockBackend
from arke.compiler.passes import PassPipeline, SSAValidationPass, ShapeInferencePass
from arke.ir.graph import IRGraph, IRNode
from arke.ir.ops.interpreter import INTERPRETER
from arke.ir.ops.registry import REGISTRY
from arke.ir.ops.shape_engine import SHAPE_ENGINE


# ── Input generation for all 45 ops ──────────────────────────

def _gen_inputs(op_name: str, shapes: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Generate valid inputs for an operator given shapes."""
    op = REGISTRY.get(op_name)
    inputs = {}

    for inp_name, shape in shapes.items():
        if op.input_gen and inp_name in op.input_gen.distributions:
            dist = op.input_gen.distributions[inp_name]
            rng = op.input_gen.ranges.get(inp_name, (0, 10))
            if dist == "randint":
                lo, hi = int(rng[0]), int(rng[1])
                if hi <= lo:
                    hi = lo + 1
                inputs[inp_name] = torch.randint(lo, hi, shape)
            elif dist == "uniform":
                inputs[inp_name] = torch.empty(shape).uniform_(rng[0], rng[1])
            elif dist == "ones":
                inputs[inp_name] = torch.ones(shape)
            elif dist == "bool_mask":
                inputs[inp_name] = torch.randint(0, 2, shape, dtype=torch.bool)
            elif dist == "eye":
                inputs[inp_name] = torch.eye(shape[-1]).expand(shape)
            else:
                inputs[inp_name] = torch.randn(shape)
        else:
            inputs[inp_name] = torch.randn(shape)

    return inputs


# Shape configs for all 45 ops (ST1 = small, ST2 = medium)
OP_SHAPES = {
    # OT0: Elementwise
    "relu": {"X": [4, 64]},
    "gelu": {"X": [4, 64]},
    "silu": {"X": [4, 64]},
    "tanh": {"X": [4, 64]},
    "sigmoid": {"X": [4, 64]},
    "neg": {"X": [4, 64]},
    "exp": {"X": [4, 64]},
    "rsqrt": {"X": [4, 64]},
    "cast": {"X": [4, 64]},
    "add": {"A": [4, 64], "B": [4, 64]},
    "mul": {"A": [4, 64], "B": [4, 64]},
    "where_": {"cond": [4, 64], "A": [4, 64], "B": [4, 64]},

    # OT1: Reductions
    "softmax": {"X": [4, 128]},
    "layernorm": {"X": [4, 128], "W": [128], "B": [128]},
    "rmsnorm": {"X": [4, 128], "W": [128]},
    "reduce_sum": {"X": [4, 64]},
    "reduce_max": {"X": [4, 64]},
    "reduce_mean": {"X": [4, 64]},
    "argmax": {"X": [4, 64]},
    "topk": {"X": [4, 64]},
    "cumsum": {"X": [4, 64]},

    # OT2: Dense compute + data movement
    "matmul": {"A": [64, 32], "B": [32, 128]},
    "batch_matmul": {"A": [4, 16, 32], "B": [4, 32, 64]},
    "grouped_matmul": {"X": [4, 16, 32], "W": [8, 32, 64], "indices": [4]},
    "transpose": {"X": [8, 16]},
    "concat": {"A": [4, 8], "B": [4, 12]},
    "split": {"X": [4, 16]},
    "gather": {"X": [4, 32], "idx": [4, 8]},
    "scatter": {"X": [4, 32], "idx": [4, 8], "src": [4, 8]},
    "embedding": {"indices": [2, 8], "weight": [100, 32]},
    "permute": {"X": [2, 4, 8, 16]},
    "copy_": {"X": [4, 64]},

    # OT3: Gated + fused
    "swiglu": {"X": [4, 128]},
    "geglu": {"X": [4, 128]},
    "rmsnorm_residual": {"X": [4, 64], "residual": [4, 64], "W": [64]},
    "rope": {"X": [1, 2, 8, 32], "cos": [8, 16], "sin": [8, 16]},
    "cross_entropy": {"logits": [8, 100], "labels": [8]},
    "fused_linear_cross_entropy": {"X": [8, 64], "W": [100, 64], "labels": [8]},
    "quantize_per_token": {"X": [4, 64]},
    "dequantize_per_channel": {"X_int8": [4, 64], "scale": [64], "zero_point": [64]},

    # OT4: Attention
    "flash_attention": {"Q": [1, 2, 16, 32], "K": [1, 2, 16, 32], "V": [1, 2, 16, 32]},
    "grouped_query_attention": {"Q": [1, 8, 16, 32], "K": [1, 2, 16, 32], "V": [1, 2, 16, 32]},
    "multi_latent_attention": {"Q": [1, 8, 16, 32], "KV_compressed": [1, 16, 64], "W_uk": [64, 8, 32], "W_uv": [64, 8, 32]},
    "cross_attention": {"Q": [1, 2, 16, 32], "K": [1, 2, 32, 32], "V": [1, 2, 32, 32]},
    "paged_attention": {"Q": [1, 2, 1, 32], "K_cache": [8, 16, 2, 32], "V_cache": [8, 16, 2, 32], "block_table": [1, 4]},
}

# Attrs overrides
OP_ATTRS = {
    "cast": {"target_dtype": "float16"},
    "topk": {"k": 5},
    "permute": {"dims": [0, 2, 1, 3]},
}

# Expected output shapes
EXPECTED_SHAPES = {
    "matmul": [64, 128],
    "batch_matmul": [4, 16, 64],
    "grouped_matmul": [4, 16, 64],
    "relu": [4, 64], "gelu": [4, 64], "silu": [4, 64], "tanh": [4, 64],
    "sigmoid": [4, 64], "neg": [4, 64], "exp": [4, 64], "rsqrt": [4, 64],
    "cast": [4, 64],
    "add": [4, 64], "mul": [4, 64], "where_": [4, 64],
    "softmax": [4, 128],
    "layernorm": [4, 128], "rmsnorm": [4, 128],
    "reduce_sum": [4], "reduce_max": [4], "reduce_mean": [4],
    "argmax": [4], "topk": [4, 5], "cumsum": [4, 64],
    "transpose": [16, 8],
    "concat": [4, 20], "split": [4, 8],
    "gather": [4, 8], "scatter": [4, 32],
    "embedding": [2, 8, 32],
    "permute": [2, 8, 4, 16],
    "copy_": [4, 64],
    "swiglu": [4, 64], "geglu": [4, 64],
    "rmsnorm_residual": [4, 64],
    "rope": [1, 2, 8, 32],
    "cross_entropy": [], "fused_linear_cross_entropy": [],
    "quantize_per_token": [4, 64],
    "dequantize_per_channel": [4, 64],
    "flash_attention": [1, 2, 16, 32],
    "grouped_query_attention": [1, 8, 16, 32],
    "multi_latent_attention": [1, 8, 16, 32],
    "cross_attention": [1, 2, 16, 32],
    "paged_attention": [1, 2, 1, 32],
}


# ── Verify we cover all 45 ops ───────────────────────────────

def test_coverage_all_45_ops():
    """Verify OP_SHAPES covers every op in the registry."""
    registry_ops = {op.name for op in REGISTRY}
    shape_ops = set(OP_SHAPES.keys())
    missing = registry_ops - shape_ops
    extra = shape_ops - registry_ops
    assert not missing, f"Missing ops in OP_SHAPES: {missing}"
    assert not extra, f"Extra ops in OP_SHAPES not in registry: {extra}"
    assert len(registry_ops) == 45


# ── BL4-L1: Shape Inference ──────────────────────────────────

ALL_OP_NAMES = sorted(OP_SHAPES.keys())


@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_shape_inference(op_name):
    """Shape inference produces expected output shape for every op."""
    shapes = OP_SHAPES[op_name]
    attrs = OP_ATTRS.get(op_name, {})
    expected = EXPECTED_SHAPES.get(op_name)

    if expected is None:
        pytest.skip(f"No expected shape for {op_name}")

    result = SHAPE_ENGINE.infer(op_name, shapes, attrs)
    assert result == expected, f"{op_name}: got {result}, expected {expected}"


# ── BL4-L2: SemanticInterpreter Correctness ──────────────────

@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_interpreter_executes(op_name):
    """Every op executes without error through SemanticInterpreter."""
    shapes = OP_SHAPES[op_name]
    attrs = OP_ATTRS.get(op_name, {})

    torch.manual_seed(42)
    inputs = _gen_inputs(op_name, shapes)

    result = INTERPRETER.execute(op_name, inputs, attrs)
    assert isinstance(result, torch.Tensor), f"{op_name}: expected Tensor, got {type(result)}"

    # Check output is finite (no NaN/Inf) for float tensors
    if result.is_floating_point():
        assert torch.isfinite(result).all(), f"{op_name}: output contains NaN or Inf"


@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_interpreter_output_shape(op_name):
    """SemanticInterpreter output shape matches expected."""
    shapes = OP_SHAPES[op_name]
    attrs = OP_ATTRS.get(op_name, {})
    expected = EXPECTED_SHAPES.get(op_name)

    if expected is None:
        pytest.skip(f"No expected shape for {op_name}")

    torch.manual_seed(42)
    inputs = _gen_inputs(op_name, shapes)
    result = INTERPRETER.execute(op_name, inputs, attrs)
    assert list(result.shape) == expected, f"{op_name}: got {list(result.shape)}, expected {expected}"


# ── BL4-L3: MockBackend E2E Pipeline ─────────────────────────

@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_e2e_pipeline_mock_backend(op_name):
    """Full E2E: IR graph → SSA validation → shape inference → MockBackend execution."""
    shapes = OP_SHAPES[op_name]
    attrs = OP_ATTRS.get(op_name, {})
    op = REGISTRY.get(op_name)

    # Build graph
    graph = IRGraph(name=f"bl4_{op_name}")
    for inp_name, shape in shapes.items():
        graph.add_input(inp_name, shape=shape)
    graph.add_node(IRNode(
        id="n0", op=op_name,
        inputs={k: k for k in shapes.keys()},
        outputs=["output"],
        attrs=attrs,
    ))
    graph.set_outputs(["output"])

    # Run pipeline
    pipeline = PassPipeline("bl4")
    pipeline.add_pass(SSAValidationPass())
    pipeline.add_pass(ShapeInferencePass())
    result = pipeline.run(graph)
    assert result.success, f"{op_name}: pipeline failed: {result.error}"

    # Execute via MockBackend
    torch.manual_seed(42)
    inputs = _gen_inputs(op_name, shapes)
    mb = MockBackend()
    outputs = mb.run_graph(result.graph, inputs)
    assert "output" in outputs, f"{op_name}: missing 'output' key"

    # Validate against interpreter
    torch.manual_seed(42)
    inputs2 = _gen_inputs(op_name, shapes)
    ref = INTERPRETER.execute(op_name, inputs2, attrs)

    out = outputs["output"]
    if ref.is_floating_point():
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-5), \
            f"{op_name}: max_diff={( out - ref).abs().max().item():.6f}"
    else:
        assert torch.equal(out, ref), f"{op_name}: integer mismatch"


# ── BL4-L4: Determinism check ────────────────────────────────

@pytest.mark.parametrize("op_name", ["matmul", "softmax", "layernorm", "flash_attention", "swiglu"])
def test_deterministic_execution(op_name):
    """Same seed → same output (determinism for key ops)."""
    shapes = OP_SHAPES[op_name]
    attrs = OP_ATTRS.get(op_name, {})

    torch.manual_seed(123)
    inputs1 = _gen_inputs(op_name, shapes)
    result1 = INTERPRETER.execute(op_name, inputs1, attrs)

    torch.manual_seed(123)
    inputs2 = _gen_inputs(op_name, shapes)
    result2 = INTERPRETER.execute(op_name, inputs2, attrs)

    assert torch.equal(result1, result2), f"{op_name}: non-deterministic"
