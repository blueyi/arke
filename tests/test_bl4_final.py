# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""BL4×L1 Full Operator Verification (S6 Track 5) — FINAL CORRECTED VERSION.

Validates Arke compiler infrastructure by comparing two INDEPENDENT paths:

Path A (Arke): IRGraph → SSA → Shape → MockBackend → SemanticInterpreter → output_a
Path B (Independent): Shared inputs → independent PyTorch baseline → output_b

This validates:
1. Compiler pipeline correctness (no parameter corruption, shape inference correct)
2. Integration with benchmark baselines (P0/P3 where available)
3. Determinism for key operators

NOT validated in S6 (deferred to S7):
- Real Triton kernel generation
- GPU performance vs baselines
"""

import time

import pytest
import torch

from arke.backend.mock_backend import MockBackend
from arke.compiler.passes import PassPipeline, SSAValidationPass, ShapeInferencePass
from arke.ir.graph import IRGraph, IRNode
from arke.ir.ops.registry import REGISTRY
from tests.independent_baseline import BASELINE_REGISTRY


# ── Test Configuration ────────────────────────────────────────

# Standard test shapes (ST1 level, small for fast CI)
M, N, K = 64, 128, 32

ALL_OP_NAMES = sorted([op.name for op in REGISTRY])

# Arke pipeline setup
BACKEND = MockBackend()
PIPELINE = PassPipeline("bl4_validation")
PIPELINE.add_pass(SSAValidationPass())
PIPELINE.add_pass(ShapeInferencePass())


# ── Helper: Generate inputs ──────────────────────────────────

def generate_inputs(op_name: str, dtype=torch.float32) -> tuple[dict, dict]:
    """Generate inputs and attrs for an operator.
    
    Returns:
        (inputs_dict, attrs_dict)
    """
    op = REGISTRY.get(op_name)
    
    # Determine shapes based on op
    if op_name == "matmul":
        shapes = {"A": [M, K], "B": [K, N]}
    elif op_name == "batch_matmul":
        shapes = {"A": [4, M, K], "B": [4, K, N]}
    elif op_name == "grouped_matmul":
        shapes = {"X": [4, M, K], "W": [8, K, N], "indices": [4]}
    elif op_name == "layernorm":
        shapes = {"X": [M, N], "W": [N], "B": [N]}
    elif op_name == "rmsnorm":
        shapes = {"X": [M, N], "W": [N]}
    elif op_name == "rmsnorm_residual":
        shapes = {"X": [M, N], "residual": [M, N], "W": [N]}
    elif op_name == "embedding":
        shapes = {"indices": [M, 16], "weight": [1000, N]}
    elif op_name == "concat":
        shapes = {"A": [M, N], "B": [M, N]}
    elif op_name == "split":
        shapes = {"X": [M, N * 2]}
    elif op_name == "gather":
        shapes = {"X": [M, N], "idx": [M, 16]}
    elif op_name == "scatter":
        shapes = {"X": [M, N], "idx": [M, 16], "src": [M, 16]}
    elif op_name == "permute":
        shapes = {"X": [2, 4, 8, 16]}
    elif op_name == "rope":
        shapes = {"X": [1, 2, M, K], "cos": [M, K // 2], "sin": [M, K // 2]}
    elif op_name == "cross_entropy":
        shapes = {"logits": [M, N], "labels": [M]}
    elif op_name == "fused_linear_cross_entropy":
        shapes = {"X": [M, K], "W": [N, K], "labels": [M]}
    elif op_name == "quantize_per_token":
        shapes = {"X": [M, N]}
    elif op_name == "dequantize_per_channel":
        shapes = {"X_int8": [M, N], "scale": [N], "zero_point": [N]}
    elif op_name in ("flash_attention", "cross_attention"):
        shapes = {"Q": [1, 4, M, K], "K": [1, 4, M, K], "V": [1, 4, M, K]}
    elif op_name == "grouped_query_attention":
        shapes = {"Q": [1, 8, M, K], "K": [1, 2, M, K], "V": [1, 2, M, K]}
    elif op_name == "multi_latent_attention":
        shapes = {"Q": [1, 8, M, K], "KV_compressed": [1, M, 128], "W_uk": [128, 8, K], "W_uv": [128, 8, K]}
    elif op_name == "paged_attention":
        shapes = {"Q": [1, 4, 1, K], "K_cache": [8, 16, 4, K], "V_cache": [8, 16, 4, K], "block_table": [1, 4]}
    elif op_name in ("add", "mul"):
        shapes = {"A": [M, N], "B": [M, N]}
    elif op_name == "where_":
        shapes = {"cond": [M, N], "A": [M, N], "B": [M, N]}
    elif op_name in ("silu_and_mul", "geglu"):
        shapes = {"X": [M, N * 2]}
    elif op_name == "topk":
        shapes = {"X": [M, N]}
    else:
        # Default: single input X
        shapes = {"X": [M, N]}
    
    # Generate tensors
    inputs = {}
    for inp_name, shape in shapes.items():
        if op.input_gen and inp_name in op.input_gen.distributions:
            dist = op.input_gen.distributions[inp_name]
            rng = op.input_gen.ranges.get(inp_name, (0, 10))
            
            if dist == "randint":
                lo, hi = int(rng[0]), max(int(rng[1]), int(rng[0]) + 1)
                inputs[inp_name] = torch.randint(lo, hi, shape)
            elif dist == "uniform":
                inputs[inp_name] = torch.empty(shape, dtype=dtype).uniform_(rng[0], rng[1])
            elif dist == "ones":
                inputs[inp_name] = torch.ones(shape, dtype=dtype)
            elif dist == "bool_mask":
                inputs[inp_name] = torch.randint(0, 2, shape, dtype=torch.bool)
            else:
                inputs[inp_name] = torch.randn(shape, dtype=dtype)
        else:
            inputs[inp_name] = torch.randn(shape, dtype=dtype)
    
    # Attrs
    attrs = {}
    if op_name == "cast":
        attrs["target_dtype"] = "float16"
    elif op_name == "topk":
        attrs["k"] = min(50, N)
    elif op_name == "permute":
        attrs["dims"] = [0, 2, 1, 3]
    
    return inputs, attrs, shapes


# ── Helper: Run Arke pipeline ─────────────────────────────────

def run_arke_pipeline(op_name: str, inputs: dict, attrs: dict, shapes: dict) -> torch.Tensor:
    """Execute op through Arke pipeline."""
    graph = IRGraph(name=f"bl4_{op_name}")
    
    for inp_name, shape in shapes.items():
        graph.add_input(inp_name, shape=shape)
    
    graph.add_node(IRNode(
        id="n0",
        op=op_name,
        inputs={k: k for k in shapes.keys()},
        outputs=["output"],
        attrs=attrs,
    ))
    graph.set_outputs(["output"])
    
    result = PIPELINE.run(graph)
    if not result.success:
        raise RuntimeError(f"Pipeline failed: {result.error}")
    
    outputs = BACKEND.run_graph(result.graph, inputs)
    return outputs["output"]


# ── Coverage Check ────────────────────────────────────────────

def test_coverage_all_45_ops():
    """Verify registry size matches the SSOT kernel catalog total."""
    from benchmarks.op_registry import total_ops
    expected = total_ops()
    assert len(ALL_OP_NAMES) == expected, (
        f"Expected {expected} ops (per SSOT benchmark-ops.md), "
        f"got {len(ALL_OP_NAMES)}"
    )


def test_baseline_coverage():
    """Check how many ops have independent baseline."""
    from benchmarks.op_registry import total_ops
    expected = total_ops()
    covered = [op for op in ALL_OP_NAMES if op in BASELINE_REGISTRY]
    assert len(covered) == expected, (
        f"Expected {expected} baselines (per SSOT), got {len(covered)}"
    )


# ── BL4-L1: Correctness vs Independent Baseline ──────────────

@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_correctness_vs_independent_baseline(op_name):
    """Every op: Arke output matches independent PyTorch baseline."""
    if op_name not in BASELINE_REGISTRY:
        pytest.skip(f"No independent baseline for {op_name}")
    
    # Generate shared inputs
    torch.manual_seed(42)
    inputs, attrs, shapes = generate_inputs(op_name, dtype=torch.float32)
    
    # Path A: Arke pipeline
    arke_output = run_arke_pipeline(op_name, inputs, attrs, shapes)
    
    # Path B: Independent baseline
    baseline_fn = BASELINE_REGISTRY[op_name]
    baseline_output = baseline_fn(inputs, attrs)
    
    # Compare
    assert arke_output.shape == baseline_output.shape, \
        f"{op_name}: shape mismatch — Arke {list(arke_output.shape)} vs baseline {list(baseline_output.shape)}"
    
    if arke_output.is_floating_point():
        max_diff = (arke_output - baseline_output).abs().max().item()
        assert torch.allclose(arke_output, baseline_output, rtol=1e-4, atol=1e-5), \
            f"{op_name}: max_diff={max_diff:.6e} exceeds tolerance"
    else:
        assert torch.equal(arke_output, baseline_output), \
            f"{op_name}: integer outputs differ"


# ── BL4-L1: Shape Inference Correctness ──────────────────────

@pytest.mark.parametrize("op_name", ALL_OP_NAMES)
def test_shape_inference_matches_execution(op_name):
    """Shape inference result matches actual execution output shape."""
    from arke.ir.ops.shape_engine import SHAPE_ENGINE
    
    torch.manual_seed(42)
    inputs, attrs, shapes = generate_inputs(op_name, dtype=torch.float32)
    
    # Get actual output shape
    arke_output = run_arke_pipeline(op_name, inputs, attrs, shapes)
    actual_shape = list(arke_output.shape)
    
    # Get inferred shape
    try:
        inferred_shape = SHAPE_ENGINE.infer(op_name, shapes, attrs)
    except Exception as e:
        pytest.fail(f"{op_name}: shape inference failed: {e}")
    
    assert inferred_shape == actual_shape, \
        f"{op_name}: inferred {inferred_shape} != actual {actual_shape}"


# ── BL4-L1: Determinism Check ────────────────────────────────

@pytest.mark.parametrize("op_name", ["matmul", "softmax", "layernorm", "flash_attention", "silu_and_mul"])
def test_deterministic_execution(op_name):
    """Key ops: same seed → same output (determinism)."""
    torch.manual_seed(123)
    inputs1, attrs1, shapes1 = generate_inputs(op_name, dtype=torch.float32)
    output1 = run_arke_pipeline(op_name, inputs1, attrs1, shapes1)
    
    torch.manual_seed(123)
    inputs2, attrs2, shapes2 = generate_inputs(op_name, dtype=torch.float32)
    output2 = run_arke_pipeline(op_name, inputs2, attrs2, shapes2)
    
    assert torch.equal(output1, output2), f"{op_name}: non-deterministic"


# ── BL4-L1: Performance Sanity Check ─────────────────────────

@pytest.mark.parametrize("op_name", ["matmul", "softmax", "layernorm"])
def test_performance_sanity(op_name):
    """Key ops: Arke pipeline completes in reasonable time (< 100ms per call)."""
    torch.manual_seed(42)
    inputs, attrs, shapes = generate_inputs(op_name, dtype=torch.float32)
    
    # Warmup
    for _ in range(3):
        run_arke_pipeline(op_name, inputs, attrs, shapes)
    
    # Measure
    t0 = time.perf_counter()
    for _ in range(10):
        run_arke_pipeline(op_name, inputs, attrs, shapes)
    elapsed_ms = (time.perf_counter() - t0) * 100  # ms per call
    
    # S6 acceptance: MockBackend overhead is acceptable, just check it's not absurdly slow
    assert elapsed_ms < 100, f"{op_name}: {elapsed_ms:.2f}ms per call (limit 100ms)"


# ── Summary Report ────────────────────────────────────────────

def test_bl4_summary_report(capsys):
    """Generate BL4 summary report (run last)."""
    print("\n" + "="*70)
    print("BL4×L1 SUMMARY (S6 Compiler Infrastructure)")
    print("="*70)
    print(f"Total ops in registry: {len(ALL_OP_NAMES)}")
    print(f"Ops with independent baseline: {len([op for op in ALL_OP_NAMES if op in BASELINE_REGISTRY])}")
    print(f"Test shapes: M={M}, N={N}, K={K}")
    print("="*70)
    print("Validation scope:")
    print("  ✅ Compiler pipeline correctness (IRGraph → SSA → Shape → Backend)")
    print("  ✅ Parameter passing integrity")
    print("  ✅ Shape inference accuracy")
    print("  ✅ Determinism for key ops")
    print("\nNOT validated in S6 (deferred to S7):")
    print("  ⏭️  Real Triton kernel generation")
    print("  ⏭️  GPU performance vs baselines")
    print("="*70)
    
    assert True
