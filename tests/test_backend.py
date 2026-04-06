# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Backend Abstraction (S6 Track 3).

C3.1: ArkeBackend protocol + BackendArtifact + CompiledKernel + BackendRegistry
C3.2: TritonBackend lower/compile on matmul + softmax + layernorm
C3.3: Full E2E via pipeline + backend
C3.4: MockBackend deterministic execution on relu + add + multi-node graphs
"""

import pytest
import torch

from arke.backend.protocol import (
    ArkeBackend, BackendArtifact, BackendRegistry, CompiledKernel,
)
from arke.backend.triton_backend import TritonBackend
from arke.backend.mock_backend import MockBackend
from arke.compiler.passes import PassPipeline, SSAValidationPass, ShapeInferencePass
from arke.ir.graph import IRGraph, IRNode


# ── Helpers ─────────────────────────────────────────────────

def make_matmul_graph() -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype="float32", shape=[64, 32])
    g.add_input("B", dtype="float32", shape=[32, 128])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


def make_softmax_graph() -> IRGraph:
    g = IRGraph(name="softmax")
    g.add_input("X", dtype="float32", shape=[32, 1024])
    g.add_node(IRNode(id="n0", op="softmax", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def make_layernorm_graph() -> IRGraph:
    g = IRGraph(name="layernorm")
    g.add_input("X", dtype="float32", shape=[4, 768])
    g.add_input("W", dtype="float32", shape=[768])
    g.add_input("B", dtype="float32", shape=[768])
    g.add_node(IRNode(id="n0", op="layernorm", inputs={"X": "X", "W": "W", "B": "B"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def make_relu_graph() -> IRGraph:
    g = IRGraph(name="relu")
    g.add_input("X", dtype="float32", shape=[4, 8])
    g.add_node(IRNode(id="n0", op="relu", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def make_add_graph() -> IRGraph:
    g = IRGraph(name="add")
    g.add_input("A", dtype="float32", shape=[4, 8])
    g.add_input("B", dtype="float32", shape=[4, 8])
    g.add_node(IRNode(id="n0", op="add", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


def make_relu_matmul_graph() -> IRGraph:
    """relu(matmul(A, B))"""
    g = IRGraph(name="relu_matmul")
    g.add_input("A", dtype="float32", shape=[4, 8])
    g.add_input("B", dtype="float32", shape=[8, 16])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.add_node(IRNode(id="n1", op="relu", inputs={"X": "C"}, outputs=["D"]))
    g.set_outputs(["D"])
    return g


# ── C3.1: Protocol + Artifacts ─────────────────────────────

class TestProtocol:

    def test_triton_is_arke_backend(self):
        tb = TritonBackend()
        assert isinstance(tb, ArkeBackend)

    def test_mock_is_arke_backend(self):
        mb = MockBackend()
        assert isinstance(mb, ArkeBackend)

    def test_backend_artifact_fields(self):
        a = BackendArtifact(source_code="code", backend_name="triton", op_name="matmul")
        assert a.source_code == "code"
        assert a.backend_name == "triton"
        assert a.op_name == "matmul"

    def test_compiled_kernel_ok(self):
        k = CompiledKernel.ok(fn=lambda: None, key="value")
        assert k.success
        assert k.compiled_fn is not None
        assert k.metadata["key"] == "value"

    def test_compiled_kernel_fail(self):
        k = CompiledKernel.fail("out of memory")
        assert not k.success
        assert k.error == "out of memory"


class TestBackendRegistry:

    def test_register_and_get(self):
        reg = BackendRegistry()
        tb = TritonBackend()
        reg.register(tb, ["nvidia_sm86", "nvidia_generic"])
        assert reg.get("nvidia_sm86") is tb
        assert reg.get("nvidia_generic") is tb

    def test_get_by_name(self):
        reg = BackendRegistry()
        tb = TritonBackend()
        reg.register(tb, ["nvidia_sm86"])
        assert reg.get("triton") is tb

    def test_missing_target(self):
        reg = BackendRegistry()
        with pytest.raises(KeyError, match="No backend"):
            reg.get("ascend_910b")

    def test_list_backends(self):
        reg = BackendRegistry()
        reg.register(TritonBackend(), ["nvidia_sm86"])
        reg.register(MockBackend(), ["cpu"])
        assert sorted(reg.list_backends()) == ["mock", "triton"]

    def test_contains(self):
        reg = BackendRegistry()
        reg.register(TritonBackend(), ["nvidia_sm86"])
        assert "nvidia_sm86" in reg
        assert "ascend" not in reg


# ── C3.2: TritonBackend lower/compile ─────────────────────

class TestTritonBackend:

    def test_lower_matmul(self):
        tb = TritonBackend()
        g = make_matmul_graph()
        artifact = tb.lower(g)
        assert artifact.backend_name == "triton"
        assert "matmul" in artifact.source_code
        assert artifact.metadata["num_nodes"] == 1

    def test_lower_softmax(self):
        tb = TritonBackend()
        g = make_softmax_graph()
        artifact = tb.lower(g)
        assert "softmax" in artifact.source_code

    def test_lower_layernorm(self):
        tb = TritonBackend()
        g = make_layernorm_graph()
        artifact = tb.lower(g)
        assert "layernorm" in artifact.source_code

    def test_compile_matmul(self):
        tb = TritonBackend()
        g = make_matmul_graph()
        artifact = tb.lower(g)
        kernel = tb.compile(artifact)
        assert kernel.success
        assert kernel.backend_name == "triton"

    def test_supports_op(self):
        tb = TritonBackend()
        assert tb.supports_op("matmul")
        assert tb.supports_op("softmax")
        assert not tb.supports_op("FAKE_OP")

    def test_lower_preserves_rationale(self):
        g = IRGraph(name="with_rationale")
        g.add_input("A", shape=[64, 32])
        g.add_input("B", shape=[32, 128])
        g.add_node(IRNode(
            id="n0", op="matmul",
            inputs={"A": "A", "B": "B"}, outputs=["C"],
            rationale="Tile 128x128 for RTX 3060",
        ))
        g.set_outputs(["C"])
        tb = TritonBackend()
        artifact = tb.lower(g)
        assert "@rationale" in artifact.source_code
        assert "Tile 128x128" in artifact.source_code


# ── C3.4: MockBackend ──────────────────────────────────────

class TestMockBackend:

    def test_relu_deterministic(self):
        mb = MockBackend()
        g = make_relu_graph()
        x = torch.randn(4, 8)
        result = mb.run_graph(g, {"X": x})
        assert "Y" in result
        assert torch.allclose(result["Y"], torch.relu(x))

    def test_add_deterministic(self):
        mb = MockBackend()
        g = make_add_graph()
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        result = mb.run_graph(g, {"A": a, "B": b})
        assert "C" in result
        assert torch.allclose(result["C"], a + b)

    def test_matmul_correctness(self):
        mb = MockBackend()
        g = make_matmul_graph()
        a = torch.randn(64, 32)
        b = torch.randn(32, 128)
        result = mb.run_graph(g, {"A": a, "B": b})
        assert "C" in result
        assert torch.allclose(result["C"], a @ b, rtol=1e-3, atol=1e-5)

    def test_softmax_correctness(self):
        mb = MockBackend()
        g = make_softmax_graph()
        x = torch.randn(32, 1024)
        result = mb.run_graph(g, {"X": x})
        expected = torch.nn.functional.softmax(x, dim=-1)
        assert torch.allclose(result["Y"], expected)

    def test_multi_node_graph(self):
        """relu(matmul(A, B)) through MockBackend."""
        mb = MockBackend()
        g = make_relu_matmul_graph()
        a = torch.randn(4, 8)
        b = torch.randn(8, 16)
        result = mb.run_graph(g, {"A": a, "B": b})
        expected = torch.relu(a @ b)
        assert "D" in result
        assert torch.allclose(result["D"], expected, rtol=1e-3, atol=1e-5)

    def test_lower_produces_readable_code(self):
        mb = MockBackend()
        g = make_relu_graph()
        artifact = mb.lower(g)
        assert "relu" in artifact.source_code
        assert artifact.backend_name == "mock"

    def test_supports_op(self):
        mb = MockBackend()
        assert mb.supports_op("relu")
        assert mb.supports_op("add")
        assert not mb.supports_op("NONEXISTENT")


# ── C3.3: Full E2E Pipeline + Backend ─────────────────────

class TestE2EPipelineBackend:

    def test_matmul_e2e(self):
        """matmul.ak → pipeline → MockBackend → correct result."""
        g = make_matmul_graph()

        # Run pipeline
        pipeline = PassPipeline("e2e")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success
        assert result.artifacts["shape_map"]["C"] == [64, 128]

        # Execute via MockBackend
        mb = MockBackend()
        a = torch.randn(64, 32)
        b = torch.randn(32, 128)
        outputs = mb.run_graph(result.graph, {"A": a, "B": b})
        expected = a @ b
        assert torch.allclose(outputs["C"], expected, rtol=1e-3, atol=1e-5)

    def test_softmax_e2e(self):
        """softmax → pipeline → MockBackend → correct result."""
        g = make_softmax_graph()
        pipeline = PassPipeline("e2e")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success

        mb = MockBackend()
        x = torch.randn(32, 1024)
        outputs = mb.run_graph(result.graph, {"X": x})
        expected = torch.nn.functional.softmax(x, dim=-1)
        assert torch.allclose(outputs["Y"], expected)

    def test_relu_matmul_chain_e2e(self):
        """relu(matmul(A,B)) → pipeline → MockBackend → correct."""
        g = make_relu_matmul_graph()
        pipeline = PassPipeline("e2e")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success
        assert result.artifacts["shape_map"]["D"] == [4, 16]

        mb = MockBackend()
        a = torch.randn(4, 8)
        b = torch.randn(8, 16)
        outputs = mb.run_graph(result.graph, {"A": a, "B": b})
        expected = torch.relu(a @ b)
        assert torch.allclose(outputs["D"], expected, rtol=1e-3, atol=1e-5)

    def test_layernorm_e2e(self):
        g = make_layernorm_graph()
        pipeline = PassPipeline("e2e")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success

        mb = MockBackend()
        x = torch.randn(4, 768)
        w = torch.ones(768)
        b = torch.zeros(768)
        outputs = mb.run_graph(result.graph, {"X": x, "W": w, "B": b})
        expected = torch.nn.functional.layer_norm(x, [768], w, b)
        assert torch.allclose(outputs["Y"], expected, rtol=1e-3, atol=1e-5)
