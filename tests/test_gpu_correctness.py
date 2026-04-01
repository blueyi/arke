# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU correctness verification and trajectory export."""

import json
import os
import tempfile

import pytest

from arke.agent.session import OptimizationSession
from arke.ir.builder import KernelBuilder
from arke.learn.trajectory import TrajectoryWriter, export_session_trajectory

# ============================================================
# GPU Correctness Tests
# ============================================================

def _build_matmul_relu():
    b = KernelBuilder("test_mm_relu")
    b.param("A", [512, 256], "f16")
    b.param("B", [256, 1024], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [512, 1024], "f16")
    return b.build()


def _build_softmax():
    b = KernelBuilder("test_softmax")
    b.param("X", [512, 1024], "f16")
    s = b.op("softmax", X="X")
    b.returns(s, [512, 1024], "f16")
    return b.build()


def _build_matmul():
    b = KernelBuilder("test_mm")
    b.param("A", [512, 256], "f16")
    b.param("B", [256, 1024], "f16")
    m = b.op("matmul", A="A", B="B")
    b.returns(m, [512, 1024], "f16")
    return b.build()


@pytest.mark.skipif(
    not os.environ.get("ARKE_GPU_TESTS"),
    reason="GPU tests require ARKE_GPU_TESTS=1",
)
class TestGPUCorrectness:
    """GPU correctness: Triton kernel output vs NumPy reference."""

    def test_matmul_relu_gpu_correct(self):
        ir = _build_matmul_relu()
        session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")

        # Apply typical strategy
        for kind, params, rat in [
            ("fuse", {"nodes": ["matmul_0", "relu_1"], "type": "epilogue"}, "f"),
            ("tile", {"loop": "i", "factors": [64, 16]}, "t"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "t"),
            ("tile", {"loop": "k", "factors": [32, 16]}, "t"),
        ]:
            session.run_tool("apply_decision", {"kind": kind, "params": params, "rationale": rat})

        result = session.run_tool("verify_correctness", {"trials": 3})
        assert result["passed"] is True
        assert result["gpu_correctness"]["passed"] is True
        assert result["gpu_correctness"]["max_absolute_error"] < 1.0  # f16 reasonable

    def test_softmax_gpu_correct(self):
        ir = _build_softmax()
        session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")

        # No strategy needed for softmax (uses defaults)
        result = session.run_tool("verify_correctness", {"trials": 3})
        assert result["passed"] is True
        assert result["gpu_correctness"]["passed"] is True
        assert result["gpu_correctness"]["max_absolute_error"] < 0.01

    def test_matmul_gpu_correct(self):
        ir = _build_matmul()
        session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")

        for kind, params, rat in [
            ("tile", {"loop": "i", "factors": [64, 16]}, "t"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "t"),
            ("tile", {"loop": "k", "factors": [32, 16]}, "t"),
        ]:
            session.run_tool("apply_decision", {"kind": kind, "params": params, "rationale": rat})

        result = session.run_tool("verify_correctness", {"trials": 3})
        assert result["passed"] is True
        assert result["gpu_correctness"]["passed"] is True

    def test_compile_and_profile_returns_baseline(self):
        ir = _build_matmul_relu()
        session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")

        for kind, params, rat in [
            ("fuse", {"nodes": ["matmul_0", "relu_1"], "type": "epilogue"}, "f"),
            ("tile", {"loop": "i", "factors": [64, 16]}, "t"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "t"),
            ("tile", {"loop": "k", "factors": [32, 16]}, "t"),
        ]:
            session.run_tool("apply_decision", {"kind": kind, "params": params, "rationale": rat})

        result = session.run_tool("compile_and_profile", {"warmup": 3, "runs": 10})
        assert result["success"] is True
        perf = result["performance"]
        assert perf["latency_us"] > 0
        assert perf["tflops"] > 0
        assert perf["vs_baseline"] is not None
        assert perf["vs_baseline"] > 0.3  # At least 30% of cuBLAS


# ============================================================
# Trajectory Export Tests
# ============================================================

class TestTrajectoryExport:
    """Tests for trajectory recording and JSONL export."""

    def test_trajectory_writer_basic(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name

        try:
            with TrajectoryWriter(path) as writer:
                writer.write_header({"kernel_id": "test", "target_hw": "nvidia_ampere"})
                writer.write_action(1, "analyze_compute", {})
                writer.write_result(1, True, {"bottleneck": "compute_bound"})
                writer.write_action(2, "apply_decision", {"kind": "tile"})
                writer.write_result(2, True, {"success": True})

            # Read and verify
            lines = open(path).readlines()
            assert len(lines) == 5

            header = json.loads(lines[0])
            assert header["event_type"] == "header"
            assert header["kernel_id"] == "test"

            action = json.loads(lines[1])
            assert action["event_type"] == "action"
            assert action["tool"] == "analyze_compute"
        finally:
            os.unlink(path)

    def test_export_session_trajectory(self):
        ir = _build_matmul()
        session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")

        # Make some decisions
        session.run_tool("analyze_compute", {})
        session.run_tool("apply_decision", {
            "kind": "tile", "params": {"loop": "i", "factors": [64, 16]},
            "rationale": "test"
        })

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            export_session_trajectory(
                session.export_trajectory(),
                session.summary(),
                path,
            )

            lines = open(path).readlines()
            assert len(lines) >= 3  # header + at least 2 actions

            header = json.loads(lines[0])
            assert header["event_type"] == "header"
            assert header["kernel_id"] == "test_mm"
        finally:
            os.unlink(path)

    def test_trajectory_roundtrip(self):
        """Write and read back trajectory records."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            with TrajectoryWriter(path) as writer:
                writer.write_header({"version": "1.0"})
                writer.write_action(1, "tile", {"loop": "i", "factors": [64]})
                writer.write_result(1, True, {"step": 1})
                writer.write_observation(2, {"decisions": 1, "shared_memory": 8192})

            lines = [json.loads(line) for line in open(path)]
            assert len(lines) == 4
            assert lines[0]["event_type"] == "header"
            assert lines[1]["event_type"] == "action"
            assert lines[2]["event_type"] == "result"
            assert lines[3]["event_type"] == "observation"
        finally:
            os.unlink(path)
