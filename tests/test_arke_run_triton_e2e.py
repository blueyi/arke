# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for ``arke run --backend triton`` CLI path.

T2 coverage: the ``_run_registry_backend`` dispatch in ``arke.agent.backends``
was previously broken by two bugs:
  1. IRNode inputs used hardcoded ["x","y","z","w"] keys (lowered) that didn't
     match OpSchema input names ("A", "B", "X", etc.) → KeyError in triton
     wrapper / interpreter binding.
  2. Test inputs were numpy arrays, not torch tensors → triton backend choked.

These tests verify the fix at the API level (``run_backend()``) and at the
CLI level (``_cmd_run()`` via ``main()``).
"""

import json
import subprocess
import sys

import pytest
import torch

from arke.agent.backends import BackendResult, run_backend


# ── Helpers ───────────────────────────────────────────────────

def _shapes_for_op(op: str) -> dict[str, list[int]]:
    """Canonical shapes mirroring ``_shapes_for`` in run_live_optimize."""
    if op in ("matmul", "batch_matmul"):
        return {"A": [128, 128], "B": [128, 128]}
    if op in ("add", "mul"):
        return {"A": [128, 128], "B": [128, 128]}
    if op in ("rmsnorm",):
        return {"X": [128, 64], "W": [64]}
    if op in ("layernorm",):
        return {"X": [128, 64], "W": [64], "B": [64]}
    # softmax, relu, gelu, silu, sigmoid, etc. — single tensor
    return {"X": [128, 128]}


# ── API-level tests (run_backend directly) ────────────────────

@pytest.mark.gpu
class TestRunBackendTriton:
    """Test ``run_backend('triton', ...)`` for core ops."""

    @pytest.mark.parametrize("op", ["relu", "softmax", "matmul", "add"])
    def test_core_ops_succeed(self, op: str) -> None:
        shapes = _shapes_for_op(op)
        result = run_backend("triton", op_name=op, shapes=shapes)
        assert isinstance(result, BackendResult)
        assert result.success, f"{op}: {result.message}"
        assert result.mode == "registry"
        assert result.backend == "triton"

    def test_result_has_output_keys(self) -> None:
        result = run_backend("triton", op_name="relu", shapes={"X": [64, 64]})
        assert result.success
        assert "output_keys" in result.detail
        assert len(result.detail["output_keys"]) >= 1

    @pytest.mark.parametrize("op", ["gelu", "silu", "sigmoid"])
    def test_elementwise_unary_ops(self, op: str) -> None:
        shapes = _shapes_for_op(op)
        result = run_backend("triton", op_name=op, shapes=shapes)
        assert isinstance(result, BackendResult)
        assert result.success, f"{op}: {result.message}"

    def test_mul_binary_op(self) -> None:
        result = run_backend("triton", op_name="mul",
                             shapes={"A": [64, 64], "B": [64, 64]})
        assert result.success, f"mul: {result.message}"

    def test_rmsnorm(self) -> None:
        result = run_backend("triton", op_name="rmsnorm",
                             shapes={"X": [128, 64], "W": [64]})
        assert result.success, f"rmsnorm: {result.message}"

    def test_layernorm(self) -> None:
        result = run_backend("triton", op_name="layernorm",
                             shapes={"X": [128, 64], "W": [64], "B": [64]})
        assert result.success, f"layernorm: {result.message}"

    def test_unknown_backend_errors(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent backend"):
            run_backend("nonexistent_backend_xyz", op_name="relu",
                        shapes={"X": [64, 64]})

    def test_unsupported_op_returns_failure(self) -> None:
        result = run_backend("triton", op_name="totally_fake_op_xyz",
                             shapes={"X": [64, 64]})
        assert not result.success
        assert "does not support" in result.message or "Unknown" in result.message or "error" in result.message.lower()


# ── CLI-level tests (subprocess, verifies argument parsing) ───

@pytest.mark.gpu
class TestArkeRunCLITriton:
    """Test ``arke run --backend triton`` end-to-end through the CLI."""

    @pytest.mark.parametrize("op,shape_arg", [
        ("relu", "128,128"),
        ("matmul", "128,128,128"),
        ("softmax", "128,128"),
        ("add", "128,128"),
    ])
    def test_cli_core_ops(self, op: str, shape_arg: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c",
             f"from arke.cli import main; import sys; "
             f"sys.argv=['arke','run','--kernel','{op}','--shape','{shape_arg}',"
             f"'--backend','triton','--json']; main()"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed for {op}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["backend"] == "triton"
        assert data["mode"] == "registry"

    def test_cli_json_output_structure(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c",
             "from arke.cli import main; import sys; "
             "sys.argv=['arke','run','--kernel','relu','--shape','64,64',"
             "'--backend','triton','--json']; main()"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        for key in ("backend", "op_name", "success", "mode", "message"):
            assert key in data, f"Missing key {key} in JSON output"
