# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for .akir file format — JSON serialization and round-trip.

Tests:
    - Round-trip: .ak → compile → save .akir → load .akir → execute → same result
    - Format validation (version, format field)
    - Strategy included/excluded
    - All 46 .ak files round-trip through .akir
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from arke.compiler.pipeline import ArkePipeline, CompilationResult
from arke.compiler.validator import validate_semantic_ir
from arke.ir.akir import (
    AKIR_FORMAT,
    AKIR_VERSION,
    akir_from_dict,
    akir_to_dict,
    load_akir,
    save_akir,
)
from arke.ir.semantic import (
    Node,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.strategy import StrategyIR

# ---- Fixtures ----

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))


@pytest.fixture
def pipeline():
    return ArkePipeline()


@pytest.fixture
def tmp_akir(tmp_path):
    """Return a temporary .akir file path."""
    return str(tmp_path / "test.akir")


# ---- Helpers ----

def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _make_simple_ir():
    """Create a minimal SemanticIR for testing."""
    ir = SemanticIR(kernel_id="test_relu")
    ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
    ir.add_node(Node(
        id="n0",
        op="relu",
        inputs={"X": ParamRef(name="X")},
        output=TensorDesc(shape=[4, 8], dtype="f32"),
        semantics=Semantics(computation="relu(X)"),
    ))
    ir.return_node = "n0"
    return ir


# ============================================================
# Test Class: Format Validation
# ============================================================


class TestAkirFormat:
    """Test .akir format metadata and validation."""

    def test_akir_to_dict_has_format(self):
        """akir_to_dict should include 'format' field."""
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert d["format"] == AKIR_FORMAT

    def test_akir_to_dict_has_version(self):
        """akir_to_dict should include 'version' field."""
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert d["version"] == AKIR_VERSION

    def test_akir_to_dict_has_semantic_ir(self):
        """akir_to_dict should include 'semantic_ir' dict."""
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert "semantic_ir" in d
        assert isinstance(d["semantic_ir"], dict)
        assert d["semantic_ir"]["kernel_id"] == "test_relu"

    def test_akir_to_dict_strategy_null(self):
        """akir_to_dict with no strategy should have null strategy_ir."""
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert d["strategy_ir"] is None

    def test_akir_to_dict_strategy_present(self):
        """akir_to_dict with strategy should include strategy_ir dict."""
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4], rationale="test tile")
        d = akir_to_dict(ir, strategy)
        assert d["strategy_ir"] is not None
        assert isinstance(d["strategy_ir"], dict)
        assert d["strategy_ir"]["kernel_id"] == "test_relu"

    def test_akir_from_dict_invalid_format(self):
        """akir_from_dict should reject invalid format."""
        d = {"format": "wrong", "version": "1.0.0", "semantic_ir": {}, "strategy_ir": None}
        with pytest.raises(ValueError, match="Invalid .akir format"):
            akir_from_dict(d)

    def test_akir_from_dict_missing_version(self):
        """akir_from_dict should reject missing version."""
        d = {"format": "akir", "semantic_ir": {}, "strategy_ir": None}
        with pytest.raises(ValueError, match="Missing 'version'"):
            akir_from_dict(d)

    def test_akir_from_dict_missing_semantic_ir(self):
        """akir_from_dict should reject missing semantic_ir."""
        d = {"format": "akir", "version": "1.0.0", "strategy_ir": None}
        with pytest.raises(ValueError, match="Missing 'semantic_ir'"):
            akir_from_dict(d)


# ============================================================
# Test Class: Dict Round-Trip
# ============================================================


class TestDictRoundTrip:
    """Test akir_to_dict/akir_from_dict round-trip."""

    def test_round_trip_no_strategy(self):
        """SemanticIR without strategy should round-trip."""
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        ir_rt, strat_rt = akir_from_dict(d)

        assert strat_rt is None
        assert ir_rt.kernel_id == "test_relu"
        assert len(ir_rt.nodes) == 1
        assert ir_rt.nodes[0].op == "relu"
        assert ir_rt.return_node == "n0"

    def test_round_trip_with_strategy(self):
        """SemanticIR + StrategyIR should round-trip."""
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4], rationale="test tile")

        d = akir_to_dict(ir, strategy)
        ir_rt, strat_rt = akir_from_dict(d)

        assert strat_rt is not None
        assert strat_rt.kernel_id == "test_relu"
        assert strat_rt.target_hw == "nvidia_ampere"
        assert len(strat_rt.decisions) == 1
        assert strat_rt.decisions[0].kind == "tile"

    def test_round_trip_json_serializable(self):
        """akir_to_dict output should be JSON-serializable."""
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        d = akir_to_dict(ir, strategy)
        # Should not raise
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)
        assert parsed["format"] == "akir"


# ============================================================
# Test Class: File I/O
# ============================================================


class TestFileIO:
    """Test save_akir/load_akir file operations."""

    def test_save_and_load(self, tmp_akir):
        """save_akir then load_akir should round-trip."""
        ir = _make_simple_ir()
        save_akir(ir, None, tmp_akir)

        ir_rt, strat_rt = load_akir(tmp_akir)
        assert strat_rt is None
        assert ir_rt.kernel_id == "test_relu"
        assert len(ir_rt.nodes) == 1

    def test_save_and_load_with_strategy(self, tmp_akir):
        """save_akir with strategy then load_akir should round-trip."""
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4])
        save_akir(ir, strategy, tmp_akir)

        ir_rt, strat_rt = load_akir(tmp_akir)
        assert strat_rt is not None
        assert strat_rt.kernel_id == "test_relu"

    def test_load_nonexistent_file(self):
        """load_akir with non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_akir("/nonexistent/path.akir")

    def test_load_invalid_json(self, tmp_path):
        """load_akir with invalid JSON should raise."""
        bad_file = str(tmp_path / "bad.akir")
        with open(bad_file, "w") as f:
            f.write("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_akir(bad_file)

    def test_file_is_valid_json(self, tmp_akir):
        """Saved .akir file should be valid JSON."""
        ir = _make_simple_ir()
        save_akir(ir, None, tmp_akir)

        with open(tmp_akir) as f:
            data = json.load(f)
        assert data["format"] == "akir"
        assert data["version"] == "1.0.0"


# ============================================================
# Test Class: CompilationResult Integration
# ============================================================


class TestCompilationResultAkir:
    """Test CompilationResult.save_akir and ArkePipeline.load_akir."""

    def test_result_save_akir(self, pipeline, tmp_akir):
        """CompilationResult.save_akir should save to file."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        # Verify file exists and is valid
        with open(tmp_akir) as f:
            data = json.load(f)
        assert data["format"] == "akir"
        assert data["semantic_ir"]["kernel_id"] == "relu_kernel"

    def test_result_save_akir_none_ir(self):
        """save_akir should raise if semantic_ir is None."""
        result = CompilationResult()
        with pytest.raises(ValueError, match="semantic_ir is None"):
            result.save_akir("/tmp/should_not_exist.akir")

    def test_pipeline_load_akir(self, pipeline, tmp_akir):
        """ArkePipeline.load_akir should load and create CompilationResult."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.success
        assert loaded.semantic_ir is not None
        assert loaded.kernel_name == "relu_kernel"

    def test_pipeline_load_akir_nonexistent(self):
        """ArkePipeline.load_akir with bad path should return errors."""
        loaded = ArkePipeline.load_akir("/nonexistent/path.akir")
        assert not loaded.success
        assert len(loaded.errors) > 0

    def test_pipeline_load_akir_preserves_strategy(self, pipeline, tmp_akir):
        """load_akir should preserve StrategyIR."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        assert result.strategy_ir is not None
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.strategy_ir is not None
        assert loaded.strategy_ir.kernel_id == result.strategy_ir.kernel_id
        assert loaded.strategy_ir.target_hw == result.strategy_ir.target_hw


# ============================================================
# Test Class: E2E Round-Trip (.ak → .akir → execute)
# ============================================================


class TestE2ERoundTrip:
    """Full round-trip: .ak → compile → save .akir → load .akir → execute."""

    def test_round_trip_relu(self, pipeline, tmp_akir):
        """relu: compile → save → load → execute → same result."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.success

        X = torch.randn(128, 3072, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"X": X})
        out_loaded = pipeline.execute(loaded, {"X": X})

        assert torch.allclose(
            out_orig["output"], out_loaded["output"]
        ), f"Round-trip changed relu result: max_diff={_diff(out_orig['output'], out_loaded['output']):.2e}"

    def test_round_trip_matmul(self, pipeline, tmp_akir):
        """matmul: compile → save → load → execute → same result."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.success

        A = torch.randn(1024, 1024, dtype=torch.float32)
        B = torch.randn(1024, 1024, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"A": A, "B": B})
        out_loaded = pipeline.execute(loaded, {"A": A, "B": B})

        assert torch.allclose(
            out_orig["output"], out_loaded["output"]
        ), "Round-trip changed matmul result"

    def test_round_trip_matmul_gelu(self, pipeline, tmp_akir):
        """matmul_gelu (multi-node): compile → save → load → execute → same result."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.success

        X = torch.randn(128, 768, dtype=torch.float32)
        W = torch.randn(768, 3072, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"X": X, "W": W})
        out_loaded = pipeline.execute(loaded, {"X": X, "W": W})

        assert torch.allclose(
            out_orig["output"], out_loaded["output"], rtol=1e-4, atol=1e-5
        ), "Round-trip changed matmul_gelu result"

    def test_round_trip_strategy_included(self, pipeline, tmp_akir):
        """File with strategy should preserve strategy through .akir round-trip."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        assert result.strategy_ir is not None
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.strategy_ir is not None
        assert len(loaded.strategy_ir.decisions) == len(result.strategy_ir.decisions)

    def test_round_trip_strategy_excluded(self, pipeline, tmp_akir):
        """File without strategy should have null strategy in .akir."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success
        assert result.strategy_ir is None
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.strategy_ir is None


# ============================================================
# Test Class: All 46 .ak Files Round-Trip
# ============================================================


class TestAllFilesRoundTrip:
    """All 46 .ak files should round-trip through .akir format."""

    @pytest.mark.parametrize(
        "ak_file",
        ALL_AK_FILES,
        ids=[f.stem for f in ALL_AK_FILES],
    )
    def test_akir_round_trip(self, pipeline, ak_file, tmp_path):
        """Each .ak file should round-trip through .akir."""
        result = pipeline.compile_file(str(ak_file))
        assert result.success, (
            f"Compilation failed for {ak_file.name}: {result.errors}"
        )

        # Save to .akir
        akir_path = str(tmp_path / f"{ak_file.stem}.akir")
        result.save_akir(akir_path)

        # Load from .akir
        loaded = ArkePipeline.load_akir(akir_path)
        assert loaded.success, (
            f"Load failed for {ak_file.stem}.akir: {loaded.errors}"
        )

        # Verify structural equality
        assert loaded.kernel_name == result.kernel_name
        assert loaded.semantic_ir is not None
        assert len(loaded.semantic_ir.nodes) == len(result.semantic_ir.nodes)
        assert len(loaded.semantic_ir.params) == len(result.semantic_ir.params)
        assert loaded.semantic_ir.return_node == result.semantic_ir.return_node

        # Verify strategy preservation
        if result.strategy_ir is not None:
            assert loaded.strategy_ir is not None
            assert loaded.strategy_ir.kernel_id == result.strategy_ir.kernel_id
        else:
            assert loaded.strategy_ir is None


# ============================================================
# Test Class: CLI Integration
# ============================================================


class TestCLI:
    """Test CLI compile subcommand."""

    def test_cli_compile_to_stdout(self):
        """arke compile <file> should print JSON to stdout."""
        import subprocess

        ak_file = str(OPERATORS_DIR / "00_relu.ak")
        proc = subprocess.run(
            ["/home/blueyi/.venvs/arke/bin/python", "-m", "arke.cli", "compile", ak_file],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["format"] == "akir"
        assert data["version"] == "1.0.0"
        assert data["semantic_ir"]["kernel_id"] == "relu_kernel"

    def test_cli_compile_to_file(self, tmp_path):
        """arke compile <file> -o <output> should create .akir file."""
        import subprocess

        ak_file = str(OPERATORS_DIR / "01_matmul.ak")
        out_file = str(tmp_path / "matmul.akir")
        proc = subprocess.run(
            ["/home/blueyi/.venvs/arke/bin/python", "-m", "arke.cli", "compile", ak_file, "-o", out_file],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
        assert os.path.exists(out_file)

        with open(out_file) as f:
            data = json.load(f)
        assert data["format"] == "akir"

    def test_cli_compile_invalid_file(self):
        """arke compile with invalid file should return non-zero exit code."""
        import subprocess

        proc = subprocess.run(
            ["/home/blueyi/.venvs/arke/bin/python", "-m", "arke.cli", "compile", "/nonexistent/file.ak"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
