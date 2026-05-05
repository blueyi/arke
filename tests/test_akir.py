# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for .akir file format — JSON serialization and multi-layer round-trip."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from arke.compiler.pipeline import ArkePipeline, CompilationResult
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

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))


@pytest.fixture
def pipeline():
    return ArkePipeline()


@pytest.fixture
def tmp_akir(tmp_path):
    return str(tmp_path / "test.akir")


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _make_simple_ir():
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


class TestAkirFormat:
    def test_akir_to_dict_has_format(self):
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert d["format"] == AKIR_FORMAT

    def test_akir_to_dict_has_version(self):
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert d["version"] == AKIR_VERSION

    def test_akir_to_dict_has_all_layer_keys(self):
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        assert set(d.keys()) == {
            "format",
            "version",
            "semantic_ir",
            "strategy_ir",
            "schedule_ir",
            "instruction_ir",
        }

    def test_akir_to_dict_strategy_present(self):
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4], rationale="test tile")
        d = akir_to_dict(ir, strategy)
        assert d["strategy_ir"] is not None
        assert d["strategy_ir"]["kernel_id"] == "test_relu"

    def test_akir_from_dict_invalid_format(self):
        d = {"format": "wrong", "version": AKIR_VERSION, "semantic_ir": {}, "strategy_ir": None, "schedule_ir": None, "instruction_ir": None}
        with pytest.raises(ValueError, match="Invalid .akir format"):
            akir_from_dict(d)

    def test_akir_from_dict_invalid_version(self):
        d = {"format": "akir", "version": "1.0.0", "semantic_ir": {}, "strategy_ir": None, "schedule_ir": None, "instruction_ir": None}
        with pytest.raises(ValueError, match="Unsupported .akir version"):
            akir_from_dict(d)

    def test_akir_from_dict_rejects_legacy_nested_ir_versions(self):
        d = {
            "format": AKIR_FORMAT,
            "version": AKIR_VERSION,
            "semantic_ir": {
                "version": "1.0.0",
                "kernel_id": "test_relu",
                "params": [],
                "nodes": [],
                "edges": [],
                "return_node": "",
            },
            "strategy_ir": None,
            "schedule_ir": None,
            "instruction_ir": None,
        }
        with pytest.raises(ValueError, match="Unsupported SemanticIR version"):
            akir_from_dict(d)

    def test_akir_from_dict_missing_semantic_ir(self):
        d = {"format": "akir", "version": AKIR_VERSION, "strategy_ir": None, "schedule_ir": None, "instruction_ir": None}
        with pytest.raises(ValueError, match="Missing 'semantic_ir'"):
            akir_from_dict(d)


class TestDictRoundTrip:
    def test_round_trip_no_strategy(self):
        ir = _make_simple_ir()
        d = akir_to_dict(ir, None)
        ir_rt, strat_rt, schedule_rt, instruction_rt = akir_from_dict(d)

        assert strat_rt is None
        assert schedule_rt is None
        assert instruction_rt is None
        assert ir_rt.kernel_id == "test_relu"
        assert len(ir_rt.nodes) == 1
        assert ir_rt.nodes[0].op == "relu"
        assert ir_rt.return_node == "n0"

    def test_round_trip_with_strategy(self):
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4], rationale="test tile")

        d = akir_to_dict(ir, strategy)
        ir_rt, strat_rt, schedule_rt, instruction_rt = akir_from_dict(d)

        assert strat_rt is not None
        assert strat_rt.kernel_id == "test_relu"
        assert strat_rt.target_hw == "nvidia_ampere"
        assert len(strat_rt.decisions) == 1
        assert strat_rt.decisions[0].kind == "tile"
        assert schedule_rt is None
        assert instruction_rt is None

    def test_round_trip_json_serializable(self):
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        d = akir_to_dict(ir, strategy)
        parsed = json.loads(json.dumps(d, indent=2))
        assert parsed["format"] == "akir"
        assert parsed["version"] == AKIR_VERSION


class TestFileIO:
    def test_save_and_load(self, tmp_akir):
        ir = _make_simple_ir()
        save_akir(ir, None, tmp_akir)

        ir_rt, strat_rt, schedule_rt, instruction_rt = load_akir(tmp_akir)
        assert strat_rt is None
        assert schedule_rt is None
        assert instruction_rt is None
        assert ir_rt.kernel_id == "test_relu"
        assert len(ir_rt.nodes) == 1

    def test_save_and_load_with_strategy(self, tmp_akir):
        ir = _make_simple_ir()
        strategy = StrategyIR(kernel_id="test_relu", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4])
        save_akir(ir, strategy, tmp_akir)

        ir_rt, strat_rt, schedule_rt, instruction_rt = load_akir(tmp_akir)
        assert strat_rt is not None
        assert strat_rt.kernel_id == "test_relu"
        assert schedule_rt is None
        assert instruction_rt is None

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_akir("/nonexistent/path.akir")

    def test_load_invalid_json(self, tmp_path):
        bad_file = str(tmp_path / "bad.akir")
        with open(bad_file, "w") as f:
            f.write("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_akir(bad_file)

    def test_file_is_valid_json(self, tmp_akir):
        ir = _make_simple_ir()
        save_akir(ir, None, tmp_akir)

        with open(tmp_akir) as f:
            data = json.load(f)
        assert data["format"] == "akir"
        assert data["version"] == AKIR_VERSION
        assert "schedule_ir" in data
        assert "instruction_ir" in data


class TestCompilationResultAkir:
    def test_result_save_akir(self, pipeline, tmp_akir):
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        with open(tmp_akir) as f:
            data = json.load(f)
        assert data["format"] == "akir"
        assert data["semantic_ir"]["kernel_id"] == "relu_kernel"
        assert data["schedule_ir"] is not None
        assert data["instruction_ir"] is not None

    def test_result_save_akir_none_ir(self):
        result = CompilationResult()
        with pytest.raises(ValueError, match="semantic_ir is None"):
            result.save_akir("/tmp/should_not_exist.akir")

    def test_pipeline_load_akir(self, pipeline, tmp_akir):
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.success
        assert loaded.semantic_ir is not None
        assert loaded.schedule_ir is not None
        assert loaded.instruction_ir is not None
        assert loaded.kernel_name == "relu_kernel"

    def test_pipeline_load_akir_preserves_all_layers(self, pipeline, tmp_akir):
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        assert result.strategy_ir is not None
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.strategy_ir is not None
        assert loaded.schedule_ir is not None
        assert loaded.instruction_ir is not None
        assert loaded.strategy_ir.kernel_id == result.strategy_ir.kernel_id
        assert loaded.schedule_ir.to_dict() == result.schedule_ir.to_dict()
        assert loaded.instruction_ir.to_dict() == result.instruction_ir.to_dict()


class TestE2ERoundTrip:
    def test_round_trip_relu(self, pipeline, tmp_akir):
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

    def test_round_trip_strategy_excluded(self, pipeline, tmp_akir):
        src = (
            "kernel matmul_no_strategy(\n"
            "    A: Tensor<[1024, 1024], f16>,\n"
            "    B: Tensor<[1024, 1024], f16>\n"
            ") -> Tensor<[1024, 1024], f16>\n"
            "{\n"
            "    let C = matmul(A=A, B=B);\n"
            "    return C;\n"
            "}\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".ak", delete=False) as fh:
            fh.write(src)
            path = fh.name
        try:
            result = pipeline.compile_file(path)
        finally:
            os.unlink(path)
        assert result.success
        assert result.strategy_ir is None
        result.save_akir(tmp_akir)

        loaded = ArkePipeline.load_akir(tmp_akir)
        assert loaded.strategy_ir is None
        assert loaded.schedule_ir is None
        assert loaded.instruction_ir is None


class TestAllFilesRoundTrip:
    @pytest.mark.parametrize(
        "ak_file",
        ALL_AK_FILES,
        ids=[f.stem for f in ALL_AK_FILES],
    )
    def test_akir_round_trip(self, pipeline, ak_file, tmp_path):
        result = pipeline.compile_file(str(ak_file))
        assert result.success, (
            f"Compilation failed for {ak_file.name}: {result.errors}"
        )

        akir_path = str(tmp_path / f"{ak_file.stem}.akir")
        result.save_akir(akir_path)

        loaded = ArkePipeline.load_akir(akir_path)
        assert loaded.success, (
            f"Load failed for {ak_file.stem}.akir: {loaded.errors}"
        )

        assert loaded.kernel_name == result.kernel_name
        assert loaded.semantic_ir is not None
        assert len(loaded.semantic_ir.nodes) == len(result.semantic_ir.nodes)
        assert len(loaded.semantic_ir.params) == len(result.semantic_ir.params)
        assert loaded.semantic_ir.return_node == result.semantic_ir.return_node

        if result.strategy_ir is not None:
            assert loaded.strategy_ir is not None
            assert loaded.strategy_ir.kernel_id == result.strategy_ir.kernel_id
            assert loaded.schedule_ir is not None
            assert loaded.instruction_ir is not None
        else:
            assert loaded.strategy_ir is None


class TestCLI:
    def test_cli_compile_to_stdout(self):
        import subprocess

        ak_file = str(OPERATORS_DIR / "00_relu.ak")
        proc = subprocess.run(
            ["/home/blueyi/.venvs/arke/bin/python", "-m", "arke.cli", "compile", ak_file],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["format"] == "akir"
        assert data["version"] == AKIR_VERSION
        assert data["semantic_ir"]["kernel_id"] == "relu_kernel"
        assert data["schedule_ir"] is not None
        assert data["instruction_ir"] is not None

    def test_cli_compile_to_file(self, tmp_path):
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
        assert data["version"] == AKIR_VERSION

    def test_cli_compile_invalid_file(self):
        import subprocess

        proc = subprocess.run(
            ["/home/blueyi/.venvs/arke/bin/python", "-m", "arke.cli", "compile", "/nonexistent/file.ak"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
