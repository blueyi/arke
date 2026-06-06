# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Stage 8 autonomous optimize MVP."""

from __future__ import annotations

import json
from pathlib import Path

from arke.agent.optimize import (
    HeuristicStrategyGenerator,
    OptimizeInputRouter,
    optimize,
    optimize_file,
)
from arke.compiler.pipeline import ArkePipeline

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"


def test_heuristic_strategy_generator_emits_bounded_matmul_strategy():
    result = ArkePipeline().compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
    assert result.success, result.errors
    assert result.semantic_ir is not None

    strategy = HeuristicStrategyGenerator().generate(result.semantic_ir)

    assert strategy.kernel_id == "matmul"
    assert strategy.target_hw == "nvidia_ampere"
    assert len(strategy.decisions) >= 6
    assert {decision.kind for decision in strategy.decisions} >= {
        "tile",
        "reorder",
        "parallel",
        "place",
        "compute",
    }
    assert "triton" not in json.dumps(strategy.to_dict()).lower()


def test_optimize_file_records_three_compile_profile_adjust_cycles(tmp_path):
    out_dir = tmp_path / "optimize"
    result = optimize_file(
        OPERATORS_DIR / "01_matmul.ak",
        output_dir=out_dir,
        cycles=3,
    )

    assert result.success, result.errors
    assert result.input_kind == "ak_file"
    assert result.normalized_source_path == str(out_dir / "input.ak")
    assert result.cycles_completed == 3
    assert result.decision_count >= 6
    assert Path(result.strategy_path).exists()
    assert Path(result.akir_path).exists()
    assert Path(result.summary_path).exists()

    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    # D8-F3 trajectory v1.0: first line MUST be a `header` record pinning
    # the contract id; subsequent cycle order is expressed via the
    # `compile` / `profile` / `adjust` record kinds in stream order.
    assert events[0]["kind"] == "header"
    assert events[0]["data"]["contract_id"] == "arke-trajectory-v1.0.0"
    assert events[0]["data"]["trajectory_version"] == "1.0.0"
    cycle_kinds = [
        event["kind"] for event in events
        if event.get("kind") in {"compile", "profile", "adjust"}
    ]
    assert cycle_kinds == ["compile", "profile", "adjust"] * 3
    # Trajectory must end with the terminal `done` record.
    assert events[-1]["kind"] == "done"

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["success"] is True
    assert summary["input_kind"] == "ak_file"
    assert summary["normalized_source_path"] == str(out_dir / "input.ak")
    assert summary["cycles_completed"] == 3
    assert summary["decision_count"] == result.decision_count


def test_optimize_routes_structured_args_to_compile_ready_source(tmp_path):
    result = optimize(
        kernel="matmul",
        shape="16,32,64",
        output_dir=tmp_path / "structured",
        cycles=1,
    )

    assert result.success, result.errors
    assert result.input_kind == "structured_args"
    assert result.kernel_id == "matmul_kernel"
    assert result.cycles_completed == 1

    normalized = Path(result.normalized_source_path or "")
    assert normalized.exists()
    source = normalized.read_text()
    assert "kernel matmul_kernel" in source
    assert "A: Tensor<[16, 64], f16>" in source
    assert "B: Tensor<[64, 32], f16>" in source


def test_optimize_routes_natural_language_input(tmp_path):
    result = optimize(
        "optimize relu for shape 1024x2048 fp16",
        output_dir=tmp_path / "nl",
        cycles=1,
    )

    assert result.success, result.errors
    assert result.input_kind == "natural_language"
    assert result.kernel_id == "relu_kernel"
    assert result.source_text_path is not None
    assert Path(result.source_text_path).read_text() == "optimize relu for shape 1024x2048 fp16"

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["input_kind"] == "natural_language"
    assert summary["source_text_path"] == result.source_text_path


def test_optimize_routes_code_snippet_and_preserves_function_name(tmp_path):
    snippet = (
        "def fused(x, w): return torch.nn.functional.gelu("
        "torch.matmul(x, w))  # m=16 n=32 k=64"
    )
    result = optimize(snippet, output_dir=tmp_path / "code", cycles=1)

    assert result.success, result.errors
    assert result.input_kind == "code_snippet"
    assert result.kernel_id == "fused"
    assert Path(result.source_text_path or "").read_text() == snippet
    assert "kernel fused" in Path(result.normalized_source_path or "").read_text()


def test_optimize_input_router_requires_disambiguation_for_unknown_text():
    router = OptimizeInputRouter()
    try:
        router.route("please make this faster")
    except ValueError as exc:
        assert "Could not infer optimize input op" in str(exc)
    else:
        raise AssertionError("expected unknown natural-language input to fail")
