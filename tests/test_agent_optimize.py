# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Stage 8 autonomous optimize MVP."""

from __future__ import annotations

import json
from pathlib import Path

from arke.agent.optimize import HeuristicStrategyGenerator, optimize_file
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
    assert result.cycles_completed == 3
    assert result.decision_count >= 6
    assert Path(result.strategy_path).exists()
    assert Path(result.akir_path).exists()
    assert Path(result.summary_path).exists()

    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    actions = [event for event in events if event.get("event_type") == "action"]
    assert [event["tool"] for event in actions] == ["compile", "profile", "adjust"] * 3

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["success"] is True
    assert summary["cycles_completed"] == 3
    assert summary["decision_count"] == result.decision_count
