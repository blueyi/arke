# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.compiler.pipeline import ArkePipeline
from arke.ir.akir import akir_from_dict, akir_to_dict
from arke.lang.grammar import parse_file


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS_DIR = REPO_ROOT / "examples" / "operators"
ALL_STAGE7_AK_FILES = sorted(OPERATORS_DIR.rglob("*.ak"))
ALL_STAGE7_STRATEGY_AK_FILES = [
    path for path in ALL_STAGE7_AK_FILES if parse_file(str(path)).strategies
]


class TestStage7RoundTrip:
    def test_all_stage7_examples_compile(self):
        pipeline = ArkePipeline()
        assert len(ALL_STAGE7_AK_FILES) >= 49

        for ak_file in ALL_STAGE7_AK_FILES:
            result = pipeline.compile_file(str(ak_file))
            assert result.success, f"{ak_file.name}: {result.errors}"
            assert result.semantic_ir is not None, f"{ak_file.name}: missing SemanticIR"

    def test_all_stage7_strategy_examples_have_strategy_ir(self):
        pipeline = ArkePipeline()
        assert len(ALL_STAGE7_STRATEGY_AK_FILES) >= 48

        for ak_file in ALL_STAGE7_STRATEGY_AK_FILES:
            result = pipeline.compile_file(str(ak_file))
            assert result.success, f"{ak_file.name}: {result.errors}"
            assert result.strategy_ir is not None, f"{ak_file.name}: missing StrategyIR"

    def test_all_stage7_strategy_examples_akir_round_trip(self):
        pipeline = ArkePipeline()

        for ak_file in ALL_STAGE7_STRATEGY_AK_FILES:
            result = pipeline.compile_file(str(ak_file))
            assert result.success, f"{ak_file.name}: {result.errors}"
            assert result.semantic_ir is not None
            assert result.strategy_ir is not None

            payload = akir_to_dict(
                result.semantic_ir,
                result.strategy_ir,
                schedule_ir=result.schedule_ir,
                instruction_ir=result.instruction_ir,
            )
            sem2, strat2, sched2, instr2 = akir_from_dict(payload)

            assert sem2.kernel_id == result.semantic_ir.kernel_id, ak_file.name
            assert len(sem2.nodes) == len(result.semantic_ir.nodes), ak_file.name
            assert len(sem2.params) == len(result.semantic_ir.params), ak_file.name
            assert strat2 is not None, ak_file.name
            assert strat2.kernel_id == result.strategy_ir.kernel_id, ak_file.name
            assert len(strat2.decisions) == len(result.strategy_ir.decisions), ak_file.name
            assert sched2 is not None, ak_file.name
            assert instr2 is not None, ak_file.name
