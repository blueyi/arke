# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.lowering import strategy_to_schedule
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


class TestStage7CompileAdviceProvenance:
    def test_compile_advice_metadata_flows_into_schedule(self):
        semantic = SemanticIR(kernel_id="flash_attention_dispatch")
        strategy = StrategyIR(
            kernel_id="flash_attention_dispatch",
            target_hw="nvidia_ampere",
            metadata={
                "compile_advice": {
                    "allow_compile": False,
                    "reason": "memory preflight on 6GB gpu",
                    "strategy_hint": "prefer paged kv",
                }
            },
        )
        schedule = strategy_to_schedule(semantic, strategy)
        assert schedule.metadata["compile_advice"]["allow_compile"] is False
        assert any(r.source_kind == "advice" for r in schedule.provenance)
        assert any("compile_advice:False" == r.effect for r in schedule.provenance)

    def test_non_attention_advice_does_not_force_long_context_tiles(self):
        semantic = SemanticIR(kernel_id="relu_kernel")
        strategy = StrategyIR(
            kernel_id="relu_kernel",
            target_hw="nvidia_ampere",
            metadata={
                "compile_advice": {
                    "allow_compile": False,
                    "reason": "generic pressure",
                    "strategy_hint": "be conservative",
                }
            },
        )
        schedule = strategy_to_schedule(semantic, strategy)
        assert schedule.get_loop("Br") is None
        assert schedule.get_loop("Bc") is None
