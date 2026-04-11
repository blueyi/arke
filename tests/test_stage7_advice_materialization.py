# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.lowering import strategy_to_schedule
from arke.ir.semantic import Node, Param, ParamRef, SemanticIR, Semantics, SymbolicDim, TensorDesc
from arke.ir.strategy import StrategyIR


class TestStage7AdviceMaterialization:
    def test_long_context_attention_advice_materializes_schedule_hints(self):
        semantic = SemanticIR(kernel_id="flash_attention_dispatch")
        semantic.add_symbolic_dim(SymbolicDim(name="S", min=512, max=32768))
        semantic.add_param(Param(name="Q", shape=[1, 12, SymbolicDim(name="S"), 64], dtype="f16"))
        semantic.add_param(Param(name="K", shape=[1, 12, SymbolicDim(name="S"), 64], dtype="f16"))
        semantic.add_param(Param(name="V", shape=[1, 12, SymbolicDim(name="S"), 64], dtype="f16"))
        semantic.add_node(Node(
            id="n1",
            op="flash_attention",
            inputs={"Q": ParamRef("Q"), "K": ParamRef("K"), "V": ParamRef("V")},
            output=TensorDesc(shape=[1, 12, SymbolicDim(name="S"), 64], dtype="f16"),
            semantics=Semantics(computation="flash_attention(Q,K,V)"),
        ))
        strategy = StrategyIR(
            kernel_id="flash_attention_dispatch",
            target_hw="nvidia_ampere",
            metadata={
                "compile_advice": {
                    "allow_compile": False,
                    "reason": "memory preflight on 6GB gpu",
                    "strategy_hint": "prefer smaller tiles and lower shared memory",
                }
            },
        )

        schedule = strategy_to_schedule(semantic, strategy)
        loop_names = {loop.loop for loop in schedule.loop_nests}
        assert "Br" in loop_names
        assert "Bc" in loop_names
        assert schedule.resources.shared_memory == 32768
        assert any(r.effect == "materialized:long-context-attention-guard" for r in schedule.provenance)
