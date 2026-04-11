# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.pipeline import ArkePipeline


class TestStage7StrategySynthesis:
    def test_compile_advice_synthesizes_conditional_strategy_for_long_context_attention(self):
        source = '''
kernel flash_attention_dispatch(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where B: dynamic(min=1, max=4), H: static, S: dynamic(min=512, max=32768), D: static
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy flash_attention_dispatch for target("nvidia_ampere") {
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors

        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "memory preflight on 6GB gpu",
            "strategy_hint": "prefer conditional strategy",
        }
        # rerun synthesis+lowering path manually to mirror compile pipeline stage
        from arke.compiler.pipeline import _synthesize_strategy_from_compile_advice
        from arke.compiler.lowering import lower_full_stack
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        schedule, instruction = lower_full_stack(result.semantic_ir, result.strategy_ir)

        assert result.strategy_ir.decisions
        cond = result.strategy_ir.decisions[0]
        assert getattr(cond, "predicate", None) == "S <= 4096"
        assert len(cond.true_decisions) == 3
        assert len(cond.false_decisions) == 3
        assert schedule is not None
        assert instruction is not None
        assert any(r.source_kind == "conditional" for r in schedule.provenance)

    def test_explicit_strategy_is_not_overwritten_by_synthesis(self):
        source = '''
kernel flash_attention_dispatch(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where B: dynamic(min=1, max=4), H: static, S: dynamic(min=512, max=32768), D: static
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy flash_attention_dispatch for target("nvidia_ampere") {
    tile(loop="Br", factors=[96]) @rationale("user strategy");
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "memory preflight on 6GB gpu",
            "strategy_hint": "prefer conditional strategy",
        }
        from arke.compiler.pipeline import _synthesize_strategy_from_compile_advice
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        assert len(result.strategy_ir.decisions) == 1
        assert getattr(result.strategy_ir.decisions[0], "kind", None) == "tile"
