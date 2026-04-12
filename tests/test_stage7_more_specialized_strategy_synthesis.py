# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.pipeline import ArkePipeline, _synthesize_strategy_from_compile_advice


class TestStage7MoreSpecializedStrategySynthesis:
    def test_cross_attention_synthesizes_asymmetric_kv_strategy(self):
        source = '''
kernel cross_attention_dispatch(
    Q: Tensor<[B, H, S_q, D], f16>,
    K: Tensor<[B, H, S_kv, D], f16>,
    V: Tensor<[B, H, S_kv, D], f16>
) -> Tensor<[B, H, S_q, D], f16>
where B: dynamic(min=1, max=8), H: static, S_q: dynamic(min=1, max=2048), S_kv: dynamic(min=128, max=32768), D: static
{
    let O = cross_attention(Q=Q, K=K, V=V);
    return O;
}

strategy cross_attention_dispatch for target("nvidia_ampere") {
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "kv cache pressure",
            "strategy_hint": "prefer smaller kv tile",
        }
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        cond = result.strategy_ir.decisions[0]
        assert getattr(cond, "predicate", None) == "S_kv <= 4096"
        assert any(d.params.get("loop") == "S_q" for d in cond.true_decisions)
        assert any(d.params.get("loop") == "S_kv" for d in cond.false_decisions)

    def test_rope_synthesizes_dim_aware_strategy(self):
        source = '''
kernel rope_dispatch(
    X: Tensor<[B, S, H, D], f16>
) -> Tensor<[B, S, H, D], f16>
where B: dynamic(min=1, max=8), S: dynamic(min=128, max=32768), H: static, D: dynamic(min=32, max=256)
{
    let Y = rope(X=X);
    return Y;
}

strategy rope_dispatch for target("nvidia_ampere") {
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "vector pressure",
            "strategy_hint": "prefer narrower rope vector width",
        }
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        cond = result.strategy_ir.decisions[0]
        assert getattr(cond, "predicate", None) == "D <= 128"
        assert any(d.params.get("loop") == "D" for d in cond.true_decisions)
        assert any(d.kind == "compute" for d in cond.false_decisions)
