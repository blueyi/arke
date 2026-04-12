# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.pipeline import ArkePipeline, _synthesize_strategy_from_compile_advice


class TestStage7SpecializedStrategySynthesis:
    def test_paged_attention_synthesizes_page_aware_strategy(self):
        source = '''
kernel paged_attention_dispatch(
    Q: Tensor<[B, H, 1, D], f16>,
    K_cache: Tensor<[NB, BS, H, D], f16>,
    V_cache: Tensor<[NB, BS, H, D], f16>,
    BlockTable: Tensor<[B, MB], i32>
) -> Tensor<[B, H, 1, D], f16>
where B: dynamic(min=1, max=16), H: static, D: static, NB: dynamic(min=1, max=4096), BS: static, MB: dynamic(min=1, max=512)
{
    let O = paged_attention(Q=Q, K_cache=K_cache, V_cache=V_cache, block_table=BlockTable);
    return O;
}

strategy paged_attention_dispatch for target("nvidia_ampere") {
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "page cache pressure",
            "strategy_hint": "prefer smaller page windows",
        }
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        cond = result.strategy_ir.decisions[0]
        assert getattr(cond, "predicate", None) == "NB <= 1024 and MB <= 128"
        assert any(d.params.get("loop") == "NB" for d in cond.true_decisions)
        assert any(d.params.get("loop") == "MB" for d in cond.true_decisions)

    def test_mla_synthesizes_compressed_kv_strategy(self):
        source = '''
kernel mla_dispatch(
    Q: Tensor<[B, H, S, D], f16>,
    KV_compressed: Tensor<[B, S, D_c], f16>,
    W_uk: Tensor<[D_c, H, D], f16>,
    W_uv: Tensor<[D_c, H, D], f16>
) -> Tensor<[B, H, S, D], f16>
where B: dynamic(min=1, max=4), H: static, S: dynamic(min=128, max=8192), D: static, D_c: dynamic(min=16, max=128)
{
    let O = multi_latent_attention(Q=Q, KV_compressed=KV_compressed, W_uk=W_uk, W_uv=W_uv);
    return O;
}

strategy mla_dispatch for target("nvidia_ampere") {
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        result.strategy_ir.metadata["compile_advice"] = {
            "allow_compile": False,
            "reason": "compressed kv pressure",
            "strategy_hint": "prefer compact kv branch",
        }
        _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
        cond = result.strategy_ir.decisions[0]
        assert getattr(cond, "predicate", None) == "D_c <= 64"
        assert any(d.params.get("loop") == "D_c" for d in cond.true_decisions)
        assert any(d.kind == "compute" for d in cond.false_decisions)
