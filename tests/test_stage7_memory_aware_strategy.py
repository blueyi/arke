# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.pipeline import ArkePipeline
from arke.ir.semantic import SymbolicDim


class TestStage7MemoryAwareStrategy:
    def test_ot4_long_context_conditional_strategy_lowers(self):
        source = '''
kernel flash_attention_dispatch(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where B: dynamic(min=1, max=4, default=1), H: static, S: dynamic(min=512, max=32768, multiple_of=128, default=8192), D: static
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy flash_attention_dispatch for target("nvidia_ampere") {
    when S <= 4096 {
        tile(loop="Br", factors=[128])
            @rationale("smaller context uses wider query tiles");
        tile(loop="Bc", factors=[128])
            @rationale("smaller context uses wider kv tiles");
        compute(warps=8, num_stages=2, shared_memory=65536)
            @rationale("higher smem budget still fits 6GB regime for this branch");
    } otherwise {
        tile(loop="Br", factors=[64])
            @rationale("long context reduces tile footprint to control memory pressure");
        tile(loop="Bc", factors=[64])
            @rationale("smaller kv tiles reduce online softmax state pressure");
        compute(warps=4, num_stages=2, shared_memory=32768)
            @rationale("fallback branch lowers smem demand for long-context runs on 6GB VRAM");
    }
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        assert result.semantic_ir is not None
        assert result.strategy_ir is not None
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None

        dims = {d.name: d for d in result.semantic_ir.symbolic_dims}
        assert "S" in dims
        assert dims["S"].max == 32768
        assert dims["S"].multiple_of == 128

        loop_names = {loop.loop for loop in result.schedule_ir.loop_nests}
        assert "Br" in loop_names
        assert "Bc" in loop_names
        assert any(record.source_kind == "conditional" for record in result.schedule_ir.provenance)
        assert result.schedule_ir.resources.shared_memory == 65536

    def test_symbolic_dim_bounds_can_encode_memory_relevant_shape_family(self):
        source = '''
kernel paged_attention_dispatch(
    Q: Tensor<[B, H, 1, D], f16>,
    K_cache: Tensor<[NB, BS, H, D], f16>,
    V_cache: Tensor<[NB, BS, H, D], f16>,
    BlockTable: Tensor<[B, MB], i32>
) -> Tensor<[B, H, 1, D], f16>
where
    B: dynamic(min=1, max=16, default=4),
    H: static,
    D: static,
    NB: dynamic(min=1, max=4096),
    BS: static,
    MB: dynamic(min=1, max=512)
{
    let O = paged_attention(Q=Q, K_cache=K_cache, V_cache=V_cache, block_table=BlockTable);
    return O;
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        assert result.semantic_ir is not None
        dims = {d.name: d for d in result.semantic_ir.symbolic_dims}
        assert isinstance(dims["NB"], SymbolicDim)
        assert dims["NB"].max == 4096
        assert dims["MB"].max == 512
        assert dims["B"].default == 4
