# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.compiler.pipeline import ArkePipeline


class TestStage7L2FusionSurface:
    def test_matmul_gelu_surface_is_representable(self):
        source = '''
kernel matmul_gelu_kernel(
    A: Tensor<[M, K], f16>,
    B: Tensor<[K, N], f16>
) -> Tensor<[M, N], f16>
where M: dynamic(max=4096), K: static, N: dynamic(max=4096)
{
    let C = matmul(A=A, B=B);
    let Y = gelu(X=C);
    return Y;
}

strategy matmul_gelu_kernel for target("nvidia_ampere") {
    fuse(ops=["matmul", "gelu"], fusion_type="epilogue")
        @rationale("stage7 l2 fusion slot: matmul_gelu");
    compute(warps=4, num_stages=3)
        @rationale("resource plan for fused epilogue path");
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        assert result.strategy_ir is not None
        assert any(getattr(d, "kind", None) == "fuse" for d in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.schedule_ir.fusion_groups

    def test_gated_activation_l2_fusion_surfaces_are_representable(self):
        for path, expected_ops in (
            ("examples/operators/19_silu_and_mul.ak", ["silu", "mul"]),
            ("examples/operators/20_geglu.ak", ["gelu", "mul"]),
        ):
            result = ArkePipeline().compile_file(path)
            assert result.success, result.errors
            assert result.semantic_ir is not None
            assert result.strategy_ir is not None
            assert any(getattr(d, "kind", None) == "fuse" for d in result.strategy_ir.decisions)
            assert result.schedule_ir is not None
            assert result.schedule_ir.fusion_groups
            assert result.schedule_ir.fusion_groups[0].ops == expected_ops
            assert result.schedule_ir.fusion_groups[0].fusion_type == "epilogue"

    def test_linear_ce_surface_is_representable(self):
        result = ArkePipeline().compile_file("examples/operators/41_fused_linear_cross_entropy.ak")
        assert result.success, result.errors
        assert result.semantic_ir is not None
        assert any(node.op == "fused_linear_cross_entropy" for node in result.semantic_ir.nodes)
        assert result.strategy_ir is not None
        assert any(getattr(d, "kind", None) == "compute" for d in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.instruction_ir is not None

    def test_qkv_fa_surface_is_representable(self):
        source = '''
kernel qkv_fa_kernel(
    X: Tensor<[B, S, D], f16>,
    Wq: Tensor<[D, H], f16>,
    Wk: Tensor<[D, H], f16>,
    Wv: Tensor<[D, H], f16>
) -> _
where B: dynamic(max=64), S: dynamic(max=8192), D: static, H: static
{
    let Q = matmul(A=X, B=Wq);
    let K = matmul(A=X, B=Wk);
    let V = matmul(A=X, B=Wv);
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy qkv_fa_kernel for target("nvidia_ampere") {
    fuse(ops=["matmul", "matmul", "matmul", "flash_attention"], fusion_type="producer_consumer")
        @rationale("stage7 l2 fusion slot: qkv_fa");
    compute(warps=4, num_stages=2, shared_memory=65536)
        @rationale("shared-memory-heavy attention staging on 6GB device");
}
        '''
        result = ArkePipeline().compile_string(source)
        assert result.success, result.errors
        assert result.semantic_ir is not None
        ops = [node.op for node in result.semantic_ir.nodes]
        assert ops.count("matmul") == 3
        assert "flash_attention" in ops
        assert result.strategy_ir is not None
        assert any(getattr(d, "kind", None) == "fuse" for d in result.strategy_ir.decisions)
        assert result.schedule_ir is not None
        assert result.schedule_ir.fusion_groups
        fusion_ops = result.schedule_ir.fusion_groups[0].ops
        assert "flash_attention" in fusion_ops
