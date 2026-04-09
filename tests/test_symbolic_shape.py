# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from arke.ir.semantic import Node, SemanticIR, SymbolicDim
from arke.lang.grammar import parse_file, parse_string
from arke.ir.converters import ast_to_semantic

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS_DIR = REPO_ROOT / "examples" / "operators"
PRODUCTION_DIR = OPERATORS_DIR / "production"

SYMBOLIC_EXAMPLES = sorted(list(OPERATORS_DIR.glob("*.ak")) + list(PRODUCTION_DIR.glob("*.ak")))


def _collect_symbolic_dims(ir: SemanticIR) -> set[str]:
    return {d.name for d in ir.symbolic_dims}


class TestSymbolicShapeCore:
    def test_where_clause_lowers_to_symbolic_dims(self):
        program = parse_string(
            '''
kernel dynamic_matmul(
    A: Tensor<[M, K], f16>,
    B: Tensor<[K, N], f16>
) -> _
where M: dynamic(max=4096), K: static, N: dynamic(min=64, multiple_of=32, default=128)
{
    let C = matmul(A=A, B=B);
    return C;
}
            '''
        )
        ir = ast_to_semantic(program.kernels[0])
        dims = {d.name: d for d in ir.symbolic_dims}
        assert set(dims) == {"M", "K", "N"}
        assert dims["M"].max == 4096
        assert dims["K"].is_static is True
        assert dims["N"].min == 64
        assert dims["N"].multiple_of == 32
        assert dims["N"].default == 128

    def test_symbolic_dims_reach_params_and_output_nodes(self):
        program = parse_string(
            '''
kernel dynamic_softmax(
    X: Tensor<[B, S, D], f16>
) -> _
where B: dynamic(max=16), S: dynamic(max=8192), D: static
{
    let Y = softmax(X=X);
    return Y;
}
            '''
        )
        ir = ast_to_semantic(program.kernels[0])
        assert _collect_symbolic_dims(ir) == {"B", "S", "D"}
        assert any(isinstance(dim, SymbolicDim) and dim.name == "S" for dim in ir.params[0].shape)
        node = ir.nodes[0]
        assert isinstance(node, Node)
        assert any(isinstance(dim, SymbolicDim) and dim.name == "S" for dim in node.output.shape)


class TestStage7SymbolicCoverage:
    @pytest.mark.parametrize("ak_file", SYMBOLIC_EXAMPLES, ids=lambda p: p.name)
    def test_examples_parse_and_lower(self, ak_file: Path):
        program = parse_file(ak_file)
        assert len(program.kernels) >= 1
        for kernel in program.kernels:
            ir = ast_to_semantic(kernel)
            assert ir.kernel_id
            assert len(ir.params) >= 1
            assert len(ir.nodes) >= 1

    def test_production_examples_cover_attention_family(self):
        names = {p.name for p in PRODUCTION_DIR.glob("*.ak")}
        assert "flash_attention_st4.ak" in names
        assert "paged_attention_st4.ak" in names
        assert "fused_linear_cross_entropy_st4.ak" in names
