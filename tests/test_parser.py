# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from arke.lang.ast import (
    Annotation,
    BoolCondition,
    CompareCondition,
    InferType,
    LetStmt,
    Program,
    ScalarType,
    StrategyStmt,
    TensorType,
    TupleType,
    WhenBlock,
)
from arke.lang.grammar import parse_file, parse_string


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "operators"


def test_parse_all_operator_examples() -> None:
    files = sorted(EXAMPLES_DIR.glob("*.ak"))
    assert len(files) == 46, "expected 46 operator example files"

    for path in files:
        program = parse_file(path)
        assert isinstance(program, Program), f"failed to parse {path.name}"
        assert len(program.kernels) >= 1, f"no kernel found in {path.name}"


def test_parse_basic_kernel() -> None:
    source = """
    kernel relu_kernel(
        X: Tensor<[128, 3072], f16>
    ) -> Tensor<[128, 3072], f16> {
        let Y = relu(X=X);
        return Y;
    }
    """
    program = parse_string(source)

    assert len(program.kernels) == 1
    kernel = program.kernels[0]

    assert kernel.name == "relu_kernel"
    assert len(kernel.params) == 1
    assert kernel.params[0].name == "X"
    assert kernel.params[0].type == TensorType(
        shape=[128, 3072],
        dtype=ScalarType("f16"),
        layout="row_major",
    )
    assert kernel.return_type == TensorType(
        shape=[128, 3072],
        dtype=ScalarType("f16"),
        layout="row_major",
    )
    assert len(kernel.body) == 2
    assert isinstance(kernel.body[0], LetStmt)
    assert kernel.body[0].lhs == "Y"
    assert kernel.body[0].op_call.op == "relu"
    assert kernel.body[0].op_call.args == [("X", "X")]
    assert kernel.body[1].value == "Y"


def test_parse_strategy_with_annotations() -> None:
    source = """
    strategy relu_kernel_strategy for target("nvidia_ampere") {
        tile(loop="row", factors=[4])
            @rationale("4 rows per block — fully memory-bound, maximize occupancy");
        launch_config(num_warps=4, num_stages=1)
            @meta(kind="launch")
            @rationale("single-stage: pure elementwise, no arithmetic pipeline benefit");
    }
    """
    program = parse_string(source)

    assert len(program.strategies) == 1
    strategy = program.strategies[0]
    assert strategy.name == "relu_kernel_strategy"
    assert strategy.target == "nvidia_ampere"
    assert len(strategy.body) == 2

    stmt1 = strategy.body[0]
    assert isinstance(stmt1, StrategyStmt)
    assert stmt1.directive == "tile"
    assert stmt1.kwargs == {"loop": "row", "factors": [4]}
    assert len(stmt1.annotations) == 1
    assert stmt1.annotations[0].name == "rationale"
    assert stmt1.annotations[0].args == ["4 rows per block — fully memory-bound, maximize occupancy"]

    stmt2 = strategy.body[1]
    assert isinstance(stmt2, StrategyStmt)
    assert stmt2.directive == "launch_config"
    assert stmt2.kwargs == {"num_warps": 4, "num_stages": 1}
    assert [a.name for a in stmt2.annotations] == ["meta", "rationale"]


def test_parse_where_clause_and_kernel_annotations() -> None:
    source = """
    @constraint(dtypes="f16|bf16|f32")
    @meta(category="OT1", fusion_hint="epilogue")
    kernel matmul_gelu(
        X: Tensor<[M, K], f16>,
        W: Tensor<[K, N], f16>
    ) -> _
    where M: dynamic(max=4096), K: static, N: dynamic(min=64, multiple_of=32, default=128)
    {
        let Z = matmul(A=X, B=W);
        let Y = gelu(X=Z);
        return Y;
    }
    """
    program = parse_string(source)
    kernel = program.kernels[0]

    assert [a.name for a in kernel.annotations] == ["constraint", "meta"]
    assert kernel.annotations[0].args == [("dtypes", "f16|bf16|f32")]
    assert kernel.annotations[1].args == [("category", "OT1"), ("fusion_hint", "epilogue")]
    assert isinstance(kernel.return_type, InferType)
    assert kernel.where_clause is not None
    dims = kernel.where_clause.dims
    assert len(dims) == 3
    assert dims[0].name == "M"
    assert dims[0].kind == "dynamic"
    assert dims[0].opts == {"max": 4096}
    assert dims[1].name == "K"
    assert dims[1].kind == "static"
    assert dims[1].opts == {}
    assert dims[2].name == "N"
    assert dims[2].kind == "dynamic"
    assert dims[2].opts == {"min": 64, "multiple_of": 32, "default": 128}


def test_parse_tuple_destructuring_and_multi_return() -> None:
    source = """
    kernel top_candidates(
        scores: Tensor<[B, V], f16>
    ) -> (_, _)
    where B: dynamic(max=512), V: static
    {
        let (values, indices) = topk(X=scores, k=50);
        return (values, indices);
    }
    """
    program = parse_string(source)
    kernel = program.kernels[0]

    assert isinstance(kernel.return_type, TupleType)
    assert len(kernel.return_type.types) == 2
    assert all(isinstance(t, InferType) for t in kernel.return_type.types)

    let_stmt = kernel.body[0]
    assert isinstance(let_stmt, LetStmt)
    assert let_stmt.lhs == ["values", "indices"]
    assert let_stmt.op_call.op == "topk"
    assert let_stmt.op_call.args == [("X", "scores"), ("k", 50)]

    ret_stmt = kernel.body[1]
    assert ret_stmt.value == ["values", "indices"]


def test_parse_import_statements() -> None:
    source = 'import "arke://ops/normalization" as norm;\nimport "arke://ops/math";\n'
    program = parse_string(source)

    assert len(program.imports) == 2
    assert program.imports[0].path == "arke://ops/normalization"
    assert program.imports[0].alias == "norm"
    assert program.imports[1].path == "arke://ops/math"
    assert program.imports[1].alias is None


def test_parse_when_otherwise_strategy_blocks() -> None:
    source = """
    strategy sdpa_strategy for target("nvidia_ampere") {
        when S <= 512 {
            tile(loop="S", factors=[64])
                @rationale("short seqlen");
            compute(parallelism=32, pipeline_depth=2);
        }
        when S <= 2048 {
            tile(loop="S", factors=[128])
                @rationale("medium seqlen");
        }
        otherwise {
            tile(loop="S", factors=[256])
                @rationale("long seqlen");
        }
        parallel(loops=["B", "H"], mapping={"B": "blockIdx.x", "H": "blockIdx.y"});
    }
    """
    program = parse_string(source)
    strategy = program.strategies[0]

    assert len(strategy.body) == 2
    block = strategy.body[0]
    assert isinstance(block, WhenBlock)
    assert len(block.arms) == 2
    assert block.otherwise_body is not None

    cond1, body1 = block.arms[0]
    assert isinstance(cond1, CompareCondition)
    assert cond1.ident == "S"
    assert cond1.op == "<="
    assert cond1.value == 512
    assert len(body1) == 2
    assert all(isinstance(stmt, StrategyStmt) for stmt in body1)

    cond2, body2 = block.arms[1]
    assert isinstance(cond2, CompareCondition)
    assert cond2.value == 2048
    assert len(body2) == 1

    assert len(block.otherwise_body) == 1
    assert block.otherwise_body[0].directive == "tile"

    tail = strategy.body[1]
    assert isinstance(tail, StrategyStmt)
    assert tail.directive == "parallel"
    assert tail.kwargs == {
        "loops": ["B", "H"],
        "mapping": {"B": "blockIdx.x", "H": "blockIdx.y"},
    }


def test_parse_boolean_conditions() -> None:
    source = """
    strategy s for target("nvidia_ampere") {
        when (S <= 512) and (B <= 8) {
            tile(loop="S", factors=[64]);
        }
    }
    """
    program = parse_string(source)
    block = program.strategies[0].body[0]
    cond, _body = block.arms[0]

    assert isinstance(cond, BoolCondition)
    assert cond.op == "and"
    assert isinstance(cond.left, CompareCondition)
    assert isinstance(cond.right, CompareCondition)
    assert cond.left.ident == "S"
    assert cond.right.ident == "B"


@pytest.mark.parametrize(
    "source",
    [
        "kernel broken(X: Tensor<[128], f16>) Tensor<[128], f16> { return X; }",  # missing ->
        "strategy s for target(\"nvidia_ampere\") { when S <= { tile(loop=\"S\", factors=[64]); } }",  # bad condition
        "import \"arke://ops/math\" as ;",  # missing alias
        "kernel k(X: Tensor<[128], f16>) -> Tensor<[128], f16> { let Y = relu(X=X) return Y; }",  # missing semicolon
        "kernel k(X: Tensor<[128], f16>) -> Tensor<[128], f16> where M: dynamic(max=) { return X; }",  # bad where clause
    ],
)
def test_parse_invalid_syntax(source: str) -> None:
    with pytest.raises(Exception):
        parse_string(source)
