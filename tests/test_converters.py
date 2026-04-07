# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for AST -> IR converters (arke/ir/converters.py)."""

import glob
import json

import pytest

from arke.lang.grammar import parse_file, parse_string
from arke.ir.converters import ast_to_semantic, ast_to_strategy
from arke.ir.semantic import (
    ConditionalNode,
    MultiOutputNode,
    Node,
    NodeRef,
    ParamRef,
    SemanticIR,
    SymbolicDim,
)
from arke.ir.strategy import (
    ConditionalDecision,
    Decision,
    StrategyIR,
)


# ============================================================
# AST -> SemanticIR
# ============================================================

class TestAstToSemantic:
    def test_relu_kernel(self):
        prog = parse_string('''
kernel relu_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = relu(X=X);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        assert sem.kernel_id == "relu_kernel"
        assert sem.version == "1.0.0"
        assert len(sem.params) == 1
        assert sem.params[0].name == "X"
        assert sem.params[0].dtype == "f16"
        assert sem.params[0].shape == [128, 3072]
        assert len(sem.nodes) == 1
        node = sem.nodes[0]
        assert node.op == "relu"
        assert "X" in node.inputs
        assert isinstance(node.inputs["X"], ParamRef)
        assert sem.return_node == node.id
        assert len(sem.edges) == 1

    def test_matmul_kernel(self):
        prog = parse_string('''
kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A=A, B=B);
    return C;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        assert sem.kernel_id == "matmul"
        assert len(sem.params) == 2
        assert len(sem.nodes) == 1
        node = sem.nodes[0]
        assert node.op == "matmul"
        assert "A" in node.inputs
        assert "B" in node.inputs
        # Check matmul semantics from registry
        assert "sum" in node.semantics.computation.lower() or "matmul" in node.semantics.computation.lower()
        assert sem.return_node == node.id

    def test_two_node_chain(self):
        """matmul -> gelu chain should produce two nodes with correct edges."""
        prog = parse_string('''
kernel matmul_gelu(
    A: Tensor<[128, 768], f16>,
    B: Tensor<[768, 256], f16>
) -> Tensor<[128, 256], f16> {
    let C = matmul(A=A, B=B);
    let Y = gelu(X=C);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        assert len(sem.nodes) == 2
        n0, n1 = sem.nodes
        assert n0.op == "matmul"
        assert n1.op == "gelu"
        # gelu's input should be a NodeRef to matmul
        assert isinstance(n1.inputs["X"], NodeRef)
        assert n1.inputs["X"].id == n0.id
        assert sem.return_node == n1.id

    def test_softmax_with_attrs(self):
        """Softmax call with axis=-1 should preserve attr."""
        prog = parse_string('''
kernel softmax(
    X: Tensor<[2048, 1024], f16>
) -> Tensor<[2048, 1024], f16> {
    let Y = softmax(X=X, axis=-1);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        node = sem.nodes[0]
        assert node.op == "softmax"
        assert node.attrs.get("axis") == -1

    def test_json_round_trip_after_conversion(self):
        """Convert AST -> IR -> JSON -> IR -> JSON should be stable."""
        prog = parse_string('''
kernel relu_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = relu(X=X);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        j1 = sem.to_json()
        sem2 = SemanticIR.from_json(j1)
        j2 = sem2.to_json()
        assert j1 == j2

    def test_edges_from_params(self):
        """Edges from params should use 'param:' prefix."""
        prog = parse_string('''
kernel add_kernel(
    X: Tensor<[128, 768], f16>,
    Y: Tensor<[128, 768], f16>
) -> Tensor<[128, 768], f16> {
    let Z = add(X=X, Y=Y);
    return Z;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        param_edges = [e for e in sem.edges if e.from_node.startswith("param:")]
        assert len(param_edges) == 2


# ============================================================
# AST -> StrategyIR
# ============================================================

class TestAstToStrategy:
    def test_relu_strategy(self):
        prog = parse_string('''
kernel relu_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = relu(X=X);
    return Y;
}

strategy relu_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage");
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        assert strat.kernel_id == "relu_kernel_strategy"
        assert strat.target_hw == "nvidia_ampere"
        assert strat.decision_count == 2

        # First decision: tile
        d0 = strat.decisions[0]
        assert d0.kind == "tile"
        assert d0.params["loop"] == "row"
        assert d0.params["factors"] == [4]
        assert d0.rationale is not None
        assert "4 rows" in d0.rationale.text
        assert d0.level == 1

        # Second: launch_config migrated to compute_resource
        d1 = strat.decisions[1]
        assert d1.kind == "compute_resource"
        assert d1.level == 2
        assert d1.params["warps"] == 4
        assert d1.params["stages"] == 1

    def test_rationale_preserved(self):
        """@rationale annotations should become Rationale objects."""
        prog = parse_string('''
kernel k(X: Tensor<[128], f16>) -> Tensor<[128], f16> {
    let Y = relu(X=X);
    return Y;
}
strategy s for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("this is the reason");
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        assert strat.decisions[0].rationale is not None
        assert strat.decisions[0].rationale.text == "this is the reason"

    def test_json_round_trip_after_conversion(self):
        """Convert AST -> StrategyIR -> JSON -> StrategyIR -> JSON should be stable."""
        prog = parse_string('''
kernel k(X: Tensor<[128], f16>) -> Tensor<[128], f16> {
    let Y = relu(X=X);
    return Y;
}
strategy s for target("nvidia_ampere") {
    tile(loop="M", factors=[64])
        @rationale("cache aligned");
    place(tensor="X_tile", memory="shared");
    launch_config(num_warps=4, num_stages=2);
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        j1 = strat.to_json()
        strat2 = StrategyIR.from_json(j1)
        j2 = strat2.to_json()
        assert j1 == j2

    def test_multiple_decisions(self):
        """Strategy with many directives should preserve all decisions."""
        prog = parse_string('''
kernel k(X: Tensor<[128, 768], f16>, W: Tensor<[768, 768], f16>) -> Tensor<[128, 768], f16> {
    let Y = matmul(A=X, B=W);
    return Y;
}
strategy s for target("nvidia_ampere") {
    tile(loop="M", factors=[64]);
    tile(loop="N", factors=[64]);
    tile(loop="K", factors=[16]);
    reorder(order=["M", "N", "K"]);
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"});
    place(tensor="A_tile", memory="shared");
    place(tensor="B_tile", memory="shared");
    launch_config(num_warps=4, num_stages=2);
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        assert strat.decision_count == 8


# ============================================================
# Parse all 46 .ak files
# ============================================================

class TestParseAll46:
    def test_all_ak_files_to_semantic(self):
        """Parse all 46 .ak files -> AST -> SemanticIR without errors."""
        files = sorted(glob.glob("examples/operators/*.ak"))
        assert len(files) == 46, f"Expected 46 .ak files, found {len(files)}"
        for f in files:
            prog = parse_file(f)
            for k in prog.kernels:
                sem = ast_to_semantic(k)
                assert sem.kernel_id != ""
                assert len(sem.params) >= 1
                assert len(sem.nodes) >= 1

    def test_all_ak_files_strategies(self):
        """Parse all .ak files with strategies -> StrategyIR without errors."""
        files = sorted(glob.glob("examples/operators/*.ak"))
        strategy_count = 0
        for f in files:
            prog = parse_file(f)
            for s in prog.strategies:
                strat = ast_to_strategy(s)
                assert strat.kernel_id != ""
                assert strat.target_hw != ""
                strategy_count += 1
        # Most files have strategies
        assert strategy_count > 20

    def test_all_ak_files_json_round_trip(self):
        """All .ak -> SemanticIR -> JSON -> SemanticIR -> JSON should be stable."""
        files = sorted(glob.glob("examples/operators/*.ak"))
        for f in files:
            prog = parse_file(f)
            for k in prog.kernels:
                sem = ast_to_semantic(k)
                j1 = sem.to_json()
                sem2 = SemanticIR.from_json(j1)
                j2 = sem2.to_json()
                assert j1 == j2, f"JSON round-trip failed for {f} kernel {k.name}"


# ============================================================
# Where clause -> SymbolicDim propagation
# ============================================================

class TestWhereClausePropagation:
    def test_symbolic_dims_from_shape(self):
        """Symbolic dim names in param shapes should be collected."""
        prog = parse_string('''
kernel dynamic_kernel(
    X: Tensor<[M, N], f16>
) -> Tensor<[M, N], f16> {
    let Y = relu(X=X);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        sym_names = {sd.name for sd in sem.symbolic_dims}
        assert "M" in sym_names
        assert "N" in sym_names
        # Param shapes should contain SymbolicDim
        assert isinstance(sem.params[0].shape[0], SymbolicDim)
        assert isinstance(sem.params[0].shape[1], SymbolicDim)
