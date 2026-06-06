# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob

from arke.ir.converters import ast_to_semantic, ast_to_strategy
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR
from arke.lang.grammar import parse_file, parse_string


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
    compute(warps=4, num_stages=1, shared_memory=0)
        @rationale("single-stage");
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        assert strat.kernel_id == "relu_kernel"
        assert strat.target_hw == "nvidia_ampere"
        assert strat.decision_count == 2

        d0 = strat.decisions[0]
        assert d0.kind == "tile"
        assert d0.params["loop"] == "row"
        assert d0.params["factors"] == [4]
        assert d0.rationale is not None
        assert "4 rows" in d0.rationale.text
        assert d0.level == 1

        d1 = strat.decisions[1]
        assert d1.kind == "compute"
        assert d1.level == 2
        assert d1.params["warps"] == 4
        assert d1.params["num_stages"] == 1
        assert d1.params["shared_memory"] == 0

    def test_rationale_preserved(self):
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
        prog = parse_string('''
kernel k(X: Tensor<[128], f16>) -> Tensor<[128], f16> {
    let Y = relu(X=X);
    return Y;
}
strategy s for target("nvidia_ampere") {
    tile(loop="M", factors=[64])
        @rationale("cache aligned");
    place(tensor="X_tile", memory="shared");
    compute(warps=8, num_stages=2, shared_memory=49152);
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        j1 = strat.to_json()
        strat2 = StrategyIR.from_json(j1)
        j2 = strat2.to_json()
        assert j1 == j2

    def test_v2_compute_directive_preserved(self):
        prog = parse_string("""
strategy s for target("nvidia_ampere") {
    compute(warps=8, num_stages=3, shared_memory=49152)
        @rationale("3-stage pipeline for memory latency hiding");
}
        """)
        strat = ast_to_strategy(prog.strategies[0])
        d0 = strat.decisions[0]
        assert d0.kind == "compute"
        assert d0.level == 2
        assert d0.params == {
            "warps": 8,
            "num_stages": 3,
            "shared_memory": 49152,
        }
        assert d0.rationale is not None

    def test_multiple_decisions(self):
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
    compute(warps=8, num_stages=2, shared_memory=49152);
}
        ''')
        strat = ast_to_strategy(prog.strategies[0])
        assert strat.decision_count == 8


class TestParseAllExamples:
    def test_all_ak_files_to_semantic(self):
        from benchmarks.op_registry import total_ops
        files = sorted(glob.glob("examples/operators/*.ak"))
        assert len(files) >= total_ops()  # ≥ SSOT catalog total
        for f in files:
            prog = parse_file(f)
            for k in prog.kernels:
                sem = ast_to_semantic(k)
                assert sem.kernel_id != ""
                assert len(sem.params) >= 1
                assert len(sem.nodes) >= 1

    def test_all_ak_files_strategies(self):
        files = sorted(glob.glob("examples/operators/*.ak"))
        strategy_count = 0
        for f in files:
            prog = parse_file(f)
            for s in prog.strategies:
                strat = ast_to_strategy(s)
                assert strat.kernel_id != ""
                assert strat.target_hw != ""
                strategy_count += 1
        assert strategy_count >= 45

    def test_all_ak_files_json_round_trip(self):
        files = sorted(glob.glob("examples/operators/*.ak"))
        for f in files:
            prog = parse_file(f)
            for k in prog.kernels:
                sem = ast_to_semantic(k)
                j1 = sem.to_json()
                sem2 = SemanticIR.from_json(j1)
                j2 = sem2.to_json()
                assert j1 == j2, f"JSON round-trip failed for {f} kernel {k.name}"
