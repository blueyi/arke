# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.ir.converters import ast_to_strategy
from arke.ir.strategy import ConditionalDecision, Decision
from arke.lang.grammar import parse_string


class TestStrategyConverter:
    def test_when_otherwise_converts_to_conditional_decision(self):
        program = parse_string(
            '''
strategy k_strategy for target("nvidia_ampere") {
    when S <= 512 {
        tile(loop="S", factors=[256]) @rationale("small sequence tile");
    } otherwise {
        tile(loop="S", factors=[512]) @rationale("large sequence tile");
    }
}
            '''
        )
        ir = ast_to_strategy(program.strategies[0])
        assert len(ir.decisions) == 1
        cd = ir.decisions[0]
        assert isinstance(cd, ConditionalDecision)
        assert cd.predicate == "S <= 512"
        assert len(cd.true_decisions) == 1
        assert len(cd.false_decisions) == 1
        assert cd.true_decisions[0].rationale is not None
        assert cd.false_decisions[0].rationale is not None

    def test_compute_directive_lowers_to_l2_decision(self):
        program = parse_string(
            '''
strategy k_strategy for target("nvidia_ampere") {
    compute(num_warps=4, num_stages=2, shared_memory=65536)
        @rationale("resource planning");
}
            '''
        )
        ir = ast_to_strategy(program.strategies[0])
        assert len(ir.decisions) == 1
        d = ir.decisions[0]
        assert isinstance(d, Decision)
        assert d.kind == "compute"
        assert d.level == 2
        assert d.params["num_warps"] == 4
        assert d.params["num_stages"] == 2
        assert d.params["shared_memory"] == 65536
        assert d.rationale is not None
