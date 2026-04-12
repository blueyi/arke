# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from arke.lang.grammar import parse_string
from arke.lang.ast import StrategyDef, WhenBlock, CompareCondition


def test_when_dim_function_condition_parses_to_symbolic_dim_compare():
    program = parse_string(
        '''
strategy conditional_relu_strategy for target("nvidia_ampere") {
    when dim("B") <= 128 {
        tile(dim="B", factors=[64]);
    } otherwise {
        tile(dim="B", factors=[128]);
    }
}
        '''
    )

    strategy = next(item for item in program.strategies if isinstance(item, StrategyDef))
    when_block = next(item for item in strategy.body if isinstance(item, WhenBlock))
    condition = when_block.arms[0][0]

    assert isinstance(condition, CompareCondition)
    assert condition.ident == "B"
    assert condition.op == "<="
    assert condition.value == 128
