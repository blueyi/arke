# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.compiler_advice import compile_advice_for_op
from benchmarks.hardware import HardwareInfo
from benchmarks.shapes import AttentionShape, GroupedMatmulShape


class TestArkeRunnerAdvice:
    def test_runner_side_advice_matches_ot4_memory_pressure(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        advice = compile_advice_for_op(hw, "paged_attention", shape)
        assert advice.allow_compile is False
        assert advice.reason

    def test_runner_side_advice_matches_ot2_grouped_matmul_pressure(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = GroupedMatmulShape(tag="ds-v3-moe", B=4, E=256, M=512, K=7168, N=7168)
        advice = compile_advice_for_op(hw, "grouped_matmul", shape)
        assert advice.allow_compile is False
        assert "expert sharding" in advice.strategy_hint
