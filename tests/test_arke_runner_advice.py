# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.compiler_advice import compile_advice_for_op
from benchmarks.hardware import HardwareInfo
from benchmarks.shapes import AttentionShape


class TestArkeRunnerAdvice:
    def test_runner_side_advice_matches_ot4_memory_pressure(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        advice = compile_advice_for_op(hw, "paged_attention", shape)
        assert advice.allow_compile is False
        assert advice.reason
