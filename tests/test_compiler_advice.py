# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.compiler_advice import compile_advice_for_op
from benchmarks.hardware import HardwareInfo
from benchmarks.shapes import AttentionShape


class TestCompilerAdvice:
    def test_compile_advice_blocks_large_attention_on_6gb(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        advice = compile_advice_for_op(hw, "flash_attention", shape)
        assert advice.allow_compile is False
        assert "memory-aware dispatch" in advice.strategy_hint

    def test_compile_advice_allows_small_attention(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="gpt2-sm-128", B=1, H=12, S=128, D=64)
        advice = compile_advice_for_op(hw, "flash_attention", shape)
        assert advice.allow_compile is True
