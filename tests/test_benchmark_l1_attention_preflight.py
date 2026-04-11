# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.hardware import HardwareInfo
from benchmarks.memory_policy import maybe_attention_preflight
from benchmarks.shapes import AttentionShape


class TestBenchmarkL1AttentionPreflight:
    def test_attention_family_preflight_skips_large_shape(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        status = maybe_attention_preflight(hw, "flash_attention", shape)
        assert status is not None
        assert status.status == "skipped"

    def test_non_attention_op_has_no_preflight(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        status = maybe_attention_preflight(hw, "matmul", shape)
        assert status is None
