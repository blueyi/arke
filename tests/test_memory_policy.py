# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.hardware import HardwareInfo
from benchmarks.memory_policy import attention_preflight, estimate_attention_bytes


class TestMemoryPolicy:
    def test_attention_preflight_allows_small_case(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, estimate = attention_preflight(
            hw,
            batch=1,
            heads=12,
            seq=128,
            head_dim=64,
        )
        assert status.status == "ok"
        assert estimate.bytes_required < estimate.bytes_budget

    def test_attention_preflight_skips_large_case(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, estimate = attention_preflight(
            hw,
            batch=4,
            heads=32,
            seq=32768,
            head_dim=128,
        )
        assert status.status == "skipped"
        assert status.retryable is True
        assert estimate.bytes_required > estimate.bytes_budget

    def test_attention_estimate_grows_with_sequence_quadratically(self):
        small = estimate_attention_bytes(batch=1, heads=8, seq=512, head_dim=64)
        large = estimate_attention_bytes(batch=1, heads=8, seq=4096, head_dim=64)
        assert large > small * 10
