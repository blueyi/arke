# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.bench_l2 import ALL_FUSED_OPS, run_fused_op
from benchmarks.hardware import HardwareInfo
from benchmarks.memory_policy import attention_preflight


class TestBenchmarkL2QkvFa:
    def test_qkv_fa_is_registered(self):
        assert "qkv_fa" in ALL_FUSED_OPS

    def test_qkv_fa_uses_attention_shape_filter(self):
        results = run_fused_op(
            "qkv_fa",
            warmup=1,
            reps=1,
            shape_tags=["gpt2-sm-128"],
        )
        assert results
        assert {r.shape_tag for r in results} == {"gpt2-sm-128"}
        assert all(r.op == "qkv_fa" for r in results)

    def test_qkv_fa_memory_preflight_identifies_st4_pressure(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, _estimate = attention_preflight(
            hw,
            batch=4,
            heads=32,
            seq=32768,
            head_dim=128,
        )
        assert status.status == "skipped"
