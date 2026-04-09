# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.bench_l2 import ALL_FUSED_OPS, run_fused_op


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
