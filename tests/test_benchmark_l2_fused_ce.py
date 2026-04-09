# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.bench_l2 import run_fused_op


class TestBenchmarkL2FusedCE:
    def test_fused_linear_cross_entropy_uses_benchmark_shape(self):
        results = run_fused_op(
            "fused_linear_cross_entropy",
            warmup=1,
            reps=1,
            shape_tags=["gpt2-seq128"],
        )
        assert results
        assert {r.shape_tag for r in results} == {"gpt2-seq128"}
        assert all(r.op == "fused_linear_cross_entropy" for r in results)
