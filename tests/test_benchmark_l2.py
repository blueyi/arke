# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.bench_l2 import run_fused_op


class TestBenchmarkL2:
    def test_gated_fused_ops_use_benchmark_shapes(self):
        silu_and_mul_results = run_fused_op("silu_and_mul", warmup=1, reps=1, shape_tags=["gpt2-sm"])
        geglu_results = run_fused_op("geglu", warmup=1, reps=1, shape_tags=["gpt2-sm"])

        assert silu_and_mul_results
        assert geglu_results
        assert {r.shape_tag for r in silu_and_mul_results} == {"gpt2-sm"}
        assert {r.shape_tag for r in geglu_results} == {"gpt2-sm"}
        assert all(r.op == "silu_and_mul" for r in silu_and_mul_results)
        assert all(r.op == "geglu" for r in geglu_results)
