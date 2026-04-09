# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.bench_l2 import run_fused_op


class TestBenchmarkL2:
    def test_gated_fused_ops_use_benchmark_shapes(self):
        swiglu_results = run_fused_op("swiglu", warmup=1, reps=1, shape_tags=["gpt2-sm"])
        geglu_results = run_fused_op("geglu", warmup=1, reps=1, shape_tags=["gpt2-sm"])

        assert swiglu_results
        assert geglu_results
        assert {r.shape_tag for r in swiglu_results} == {"gpt2-sm"}
        assert {r.shape_tag for r in geglu_results} == {"gpt2-sm"}
        assert all(r.op == "swiglu" for r in swiglu_results)
        assert all(r.op == "geglu" for r in geglu_results)
