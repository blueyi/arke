# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for G9[2] Arke-vs-LLM-direct comparison harness."""

from __future__ import annotations

from benchmarks.compare_arke_vs_direct import (
    ARKE_CORRECTNESS_MIN,
    PERF_GEOMEAN_MIN,
    TOKEN_RATIO_MAX,
    ComparisonReport,
    OpComparison,
    _geomean,
)


def test_locked_thresholds_match_plan():
    # Mirror of docs/phase1/stage9-plan.md G9[2] — must not drift.
    assert PERF_GEOMEAN_MIN == 1.05
    assert abs(TOKEN_RATIO_MAX - 0.70) < 1e-9
    assert ARKE_CORRECTNESS_MIN == 1.0


def test_geomean_basic():
    assert _geomean([1.0, 4.0]) == 2.0
    assert _geomean([]) is None
    assert _geomean([0.0, -1.0]) is None


def test_perf_ratio_arke_faster():
    c = OpComparison(op="matmul", shape=(256, 256, 256),
                     arke_latency_us=50.0, direct_latency_us=100.0)
    # direct slower → ratio 2.0 (Arke 2× faster)
    assert c.perf_ratio == 2.0


def test_token_ratio_arke_cheaper():
    c = OpComparison(op="matmul", shape=(256, 256, 256),
                     arke_tokens=0, direct_tokens=500)
    assert c.token_ratio == 0.0  # structured path emits no per-kernel LLM tokens


def test_perf_ratio_none_when_missing():
    c = OpComparison(op="matmul", shape=(1, 1, 1))
    assert c.perf_ratio is None
    assert c.token_ratio is None


def test_report_dataclass_defaults():
    r = ComparisonReport()
    assert r.scored_ops == 0
    assert r.passed is False
    assert r.thresholds["perf_geomean_min"] == 1.05
    assert r.thresholds["token_ratio_max"] == 0.70
