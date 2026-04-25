# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.hardware import HardwareInfo
from benchmarks.memory_policy import (
    attention_preflight,
    dense_matmul_preflight,
    estimate_attention_bytes,
    estimate_dense_matmul_bytes,
    estimate_linear_ce_bytes,
    linear_ce_preflight,
    maybe_attention_preflight,
    maybe_memory_preflight,
)
from benchmarks.shapes import AttentionShape, MatmulShape, Shape2D


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

    def test_dense_matmul_preflight_skips_extreme_output_on_6gb(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, estimate = dense_matmul_preflight(
            hw,
            m=32768,
            n=151936,
            k=7168,
        )
        assert status.status == "skipped"
        assert status.retryable is True
        assert estimate.category == "dense_matmul"
        assert estimate.bytes_required > estimate.bytes_budget

    def test_dense_matmul_estimate_accounts_for_grouped_expert_weights(self):
        single = estimate_dense_matmul_bytes(m=512, n=5120, k=5120, batch=4, weight_copies=1)
        grouped = estimate_dense_matmul_bytes(m=512, n=5120, k=5120, batch=4, weight_copies=64)
        assert grouped > single

    def test_linear_ce_preflight_tracks_logits_memory_pressure(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, estimate = linear_ce_preflight(
            hw,
            tokens=8192,
            hidden=4096,
            vocab=128256,
        )
        assert status.status == "skipped"
        assert estimate.category == "linear_ce"
        assert estimate.bytes_required == estimate_linear_ce_bytes(
            tokens=8192,
            hidden=4096,
            vocab=128256,
        )

    def test_maybe_memory_preflight_handles_dense_family(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = MatmulShape(tag="too-wide", M=32768, N=151936, K=7168)
        preflight = maybe_memory_preflight(hw, "matmul", shape)
        assert preflight is not None
        assert preflight.status.status == "skipped"
        assert preflight.estimate.category == "dense_matmul"

    def test_maybe_memory_preflight_handles_linear_ce_family(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = Shape2D(tag="llama3-seq8k", M=8192, N=4096)
        preflight = maybe_memory_preflight(hw, "linear_ce", shape)
        assert preflight is not None
        assert preflight.status.status == "skipped"
        assert preflight.estimate.category == "linear_ce"

    def test_maybe_attention_preflight_handles_attention_family(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="st4-long-32k", B=4, H=32, S=32768, D=128)
        status = maybe_attention_preflight(hw, "paged_attention", shape)
        assert status is not None
        assert status.status == "skipped"
