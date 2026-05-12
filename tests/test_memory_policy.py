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

    # ── a9: baseline-aware attention peak (Q1) ──────────────────

    def test_attention_estimate_drops_score_buffer_when_fused(self):
        """Fused kernels (flash, liger, flaggems) never materialize [B,H,S,S]."""
        bh_s_d = dict(batch=2, heads=16, seq=8192, head_dim=128)
        materialized = estimate_attention_bytes(**bh_s_d, materialize_scores=True)
        fused = estimate_attention_bytes(**bh_s_d, materialize_scores=False)
        # At S=8192, score buffer dominates: 2*16*8192*8192*2 = ~4.3 GB
        # whereas QKV+output is only 2*16*8192*128*2 * 4 = ~268 MB.
        assert materialized > fused * 10
        # Sanity: fused estimate is just QKV (×3) + output
        expected_fused = 2 * 16 * 8192 * 128 * 2 * 3 + 2 * 16 * 8192 * 128 * 2
        assert fused == expected_fused

    def test_maybe_memory_preflight_uses_baseline_for_attention(self):
        """A shape that would be skipped under torch-eager passes under fused."""
        hw = HardwareInfo(gpu_memory_mb=6143)
        # B=2 H=16 S=4096 D=128: scores = 2*16*4096*4096*2 = 1 GB, QKV+out = 96 MB.
        # Budget @ 0.55 ratio = ~3.4 GB; total materialized ~1.1 GB (passes),
        # but bump to S=8192 and materialized hits ~4.4 GB (>budget).
        shape = AttentionShape(tag="oom-edge", B=2, H=16, S=8192, D=128)

        eager = maybe_memory_preflight(hw, "flash_attention", shape, baseline="PyTorch-eager")
        fused = maybe_memory_preflight(hw, "flash_attention", shape, baseline="Liger-Kernel")
        assert eager is not None and fused is not None
        assert eager.status.status == "skipped"
        assert fused.status.status == "ok"
        assert eager.estimate.bytes_required > fused.estimate.bytes_required * 10
        assert fused.estimate.category == "attention_fused"

    def test_score_materialization_baseline_set_case_insensitive(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="t", B=2, H=16, S=8192, D=128)
        for name in ["PyTorch-eager", "pytorch-eager", "_torch_reference"]:
            pf = maybe_memory_preflight(hw, "flash_attention", shape, baseline=name)
            assert pf is not None
            assert pf.estimate.category == "attention", f"{name!r} should be materialized"

    def test_no_baseline_defaults_conservative(self):
        """Backward-compat: callers that don't pass baseline get the old behavior."""
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = AttentionShape(tag="t", B=2, H=16, S=8192, D=128)
        pf = maybe_memory_preflight(hw, "flash_attention", shape)
        assert pf is not None
        assert pf.estimate.category == "attention"  # materialized = default

    # ── a9: topk/argmax/cumsum N cap (Q3) ────────────────────────

    def test_topk_preflight_skips_giant_n(self):
        """Large [M,N] topk with fp32 sort workspace exceeds 6 GB budget."""
        from benchmarks.memory_policy import estimate_topk_like_bytes, topk_like_preflight
        hw = HardwareInfo(gpu_memory_mb=6143)
        # M=4096, N=524288 fp16: input 4 GB, workspace 8 GB → 12 GB total
        status, estimate = topk_like_preflight(hw, m=4096, n=524288)
        assert status.status == "skipped"
        assert status.retryable is True
        assert estimate.category == "topk_like"
        assert estimate.bytes_required == estimate_topk_like_bytes(m=4096, n=524288)

    def test_topk_preflight_allows_normal_shape(self):
        from benchmarks.memory_policy import topk_like_preflight
        hw = HardwareInfo(gpu_memory_mb=6143)
        status, estimate = topk_like_preflight(hw, m=1024, n=4096)
        assert status.status == "ok"
        assert estimate.bytes_required < estimate.bytes_budget

    def test_maybe_memory_preflight_handles_topk_family(self):
        hw = HardwareInfo(gpu_memory_mb=6143)
        shape = Shape2D(tag="oom-topk", M=4096, N=524288)
        for op in ("topk", "argmax", "cumsum"):
            pf = maybe_memory_preflight(hw, op, shape)
            assert pf is not None, f"op={op}"
            assert pf.status.status == "skipped"
            assert pf.estimate.category == "topk_like"
