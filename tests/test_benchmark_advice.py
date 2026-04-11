# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.advice import summarize_status_rows


class TestBenchmarkAdvice:
    def test_advice_mentions_memory_policy_for_skipped_rows(self):
        advice = summarize_status_rows(
            [
                {"status": "skipped", "shape_tag": "llama-long-8k"},
                {"status": "ok", "shape_tag": "gpt2-sm-128"},
            ],
            gpu_memory_mb=6143,
        )
        assert advice
        assert any(a.kind == "memory-policy" for a in advice)
        assert any(a.kind == "long-context" for a in advice)

    def test_advice_mentions_runtime_oom(self):
        advice = summarize_status_rows(
            [{"status": "oom", "shape_tag": "st4-long-32k"}],
            gpu_memory_mb=6143,
        )
        assert any(a.kind == "runtime-oom" for a in advice)
