# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.advice import build_agent_advice_summary, summarize_status_rows


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

    def test_build_agent_advice_summary_returns_structured_focus(self):
        summary = build_agent_advice_summary(
            [
                {"status": "skipped", "shape_tag": "llama-long-8k"},
                {"status": "oom", "shape_tag": "st4-long-32k"},
                {"status": "ok", "shape_tag": "gpt2-sm-128"},
            ],
            gpu_memory_mb=6143,
        )
        assert summary["counts"]["rows"] == 3
        assert summary["counts"]["skipped"] == 1
        assert summary["counts"]["oom"] == 1
        assert "long-context" in summary["recommended_focus"]
        assert any(item["kind"] == "runtime-oom" for item in summary["advice"])
