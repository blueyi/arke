# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.agent.tools import BenchmarkAdviceSummaryTool


class TestAgentBenchmarkAdviceTool:
    def test_benchmark_advice_summary_tool_reads_perf_csv(self, tmp_path: Path):
        csv_path = tmp_path / "PERF_ALL.csv"
        csv_path.write_text(
            "operator,shape_tag,status\n"
            "flash_attention,llama-long-8k,skipped\n"
            "flash_attention,st4-long-32k,oom\n"
            "relu,square-1k,ok\n"
        )
        tool = BenchmarkAdviceSummaryTool()
        result = tool.execute({"csv_path": str(csv_path), "gpu_memory_mb": 6143})
        assert result.success is True
        assert result.data["counts"]["rows"] == 3
        assert "long-context" in result.data["recommended_focus"]
