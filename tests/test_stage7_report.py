# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from benchmarks.stage7_report import collect_stage7_operator_shape_stats


class TestStage7Report:
    def test_collect_stage7_operator_shape_stats_groups_by_layer_op_and_shape(self, tmp_path: Path):
        l1 = tmp_path / "l1"
        l1.mkdir(parents=True)
        (l1 / "PERF_ALL.csv").write_text(
            "operator,shape_tag,status\n"
            "relu,square-1k,ok\n"
            "relu,square-2k,ok\n"
            "flash_attention,llama-long-8k,skipped\n"
        )
        l2 = tmp_path / "l2"
        l2.mkdir(parents=True)
        (l2 / "PERF_ALL.csv").write_text(
            "operator,shape_tag,status\n"
            "matmul_relu,square-1k,ok\n"
            "qkv_fa,st4-long-32k,skipped\n"
        )
        report = collect_stage7_operator_shape_stats(tmp_path)
        assert report["l1"]["relu"]["shape_count"] == 2
        assert report["l1"]["flash_attention"]["status_counts"]["skipped"] == 1
        assert report["l2"]["qkv_fa"]["shape_tags"] == ["st4-long-32k"]
