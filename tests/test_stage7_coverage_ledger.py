# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.stage7_coverage_ledger import build_stage7_coverage_ledger


def test_build_stage7_coverage_ledger_links_matrix_examples_and_artifacts(tmp_path: Path):
    matrix = {
        "stage": "S7",
        "gate": "G7",
        "l1": [
            {
                "op": "relu",
                "ot_tier": 0,
                "layer": "l1",
                "shape_count_required": 2,
                "shape_tags_required": ["square-1k", "gpt2-hidden"],
            }
        ],
        "l2": [
            {
                "op": "matmul_gelu",
                "ot_tier": 3,
                "layer": "l2",
                "shape_count_required": 1,
                "shape_tags_required": ["gpt2-ffn"],
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))

    examples_root = tmp_path / "examples"
    (examples_root / "operators").mkdir(parents=True)
    (examples_root / "operators" / "00_relu.ak").write_text(
        "kernel relu(X: Tensor<[M], f16>) -> Tensor<[M], f16> where M: dynamic(max=4096) { let Y = relu(X=X); return Y; }"
    )
    (examples_root / "operators" / "05_matmul_gelu.ak").write_text(
        "kernel matmul_gelu(X: Tensor<[M, K], f16>, W: Tensor<[K, N], f16>) -> Tensor<[M, N], f16> where M: dynamic(max=4096), K: static, N: dynamic(max=4096) { let Z = matmul(A=X, B=W); let Y = gelu(X=Z); return Y; }\nstrategy matmul_gelu_strategy for target(\"nvidia_ampere\") { fuse(ops=[\"matmul\", \"gelu\"], fusion_type=\"epilogue\") @rationale(\"fused\"); }"
    )

    results_root = tmp_path / "results"
    (results_root / "l1").mkdir(parents=True)
    (results_root / "l2").mkdir(parents=True)
    (results_root / "l1" / "PERF_ALL.csv").write_text(
        "operator,shape_tag,correctness_status,perf_target,perf_actual,perf_pass,perf_gap\n"
        "relu,square-1k,pass,1.0,1.1,true,0.1\n"
    )
    (results_root / "l2" / "PERF_ALL.csv").write_text(
        "operator,shape_tag,correctness_status,perf_target,perf_actual,perf_pass,perf_gap\n"
        "matmul_gelu,gpt2-ffn,pass,1.0,1.2,true,0.2\n"
    )

    report = build_stage7_coverage_ledger(
        matrix_path=matrix_path,
        examples_root=examples_root,
        results_root=results_root,
    )

    assert report["summary"]["l1"]["entries"] == 1
    assert report["summary"]["l2"]["entries"] == 1
    assert report["summary"]["l1"]["with_examples"] == 1
    assert report["summary"]["l2"]["with_strategy_examples"] == 1

    relu = report["l1"][0]
    assert relu["example"]["found"] is True
    assert relu["example"]["relative_path"].endswith("00_relu.ak")
    assert relu["pipeline"]["semantic_ok"] is True
    assert relu["pipeline"]["strategy_ok"] is False
    assert relu["evidence"]["observed_shape_tags"] == ["square-1k"]
    assert relu["evidence"]["correctness_evidence_present"] is True
    assert relu["evidence"]["performance_evidence_present"] is True

    matmul_gelu = report["l2"][0]
    assert matmul_gelu["example"]["found"] is True
    assert matmul_gelu["pipeline"]["semantic_ok"] is True
    assert matmul_gelu["pipeline"]["strategy_ok"] is True
    assert matmul_gelu["pipeline"]["has_fusion_decision"] is True
    assert matmul_gelu["evidence"]["observed_shape_tags"] == ["gpt2-ffn"]
    assert matmul_gelu["evidence"]["missing_shape_tags"] == []