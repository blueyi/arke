# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage7_coverage_ledger_cli_writes_output(tmp_path: Path):
    matrix = {
        "stage": "S7",
        "gate": "G7",
        "l1": [
            {
                "op": "relu",
                "ot_tier": 0,
                "layer": "l1",
                "shape_count_required": 1,
                "shape_tags_required": ["square-1k"],
            }
        ],
        "l2": [],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))

    examples_root = tmp_path / "examples"
    (examples_root / "operators").mkdir(parents=True)
    (examples_root / "operators" / "00_relu.ak").write_text(
        "kernel relu(X: Tensor<[M], f16>) -> Tensor<[M], f16> where M: dynamic(max=4096) { let Y = relu(X=X); return Y; }"
    )

    results_root = tmp_path / "results"
    (results_root / "l1").mkdir(parents=True)
    (results_root / "l1" / "PERF_ALL.csv").write_text(
        "operator,shape_tag,correctness_status,perf_target,perf_actual,perf_pass,perf_gap\n"
        "relu,square-1k,pass,1.0,1.1,true,0.1\n"
    )

    output_path = tmp_path / "ledger.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.stage7_coverage_ledger",
            "--matrix",
            str(matrix_path),
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(results_root),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert proc.returncode == 0, proc.stderr
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["summary"]["l1"]["with_examples"] == 1
    assert "coverage ledger" in proc.stdout.lower()