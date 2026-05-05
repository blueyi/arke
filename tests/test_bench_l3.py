# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Stage 8 L3 benchmark artifact contract."""

from __future__ import annotations

import json

from benchmarks.bench_l3 import _load_mock_model, build_summary, run_l3, run_l3_single


def test_run_l3_single_mock_contract_cpu():
    results = run_l3_single(
        8,
        model_name="mock-gpt2",
        model_loader=_load_mock_model,
        device="cpu",
        dtype="float32",
        warmup=1,
        runs=2,
    )

    assert [row.mode for row in results] == ["eager", "torch.compile"]
    eager, compiled = results
    assert eager.status == "ok"
    assert eager.correct is True
    assert eager.top1_match is True
    assert compiled.status == "ok"
    assert compiled.correct is True
    assert compiled.top1_match is True
    assert compiled.ratio_vs_eager is not None


def test_build_summary_tracks_g8_gpt2_threshold():
    results = run_l3_single(
        8,
        model_name="mock-gpt2",
        model_loader=_load_mock_model,
        device="cpu",
        dtype="float32",
        warmup=1,
        runs=2,
    )
    summary = build_summary(results, target_ratio=0.0)

    assert summary["schema"] == "stage8-l3-gpt2-v1"
    assert summary["rows"] == 2
    assert summary["eager_rows"] == 1
    assert summary["compile_rows"] == 1
    assert summary["compile_success_rows"] == 1
    assert summary["g8_gpt2_pass"] is True


def test_run_l3_writes_structured_artifacts(tmp_path):
    results = run_l3(
        [8],
        output_dir=str(tmp_path),
        device="cpu",
        dtype="float32",
        warmup=1,
        runs=2,
        mock=True,
    )

    assert len(results) == 2
    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in [
        "config.json",
        "hardware.json",
        "sources.json",
        "gpt2_results.csv",
        "results.json",
        "summary.json",
    ]:
        assert (run_dir / name).exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["rows"] == 2
    assert summary["compile_rows"] == 1
