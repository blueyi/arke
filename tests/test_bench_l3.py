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


# --- Regression tests for the --modes alias normalization fix (2026-05-17) ---


def test_normalize_mode_accepts_canonical_names():
    """Canonical names round-trip unchanged."""
    from benchmarks.bench_l3 import _normalize_mode, MODE_EAGER, MODE_TORCH_COMPILE

    assert _normalize_mode("eager") == MODE_EAGER
    assert _normalize_mode("torch.compile") == MODE_TORCH_COMPILE


def test_normalize_mode_accepts_underscore_alias():
    """`torch_compile` (shell-friendly spelling) maps to canonical `torch.compile`.

    This is the exact alias that caused the 2026-05-17 bug: a CLI user passed
    `--modes eager,torch_compile`, the CSV row was written with
    `mode='torch_compile'`, but `build_summary` filtered on `'torch.compile'`,
    so `compile_rows=0` in summary.json despite a successful compile run.
    """
    from benchmarks.bench_l3 import _normalize_mode, MODE_TORCH_COMPILE

    assert _normalize_mode("torch_compile") == MODE_TORCH_COMPILE
    assert _normalize_mode("torch-compile") == MODE_TORCH_COMPILE
    assert _normalize_mode("torchcompile") == MODE_TORCH_COMPILE
    # Case-insensitive
    assert _normalize_mode("Torch_Compile") == MODE_TORCH_COMPILE
    # Whitespace tolerated
    assert _normalize_mode("  torch_compile  ") == MODE_TORCH_COMPILE


def test_normalize_mode_rejects_unknown():
    """Unknown modes fail loud — better than silently empty filter results."""
    import pytest
    from benchmarks.bench_l3 import _normalize_mode

    with pytest.raises(ValueError, match="Unknown bench_l3 mode"):
        _normalize_mode("eager-mode")  # not a registered alias
    with pytest.raises(ValueError, match="Unknown bench_l3 mode"):
        _normalize_mode("inductor")


def test_normalize_modes_handles_list():
    from benchmarks.bench_l3 import _normalize_modes, MODE_EAGER, MODE_TORCH_COMPILE

    assert _normalize_modes(["eager", "torch_compile"]) == [
        MODE_EAGER,
        MODE_TORCH_COMPILE,
    ]


def test_build_summary_uses_canonical_mode_constants():
    """End-to-end: ensure compile_rows is populated regardless of which alias
    the user originally passed to the CLI.

    Constructs E2EResult rows directly with the canonical mode names (matching
    what _normalize_modes produces after CLI parsing) and verifies the summary
    filter catches them.
    """
    from benchmarks.bench_l3 import (
        E2EResult,
        MODE_EAGER,
        MODE_TORCH_COMPILE,
        build_summary,
    )

    results = [
        E2EResult(
            model="gpt2", seq_len=128, mode=MODE_EAGER, source="eager-src",
            status="ok", mean_ms=10.0, correct=True, top1_match=True,
            ratio_vs_eager=1.0,
        ),
        E2EResult(
            model="gpt2", seq_len=128, mode=MODE_TORCH_COMPILE, source="tc-src",
            status="ok", mean_ms=9.0, correct=True, top1_match=True,
            ratio_vs_eager=10.0 / 9.0,
        ),
    ]
    summary = build_summary(results, target_ratio=0.95)
    assert summary["eager_rows"] == 1
    assert summary["compile_rows"] == 1, (
        "compile_rows must equal the number of MODE_TORCH_COMPILE rows; "
        f"got {summary['compile_rows']}"
    )
    assert summary["compile_success_rows"] == 1
    assert summary["g8_gpt2_pass"] is True
