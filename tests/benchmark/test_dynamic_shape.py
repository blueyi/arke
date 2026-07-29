# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dynamic-shape (Performance Cliff) benchmark track.

Covers the CPU-safe surface: sweep definitions, specialization-key math,
summary statistics, CSV output, and the frozen-layer contract (no gate
threshold anywhere in the module). The GPU sweep itself is exercised by a
single smoke test that skips without CUDA + Triton.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from benchmarks.dynamic_shape import (
    DEFAULT_OPS,
    SHAPE_SWEEPS,
    CliffRow,
    _div_class,
    _next_pow2,
    _spec_key,
    cliff_ratio,
    geomean,
    summarize,
    write_csv,
)


# ── bucket / specialization math ────────────────────────────────────────────


@pytest.mark.parametrize(
    "x,expected",
    [(1, 1), (2, 2), (7, 7), (16, 16), (17, 32), (100, 128),
     (128, 128), (200, 256), (513, 1024), (4096, 4096)],
)
def test_next_pow2(x: int, expected: int) -> None:
    assert _next_pow2(x) == expected


def test_next_pow2_matches_template() -> None:
    """The local mirror must agree with the K-H3.1 template helper."""
    template = (
        Path(__file__).resolve().parents[2]
        / "arke" / "backend" / "triton_templates" / "matmul.py.j2"
    ).read_text(encoding="utf-8")
    assert "def _next_pow2(" in template
    # Spot-check the semantic contract on the shared domain.
    for x in (1, 5, 16, 17, 100, 513, 2048, 5000):
        b = _next_pow2(x)
        assert b >= x
        assert b <= 16 or (b & (b - 1)) == 0  # pow2 above the small-linear zone


def test_div_class() -> None:
    assert _div_class(1) == "1"
    assert _div_class(16) == "d16"
    assert _div_class(4096) == "d16"
    assert _div_class(7) == "d1"
    assert _div_class(100) == "d1"


def test_spec_key_matmul_bucket_semantics() -> None:
    """matmul: shapes in the same pow2 bucket with the same divisibility
    classes share a spec; crossing a bucket boundary changes it."""
    # 48 and 64: same bucket (64) but 48 % 16 == 0 too → same div class → share.
    assert _spec_key("matmul", 48, 4096, 4096) == _spec_key("matmul", 64, 4096, 4096)
    # 100 vs 128: same bucket (128) but 100 is d1, 128 is d16 → differ.
    assert _spec_key("matmul", 100, 4096, 4096) != _spec_key("matmul", 128, 4096, 4096)
    # 128 vs 256: different bucket → differ.
    assert _spec_key("matmul", 128, 4096, 4096) != _spec_key("matmul", 256, 4096, 4096)


def test_spec_key_softmax_exact_vs_bucketed() -> None:
    """softmax: BLOCK derives from next_pow2(N); div classes from exact N."""
    # 256 vs 512: different BLOCK → differ.
    assert _spec_key("softmax", 32, 256, 0) != _spec_key("softmax", 32, 512, 0)
    # 384 and 512 share BLOCK=512 and are both %16==0 → predicted shared.
    assert _spec_key("softmax", 32, 384, 0) == _spec_key("softmax", 32, 512, 0)
    # 700 (d1) vs 704 (d16): same BLOCK=1024, different div class → differ.
    assert _spec_key("softmax", 32, 700, 0) != _spec_key("softmax", 32, 704, 0)


def test_spec_key_rmsnorm_exact_n() -> None:
    """rmsnorm keys on exact N; varying M only changes its div class."""
    assert _spec_key("rmsnorm", 256, 4096, 0) == _spec_key("rmsnorm", 512, 4096, 0)
    assert _spec_key("rmsnorm", 200, 4096, 0) != _spec_key("rmsnorm", 256, 4096, 0)


# ── sweep definitions ───────────────────────────────────────────────────────


def test_default_ops_have_sweeps() -> None:
    for op in DEFAULT_OPS:
        assert op in SHAPE_SWEEPS
        sweep = SHAPE_SWEEPS[op]()
        assert len(sweep) >= 8, f"{op}: sweep too short to expose a curve"


def test_sweeps_cross_spec_boundaries() -> None:
    """Each sweep must contain BOTH shared-spec and new-spec transitions,
    otherwise the cliff/warm contrast cannot be measured."""
    for op, sweep_fn in SHAPE_SWEEPS.items():
        seen: set[str] = set()
        new_flags = []
        for _, M, N, K in sweep_fn():
            key = _spec_key(op, M, N, K)
            new_flags.append(key not in seen)
            seen.add(key)
        assert any(new_flags[1:]), f"{op}: no new-spec transition after first"
        assert not all(new_flags), f"{op}: no shared-spec repeat"


def test_sweeps_use_non_pow2_sizes() -> None:
    """Real dynamic workloads are unaligned; the sweep must include
    non-power-of-two sizes in the varying dim."""
    from benchmarks.dynamic_shape import _VARYING_DIM

    for op, sweep_fn in SHAPE_SWEEPS.items():
        idx = {"M": 1, "N": 2, "K": 3}[_VARYING_DIM[op]]
        vals = [row[idx] for row in sweep_fn()]
        assert any(v > 16 and (v & (v - 1)) != 0 for v in vals), (
            f"{op}: sweep only has power-of-two sizes"
        )


# ── statistics ──────────────────────────────────────────────────────────────


def test_cliff_ratio_basic() -> None:
    assert cliff_ratio(10.0, 2.0) == pytest.approx(5.0)
    assert math.isinf(cliff_ratio(1.0, 0.0))


def test_geomean() -> None:
    assert geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert geomean([2.0, float("nan"), 8.0]) == pytest.approx(4.0)
    assert math.isnan(geomean([]))


def _row(tag: str, cold: float, warm: float, new_spec: bool) -> CliffRow:
    return CliffRow("matmul", tag, 128, 4096, 4096, "cfg128", new_spec,
                    cold, warm, cliff_ratio(cold, warm))


def test_summarize_splits_spec_classes() -> None:
    rows = [
        _row("a", 100.0, 1.0, True),    # cliff 100×
        _row("b", 1.2, 1.0, False),     # warm 1.2×
        _row("c", 50.0, 1.0, True),     # cliff 50×
        _row("d", 1.0, 1.0, False),     # warm 1.0×
    ]
    s = summarize(rows)
    assert s["n_ok"] == 4
    assert s["n_new_spec"] == 2
    assert s["n_same_spec"] == 2
    assert s["new_spec_geomean"] == pytest.approx(math.sqrt(100 * 50))
    assert s["same_spec_geomean"] == pytest.approx(math.sqrt(1.2))
    assert s["cliff_ratio_max"] == pytest.approx(100.0)


def test_summarize_excludes_failed_rows() -> None:
    bad = CliffRow("matmul", "x", 1, 1, 1, "k", True, 0.0, 0.0, float("nan"),
                   status="oom", reason="boom")
    s = summarize([_row("a", 2.0, 1.0, True), bad])
    assert s["n_shapes"] == 2
    assert s["n_ok"] == 1
    assert s["cliff_ratio_geomean"] == pytest.approx(2.0)


# ── frozen-layer contract ───────────────────────────────────────────────────


def test_no_gate_threshold_in_module() -> None:
    """The track must NOT bake in a pass/fail threshold — gate scoring
    semantics are a frozen-layer decision. Guard against accidental drift."""
    src = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "dynamic_shape.py"
    ).read_text(encoding="utf-8")
    for token in ("PASS_THRESHOLD", "GATE_THRESHOLD", "CLIFF_LIMIT",
                  "def gate(", "passed ="):
        assert token not in src, f"frozen-layer leak: {token!r} in dynamic_shape.py"


# ── CSV output ──────────────────────────────────────────────────────────────


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    rows = [_row("a", 10.0, 2.0, True), _row("b", 2.0, 2.0, False)]
    out = tmp_path / "sub" / "matmul_cliff.csv"
    write_csv(rows, out)
    with out.open(newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == 2
    assert back[0]["shape_tag"] == "a"
    assert float(back[0]["cliff_ratio"]) == pytest.approx(5.0)
    assert back[1]["new_spec"] == "False"


# ── GPU smoke ───────────────────────────────────────────────────────────────


def _cuda_and_triton_available() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_and_triton_available(),
                    reason="requires CUDA + Triton")
def test_run_sweep_smoke_softmax() -> None:
    """Tiny end-to-end run through the production wrapper (2 shapes)."""
    import benchmarks.dynamic_shape as ds

    orig = ds.SHAPE_SWEEPS["softmax"]
    ds.SHAPE_SWEEPS["softmax"] = lambda: [("n128", 8, 128, 0), ("n130", 8, 130, 0)]
    try:
        rows = ds.run_sweep("softmax", warm_reps=5)
    finally:
        ds.SHAPE_SWEEPS["softmax"] = orig

    assert len(rows) == 2
    for r in rows:
        assert r.status == "ok", r.reason
        assert r.first_call_ms > 0
        assert r.steady_ms > 0
        assert math.isfinite(r.cliff_ratio)
    # n128 vs n130: BLOCK differs (128 vs 256) → both predicted new specs.
    assert rows[0].new_spec and rows[1].new_spec
