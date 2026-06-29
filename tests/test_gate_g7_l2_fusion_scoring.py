"""Verify gate_g7 L2 fusion scoring under D-L2-a (RFC §4).

The L2 fusion path emits 3 rows per (op, shape_tag): separate / liger / arke.
After D-L2-a, the gate must:
  - recognize the `liger` row as the Triton-only denominator,
  - score only the `arke` row against it,
  - skip the `separate` row,
  - count the fusion as evaluable (total > 0).
"""
from benchmarks.gate_g7 import (
    _is_non_arke_baseline,
    _row_is_triton_denominator,
    _build_triton_ref_index,
)


def _l2_row(op, tag, approach, lat, golden="", backend=""):
    return {
        "operator": op, "shape_tag": tag, "approach": approach,
        "latency_us": str(lat), "status": "ok",
        "golden_runner": golden, "backend": backend,
        "baseline": "",
    }


def test_l2_liger_is_triton_denominator():
    liger = _l2_row("silu_and_mul", "gpt2-sm", "liger", 113.1, golden="liger", backend="triton")
    assert _row_is_triton_denominator(liger) is True


def test_l2_separate_and_liger_skipped_as_non_sut():
    sep = _l2_row("silu_and_mul", "gpt2-sm", "separate", 31.6)
    liger = _l2_row("silu_and_mul", "gpt2-sm", "liger", 113.1, golden="liger", backend="triton")
    arke = _l2_row("silu_and_mul", "gpt2-sm", "arke", 55.0, backend="triton")
    assert _is_non_arke_baseline(sep) is True       # eager reference, not SUT
    assert _is_non_arke_baseline(liger) is True      # denominator, not SUT
    assert _is_non_arke_baseline(arke) is False      # the SUT


def test_l2_index_picks_liger_latency():
    rows = [
        ("l2", _l2_row("silu_and_mul", "gpt2-sm", "separate", 31.6)),
        ("l2", _l2_row("silu_and_mul", "gpt2-sm", "liger", 113.1, golden="liger", backend="triton")),
        ("l2", _l2_row("silu_and_mul", "gpt2-sm", "arke", 55.0, backend="triton")),
    ]
    idx = _build_triton_ref_index(rows)
    assert idx.get(("l2", "silu_and_mul", "gpt2-sm")) == 113.1


def test_l2_arke_beats_liger_passes():
    # arke 55.0 <= liger 113.1 * 1.03 -> pass
    rows = [
        ("l2", _l2_row("silu_and_mul", "gpt2-sm", "liger", 113.1, golden="liger", backend="triton")),
        ("l2", _l2_row("silu_and_mul", "gpt2-sm", "arke", 55.0, backend="triton")),
    ]
    idx = _build_triton_ref_index(rows)
    ref = idx[("l2", "silu_and_mul", "gpt2-sm")]
    arke_lat = 55.0
    assert arke_lat <= ref * 1.03  # passes Same-Backend Triton Fairness
