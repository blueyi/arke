# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import benchmarks.gate_g7 as gate_g7
from benchmarks.gate_g7 import check_stage7_track6_artifacts, run_g7


REQUIRED_ROOT_ARTIFACTS = (
    "coverage_gap.json",
    "audit_report.json",
    "stage7_operator_shape_stats.json",
    "dashboard.json",
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _build_track6_tree(root: Path) -> None:
    for layer in ("l1", "l2"):
        layer_dir = root / layer
        _touch(layer_dir / "config.json")
        _touch(layer_dir / "hardware.json")
        _touch(layer_dir / "sources.json")
        _touch(layer_dir / "summary.json")
        _touch(layer_dir / "PERF_ALL.csv")
    for artifact in REQUIRED_ROOT_ARTIFACTS:
        _touch(root / artifact)


def test_check_stage7_track6_artifacts_accepts_complete_tree(tmp_path: Path):
    _build_track6_tree(tmp_path)

    ok, detail = check_stage7_track6_artifacts(tmp_path)

    assert ok is True
    assert "dashboard.json" in detail
    assert "coverage_gap.json" in detail


def test_check_stage7_track6_artifacts_requires_root_dashboard_artifacts(tmp_path: Path):
    _build_track6_tree(tmp_path)
    (tmp_path / "dashboard.json").unlink()

    ok, detail = check_stage7_track6_artifacts(tmp_path)

    assert ok is False
    assert "dashboard.json" in detail


def test_check_bl5_coverage_evidence_accepts_complete_synthetic_report(monkeypatch):
    coverage = {
        "stage": "S7",
        "gate": "G7",
        "l1": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 1,
            "shapes_required_total": 1,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 1.0,
            "per_op": [
                {
                    "op": "relu",
                    "missing_shape_tags": [],
                    "missing_perf_target_fields": [],
                }
            ],
        },
        "l2": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 1,
            "shapes_required_total": 1,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 1.0,
            "per_op": [
                {
                    "op": "matmul_relu",
                    "missing_shape_tags": [],
                    "missing_perf_target_fields": [],
                }
            ],
        },
    }
    audit = {
        "summary": {
            "l1": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
            "l2": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
        }
    }

    def fake_load_json_artifact(path: Path):
        if path.name == "coverage_gap.json":
            return coverage, None
        if path.name == "audit_report.json":
            return audit, None
        return None, f"unexpected artifact: {path}"

    monkeypatch.setattr(gate_g7, "_load_json_artifact", fake_load_json_artifact)

    ok, detail = gate_g7._check_bl5_coverage_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "complete" in detail


def test_check_bl5_coverage_evidence_rejects_gap(monkeypatch):
    coverage = {
        "stage": "S7",
        "gate": "G7",
        "l1": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 0,
            "shapes_required_total": 2,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 0.5,
            "per_op": [
                {
                    "op": "relu",
                    "missing_shape_tags": ["shape-b"],
                    "missing_perf_target_fields": [],
                }
            ],
        },
        "l2": {
            "ops_total": 0,
            "ops_with_any_evidence": 0,
            "ops_fully_covered": 0,
            "shapes_required_total": 0,
            "shapes_observed_total": 0,
            "shape_coverage_ratio": 0.0,
            "per_op": [],
        },
    }
    audit = {
        "summary": {
            "l1": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": ["relu"],
                "unsupported_surface_cases": [],
            },
            "l2": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
        }
    }

    def fake_load_json_artifact(path: Path):
        if path.name == "coverage_gap.json":
            return coverage, None
        if path.name == "audit_report.json":
            return audit, None
        return None, f"unexpected artifact: {path}"

    monkeypatch.setattr(gate_g7, "_load_json_artifact", fake_load_json_artifact)

    ok, detail = gate_g7._check_bl5_coverage_evidence(Path("/tmp/unused"))

    assert ok is False
    assert "full-shape coverage" in detail or "missing_full_shape_evidence" in detail


def test_check_bl5_correctness_evidence_respects_memory_exclusions(monkeypatch):
    rows = [
        (
            "l1",
            {
                "operator": "relu",
                "shape_tag": "square-1k",
                "status": "ok",
                "correctness_status": "ok",
                "allclose": "true",
                "memory_policy": "",
                "reason": "",
                "correctness_reason": "",
            },
        ),
        (
            "l1",
            {
                "operator": "flash_attention",
                "shape_tag": "tiny",
                "status": "skipped",
                "correctness_status": "skipped",
                "allclose": "",
                "memory_policy": "attention",
                "memory_bytes_required": "2048",
                "memory_bytes_budget": "1024",
                "memory_ratio": "2.0",
                "reason": "OOM preflight",
                "correctness_reason": "OOM preflight",
            },
        ),
        (
            "l2",
            {
                "operator": "matmul_relu",
                "shape_tag": "gpt2-sm",
                "status": "ok",
                "correctness_status": "ok",
                "allclose": "true",
                "memory_policy": "",
                "reason": "",
                "correctness_reason": "",
            },
        ),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))

    ok, detail = gate_g7._check_bl5_correctness_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "memory_excluded=1" in detail


def test_check_bl5_performance_evidence_enforces_group_targets(monkeypatch):
    rows = [
        ("l1", {"operator": "relu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "gelu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "matmul", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "softmax", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "rope", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "flash_attention", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_relu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_gelu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "swiglu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "geglu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "linear_ce", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "qkv_fa", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))
    monkeypatch.setattr(
        gate_g7,
        "_load_l1_ot_map",
        lambda _matrix: {
            "relu": 0,
            "gelu": 0,
            "matmul": 2,
            "softmax": 1,
            "rope": 3,
            "flash_attention": 4,
        },
    )

    ok, detail = gate_g7._check_bl5_performance_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "weighted_score=1.0000" in detail
    assert "L2 fusions=6" in detail


def test_check_bl5_performance_evidence_rejects_gap(monkeypatch):
    rows = [
        ("l1", {"operator": "relu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "false", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_relu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "false", "memory_policy": "", "reason": "", "correctness_reason": ""}),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))
    monkeypatch.setattr(gate_g7, "_load_l1_ot_map", lambda _matrix: {"relu": 0})

    ok, detail = gate_g7._check_bl5_performance_evidence(Path("/tmp/unused"))

    assert ok is False
    assert "weighted_score" in detail


def test_run_g7_returns_standard_gate_summary(monkeypatch):
    """Guard the G7 runner against drifting from benchmarks.gate.GateSummary."""
    import benchmarks.gate_g7 as gate_g7

    monkeypatch.setattr(gate_g7, "_check_spec_docs", lambda: (True, "docs ok"))
    monkeypatch.setattr(gate_g7, "_run_pytest", lambda _args: (True, "tests ok"))
    monkeypatch.setattr(gate_g7, "_check_all_examples_compile", lambda: (True, "examples ok"))
    monkeypatch.setattr(gate_g7, "_check_benchmark_artifacts", lambda: (True, "artifacts ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_coverage_evidence", lambda: (True, "coverage ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_correctness_evidence", lambda: (True, "correctness ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_performance_evidence", lambda: (True, "performance ok"))

    summary = run_g7(tier=1)

    assert summary.gate == "G7"
    assert summary.tier == 1
    assert summary.total == len(summary.results)
    assert summary.passed == summary.total
    assert summary.failed == 0
    assert summary.to_dict()["pass_rate"] == "100.0%"
    criteria = {result.criterion for result in summary.results}
    assert {"G7.8b", "G7.8c", "G7.8d"}.issubset(criteria)


def _write_perf_all(path: Path, rows: list[dict[str, str]]) -> None:
    import csv
    fieldnames = [
        "operator",
        "shape_tag",
        "baseline",
        "status",
        "correctness_status",
        "allclose",
        "perf_pass",
        "memory_policy",
        "memory_ratio",
        "memory_bytes_required",
        "memory_bytes_budget",
        "reason",
        "correctness_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def test_correctness_evidence_exempts_golden_unavailable(tmp_path: Path):
    """Per Golden Kernel protocol: golden_unavailable_pending_baseline rows are
    audit-only gaps, not correctness failures. Gate must NOT count them as fails.
    """
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # Two ok rows.
            {"operator": "relu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            {"operator": "gelu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            # Two golden-unavailable rows (e.g. MLA / paged_attention). These must
            # be EXEMPTED from correctness fail counting per protocol.
            {"operator": "multi_latent_attention", "shape_tag": "ds-v2-mla-512",
             "status": "ok", "correctness_status": "golden_unavailable_pending_baseline",
             "allclose": "", "perf_pass": "true"},
            {"operator": "paged_attention", "shape_tag": "paged-512",
             "status": "ok", "correctness_status": "golden_unavailable_pending_baseline",
             "allclose": "", "perf_pass": "true"},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is True, f"correctness should pass with only golden-unavailable rows: {detail}"
    assert "golden_exempted=2" in detail
    assert "checked=2" in detail


def test_correctness_evidence_still_fails_on_real_regressions(tmp_path: Path):
    """Exemption must be surgical: actual correctness regressions still fail."""
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # One real failure (allclose=false on an ok run).
            {"operator": "matmul", "shape_tag": "square-1k", "status": "ok",
             "correctness_status": "ok", "allclose": "false", "perf_pass": "false"},
            # One exempt row (should not mask the real failure above).
            {"operator": "multi_latent_attention", "shape_tag": "ds-v2-mla-512",
             "status": "ok", "correctness_status": "golden_unavailable_pending_baseline",
             "allclose": "", "perf_pass": "true"},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is False
    assert "allclose=false" in detail
    assert "golden_exempted=1" in detail


def test_is_golden_unavailable_helper():
    assert gate_g7._is_golden_unavailable(
        {"correctness_status": "golden_unavailable_pending_baseline"}
    )
    assert gate_g7._is_golden_unavailable(
        {"correctness_status": "GOLDEN_UNAVAILABLE_PENDING_BASELINE"}
    )
    assert not gate_g7._is_golden_unavailable({"correctness_status": "ok"})
    assert not gate_g7._is_golden_unavailable({"correctness_status": ""})
    assert not gate_g7._is_golden_unavailable({"correctness_status": "error"})


def test_is_typed_unsupported_helper():
    """Typed-decline reasons are exempted; untyped/empty `unsupported` is not."""
    # Recognised typed-decline reason templates.
    typed_cases = [
        "Liger-Kernel.get_fn declined rope@non-align-1",
        "runner Liger-Kernel does not implement run_with_inputs for rope",
        "RoPE requires even head_dim; got 65 (odd) for shape (13, 127, 65)",
        "mathematically ill-defined for this shape",
        "No correctness probe for fused op: linear_ce",
    ]
    for reason in typed_cases:
        assert gate_g7._is_typed_unsupported(
            {"correctness_status": "unsupported", "correctness_reason": reason}
        ), f"should be typed: {reason}"

    # Negative cases — must NOT be exempted.
    assert not gate_g7._is_typed_unsupported(
        {"correctness_status": "unsupported", "correctness_reason": ""}
    ), "empty reason — silent decline not allowed"
    assert not gate_g7._is_typed_unsupported(
        {"correctness_status": "unsupported", "correctness_reason": "something broke"}
    ), "free-form reason — silent decline not allowed"
    assert not gate_g7._is_typed_unsupported(
        {"correctness_status": "ok", "correctness_reason": "RoPE requires even head_dim"}
    ), "status must be unsupported, not just any status"
    assert not gate_g7._is_typed_unsupported(
        {"correctness_status": "error", "correctness_reason": "RoPE requires even head_dim"}
    ), "errors are never typed unsupported"
    assert not gate_g7._is_typed_unsupported({"correctness_status": ""})


def test_correctness_evidence_exempts_typed_unsupported(tmp_path: Path):
    """Typed-decline unsupported rows are audit-only, NOT correctness failures."""
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # Two real ok rows.
            {"operator": "relu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            {"operator": "gelu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            # Runner-side typed decline (Liger doesn't implement rope probe).
            {"operator": "rope", "shape_tag": "gpt2-sm-128", "status": "ok",
             "correctness_status": "unsupported",
             "correctness_reason": "runner Liger-Kernel does not implement run_with_inputs for rope",
             "allclose": "", "perf_pass": "true"},
            # Op math guard (odd head_dim).
            {"operator": "rope", "shape_tag": "non-align-1", "status": "unsupported",
             "correctness_status": "unsupported",
             "correctness_reason": "RoPE requires even head_dim; got 65 (odd) for shape (13, 127, 65) — mathematically ill-defined",
             "allclose": "", "perf_pass": ""},
            # Probe-infra gap.
            {"operator": "linear_ce", "shape_tag": "ce-512", "status": "ok",
             "correctness_status": "unsupported",
             "correctness_reason": "No correctness probe for fused op: linear_ce",
             "allclose": "", "perf_pass": "true"},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is True, f"correctness should pass with typed-unsupported rows: {detail}"
    assert "typed_unsupported=3" in detail
    assert "checked=2" in detail


def test_correctness_evidence_still_fails_on_untyped_unsupported(tmp_path: Path):
    """Untyped/empty-reason `unsupported` rows MUST still count as failures —
    otherwise a runner could silently opt out of correctness checking."""
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # One real ok row.
            {"operator": "relu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            # Untyped unsupported — no reason given. MUST fail.
            {"operator": "foo", "shape_tag": "s1", "status": "ok",
             "correctness_status": "unsupported", "correctness_reason": "",
             "allclose": "", "perf_pass": "true"},
            # Free-form reason that doesn't match any typed pattern. MUST fail.
            {"operator": "bar", "shape_tag": "s1", "status": "ok",
             "correctness_status": "unsupported",
             "correctness_reason": "something went wrong",
             "allclose": "", "perf_pass": "true"},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is False
    assert "correctness=unsupported" in detail
    assert "typed_unsupported=0" in detail


def test_perf_evidence_excludes_typed_unsupported(tmp_path: Path):
    """Typed-unsupported rows have no Arke-vs-baseline perf comparison and must
    be audit-only excluded from perf scoring, not flagged as malformed."""
    root = tmp_path
    # Need ot_map; bypass by writing a matrix path with relu mapped to OT0/1.
    matrix_path = root / "matrix.json"
    matrix_path.write_text(
        '{"l1": [{"op": "relu", "ot_tier": 0}, {"op": "rope", "ot_tier": 4}]}'
    )
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            {"operator": "relu", "shape_tag": "s1", "status": "ok",
             "correctness_status": "ok", "allclose": "true", "perf_pass": "true"},
            # Typed unsupported with empty perf_pass — should be EXCLUDED, not malformed.
            {"operator": "rope", "shape_tag": "non-align-1", "status": "unsupported",
             "correctness_status": "unsupported",
             "correctness_reason": "RoPE requires even head_dim; got 65 (odd)",
             "allclose": "", "perf_pass": ""},
            {"operator": "rope", "shape_tag": "extreme-long", "status": "ok",
             "correctness_status": "unsupported",
             "correctness_reason": "runner Liger-Kernel does not implement run_with_inputs for rope",
             "allclose": "", "perf_pass": ""},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_performance_evidence(root, matrix_path)

    # OT0/1 will be 1/1 (relu passes); ot2/ot3 empty so still fails the gate,
    # but the critical assertion is that the typed-unsupported rows are
    # excluded, NOT reported as malformed.
    assert "malformed/non-ok perf rows" not in detail, (
        f"typed-unsupported rows must not be flagged malformed: {detail}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Q2a — non-Arke baseline rows are oracles, not the SUT. Their crashes must
# never enter the correctness `bad_rows` denominator (or Arke is held
# responsible for PyTorch-eager / cuBLAS / FlagGems failures).
# ──────────────────────────────────────────────────────────────────────────
def test_correctness_evidence_skips_non_arke_baseline_failures(tmp_path: Path):
    """A baseline crash on a reference runner (e.g. PyTorch-eager ptxas SIGKILL
    on extreme-wide topk) is NOT an Arke regression. Only the Arke row counts.
    """
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # Arke row passes — system-under-test is clean.
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "Arke",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
            # PyTorch-eager (oracle) crashed in compiler — must not fail Arke.
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "PyTorch-eager",
             "status": "ok", "correctness_status": "error",
             "correctness_reason": "ptxas died with -9 (SIGKILL)",
             "allclose": "", "perf_pass": ""},
            # FlagGems (ladder reference) declined this shape.
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "FlagGems",
             "status": "unsupported",
             "correctness_status": "unsupported",
             "correctness_reason": "FlagGems.get_fn declined topk@extreme-wide",
             "allclose": "", "perf_pass": ""},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is True, f"non-Arke baseline failures must be skipped: {detail}"
    assert "non_arke_baseline_skipped=2" in detail, detail
    assert "checked=1" in detail, detail


def test_correctness_evidence_still_fails_on_arke_baseline_failure(tmp_path: Path):
    """The Arke row is the source of truth. If Arke crashes, gate must fail
    even if oracle baselines all pass.
    """
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            {"operator": "matmul", "shape_tag": "square-1k", "baseline": "Arke",
             "status": "ok", "correctness_status": "ok",
             "allclose": "false", "perf_pass": "false"},
            {"operator": "matmul", "shape_tag": "square-1k", "baseline": "cuBLAS/cuDNN",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
        ],
    )
    _write_perf_all(root / "l2" / "PERF_ALL.csv", [])

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)

    assert ok is False, f"Arke regression must fail: {detail}"
    assert "failures=1" in detail, detail


def test_correctness_evidence_treats_empty_baseline_as_arke(tmp_path: Path):
    """L2 fusion rows have empty baseline (only one row per fused op). They
    must be treated as Arke-side and counted, so a real L2 regression cannot
    hide behind a missing baseline label.
    """
    root = tmp_path
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv", []
    )
    _write_perf_all(
        root / "l2" / "PERF_ALL.csv",
        [
            {"operator": "geglu", "shape_tag": "non-align-1", "baseline": "",
             "status": "error", "correctness_status": "error",
             "correctness_reason": "Unbroadcastable Size([127, 3073])",
             "allclose": "", "perf_pass": ""},
        ],
    )

    ok, detail = gate_g7._check_bl5_correctness_evidence(root)
    assert ok is False, f"empty-baseline L2 row must NOT be skipped: {detail}"
    assert "failures=1" in detail, detail


def test_is_non_arke_baseline_helper():
    """Direct unit coverage for the helper."""
    assert gate_g7._is_non_arke_baseline({"baseline": "PyTorch-eager"}) is True
    assert gate_g7._is_non_arke_baseline({"baseline": "FlagGems"}) is True
    assert gate_g7._is_non_arke_baseline({"baseline": "cuBLAS/cuDNN"}) is True
    # Case-insensitive match for Arke.
    assert gate_g7._is_non_arke_baseline({"baseline": "Arke"}) is False
    assert gate_g7._is_non_arke_baseline({"baseline": "arke"}) is False
    assert gate_g7._is_non_arke_baseline({"baseline": "ARKE"}) is False
    # Empty/missing → Arke-side (L2 fusion path).
    assert gate_g7._is_non_arke_baseline({"baseline": ""}) is False
    assert gate_g7._is_non_arke_baseline({}) is False
    assert gate_g7._is_non_arke_baseline({"baseline": "  "}) is False


# ──────────────────────────────────────────────────────────────────────────
# Q6b — perf path must mirror the correctness path's exclusions:
#   1. non-Arke baseline crashes / declines are oracle gaps, not malformed
#   2. Arke rows with no usable perf oracle (priority-1 baseline crashed,
#      correctness verified via PyTorch-eager fallback) are audit-only
# Both eliminate the historical "malformed/non-ok perf rows=3" failure on
# extreme-flat gelu (Liger n>65k) + extreme-wide topk (PyTorch ptxas SIGKILL).
# ──────────────────────────────────────────────────────────────────────────
def test_perf_evidence_skips_non_arke_baseline_failures(tmp_path: Path):
    """Reference-baseline rows in PERF_ALL — Liger declining n>65k blocksize
    and PyTorch-eager dying with ptxas SIGKILL — are oracle gaps. They must
    NOT inflate the malformed/non-ok perf row count, mirroring the same
    treatment in `_check_bl5_correctness_evidence`.
    """
    root = tmp_path
    matrix_path = root / "matrix.json"
    matrix_path.write_text(
        '{"l1": [{"op": "gelu", "ot_tier": 1}, {"op": "topk", "ot_tier": 2}]}'
    )
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # Arke rows — system-under-test is healthy.
            {"operator": "gelu", "shape_tag": "extreme-flat", "baseline": "Arke",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "Arke",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
            # Liger declined the shape (baseline self-rejection on n>65k).
            {"operator": "gelu", "shape_tag": "extreme-flat", "baseline": "Liger-Kernel",
             "status": "error",
             "reason": "Cannot launch Triton kernel since n = 1048576 exceeds the recommended Triton blocksize = 65536.",
             "correctness_status": "error",
             "correctness_reason": "Cannot launch Triton kernel since n = 1048576 exceeds the recommended Triton blocksize = 65536.",
             "allclose": "", "perf_pass": ""},
            # PyTorch-eager ptxas SIGKILL on extreme-wide topk.
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "PyTorch-eager",
             "status": "error",
             "reason": "`ptxas` failed with error code -9",
             "correctness_status": "error",
             "correctness_reason": "`ptxas` failed with error code -9",
             "allclose": "", "perf_pass": ""},
        ],
    )
    _write_perf_all(
        root / "l2" / "PERF_ALL.csv",
        [
            # Need at least one L2 row to satisfy the "no evaluable fusion" check.
            {"operator": "geglu_fused", "shape_tag": "s1", "baseline": "",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
        ],
    )

    ok, detail = gate_g7._check_bl5_performance_evidence(root, matrix_path)

    # Critical: the two non-Arke baseline crash rows must NOT be flagged as
    # malformed/non-ok perf rows.
    assert "malformed/non-ok perf rows" not in detail, (
        f"non-Arke baseline crashes must not be flagged malformed: {detail}"
    )
    # And the counter should report we skipped exactly 2 oracle-side rows.
    assert "non_arke_baseline_skipped=2" in detail, detail


def test_perf_evidence_skips_perf_oracle_unavailable_arke_row(tmp_path: Path):
    """When the priority-1 reference baseline crashed AND Arke's correctness
    was verified via PyTorch-eager fallback (correctness_reason carries the
    'used PyTorch-eager reference fallback' marker), the Arke row has no
    usable perf ratio. It must be excluded as audit-only, not malformed.
    """
    root = tmp_path
    matrix_path = root / "matrix.json"
    matrix_path.write_text('{"l1": [{"op": "topk", "ot_tier": 2}]}')
    _write_perf_all(
        root / "l1" / "PERF_ALL.csv",
        [
            # Arke row: correctness ok via fallback, perf_pass empty (no oracle).
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "Arke",
             "status": "ok",
             "correctness_status": "ok",
             "correctness_reason": "golden_runner='FlagGems' returned None; used PyTorch-eager reference fallback",
             "allclose": "true",
             "perf_target": "1.0", "perf_actual": "", "perf_pass": ""},
            # The PyTorch-eager oracle crashed for this shape; non-Arke skip
            # already covered by the previous test.
            {"operator": "topk", "shape_tag": "extreme-wide", "baseline": "PyTorch-eager",
             "status": "error",
             "correctness_status": "error",
             "correctness_reason": "`ptxas` failed with error code -9",
             "allclose": "", "perf_pass": ""},
        ],
    )
    _write_perf_all(
        root / "l2" / "PERF_ALL.csv",
        [
            {"operator": "geglu_fused", "shape_tag": "s1", "baseline": "",
             "status": "ok", "correctness_status": "ok",
             "allclose": "true", "perf_pass": "true"},
        ],
    )

    ok, detail = gate_g7._check_bl5_performance_evidence(root, matrix_path)

    assert "malformed/non-ok perf rows" not in detail, (
        f"perf-oracle-unavailable rows must not be flagged malformed: {detail}"
    )
    assert "perf_oracle_unavailable=1" in detail, detail


def test_is_perf_oracle_unavailable_helper():
    """Direct unit coverage for the new helper."""
    # Positive: Arke row with fallback marker and empty perf fields.
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "ok",
        "correctness_reason": "golden_runner='FlagGems' returned None; used PyTorch-eager reference fallback",
        "perf_actual": "",
        "perf_pass": "",
    }) is True
    # Negative: Arke row with healthy perf_pass — the oracle worked.
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "ok",
        "correctness_reason": "used PyTorch-eager reference fallback",
        "perf_actual": "0.95",
        "perf_pass": "false",
    }) is False
    # Negative: not the Arke SUT (oracle row shouldn't trip this rule).
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "PyTorch-eager",
        "status": "ok",
        "correctness_reason": "used PyTorch-eager reference fallback",
        "perf_actual": "",
        "perf_pass": "",
    }) is False
    # Negative: status not ok (a real Arke crash must remain visible).
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "error",
        "correctness_reason": "used PyTorch-eager reference fallback",
        "perf_actual": "",
        "perf_pass": "",
    }) is False
    # Negative: empty/missing fallback marker — don't silently exempt.
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "ok",
        "correctness_reason": "",
        "perf_actual": "",
        "perf_pass": "",
    }) is False
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "ok",
        "correctness_reason": "some other reason",
        "perf_actual": "",
        "perf_pass": "",
    }) is False
    # Edge: perf_actual=N/A is treated as empty (legacy harness output).
    assert gate_g7._is_perf_oracle_unavailable({
        "baseline": "Arke",
        "status": "ok",
        "correctness_reason": "used PyTorch-eager reference fallback",
        "perf_actual": "N/A",
        "perf_pass": "",
    }) is True

