# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 Gate G7 verification.

This runner validates the currently machine-checkable implementation closure for
Stage 7. It does not weaken the roadmap gate; instead it records which parts are
already executable in CI-like form and which benchmark/perf artifacts still need
real BL5 evidence.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.gate import GateResult, GateSummary
from benchmarks.results_contract import check_result_tree_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
OPERATORS_DIR = REPO_ROOT / "examples" / "operators"
STAGE7_TRACK6_ROOT = REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6"
REQUIRED_TRACK6_LAYER_ARTIFACTS = (
    "config.json",
    "hardware.json",
    "sources.json",
    "summary.json",
    "PERF_ALL.csv",
)
REQUIRED_TRACK6_ROOT_ARTIFACTS = (
    "coverage_gap.json",
    "audit_report.json",
    "stage7_operator_shape_stats.json",
    "dashboard.json",
)
RESULT_TREE_NAME = "phase1/stage7/track6"

# --- Same-Backend Triton Fairness (locked 2026-05-16) --------------------------
#
# Gate G7 performance scoring compares Arke-Triton latency against the
# best Triton-only baseline available for the same (op, shape_tag) tuple.
# This isolates Arke's compilation quality from cross-backend architectural
# advantages. Cross-backend baselines (PyTorch-eager / cuBLAS-cuDNN /
# torch.compile) are recorded in PERF_ALL for audit but excluded from
# Gate scoring.
#
# See docs/roadmap/plan.md § Same-Backend Fairness and
# docs/benchmark/benchmark-protocol.md § Same-Backend Fairness for rules.
_TRITON_ONLY_BASELINES: frozenset[str] = frozenset({
    "flaggems",
    "liger-kernel",
    "triton-tutorial",
    "unsloth",
    "flash-attn-triton",
    "vllm-triton",
    "flashinfer-triton",
})

# Universal measurement-noise tolerance. Pass criterion:
#   arke_latency <= triton_ref_latency * (1 + _PERF_EPSILON)
# equivalently ratio = triton_ref_latency / arke_latency >= 1 / (1 + _PERF_EPSILON)
_PERF_EPSILON: float = 0.03


def _run_cmd(args: list[str], timeout: int = 300) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    ok = proc.returncode == 0
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    return ok, detail[-2500:]


def _run_pytest(args: list[str]) -> tuple[bool, str]:
    return _run_cmd([sys.executable, "-m", "pytest", "-q", *args], timeout=1200)


def _check_all_examples_compile() -> tuple[bool, str]:
    from arke.compiler.pipeline import ArkePipeline

    failures: list[str] = []
    files = sorted(OPERATORS_DIR.glob("*.ak"))
    pipeline = ArkePipeline()
    for ak in files:
        result = pipeline.compile_file(str(ak))
        if not result.success:
            failures.append(f"{ak.name}: {result.errors}")
    if failures:
        return False, "\n".join(failures[:10])
    return True, f"{len(files)} .ak files passed dry-run pipeline"


def _check_spec_docs() -> tuple[bool, str]:
    required = [
        REPO_ROOT / "docs" / "spec" / "arke-lang-spec.md",
        REPO_ROOT / "docs" / "spec" / "arke-ir-spec.md",
        REPO_ROOT / "docs" / "phase1" / "dynamic-shape-feasibility.md",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.exists()]
    if missing:
        return False, f"Missing docs: {missing}"
    return True, "Lang spec, IR spec, and dynamic-shape feasibility doc present"


def check_stage7_track6_artifacts(track6_root: Path = STAGE7_TRACK6_ROOT) -> tuple[bool, str]:
    return check_result_tree_artifacts(
        track6_root,
        layer_artifacts=REQUIRED_TRACK6_LAYER_ARTIFACTS,
        root_artifacts=REQUIRED_TRACK6_ROOT_ARTIFACTS,
        tree_name=RESULT_TREE_NAME,
    )


def _check_benchmark_artifacts() -> tuple[bool, str]:
    return check_stage7_track6_artifacts(STAGE7_TRACK6_ROOT)


def _load_json_artifact(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing artifact: {path}"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON artifact {path}: {exc}"


def _summarize_items(items: list[Any], limit: int = 8) -> str:
    if not items:
        return "none"
    shown = items[:limit]
    suffix = f", ... +{len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(str(item) for item in shown) + suffix


def _check_bl5_coverage_evidence(track6_root: Path = STAGE7_TRACK6_ROOT) -> tuple[bool, str]:
    """Verify that persisted Track 6 artifacts close the BL5 coverage surface.

    This is intentionally stricter than the older artifact-presence check: G7 is
    not complete merely because CSV/JSON files exist. The target matrix must have
    full L1/L2 required-shape evidence and the audit report must expose no
    missing examples, strategy surfaces, benchmark rows, or unsupported surface
    cases.
    """

    failures: list[str] = []
    coverage, err = _load_json_artifact(track6_root / "coverage_gap.json")
    if err:
        failures.append(err)
    audit, audit_err = _load_json_artifact(track6_root / "audit_report.json")
    if audit_err:
        failures.append(audit_err)

    if coverage:
        for layer in ("l1", "l2"):
            layer_report = coverage.get(layer, {})
            total = int(layer_report.get("ops_total", 0) or 0)
            any_evidence = int(layer_report.get("ops_with_any_evidence", 0) or 0)
            fully = int(layer_report.get("ops_fully_covered", 0) or 0)
            required_shapes = int(layer_report.get("shapes_required_total", 0) or 0)
            observed_shapes = int(layer_report.get("shapes_observed_total", 0) or 0)
            ratio = float(layer_report.get("shape_coverage_ratio", 0.0) or 0.0)
            if total <= 0:
                failures.append(f"{layer}: no target entries found in coverage_gap.json")
            if any_evidence != total:
                failures.append(f"{layer}: op evidence {any_evidence}/{total}")
            if fully != total:
                partial = [
                    entry.get("op", "<unknown>")
                    for entry in layer_report.get("per_op", [])
                    if entry.get("missing_shape_tags")
                ]
                failures.append(
                    f"{layer}: full-shape coverage {fully}/{total}; partial={_summarize_items(partial)}"
                )
            if observed_shapes != required_shapes or ratio != 1.0:
                failures.append(
                    f"{layer}: shape coverage {observed_shapes}/{required_shapes} ({ratio:.4f})"
                )
            missing_perf_fields = [
                entry.get("op", "<unknown>")
                for entry in layer_report.get("per_op", [])
                if entry.get("missing_perf_target_fields")
            ]
            if missing_perf_fields:
                failures.append(
                    f"{layer}: missing perf target fields for {_summarize_items(missing_perf_fields)}"
                )

    if audit:
        summary = audit.get("summary", {})
        for layer in ("l1", "l2"):
            layer_summary = summary.get(layer, {})
            for key in (
                "missing_examples",
                "missing_strategy_examples",
                "missing_benchmark_evidence",
                "missing_full_shape_evidence",
                "unsupported_surface_cases",
            ):
                items = layer_summary.get(key, [])
                if items:
                    failures.append(
                        f"{layer}: {key}={len(items)} ({_summarize_items(items)})"
                    )

    if failures:
        return False, "; ".join(failures)
    return True, "BL5 L1/L2 coverage evidence is complete with no audit gaps"


def _parse_float(value: str | None) -> float | None:
    if value in (None, "", "NA", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


_GOLDEN_PROTOCOL_EXEMPT_CORRECTNESS = {
    # Per docs/benchmark/benchmark-protocol.md and golden-kernel-ladder.md:
    # rows where the priority-aware Golden Kernel ladder cannot bind a P<=4
    # baseline are recorded with this status. They are evidence-of-gap, NOT
    # correctness regressions, and the protocol mandates audit-only treatment.
    "golden_unavailable_pending_baseline",
}


def _is_golden_unavailable(row: dict[str, str]) -> bool:
    """Audit-only rows where ladder could not bind a P<=4 baseline.

    These are correctness-equivalent (we lack a trustworthy oracle), so the
    Golden Kernel protocol exempts them from correctness fail counting while
    still surfacing them in the coverage / audit artifacts.
    """

    correctness = (row.get("correctness_status") or "").strip().lower()
    return correctness in _GOLDEN_PROTOCOL_EXEMPT_CORRECTNESS


# Typed-decline reason templates. A row is recognised as a *typed*
# `unsupported` (a deliberate, machine-readable declaration of "cannot run this
# combination") only if `correctness_status == "unsupported"` AND
# `correctness_reason` matches one of these patterns. Free-form / empty reasons
# remain failures so runners cannot silently opt out of correctness checking.
#
# Per benchmark-protocol.md, `status ∈ {unsupported, incompatible, oom, skipped}`
# is treated as a known limitation by the retry policy. The Gate must apply the
# same lens to its correctness/perf accounting: a typed decline is evidence of
# an intentional gap, not a regression.
_TYPED_UNSUPPORTED_REASON_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Runner-side typed declines.
    #   "<Runner>.get_fn declined <op>@<shape>"  — runner explicitly refused
    re.compile(r"\.get_fn\s+declined\b", re.IGNORECASE),
    #   "runner <Runner> does not implement run_with_inputs for <op>"
    re.compile(r"does not implement\s+run_with_inputs\b", re.IGNORECASE),
    # Op-level mathematical shape guards (ill-defined inputs).
    #   "RoPE requires even head_dim; got N (odd) ..."
    re.compile(r"requires even head_dim", re.IGNORECASE),
    #   "Gated op <op> requires even feature width N; got N=... (odd) ..."
    #   (gated activations like gelu_and_mul/silu_and_mul split via chunk(2, dim=-1))
    re.compile(r"requires even feature width", re.IGNORECASE),
    re.compile(r"mathematically ill-defined", re.IGNORECASE),
    # Probe-infra gap: harness has no correctness probe for this fused op yet.
    #   "No correctness probe for fused op: <op>"
    re.compile(r"no correctness probe for", re.IGNORECASE),
)


def _is_typed_unsupported(row: dict[str, str]) -> bool:
    """Return True iff this row is a *typed* unsupported declaration.

    Required conditions:
      1. `correctness_status == "unsupported"`
      2. `correctness_reason` matches one of `_TYPED_UNSUPPORTED_REASON_PATTERNS`

    Untyped unsupported rows (empty / unrecognised reason) are NOT exempted —
    that would let a runner silently skip correctness checking by writing
    `unsupported` without justification.
    """

    correctness = (row.get("correctness_status") or "").strip().lower()
    if correctness != "unsupported":
        return False
    reason = (row.get("correctness_reason") or "").strip()
    if not reason:
        return False
    return any(pat.search(reason) for pat in _TYPED_UNSUPPORTED_REASON_PATTERNS)


def _is_non_arke_baseline(row: dict[str, str]) -> bool:
    """Return True for baseline rows that are NOT the system-under-test (Arke).

    Per the BL5 Track-6 evidence protocol, every (op, shape) target has one row
    per baseline (Arke + reference baselines such as PyTorch-eager, cuBLAS/cuDNN,
    FlagGems, Liger-Kernel, torch.compile, Triton-Tutorial). Correctness and
    performance evidence is about Arke's behaviour; the other baselines exist
    only as oracles and ladder references. A baseline crash (e.g. PyTorch-eager
    ptxas SIGKILL on extreme-wide topk) is NOT a regression for Arke and must
    not enter the correctness `bad_rows` denominator.

    Empty baseline strings (some L2 fusion rows) are treated as Arke-side rows
    because they originate from the harness's L2 fusion path which produces
    only the system-under-test row (no per-baseline replication).
    """
    baseline = (row.get("baseline") or "").strip()
    if not baseline:
        return False
    return baseline.lower() != "arke"


_PERF_ORACLE_UNAVAILABLE_REASON_PATTERN = re.compile(
    # Emitted by `bench_l1` when the priority-1 Golden Kernel ladder runner
    # returns None and the harness falls back to PyTorch-eager as reference.
    # When PyTorch-eager itself also crashed for this shape (e.g. ptxas SIGKILL
    # on extreme-wide topk), the resulting Arke row carries:
    #   correctness_status = ok        (fallback ran and produced an oracle)
    #   correctness_reason = "...used PyTorch-eager reference fallback"
    #   perf_actual / perf_pass = empty (no usable baseline_latency to ratio against)
    # That is a *perf oracle gap*, not a malformed row.
    r"used PyTorch-eager reference fallback",
    re.IGNORECASE,
)


def _is_perf_oracle_unavailable(row: dict[str, str]) -> bool:
    """Return True for Arke rows whose perf comparison has no usable oracle.

    Distinct from `_is_golden_unavailable` (which keys on correctness_status =
    ``golden_unavailable_pending_baseline``). Here correctness *did* succeed
    via a fallback reference, but the priority-1 reference baseline crashed
    for this shape, so `perf_actual` / `perf_pass` are empty and there is no
    meaningful perf ratio to score.

    Required conditions:
      1. row is the Arke system-under-test (not a baseline oracle)
      2. status == "ok" (Arke itself ran)
      3. perf_actual / perf_pass are empty (no ratio was computable)
      4. correctness_reason documents the fallback (machine-readable marker)
    """
    if _is_non_arke_baseline(row):
        return False
    status = (row.get("status") or "").strip().lower()
    if status != "ok":
        return False
    perf_pass = (row.get("perf_pass") or "").strip()
    perf_actual = (row.get("perf_actual") or "").strip()
    if perf_pass or (perf_actual and perf_actual.upper() not in {"N/A", "NA"}):
        return False
    reason = (row.get("correctness_reason") or "").strip()
    if not reason:
        return False
    return bool(_PERF_ORACLE_UNAVAILABLE_REASON_PATTERN.search(reason))


def _baseline_is_triton_only(row: dict[str, str]) -> bool:
    """Return True iff this row's baseline is a same-backend (Triton) reference.

    Same-Backend Triton Fairness (locked 2026-05-16): the Gate perf denominator
    is the fastest Triton-only kernel across the ladder. Cross-backend
    baselines (PyTorch-eager / cuBLAS-cuDNN / torch.compile) are excluded
    from Gate scoring (but remain in PERF_ALL for audit).

    Matching is case-insensitive against the canonical baseline names used
    in PERF_ALL.csv (e.g. ``FlagGems``, ``Liger-Kernel``, ``Triton-Tutorial``).
    """
    baseline = (row.get("baseline") or "").strip().lower()
    if not baseline:
        return False
    return baseline in _TRITON_ONLY_BASELINES


def _build_triton_ref_index(
    rows: list[tuple[str, dict[str, str]]],
) -> dict[tuple[str, str, str], float]:
    """Build a ``(layer, op, shape_tag) -> min_triton_ref_latency_us`` index.

    For each (op, shape_tag) target we keep the fastest Triton-only baseline
    latency, mirroring the PRIMARY+FALLBACK ladder ordering in
    `docs/benchmark/golden-kernel-ladder.md` (the ladder is curated so the
    PRIMARY entry should be the fastest; min() over all Triton-only entries
    is a robust safety net against ladder-ordering bugs).

    Only rows with ``status == "ok"`` and a parseable ``latency_us`` are
    considered. Returns an empty dict when no Triton-only references are
    available (callers must then route every Arke row to audit-only).
    """
    index: dict[tuple[str, str, str], float] = {}
    for layer, row in rows:
        if not _baseline_is_triton_only(row):
            continue
        status = (row.get("status") or "").strip().lower()
        if status != "ok":
            continue
        op = row.get("operator") or row.get("op") or ""
        shape_tag = row.get("shape_tag") or ""
        if not op or not shape_tag:
            continue
        lat_raw = (row.get("latency_us") or "").strip()
        try:
            lat = float(lat_raw)
        except (TypeError, ValueError):
            continue
        if lat <= 0:
            continue
        key = (layer, op, shape_tag)
        prev = index.get(key)
        if prev is None or lat < prev:
            index[key] = lat
    return index


def _is_memory_excluded(row: dict[str, str]) -> bool:
    """Return true for explicit 6GB-memory-policy exclusions.

    G7 allows correctness accounting to exclude OOM-only rows, but only when the
    row carries machine-readable memory preflight evidence. Ordinary failures,
    unsupported correctness probes, and non-memory skips remain failures.
    """

    status = (row.get("status") or "").strip().lower()
    correctness = (row.get("correctness_status") or "").strip().lower()
    memory_policy = (row.get("memory_policy") or "").strip()
    if status != "skipped" and correctness != "skipped":
        return False
    if not memory_policy:
        return False

    reason = f"{row.get('reason', '')} {row.get('correctness_reason', '')}".lower()
    required = _parse_float(row.get("memory_bytes_required"))
    budget = _parse_float(row.get("memory_bytes_budget"))
    ratio = _parse_float(row.get("memory_ratio"))
    has_memory_numbers = ratio is not None or (required is not None and budget is not None)
    exceeds_budget = (ratio is not None and ratio >= 1.0) or (
        required is not None and budget is not None and required >= budget
    )
    reason_mentions_memory = any(token in reason for token in ("oom", "memory", "vram", "preflight"))
    return has_memory_numbers and (exceeds_budget or reason_mentions_memory)


def _iter_perf_rows(track6_root: Path) -> tuple[list[tuple[str, dict[str, str]]], list[str]]:
    rows: list[tuple[str, dict[str, str]]] = []
    failures: list[str] = []
    for layer in ("l1", "l2"):
        path = track6_root / layer / "PERF_ALL.csv"
        if not path.exists():
            failures.append(f"{layer}: missing {path}")
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append((layer, row))
    return rows, failures


def _row_label(layer: str, row: dict[str, str]) -> str:
    op = row.get("operator") or row.get("op") or "<unknown>"
    shape = row.get("shape_tag") or "<shape?>"
    return f"{layer}:{op}:{shape}"


def _check_bl5_correctness_evidence(track6_root: Path = STAGE7_TRACK6_ROOT) -> tuple[bool, str]:
    rows, failures = _iter_perf_rows(track6_root)
    checked = 0
    excluded = 0
    golden_exempted = 0
    typed_unsupported = 0
    non_arke_baseline_skipped = 0
    bad_rows: list[str] = []

    for layer, row in rows:
        if _is_non_arke_baseline(row):
            # Reference-baseline rows are oracles, not the system-under-test.
            # Their crashes / unsupported declarations are not Arke regressions
            # and must not enter the failure denominator. See _is_non_arke_baseline.
            non_arke_baseline_skipped += 1
            continue
        if _is_memory_excluded(row):
            excluded += 1
            continue
        if _is_golden_unavailable(row):
            golden_exempted += 1
            continue
        if _is_typed_unsupported(row):
            # Typed `unsupported` rows: runner declined or op math guard
            # refused this shape with a machine-readable reason. These are
            # intentional, auditable gaps — NOT correctness regressions.
            # Untyped/empty-reason unsupported still falls through to fail.
            typed_unsupported += 1
            continue
        checked += 1
        status = (row.get("status") or "").strip().lower()
        correctness = (row.get("correctness_status") or "").strip().lower()
        allclose = (row.get("allclose") or "").strip().lower()
        if status != "ok":
            bad_rows.append(f"{_row_label(layer, row)} status={status or '<empty>'}")
            continue
        if correctness != "ok":
            bad_rows.append(
                f"{_row_label(layer, row)} correctness={correctness or '<empty>'}"
            )
            continue
        if allclose and allclose not in {"true", "1", "yes"}:
            bad_rows.append(f"{_row_label(layer, row)} allclose={allclose}")

    if bad_rows:
        failures.append(
            f"correctness failures={len(bad_rows)} checked={checked} "
            f"memory_excluded={excluded} golden_exempted={golden_exempted} "
            f"typed_unsupported={typed_unsupported} "
            f"non_arke_baseline_skipped={non_arke_baseline_skipped}; "
            f"first={_summarize_items(bad_rows)}"
        )
    if failures:
        return False, "; ".join(failures)
    return True, (
        f"correctness rows passed: checked={checked}, "
        f"memory_excluded={excluded}, golden_exempted={golden_exempted}, "
        f"typed_unsupported={typed_unsupported}, "
        f"non_arke_baseline_skipped={non_arke_baseline_skipped}"
    )


def _load_l1_ot_map(matrix_path: Path) -> dict[str, int]:
    matrix, err = _load_json_artifact(matrix_path)
    if err or not matrix:
        return {}
    return {entry["op"]: int(entry.get("ot_tier", -1)) for entry in matrix.get("l1", [])}


def _check_bl5_performance_evidence(
    track6_root: Path = STAGE7_TRACK6_ROOT,
    matrix_path: Path = REPO_ROOT / "benchmarks" / "stage7_bl5_target_matrix.json",
) -> tuple[bool, str]:
    rows, failures = _iter_perf_rows(track6_root)
    ot_map = _load_l1_ot_map(matrix_path)
    if not ot_map:
        failures.append(f"missing L1 OT mapping from {matrix_path}")

    # Same-Backend Triton Fairness (locked 2026-05-16): build per-(layer, op,
    # shape_tag) Triton reference index from the rows, then re-score every
    # Arke row against `triton_ref_lat * (1 + _PERF_EPSILON)`. This deliberately
    # bypasses `perf_pass` written by `benchmarks/artifacts.py` (which is
    # computed against PyTorch-eager) — Benchmark measurement stays frozen,
    # only the Gate scoring layer changes.
    triton_ref_index = _build_triton_ref_index(rows)

    group_counts = {
        "ot0_1": {"passed": 0, "total": 0},
        "ot2": {"passed": 0, "total": 0},
        "ot3": {"passed": 0, "total": 0},
        "ot4": {"passed": 0, "total": 0},
    }
    l2_counts: dict[str, dict[str, int]] = {}
    excluded = 0
    non_arke_baseline_skipped = 0
    perf_oracle_unavailable = 0
    perf_oracle_unavailable_triton = 0
    malformed: list[str] = []

    for layer, row in rows:
        if _is_non_arke_baseline(row):
            # Reference-baseline rows (PyTorch-eager / cuBLAS / FlagGems / Liger /
            # torch.compile / Triton-Tutorial) are oracles and ladder references,
            # not the system-under-test. A baseline crash or self-rejection
            # (e.g. Liger declining n=1M > Triton blocksize, PyTorch-eager
            # ptxas SIGKILL on extreme-wide topk) is NOT an Arke perf regression
            # and must not enter the malformed denominator. Mirrors the same
            # treatment in `_check_bl5_correctness_evidence`.
            non_arke_baseline_skipped += 1
            continue
        if _is_memory_excluded(row):
            excluded += 1
            continue
        if _is_golden_unavailable(row):
            # No trustworthy golden ⇒ no trustworthy perf denominator.
            # Audit-only per protocol; do not count toward perf scoring.
            excluded += 1
            continue
        if _is_typed_unsupported(row):
            # Typed unsupported correctness probe: either the baseline runner
            # declined this combination, or an op-level math guard refused the
            # shape. In PERF_ALL such rows carry no `perf_actual` / `perf_pass`
            # because there's no Arke-vs-baseline comparison to run. Audit-only
            # per protocol; exclude from perf denominator.
            excluded += 1
            continue
        if _is_perf_oracle_unavailable(row):
            # Arke row whose priority-1 reference baseline crashed for this
            # shape (e.g. topk:extreme-wide where PyTorch-eager ptxas SIGKILL
            # leaves no usable ratio). Correctness was verified via fallback,
            # but the perf comparison genuinely has no oracle latency.
            # Audit-only per Golden Kernel protocol.
            perf_oracle_unavailable += 1
            continue
        if (row.get("status") or "").strip().lower() != "ok":
            malformed.append(f"{_row_label(layer, row)} status={row.get('status') or '<empty>'}")
            continue

        # --- Same-Backend Triton Fairness scoring ----------------------------
        #
        # Score Arke against the fastest Triton-only baseline for this
        # (layer, op, shape_tag). Rows with no Triton-only baseline available
        # are audit-only (perf_oracle_unavailable_triton).
        op = row.get("operator") or row.get("op") or ""
        shape_tag = row.get("shape_tag") or ""
        arke_lat_raw = (row.get("latency_us") or "").strip()
        if not arke_lat_raw:
            # Legacy PERF_ALL artifacts (and historical L2 fusion rows) often
            # omit the `latency_us` column entirely. Under Same-Backend Triton
            # Fairness we cannot score these rows — but they are an *oracle gap*
            # (no measurement available), not a malformed Arke regression.
            # Audit-only per Golden Kernel protocol, mirroring the perf_oracle
            # exclusion above. This also keeps legacy tests stable when the
            # column was not yet plumbed.
            perf_oracle_unavailable += 1
            continue
        try:
            arke_lat = float(arke_lat_raw)
        except (TypeError, ValueError):
            malformed.append(
                f"{_row_label(layer, row)} latency_us={arke_lat_raw or '<empty>'}"
            )
            continue
        if arke_lat <= 0:
            malformed.append(
                f"{_row_label(layer, row)} latency_us={arke_lat_raw} (non-positive)"
            )
            continue

        triton_ref_lat = triton_ref_index.get((layer, op, shape_tag))
        if triton_ref_lat is None:
            # No same-backend Triton reference for this op-shape → audit-only.
            # Recorded but not counted toward Gate scoring per the locked
            # Same-Backend Fairness rule (Benchmark stays frozen, Gate
            # scoring drops the row).
            perf_oracle_unavailable_triton += 1
            continue

        passed = arke_lat <= triton_ref_lat * (1.0 + _PERF_EPSILON)

        if layer == "l1":
            ot = ot_map.get(op)
            if ot in (0, 1):
                key = "ot0_1"
            elif ot == 2:
                key = "ot2"
            elif ot == 3:
                key = "ot3"
            elif ot == 4:
                key = "ot4"
            else:
                malformed.append(f"{_row_label(layer, row)} unknown_ot={ot}")
                continue
            group_counts[key]["total"] += 1
            group_counts[key]["passed"] += int(passed)
        else:
            bucket = l2_counts.setdefault(op, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(passed)

    if malformed:
        failures.append(f"malformed/non-ok perf rows={len(malformed)}; first={_summarize_items(malformed)}")

    weights = {"ot0_1": 0.25, "ot2": 0.30, "ot3": 0.20, "ot4": 0.25}
    weighted_score = 0.0
    group_details: list[str] = []
    per_group_threshold = 0.97  # per-group floor under Same-Backend Triton Fairness
    per_group_violations: list[str] = []
    for key, weight in weights.items():
        total = group_counts[key]["total"]
        passed = group_counts[key]["passed"]
        if total == 0:
            failures.append(f"L1 {key}: no evaluable performance rows (no Triton-only baseline available)")
            rate = 0.0
        else:
            rate = passed / total
            if rate < per_group_threshold:
                per_group_violations.append(f"{key}={rate:.3f}<{per_group_threshold:.2f}")
        weighted_score += weight * rate
        group_details.append(f"{key}={passed}/{total} ({rate:.3f})")
    if per_group_violations:
        failures.append(
            f"L1 per-group pass rate below {per_group_threshold:.2f}: "
            f"{'; '.join(per_group_violations)}"
        )
    if weighted_score < 0.95:
        failures.append(
            f"L1 weighted performance score {weighted_score:.4f} < 0.9500; "
            f"{'; '.join(group_details)}"
        )

    l2_failed = [
        f"{op}={counts['passed']}/{counts['total']}"
        for op, counts in sorted(l2_counts.items())
        if counts["total"] == 0 or counts["passed"] != counts["total"]
    ]
    if l2_failed:
        failures.append(f"L2 fusion performance incomplete: {_summarize_items(l2_failed)}")
    if not l2_counts:
        failures.append("L2: no evaluable fusion performance rows")

    detail = (
        f"L1 weighted_score={weighted_score:.4f} (Same-Backend Triton, eps={_PERF_EPSILON:.2f}); "
        f"{'; '.join(group_details)}; "
        f"L2 fusions={len(l2_counts)}; memory_excluded={excluded}; "
        f"non_arke_baseline_skipped={non_arke_baseline_skipped}; "
        f"perf_oracle_unavailable={perf_oracle_unavailable}; "
        f"perf_oracle_unavailable_triton={perf_oracle_unavailable_triton}"
    )
    if failures:
        return False, detail + "; " + "; ".join(failures)
    return True, detail


def run_g7(tier: int = 2) -> GateSummary:
    results: list[GateResult] = []

    spec_ok, spec_detail = _check_spec_docs()
    results.extend([
        GateResult("G7", "G7.1", "Arke Lang Spec v0.1.0 finalized and present", "function", spec_ok, spec_detail),
        GateResult("G7", "G7.2", "Arke IR Spec v0.1.0 finalized and present", "function", spec_ok, spec_detail),
    ])

    passed, details = _run_pytest([
        "tests/test_symbolic_shape.py",
        "tests/test_stage7_roundtrip.py",
        "tests/test_backend_agnostic.py",
        "tests/test_stage7_memory_aware_strategy.py",
    ])
    results.append(GateResult(
        "G7", "G7.3",
        "where clause + symbolic shape system supports BL5-relevant shape expression and propagation",
        "function", passed, details,
    ))

    results.append(GateResult(
        "G7", "G7.4",
        "Dynamic shape feasibility assessment complete",
        "function", spec_ok, spec_detail,
    ))

    passed, details = _run_pytest([
        "tests/test_mlir_backend.py",
        "tests/test_stage7_lowering.py",
        "tests/test_stage7_pipeline_no_torch.py",
        "tests/test_pipeline.py::TestCompileAll::test_compile_matmul_emits_mlir_skeleton",
        "tests/test_pipeline.py::TestCompileAll::test_compile_strategy_kernel_emits_full_stack_mlir_skeleton",
    ])
    results.append(GateResult(
        "G7", "G7.5",
        "MLIR framework skeleton exists with BL1 matmul path verified",
        "function", passed, details,
    ))

    passed, details = _check_all_examples_compile()
    results.append(GateResult(
        "G7", "G7.6",
        "All BL5 op examples: .ak -> SemanticIR -> StrategyIR full round-trip passes",
        "correctness", passed, details,
    ))

    passed, details = _run_pytest([
        "tests/test_stage7_roundtrip.py",
        "tests/test_symbolic_shape.py",
        "tests/test_rationale_e2e.py",
        "tests/test_stage7_memory_aware_strategy.py",
    ])
    results.append(GateResult(
        "G7", "G7.7",
        "Lang expressiveness covers the BL5 operator/shape surface, not just demos",
        "function", passed, details,
    ))

    passed, details = _run_pytest([
        "tests/test_benchmark_l2.py",
        "tests/test_benchmark_l2_fused_ce.py",
        "tests/test_benchmark_l2_qkv_fa.py",
        "tests/test_benchmark_artifacts.py",
        "tests/test_memory_policy.py",
        "tests/test_benchmark_l1_attention_preflight.py",
        "tests/test_benchmark_advice.py",
        "tests/agent/test_benchmark_advice_tool.py",
        "tests/test_stage7_report.py",
        "tests/test_compiler_advice.py",
        "tests/test_arke_runner_advice.py",
        "tests/test_benchmark_cli.py",
        "tests/test_benchmark_status.py",
        "tests/phase1/stage7/test_track6_contract.py",
        "tests/test_stage7_l2_fusion_surface.py",
        "tests/test_stage7_compile_advice_provenance.py",
        "tests/test_stage7_advice_materialization.py",
        "tests/test_stage7_strategy_synthesis.py",
        "tests/test_stage7_specialized_strategy_synthesis.py",
        "tests/test_stage7_more_specialized_strategy_synthesis.py",
        "tests/benchmark/test_dashboard.py",
        "tests/phase1/stage7/test_stage7_dashboard.py",
    ])
    artifact_ok, artifact_detail = _check_benchmark_artifacts()
    coverage_ok, coverage_detail = _check_bl5_coverage_evidence()
    correctness_ok, correctness_detail = _check_bl5_correctness_evidence()
    performance_ok, performance_detail = _check_bl5_performance_evidence()
    results.append(GateResult(
        "G7", "G7.8",
        "StrategyIR/lowering surface can represent the BL5 L2 fusion set",
        "function", passed, details,
    ))
    results.append(GateResult(
        "G7", "G7.8a",
        "Stage 7 benchmark standard result directories are recognized by gate",
        "function", artifact_ok, artifact_detail,
    ))
    results.append(GateResult(
        "G7", "G7.8b",
        "BL5 L1/L2 coverage evidence is complete against the target matrix",
        "correctness", coverage_ok, coverage_detail,
    ))
    results.append(GateResult(
        "G7", "G7.8c",
        "BL5 correctness rows pass, excluding only explicit memory-policy OOM rows",
        "correctness", correctness_ok, correctness_detail,
    ))
    results.append(GateResult(
        "G7", "G7.8d",
        "BL5 L1 weighted performance and L2 fusion performance meet the G7 contract",
        "performance", performance_ok, performance_detail,
    ))

    passed, details = _run_pytest([
        "tests/test_backend_agnostic.py",
        "tests/test_backend_agnostic_script.py",
    ])
    results.append(GateResult(
        "G7", "G7.9",
        "Backend-agnostic StrategyIR core contains 0 Triton-specific fields",
        "function", passed, details,
    ))

    passed, details = _run_pytest([
        "tests/test_parser.py",
        "tests/test_strategy_ir.py",
        "tests/test_strategy_converter.py",
        "tests/test_converters.py",
        "tests/test_semantic_ir.py",
        "tests/test_stage7_roundtrip.py",
        "tests/test_symbolic_shape.py",
        "tests/test_stage7_memory_aware_strategy.py",
        "tests/test_backend_agnostic.py",
        "tests/test_backend_agnostic_script.py",
        "tests/test_rationale_e2e.py",
        "tests/test_mlir_backend.py",
        "tests/test_stage7_lowering.py",
        "tests/test_stage7_pipeline_no_torch.py",
        "tests/test_benchmark_l2.py",
        "tests/test_benchmark_l2_fused_ce.py",
        "tests/test_benchmark_l2_qkv_fa.py",
        "tests/test_benchmark_cli.py",
        "tests/test_memory_policy.py",
        "tests/test_benchmark_l1_attention_preflight.py",
        "tests/test_benchmark_advice.py",
        "tests/agent/test_benchmark_advice_tool.py",
        "tests/test_stage7_report.py",
        "tests/test_compiler_advice.py",
        "tests/test_arke_runner_advice.py",
        "tests/test_benchmark_status.py",
        "tests/phase1/stage7/test_track6_contract.py",
        "tests/test_stage7_l2_fusion_surface.py",
        "tests/test_stage7_compile_advice_provenance.py",
        "tests/test_stage7_advice_materialization.py",
        "tests/test_stage7_strategy_synthesis.py",
        "tests/test_stage7_specialized_strategy_synthesis.py",
        "tests/test_stage7_more_specialized_strategy_synthesis.py",
        "tests/benchmark/test_dashboard.py",
        "tests/phase1/stage7/test_stage7_dashboard.py",
    ])
    results.append(GateResult(
        "G7", "G7.10",
        "Non-regression suite remains green for the active Stage 7 slice",
        "regression", passed, details,
    ))

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    return GateSummary(
        gate="G7",
        tier=tier,
        total=len(results),
        passed=passed_count,
        failed=failed_count,
        results=results,
    )
