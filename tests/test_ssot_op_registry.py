# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""SSOT enforcement: every op-aware module must derive from ``benchmarks.op_registry``.

This test prevents the "shadow catalog" failure mode where a module hardcodes
its own op-name list and silently drifts from ``docs/benchmark/benchmark-ops.md``.

If you're adding a new op:
  1. Add it to ``docs/benchmark/benchmark-ops.md`` (the OT Summary table).
  2. Implement it in ``tests/independent_baseline.py`` as ``baseline_<name>``.
  3. Update each baseline runner's ``supports()`` set if it has a kernel.

If this test fails, do NOT silence it — fix the drift at the source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmarks.op_registry import ALL_OPS

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_op_registry_loaded_and_nonempty():
    """Sanity: catalog parses and is non-trivial."""
    assert len(ALL_OPS) >= 40, f"Catalog suspiciously small: {len(ALL_OPS)}"
    # Spot-check a few stable ops
    for op in ("matmul", "softmax", "layernorm", "flash_attention"):
        assert op in ALL_OPS, f"Catalog missing canonical op '{op}'"


def test_independent_baseline_covers_full_catalog():
    """tests/independent_baseline.py must define a baseline_<op> for every catalog op.

    This is the correctness oracle source. Drift here = silent correctness gap.
    """
    src = (REPO_ROOT / "tests" / "independent_baseline.py").read_text()
    defined = set(re.findall(r'^def baseline_(\w+)\(', src, re.MULTILINE))
    catalog = set(ALL_OPS)
    missing = catalog - defined
    extras = defined - catalog
    assert not missing, (
        f"independent_baseline.py is missing baselines for {len(missing)} catalog "
        f"ops: {sorted(missing)}.  Add baseline_<op> functions OR remove the ops "
        f"from docs/benchmark/benchmark-ops.md."
    )
    assert not extras, (
        f"independent_baseline.py defines baseline_<op> for {len(extras)} ops "
        f"NOT in the catalog: {sorted(extras)}.  Either add them to "
        f"benchmark-ops.md or delete the orphan baselines."
    )


def test_baseline_runner_supports_subset_of_catalog():
    """Every BaselineRunner.supports() set must be a subset of the catalog.

    A runner claiming to support an op that isn't in the catalog means either
    (a) the catalog is incomplete, or (b) the runner is lying. Both are bugs.
    """
    baselines_dir = REPO_ROOT / "benchmarks" / "baselines"
    catalog = set(ALL_OPS)
    violations = []

    for f in baselines_dir.glob("*.py"):
        if f.name in ("__init__.py", "base.py", "_runtime_ctx.py"):
            continue
        txt = f.read_text()

        # Collect every op name claimed in `op in (...)` literals or
        # `_SUPPORTED_OPS = frozenset({...})` declarations.
        claimed: set[str] = set()
        for m in re.finditer(r"op\s+in\s+\(([^)]*)\)", txt):
            claimed |= {a or b for a, b in re.findall(r'"(\w+)"|\'(\w+)\'', m.group(1))}
        for m in re.finditer(r"_SUPPORTED_OPS\s*=\s*frozenset\(\{(.*?)\}\)", txt, re.DOTALL):
            claimed |= {a or b for a, b in re.findall(r'"(\w+)"|\'(\w+)\'', m.group(1))}

        # Allow a small whitelist of legacy aliases that may exist in runners
        # but aren't (yet) catalog ops. Update this list deliberately, not silently.
        ALLOWED_NON_CATALOG = {"dropout"}  # historical, scheduled for catalog inclusion
        bad = (claimed - catalog) - ALLOWED_NON_CATALOG
        if bad:
            violations.append(f"{f.name}: claims unknown ops {sorted(bad)}")

    assert not violations, (
        "Baseline runners claim support for ops not in the catalog:\n  "
        + "\n  ".join(violations)
        + "\nFix: add the op to docs/benchmark/benchmark-ops.md or remove the "
        "stale entry from the runner's supports()/__SUPPORTED_OPS."
    )


def test_no_op_registry_under_arke_ir():
    """Architecture guard: ``arke/ir/`` must not host a benchmark op catalog.

    ``arke/ir/`` hosts the **kernel-schema view** (``arke/ir/ops/``) for the
    benchmark op catalog and, in the future, the Arke-IR dialect primitives
    (load/store/arith/control flow) which live at a *lower* abstraction.
    What ``arke/ir/`` must NOT do is host an *independent* benchmark op
    catalog (i.e. a list of high-level kernel names parsed from somewhere
    other than ``docs/benchmark/benchmark-ops.md``). Aliasing the SSOT
    invites silent drift; the kernel-schema view at ``arke/ir/ops/`` is
    explicitly allowed because it derives from the SSOT (see
    ``test_ir_ops_schema_covers_kernel_catalog`` below).
    """
    bad_paths = [
        REPO_ROOT / "arke" / "ir" / "op_registry.py",
        REPO_ROOT / "arke" / "ir" / "op_catalog.py",
    ]
    found = [p for p in bad_paths if p.exists()]
    assert not found, (
        f"Found benchmark op catalog under arke/ir/: {found}\n"
        "The benchmark op catalog lives at benchmarks/op_registry.py — "
        "arke/ir/ is reserved for IR dialect primitives, which are a "
        "different abstraction. Remove the offending file."
    )


# ─────────────────────────────────────────────────────────────────────────
# Kernel-schema view ↔ SSOT coverage (Stage-7+ SSOT enforcement)
# ─────────────────────────────────────────────────────────────────────────
#
# ``arke/ir/ops/registry.REGISTRY`` is a *view* over the SSOT kernel catalog
# enriched with shape_rule / template_hint / reference_impl / input_gen
# metadata for the compiler.  It is NOT itself a SSOT — by design.  These
# tests pin the relationship: every SSOT kernel must have a schema entry,
# and no extra (shadow) kernels may sneak in.

def test_ir_ops_schema_covers_kernel_catalog():
    """``arke/ir/ops/REGISTRY`` must contain a schema for every SSOT kernel."""
    from arke.ir.ops.registry import REGISTRY
    missing = [k for k in ALL_OPS if k not in REGISTRY]
    assert not missing, (
        f"IR kernel-schema view is missing schemas for {len(missing)} SSOT "
        f"kernels: {sorted(missing)}.\n"
        "Fix: add an OpSchema entry in arke/ir/ops/catalog.py for each, OR "
        "remove the kernels from docs/benchmark/benchmark-ops.md."
    )


def test_ir_ops_schema_no_shadow_kernels():
    """``arke/ir/ops/REGISTRY`` must not contain kernels absent from the SSOT."""
    from arke.ir.ops.registry import REGISTRY
    catalog = set(ALL_OPS)
    extras = [name for name in REGISTRY.names() if name not in catalog]
    assert not extras, (
        f"IR kernel-schema view has {len(extras)} kernels not in the SSOT: "
        f"{sorted(extras)}.\n"
        "Fix: add them to docs/benchmark/benchmark-ops.md (and let the SSOT "
        "parser pick them up), or delete the orphan schemas from "
        "arke/ir/ops/catalog.py."
    )


# ─────────────────────────────────────────────────────────────────────────
# Hardcoded-count scanner (P2/P3 enforcement)
# ─────────────────────────────────────────────────────────────────────────
#
# After the op-count SSOT refactor (commits feat/op-count-ssot), no file
# outside benchmark-ops.md / op_registry.py / this test should hardcode the
# integer literal 45 *in the sense of "operator count"*.  We use a targeted
# regex matched against contexts that strongly imply op-count semantics
# (e.g. ``len(REGISTRY) == 45``, ``== 45 ops``, ``45/45 ops``, ``"45 ops"``)
# rather than scanning every literal "45" (which is noisy — log timestamps,
# tensor shapes, line widths, etc.).

_HARDCODED_OP_COUNT_PATTERNS = [
    # `len(...) == 45` or `>= 45` where ... mentions ops/registry/catalog
    re.compile(r"len\(\s*[A-Za-z_][\w\.\[\]\"']*(?:OPS|REGISTRY|CATALOG|ops|registry|catalog)[^)]*\)\s*[=!<>]=?\s*45"),
    # `== 45` followed within a few chars by "ops" / "operator"
    re.compile(r"[=!<>]=?\s*45\b[^\n]{0,40}(?:\b[Oo]ps?\b|\b[Oo]perators?\b)"),
    # narrative strings like "45 ops" / "45 operators" / "45/45"
    re.compile(r"\b45\s*(?:/\s*45)?\s*(?:[Oo]ps\b|[Oo]perators?\b)"),
    # f-string / format like `45 ops` inside quotes
    re.compile(r"['\"]\s*45\s+(?:[Oo]ps\b|[Oo]perators?\b)"),
]

# Files / globs allowed to mention "45 ops" as historical record or SSOT itself.
_HARDCODED_COUNT_ALLOWLIST = {
    # SSOT document — the only place "45" is authoritative
    "docs/benchmark/benchmark-ops.md",
    # This test file itself documents the pattern
    "tests/test_ssot_op_registry.py",
    # Run logs / archived evidence — snapshots, not normative
    "benchmark_full.log",
    "benchmark_run.log",
}


def _iter_repo_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        # Skip noisy / non-source trees
        if any(seg in rel.split("/") for seg in (
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "build", "dist", ".pytest_cache", ".mypy_cache",
            "benchmark_archive", "archive", "evidence",
        )):
            continue
        # Skip binary-ish extensions
        if path.suffix in {".pyc", ".so", ".o", ".png", ".jpg", ".jpeg",
                           ".pdf", ".pkl", ".pt", ".bin", ".npz", ".npy"}:
            continue
        if rel in _HARDCODED_COUNT_ALLOWLIST:
            continue
        # Only scan source-y trees
        if not (rel.startswith("benchmarks/") or rel.startswith("arke/")
                or rel.startswith("tests/") or rel.startswith("docs/")
                or rel.startswith("scripts/") or rel.startswith("examples/")
                or rel in ("AGENTS.md",)):
            continue
        yield path, rel


@pytest.mark.xfail(
    reason="P2-P5 in feat/op-count-ssot will clear remaining hardcoded '45' "
           "references; this guard becomes strict (xfail removed) after P5.",
    strict=False,
)
def test_no_hardcoded_op_count_outside_ssot():
    """No source/doc file may hardcode the op count "45" with op semantics.

    The SSOT for the operator total is ``docs/benchmark/benchmark-ops.md``
    (OT Summary table). Code must call ``benchmarks.op_registry.total_ops()``
    or iterate ``ALL_OPS`` instead of writing ``== 45`` / ``45 ops``.

    If this test fails, fix the offending file — do not add it to the
    allowlist unless it is genuinely an archival snapshot.
    """
    violations: list[str] = []
    for path, rel in _iter_repo_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in _HARDCODED_OP_COUNT_PATTERNS:
            for m in pat.finditer(text):
                line_no = text[:m.start()].count("\n") + 1
                violations.append(f"{rel}:{line_no}: {m.group(0)!r}")

    assert not violations, (
        f"Hardcoded op count detected in {len(violations)} location(s).\n"
        "Replace with benchmarks.op_registry.total_ops() / ALL_OPS:\n  "
        + "\n  ".join(violations[:50])
        + ("\n  ... (truncated)" if len(violations) > 50 else "")
    )
