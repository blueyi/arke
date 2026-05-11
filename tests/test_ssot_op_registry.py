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


def test_arke_ir_shim_redirects_to_canonical():
    """The arke/ir/op_registry.py shim must re-export the same ALL_OPS.

    This shim prevents agents from concluding "op_registry doesn't exist" when
    they look in the obvious arke/ir/ location and find nothing.
    """
    from arke.ir.op_registry import ALL_OPS as SHIM_ALL_OPS  # noqa: PLC0415

    assert SHIM_ALL_OPS is ALL_OPS or list(SHIM_ALL_OPS) == list(ALL_OPS), (
        "arke/ir/op_registry.py shim drifted from benchmarks/op_registry.py — "
        "the shim must re-export, not redefine."
    )
