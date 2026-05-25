# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Parse benchmark-ops.md to extract the canonical operator catalog.

╔══════════════════════════════════════════════════════════════════════════╗
║  *** SSOT — SINGLE SOURCE OF TRUTH FOR THE BENCHMARK OP CATALOG ***      ║
║                                                                          ║
║  Authoritative document : docs/benchmark/benchmark-ops.md (OT table)     ║
║  Authoritative parser   : THIS FILE (benchmarks/op_registry.py)          ║
║                                                                          ║
║  Scope: this catalog enumerates the *high-level* kernel operators that   ║
║  the Arke benchmark suite measures (matmul / flash_attention / rmsnorm / ║
║  rope / ...). The total count is whatever the OT Summary table in        ║
║  benchmark-ops.md currently declares — never hardcode it.  Use           ║
║  ``total_ops()`` to query at runtime.  It is a **kernel-level**          ║
║  abstraction tied to the benchmark/baseline-runner layer — NOT a         ║
║  compiler IR concept.                                                    ║
║                                                                          ║
║  Do NOT look for this file under ``arke/ir/``.  ``arke/ir/`` is reserved ║
║  for the future Arke-IR dialect primitives (load/store/arith/control     ║
║  flow / etc.), which are a lower-level abstraction.  IR primitives       ║
║  *lower to* this catalog's ops via the compiler — the two layers are    ║
║  intentionally decoupled and must not be aliased.                        ║
║                                                                          ║
║  Every consumer (cli.py / shapes.py / bench_l1.py / baseline runners /   ║
║  benchmark tests) MUST import from here.  Hardcoded op-name lists        ║
║  elsewhere are shadow catalogs and will silently drift.                  ║
║                                                                          ║
║  Enforcement: tests/test_ssot_op_registry.py fails if any benchmark      ║
║  module diverges (e.g. baseline runner claims an op not in this list).   ║
╚══════════════════════════════════════════════════════════════════════════╝

This module is the bridge: it reads the OT Summary table from
``docs/benchmark/benchmark-ops.md`` and exposes structured data that
``benchmarks/cli.py``, ``benchmarks/shapes.py``, and all test files consume.

If benchmark-ops.md is edited (operators added/removed/moved between tiers),
every downstream consumer picks up the change automatically at import time.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Locate the document ──────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent          # benchmarks/
_REPO_ROOT = _THIS_DIR.parent                         # arke repo root
_OPS_MD = _REPO_ROOT / "docs" / "benchmark" / "benchmark-ops.md"

# ── Parser ───────────────────────────────────────────────────────────────

# Matches rows like:
#   | **OT0** | Elementwise | 12 | `relu`, `gelu`, ... | Memory-bound ... |
_OT_ROW_RE = re.compile(
    r"\|\s*\*\*OT(\d+)\*\*\s*"      # | **OT0** |
    r"\|[^|]*"                        # | Name |
    r"\|\s*(\d+)\s*"                  # | Count |
    r"\|\s*([^|]+)"                   # | Operators (backtick-delimited) |
)

_BACKTICK_RE = re.compile(r"`([^`]+)`")


def parse_ops_md(path: Path | str | None = None) -> dict[int, list[str]]:
    """Parse the OT Summary table and return ``{tier: [op_names]}``.

    Parameters
    ----------
    path : Path or str, optional
        Override path to benchmark-ops.md.  Defaults to the repo copy.

    Returns
    -------
    dict[int, list[str]]
        Mapping from OT tier (0-4) to ordered list of operator names.

    Raises
    ------
    FileNotFoundError
        If the markdown file doesn't exist.
    ValueError
        If the parsed operator count doesn't match the declared count.
    """
    md_path = Path(path) if path is not None else _OPS_MD
    text = md_path.read_text(encoding="utf-8")

    ot_ops: dict[int, list[str]] = {}

    for m in _OT_ROW_RE.finditer(text):
        tier = int(m.group(1))
        declared_count = int(m.group(2))
        ops_cell = m.group(3)
        ops = _BACKTICK_RE.findall(ops_cell)

        if len(ops) != declared_count:
            raise ValueError(
                f"OT{tier}: declared count={declared_count} but parsed "
                f"{len(ops)} operators: {ops}"
            )

        ot_ops[tier] = ops

    if not ot_ops:
        raise ValueError(f"No OT rows found in {md_path}")

    return ot_ops


def build_op_tier(ot_ops: dict[int, list[str]] | None = None) -> dict[str, int]:
    """Build a flat ``{op_name: tier}`` mapping from parsed OT data."""
    if ot_ops is None:
        ot_ops = parse_ops_md()
    return {op: tier for tier, ops in ot_ops.items() for op in ops}


def all_ops(ot_ops: dict[int, list[str]] | None = None) -> list[str]:
    """Return a flat list of all operator names, ordered by tier."""
    if ot_ops is None:
        ot_ops = parse_ops_md()
    return [op for tier in sorted(ot_ops) for op in ot_ops[tier]]


# ── Public SSOT query API ─────────────────────────────────────────────────
#
# These functions are the *only* sanctioned way to ask "how many ops?" or
# "what ops are in OT2?". Hardcoding numbers (e.g. ``== 45``) anywhere else
# in the repo is a SSOT violation and will be caught by
# ``tests/test_ssot_op_registry.py``.

def total_ops() -> int:
    """Authoritative operator total.

    Reads from the parsed OT Summary table — never hardcode this number.

    Returns
    -------
    int
        Count of operators currently declared in benchmark-ops.md.
    """
    return len(ALL_OPS)


def ops_by_tier(tier: int) -> list[str]:
    """Return the operator list for a given OT tier.

    Parameters
    ----------
    tier : int
        OT tier index (0-4).

    Returns
    -------
    list[str]
        Operators in that tier, in declaration order. Empty list if tier
        is not present.
    """
    return list(OT_OPS.get(tier, []))


def tier_counts() -> dict[int, int]:
    """Return ``{tier: count}`` for reporting / dashboard / log lines.

    Returns
    -------
    dict[int, int]
        Mapping from OT tier to operator count, sorted by tier.
    """
    return {t: len(ops) for t, ops in sorted(OT_OPS.items())}


# ── Module-level singletons (computed once at import) ────────────────────

OT_OPS: dict[int, list[str]] = parse_ops_md()
"""Canonical OT → operator list, parsed from benchmark-ops.md."""

OP_TIER: dict[str, int] = build_op_tier(OT_OPS)
"""Canonical operator → OT tier mapping."""

ALL_OPS: list[str] = all_ops(OT_OPS)
"""Flat list of all operators, ordered by tier."""

TOTAL_OPS: int = len(ALL_OPS)
"""Total number of operators in the catalog."""
