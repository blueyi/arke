# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim — the **real** op_registry lives in ``benchmarks/op_registry.py``.

This file exists because agents and contributors intuitively look for
``arke/ir/op_registry.py`` first (since "IR" is the natural home for an op
catalog).  To prevent false-negative diagnoses (e.g. "op_registry is missing"),
this shim re-exports the canonical names and points readers at the SSOT.

╔════════════════════════════════════════════════════════════════════════╗
║  SINGLE SOURCE OF TRUTH (SSOT) FOR OPERATOR CATALOG                    ║
║                                                                        ║
║  Authoritative document : docs/benchmark/benchmark-ops.md              ║
║                            (the OT Summary table)                      ║
║  Authoritative parser   : benchmarks/op_registry.py                    ║
║                                                                        ║
║  All consumers MUST import from ``benchmarks.op_registry``:            ║
║      from benchmarks.op_registry import ALL_OPS, OT_OPS, OP_TIER       ║
║                                                                        ║
║  DO NOT redefine op-name lists / tier maps anywhere else.              ║
║  ``tests/test_ssot_op_registry.py`` enforces this invariant.           ║
╚════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# Re-export so `from arke.ir.op_registry import ALL_OPS` Just Works.
from benchmarks.op_registry import (  # noqa: F401
    ALL_OPS,
    OP_TIER,
    OT_OPS,
    TOTAL_OPS,
    parse_ops_md,
)

__all__ = ["ALL_OPS", "OP_TIER", "OT_OPS", "TOTAL_OPS", "parse_ops_md"]
