# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Health check for the Golden Kernel ladder.

Asserts the integrity of the ladder against the catalog:

  1. **Coverage** — every op in the SSOT catalog resolves to *some* Golden
     Kernel under the currently-installed runner set, OR raises
     GoldenUnavailable cleanly (caught + audited at the bench_l1 layer).
     No silent fall-through.

  2. **P5 guard** — no op resolves to a P5 runner (Arke / LLM-direct).
     P5 runners are the system *under test*, never the oracle. If the
     ladder reached P5 the bench_l1 grade would compare Arke to itself,
     which is meaningless. The ladder code raises GoldenUnavailable
     proactively when it would otherwise return a P5 — this test makes
     the guarantee testable.

If this test fails, see ``docs/benchmark/golden-kernel-ladder.md`` for the
locked per-op assignments and ``benchmarks/baselines/*.py`` ``supports()``
declarations.
"""

from __future__ import annotations

import pytest

# Trigger runner module registration via the canonical loading path
# bench_l1 uses at top of file.
import benchmarks.bench_l1  # noqa: F401
from benchmarks.golden_ladder import GoldenUnavailable, golden_runner_for
from benchmarks.op_registry import ALL_OPS


def test_no_op_resolves_to_p5_runner() -> None:
    """Arke (P5) and LLM-direct (P5) must NEVER be the Golden Kernel."""
    violations: list[tuple[str, str, int]] = []
    for op in ALL_OPS:
        try:
            r = golden_runner_for(op)
        except GoldenUnavailable:
            continue  # cleanly audited path is fine
        if r.priority >= 5:
            violations.append((op, r.name, r.priority))

    assert not violations, (
        "Ladder leaked a P5 runner as Golden Kernel for: "
        + ", ".join(f"{op} -> {name} (P{p})" for op, name, p in violations)
        + ". P5 runners are the system under test, not the oracle. "
        "Add the op to a P0-P3 runner's supports() set."
    )


def test_ladder_coverage_summary() -> None:
    """Report ladder resolution per op (non-failing; for debug visibility)."""
    resolved = 0
    unavailable: list[str] = []
    for op in ALL_OPS:
        try:
            golden_runner_for(op)
            resolved += 1
        except GoldenUnavailable:
            unavailable.append(op)

    # We tolerate some unavailables (e.g. ops that depend on FlashMLA on
    # sm<9.0 hardware), but the count must be small and known.
    # Catalog size = 45. With pytorch_eager P3 covering the OT4 fallback
    # path, all 45 ops should resolve on a working dev box.
    assert resolved >= 40, (
        f"Only {resolved}/{len(ALL_OPS)} ops resolve to a Golden Kernel. "
        f"Unavailable: {unavailable}. "
        "Check baseline runner imports in bench_l1 and supports() sets."
    )


@pytest.mark.parametrize("op", ALL_OPS)
def test_each_op_resolves_or_audits_cleanly(op: str) -> None:
    """Each op either resolves or raises GoldenUnavailable with a reason."""
    try:
        r = golden_runner_for(op)
        assert r.available, f"{op}: ladder picked {r.name} but available=False"
        assert r.supports(op), f"{op}: ladder picked {r.name} but supports({op}) is False"
        assert r.priority < 5, f"{op}: leaked P{r.priority} runner {r.name}"
    except GoldenUnavailable as e:
        assert e.op == op
        assert e.reason, f"{op}: GoldenUnavailable raised without a reason string"
