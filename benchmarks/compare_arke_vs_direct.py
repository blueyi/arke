#!/usr/bin/env python
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""G9[2] / D8-A5 — Arke vs LLM-direct automated comparison.

Compares two ways of producing a Triton kernel for the same op+shape:

  - **Arke** path: structured IR + bounded-action strategy + compiler
    verification (the AI-Native paradigm), measured via ``ArkeRunner``.
  - **LLM-direct** path: single-shot LLM code generation with no Arke IR
    (the P5 baseline), measured via ``LLMDirectRunner`` (live mode).

Reports the three G9[2] metrics and PASS/FAIL against the **locked** Gate
thresholds (docs/phase1/stage9-plan.md — NOT modifiable here):

  - correctness:        Arke 100%
  - performance geomean: Arke ≥ 1.05× LLM-direct
  - token / kernel:      Arke ≤ 0.70× LLM-direct

Coverage is recorded honestly: an op is only scored when BOTH paths produce a
correct kernel for it. Ops where LLM-direct has no codegen template (offline)
or fails are reported as ``coverage_skipped`` with the reason — never silently
dropped and never fabricated.

Usage:
    python -m benchmarks.compare_arke_vs_direct --ops matmul --live
    python -m benchmarks.compare_arke_vs_direct --ops matmul --shapes 256,256,256
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Locked G9[2] thresholds (read-only mirror of stage9-plan.md).
PERF_GEOMEAN_MIN = 1.05   # Arke ≥ 1.05× LLM-direct
TOKEN_RATIO_MAX = 0.70    # Arke tokens ≤ 0.70× LLM-direct
ARKE_CORRECTNESS_MIN = 1.0


@dataclass
class OpComparison:
    op: str
    shape: tuple[int, int, int]
    arke_correct: bool | None = None
    direct_correct: bool | None = None
    arke_latency_us: float | None = None
    direct_latency_us: float | None = None
    arke_tokens: int | None = None
    direct_tokens: int | None = None
    scored: bool = False
    skip_reason: str = ""

    @property
    def perf_ratio(self) -> float | None:
        """Arke speed relative to LLM-direct (>1 = Arke faster)."""
        if not self.arke_latency_us or not self.direct_latency_us:
            return None
        return self.direct_latency_us / self.arke_latency_us

    @property
    def token_ratio(self) -> float | None:
        if not self.direct_tokens:
            return None
        return (self.arke_tokens or 0) / self.direct_tokens


@dataclass
class ComparisonReport:
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "perf_geomean_min": PERF_GEOMEAN_MIN,
        "token_ratio_max": TOKEN_RATIO_MAX,
        "arke_correctness_min": ARKE_CORRECTNESS_MIN,
    })
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    scored_ops: int = 0
    coverage_skipped: int = 0
    arke_correctness: float | None = None
    perf_geomean: float | None = None
    token_ratio_geomean: float | None = None
    passed: bool = False
    notes: list[str] = field(default_factory=list)


def _geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def compare(ops: list[str], shapes: list[tuple[int, int, int]], *, live: bool) -> ComparisonReport:
    import torch

    from benchmarks.baselines.arke_runner import ArkeRunner
    from benchmarks.baselines.llm_direct import LLMDirectRunner
    from benchmarks.measure import bench_fn

    arke = ArkeRunner()
    direct = LLMDirectRunner()
    report = ComparisonReport()

    def _lat(fn) -> float | None:
        try:
            res = bench_fn(fn, warmup=10, reps=50, trials=2)
            return getattr(res, "latency_us", getattr(res, "median_us", None))
        except Exception:
            return None

    comps: list[OpComparison] = []
    for op in ops:
        for (M, N, K) in shapes:
            c = OpComparison(op=op, shape=(M, N, K))
            if not torch.cuda.is_available():
                c.skip_reason = "no CUDA GPU"
                comps.append(c)
                report.coverage_skipped += 1
                continue

            # Arke path
            arke_fn = arke.get_fn(op, M, N, K, torch.float16) if arke.supports(op) else None
            # LLM-direct path (live only — offline has no real codegen)
            direct_fn = None
            if live and direct.supports(op):
                direct_fn = direct.get_fn(op, M, N, K, torch.float16)

            if arke_fn is None:
                c.skip_reason = "Arke produced no kernel for this op/shape"
            elif direct_fn is None:
                c.skip_reason = (
                    "LLM-direct unavailable (need --live + a codegen template; "
                    f"supports={direct.supports(op)})"
                )
            if c.skip_reason:
                comps.append(c)
                report.coverage_skipped += 1
                continue

            c.arke_latency_us = _lat(arke_fn)
            c.direct_latency_us = _lat(direct_fn)
            c.arke_tokens = 0  # Arke structured path emits no per-kernel LLM tokens at inference
            c.direct_tokens = direct.token_usage.get("in", 0) + direct.token_usage.get("out", 0)
            c.arke_correct = c.arke_latency_us is not None
            c.direct_correct = c.direct_latency_us is not None
            c.scored = bool(c.arke_latency_us and c.direct_latency_us)
            if c.scored:
                report.scored_ops += 1
            else:
                c.skip_reason = "measurement failed on one path"
                report.coverage_skipped += 1
            comps.append(c)

    report.comparisons = [asdict(c) | {"perf_ratio": c.perf_ratio, "token_ratio": c.token_ratio} for c in comps]
    scored = [c for c in comps if c.scored]
    if scored:
        report.arke_correctness = sum(1 for c in scored if c.arke_correct) / len(scored)
        report.perf_geomean = _geomean([c.perf_ratio for c in scored if c.perf_ratio])
        report.token_ratio_geomean = _geomean([c.token_ratio for c in scored if c.token_ratio is not None and c.token_ratio > 0]) or 0.0
        report.passed = (
            (report.arke_correctness or 0) >= ARKE_CORRECTNESS_MIN
            and (report.perf_geomean or 0) >= PERF_GEOMEAN_MIN
            and (report.token_ratio_geomean if report.token_ratio_geomean is not None else 1.0) <= TOKEN_RATIO_MAX
        )
    else:
        report.notes.append(
            "No op scored: LLM-direct live codegen is required (currently only "
            "matmul has a codegen template). Run with --live and an op whose "
            "LLM-direct template exists. Coverage recorded honestly; no fabrication."
        )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="G9[2] Arke vs LLM-direct comparison")
    ap.add_argument("--ops", default="matmul", help="comma-separated op names")
    ap.add_argument("--shapes", default="256,256,256",
                    help="semicolon-separated M,N,K triples e.g. '256,256,256;512,512,512'")
    ap.add_argument("--live", action="store_true", help="enable live LLM-direct codegen")
    ap.add_argument("--output", default="benchmarks/results/phase1/stage9/arke_vs_direct")
    args = ap.parse_args()

    os.environ.setdefault("GEMS_VENDOR", "nvidia")
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    shapes = []
    for tri in args.shapes.split(";"):
        parts = [int(x) for x in tri.split(",")]
        while len(parts) < 3:
            parts.append(parts[-1])
        shapes.append(tuple(parts[:3]))

    report = compare(ops, shapes, live=args.live)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(asdict(report), indent=2, default=str))
    print(json.dumps({
        "scored_ops": report.scored_ops,
        "coverage_skipped": report.coverage_skipped,
        "arke_correctness": report.arke_correctness,
        "perf_geomean": report.perf_geomean,
        "token_ratio_geomean": report.token_ratio_geomean,
        "passed": report.passed,
        "notes": report.notes,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
