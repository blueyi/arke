# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verification-layer enhancements (D2, Leon-approved 2026-07-12).

Borrowed mechanisms from SOTA kernel-gen systems (see
docs/architecture/harness-build-vs-reuse-2026-07.md §3):

  1. robust_reward()  — CUDA Agent's discrete anti-reward-hacking schedule
                        {−1, 1, 2, 3} instead of a continuous speedup that is
                        outlier-prone and biased toward easy kernels.
  2. staged_correctness_gate() — AutoKernel's 5-stage correctness firewall
                        (smoke / shape-sweep / stability / determinism / edge),
                        all stages must pass before performance is measured.

These are *additive* Substrate utilities. They do NOT modify the frozen Façade
v1.0 tool signatures or event schema. The compile_and_profile / verify_correctness
tools and the future RL trajectory pipeline both consume them.

Design principle (from the research): the mutable code generator must be kept
separate from an immutable evaluator, and correctness must gate performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


# ─────────────────────────────────────────────────────────────────────
# 1. Robust reward (CUDA Agent, arXiv 2602.24286)
# ─────────────────────────────────────────────────────────────────────

class RobustReward(IntEnum):
    """Discrete reward tiers — anti-reward-hacking (CUDA Agent §robust reward).

    Continuous speedup reward is outlier-prone (a single huge speedup on an
    easy kernel dominates) and biases the policy toward easy wins. The discrete
    schedule rewards the *category* of outcome, not the magnitude:

      INCORRECT   = -1  : correctness failed (hard penalty, dominates all else)
      CORRECT     =  1  : correct but no meaningful speedup
      BEATS_EAGER =  2  : correct AND beats PyTorch eager by > threshold
      BEATS_BOTH  =  3  : correct AND beats BOTH eager and the strong baseline
                          (torch.compile / cuBLAS / fastest-Triton) by > threshold
    """
    INCORRECT = -1
    CORRECT = 1
    BEATS_EAGER = 2
    BEATS_BOTH = 3


def robust_reward(
    *,
    correct: bool | None,
    eager_ratio: float | None,
    strong_ratio: float | None,
    threshold: float = 0.05,
) -> RobustReward:
    """Compute the discrete robust reward for a kernel evaluation.

    Args:
        correct: V1 correctness result. None is treated as failure (honest:
            an unverifiable kernel earns no reward).
        eager_ratio: speedup vs PyTorch eager (eager_latency / kernel_latency).
            > 1.0 means the kernel is faster than eager.
        strong_ratio: speedup vs the strong baseline (torch.compile / cuBLAS /
            fastest-Triton). > 1.0 means faster than the strong reference.
        threshold: minimum relative speedup to count as "beating" (default 5%,
            matching CUDA Agent's ≥5% criterion).

    Returns:
        RobustReward tier.

    Anti-hacking rationale: correctness is checked FIRST and dominates; a fast
    but wrong kernel scores -1, never a positive reward. Speedup tiers are
    categorical so the policy cannot farm a single outlier.
    """
    if not correct:
        return RobustReward.INCORRECT
    beats_eager = eager_ratio is not None and eager_ratio > (1.0 + threshold)
    beats_strong = strong_ratio is not None and strong_ratio > (1.0 + threshold)
    if beats_eager and beats_strong:
        return RobustReward.BEATS_BOTH
    if beats_eager:
        return RobustReward.BEATS_EAGER
    return RobustReward.CORRECT


# ─────────────────────────────────────────────────────────────────────
# 2. Staged correctness gate (AutoKernel, arXiv 2603.21331)
# ─────────────────────────────────────────────────────────────────────

class GateStage(IntEnum):
    """The 5 correctness stages, in mandatory order (all must pass)."""
    SMOKE = 1          # single small shape, does it run + roughly match
    SHAPE_SWEEP = 2    # 10+ sizes across dtypes
    STABILITY = 3      # adversarial / large-magnitude inputs, no NaN/Inf
    DETERMINISM = 4    # repeated runs bitwise-identical
    EDGE = 5           # non-power-of-2 dims, degenerate shapes


@dataclass
class StageResult:
    stage: GateStage
    passed: bool
    detail: str = ""
    max_diff: float | None = None


@dataclass
class GateReport:
    """Result of the staged correctness gate.

    all_passed is True only if every attempted stage passed. first_failure
    records where it broke (None if all passed). Performance should only be
    measured when all_passed is True — this is the firewall.
    """
    all_passed: bool
    stages: list[StageResult] = field(default_factory=list)
    first_failure: GateStage | None = None

    def summary(self) -> str:
        if self.all_passed:
            return f"all {len(self.stages)} stages passed"
        return f"failed at stage {self.first_failure} ({self.first_failure.name if self.first_failure else '?'})"


def staged_correctness_gate(
    *,
    run_candidate: Callable[[dict[str, Any]], Any],
    run_reference: Callable[[dict[str, Any]], Any],
    make_inputs: Callable[[GateStage, int], dict[str, Any]],
    allclose: Callable[[Any, Any], tuple[bool, float]],
    is_finite: Callable[[Any], bool],
    equal: Callable[[Any, Any], bool],
    stages: tuple[GateStage, ...] = tuple(GateStage),
    shape_sweep_count: int = 10,
) -> GateReport:
    """Run AutoKernel's 5-stage correctness firewall.

    All callbacks are injected so this module stays backend/framework-agnostic
    (the caller supplies torch/numpy-specific comparison logic). Returns a
    GateReport; performance must only be measured when report.all_passed.

    Args:
        run_candidate: (inputs) -> candidate output for a given input dict.
        run_reference: (inputs) -> reference (golden) output.
        make_inputs: (stage, variant_index) -> input dict for that stage/variant.
        allclose: (candidate, reference) -> (is_close, max_abs_diff).
        is_finite: (output) -> True if no NaN/Inf.
        equal: (a, b) -> exact equality (for determinism check).
        stages: which stages to run (default all 5, in order).
        shape_sweep_count: number of variants for the SHAPE_SWEEP stage.

    Design: stages run in order; the first failure short-circuits (no point
    measuring perf or later stages on a broken kernel). This is the
    correctness-before-performance gate that every SOTA system enforces.
    """
    results: list[StageResult] = []

    for stage in stages:
        if stage == GateStage.SHAPE_SWEEP:
            # Multiple variants; all must pass.
            worst_diff = 0.0
            failed_variant = None
            for i in range(shape_sweep_count):
                inp = make_inputs(stage, i)
                cand = run_candidate(inp)
                ref = run_reference(inp)
                ok, diff = allclose(cand, ref)
                worst_diff = max(worst_diff, diff if diff is not None else 0.0)
                if not ok:
                    failed_variant = i
                    break
            passed = failed_variant is None
            detail = ("all variants match" if passed
                      else f"variant {failed_variant} mismatch")
            results.append(StageResult(stage, passed, detail, worst_diff))

        elif stage == GateStage.DETERMINISM:
            # Run 3× on the same input, require bitwise-identical.
            inp = make_inputs(stage, 0)
            out0 = run_candidate(inp)
            out1 = run_candidate(inp)
            out2 = run_candidate(inp)
            passed = equal(out0, out1) and equal(out1, out2)
            results.append(StageResult(
                stage, passed,
                "3 runs identical" if passed else "non-deterministic output"))

        elif stage == GateStage.STABILITY:
            # Adversarial / large-magnitude inputs; require finite + close.
            inp = make_inputs(stage, 0)
            cand = run_candidate(inp)
            ref = run_reference(inp)
            finite = is_finite(cand)
            ok, diff = allclose(cand, ref)
            passed = finite and ok
            detail = ("stable" if passed
                      else ("NaN/Inf produced" if not finite else "mismatch"))
            results.append(StageResult(stage, passed, detail, diff))

        else:
            # SMOKE and EDGE: single input, run + compare.
            inp = make_inputs(stage, 0)
            cand = run_candidate(inp)
            ref = run_reference(inp)
            ok, diff = allclose(cand, ref)
            results.append(StageResult(stage, ok,
                                       "match" if ok else "mismatch", diff))

        if not results[-1].passed:
            return GateReport(all_passed=False, stages=results,
                              first_failure=stage)

    return GateReport(all_passed=True, stages=results, first_failure=None)


# ─────────────────────────────────────────────────────────────────────
# 3. Reflexion error-trace feedback (GEAK, arXiv 2507.23194)
# ─────────────────────────────────────────────────────────────────────

# Error categories the reflector recognizes, each with a corrective hint.
# GEAK's insight: an error trace fed back with a targeted hint lets the LLM
# self-correct far better than a bare stack trace.
_REFLEXION_HINTS: dict[str, str] = {
    "compile": (
        "The generated kernel failed to COMPILE. Common causes: SSA value "
        "redefinition, undeclared variables, shared-memory over-allocation, "
        "or an illegal tile/block config. Re-read the error, fix the specific "
        "line, and prefer a smaller/simpler tiling if the config was illegal."
    ),
    "correctness": (
        "The kernel COMPILED but produced WRONG results (V1 failed). Common "
        "causes: incorrect index arithmetic, missing boundary guards for "
        "non-divisible shapes, a divergent __syncthreads() (threads that "
        "early-return skip the barrier = UB), or wrong reduction order. Check "
        "boundary handling and barrier placement first."
    ),
    "performance": (
        "The kernel is CORRECT but SLOW (V2 below target). Consider: larger "
        "tiles to amortize memory traffic, tensor-core path for matmul, "
        "shape-adaptive block size, or vectorized loads. Do NOT sacrifice "
        "correctness for speed."
    ),
    "timeout": (
        "The kernel TIMED OUT or hung. Likely an infinite loop, a deadlocked "
        "barrier (divergent __syncthreads), or a grossly oversized launch. "
        "Reduce the work per thread and verify all threads reach every barrier."
    ),
}


def classify_failure(stage: str, error_message: str) -> str:
    """Map a failure to a reflexion category (compile/correctness/performance/timeout)."""
    msg = (error_message or "").lower()
    if stage in _REFLEXION_HINTS:
        return stage
    if "timeout" in msg or "timed out" in msg or "hang" in msg:
        return "timeout"
    if "compil" in msg or "nvcc" in msg or "ptx" in msg or "redefinition" in msg:
        return "compile"
    if "correct" in msg or "mismatch" in msg or "allclose" in msg:
        return "correctness"
    return "compile"  # default: treat unknown as a compile-class problem


def reflexion_feedback(*, stage: str, error_message: str,
                       attempt: int = 1, max_attempts: int = 3) -> str:
    """Build a GEAK-style corrective feedback message for the LLM.

    Turns a raw failure into an actionable reflection the Agent can act on in
    the next turn, including the error category, a targeted hint, and the
    retry budget. Returns a string suitable for injection as a tool/user
    message in the optimization loop.
    """
    category = classify_failure(stage, error_message)
    hint = _REFLEXION_HINTS.get(category, _REFLEXION_HINTS["compile"])
    trimmed = (error_message or "").strip()
    if len(trimmed) > 800:
        trimmed = trimmed[:400] + "\n...[trimmed]...\n" + trimmed[-400:]
    return (
        f"[REFLEXION — attempt {attempt}/{max_attempts}, category={category}]\n"
        f"{hint}\n\n"
        f"Error trace:\n{trimmed}\n\n"
        f"Propose a corrected decision that addresses the root cause above. "
        f"If you have retried the same approach twice, try a fundamentally "
        f"different tiling/algorithm instead of another incremental tweak."
    )


__all__ = [
    "RobustReward",
    "robust_reward",
    "GateStage",
    "StageResult",
    "GateReport",
    "staged_correctness_gate",
    "classify_failure",
    "reflexion_feedback",
]
