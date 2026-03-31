# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — System prompts and prompt construction.

Builds the system prompt for LLM optimization sessions.
The prompt teaches the LLM how to use Arke's tool-use protocol.
"""

from __future__ import annotations

from typing import Any


def build_system_prompt(
    hw_profile: dict[str, Any],
    budget_decisions: int = 50,
    budget_compiles: int = 10,
    target_performance: float = 0.7,
) -> str:
    """Build the system prompt for an optimization session.

    Args:
        hw_profile: Hardware profile dict (from ArkeEnv)
        budget_decisions: Max optimization decisions
        budget_compiles: Max compile+profile attempts
        target_performance: Target as ratio of vendor baseline (0.7 = 70% of cuBLAS)
    """
    hw_name = hw_profile.get("display_name", hw_profile.get("name", "unknown"))
    sm_limit = hw_profile.get("constraints", {}).get("max_shared_memory_per_block", 0)
    warp_size = hw_profile.get("constraints", {}).get("warp_size", 32)
    peak_f16 = hw_profile.get("peak_tflops", {}).get("f16", 0)
    matrix_shapes = hw_profile.get("matrix_unit", {}).get("shapes", [])
    cu_count = hw_profile.get("compute_units", 0)

    matrix_info = ""
    if matrix_shapes:
        shapes_str = ", ".join(f"{s}" for s in matrix_shapes)
        matrix_info = f"\n- Tensor Core shapes: {shapes_str} — align tile sizes to these for best throughput"

    return f"""You are an expert GPU kernel optimizer working with the Arke compiler toolchain.

## Your Role

You make optimization DECISIONS. The compiler validates and executes them.
You do NOT write code — you describe strategy through structured decisions.
Every decision must include a rationale explaining your reasoning.

## Hardware Target

- **{hw_name}**
- Compute units: {cu_count}
- Shared memory per block: {sm_limit} bytes
- Warp size: {warp_size}
- Peak F16 TFLOPS: {peak_f16}{matrix_info}

## Workflow

1. **Understand**: Call `analyze_compute()` to understand the computation characteristics
2. **Plan**: Call `list_legal_actions()` to see available optimizations
3. **Decide**: Call `apply_decision()` one at a time — each decision is immediately validated
4. **Verify**: Call `verify_correctness()` to confirm numerical accuracy
5. **Measure**: Call `compile_and_profile()` to get actual GPU performance
6. **Iterate**: If performance is insufficient, `rollback()` and try alternative strategies

## Decision Priority

For compute-bound kernels (high arithmetic intensity):
  fusion → tiling (align with tensor core) → memory placement → parallelization

For memory-bound kernels (low arithmetic intensity):
  fusion → memory placement → tiling (maximize coalescing) → parallelization

## Budget

- **{budget_decisions}** optimization decisions (apply_decision calls)
- **{budget_compiles}** compile+profile attempts (expensive GPU operations)
- Target: ≥ {target_performance:.0%} of vendor baseline (cuBLAS/PyTorch)
- Use `checkpoint()` before risky decisions, `rollback()` when a direction fails

## Key Principles

1. **Tile sizes should be powers of 2** and align with warp size ({warp_size}) and tensor core shapes
2. **Fuse elementwise ops into compute ops** (epilogue fusion) to eliminate intermediate global memory writes
3. **Place frequently reused data in shared memory** — check reuse factor before placing
4. **Check shared memory usage** after tiling — must stay under {sm_limit} bytes
5. **Validate before profiling** — `verify_correctness()` is cheaper than `compile_and_profile()`
6. **Every decision needs a rationale** — explain WHY, not just WHAT

## Error Handling

- If `apply_decision()` fails validation → automatic rollback, read the violation message
- If `verify_correctness()` fails → rollback to last correct state, check accumulation dtype and boundary masking
- If performance regresses → consider rollback, analyze what changed

You will receive the kernel definition and can begin optimization."""


def build_initial_user_message(
    kernel_name: str,
    semantic_ir_summary: str,
    auto_analysis: dict[str, Any],
) -> str:
    """Build the initial user message that kicks off optimization."""
    bottleneck = auto_analysis.get("bottleneck", "unknown")
    flops = auto_analysis.get("total_flops", 0)
    fusion_opps = auto_analysis.get("fusion_opportunities", [])

    fusions_str = ""
    if fusion_opps:
        fusions_str = "\n\nFusion opportunities:\n"
        for f in fusion_opps:
            fusions_str += f"  - {f.get('nodes', [])}: {f.get('type', '')} — {f.get('reason', '')}\n"

    return f"""Optimize the kernel **{kernel_name}** for maximum GPU performance.

Computation:
{semantic_ir_summary}

Quick analysis:
- Total FLOPs: {flops:,}
- Bottleneck: {bottleneck}{fusions_str}

Begin optimization. Start with `analyze_compute()` or `list_legal_actions()` to plan your strategy."""
