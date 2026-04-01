#!/usr/bin/env python3
"""Arke Agent Demo — LLM optimizes a matmul kernel via tool-use.

This is the first real LLM integration test for Arke.
"""

import json
import logging
import sys

from arke.agent.llm_config import load_from_openclaw
from arke.agent.runner import LLMRunner
from arke.ir.builder import KernelBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_matmul_relu():
    """Build a fused matmul + relu kernel for optimization."""
    b = KernelBuilder("fused_matmul_relu")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    return b.build()


def main():
    # Load LLM config from OpenClaw
    config = load_from_openclaw()
    logger.info(f"Primary model: {config.primary}")
    logger.info(f"Providers: {list(config.providers.keys())}")

    # Build kernel
    ir = build_matmul_relu()
    logger.info(f"Kernel: {ir.kernel_id}, params: {len(ir.params)}, nodes: {len(ir.nodes)}")

    # Run optimization — use sonnet for speed (opus is too slow for long conversations)
    with LLMRunner(config, timeout=300.0) as runner:
        result = runner.optimize(
            semantic_ir=ir,
            target_hw="nvidia_ampere",
            max_turns=25,
            model_spec="api-proxy-claude/claude-sonnet-4-6",
        )

    # Report results
    print("\n" + "=" * 60)
    print(f"Model: {result.model_used}")
    print(f"Decisions: {result.decisions}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Tokens: {result.tokens_in} in / {result.tokens_out} out")
    print(f"Duration: {result.duration_seconds}s")
    print(f"Errors: {result.errors}")
    print("=" * 60)

    # Print strategy decisions
    summary = result.session_summary
    print(f"\nSession state: {summary['state']}")
    print(f"Budget: {summary['budget']}")

    # Print trajectory
    print("\nTrajectory:")
    for entry in result.trajectory:
        if entry["type"] == "action":
            print(f"  [{entry['step']}] {entry['tool']}({json.dumps(entry['params'], ensure_ascii=False)[:80]})")

    # Save full result
    output_path = "examples/agent_matmul_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": result.model_used,
            "decisions": result.decisions,
            "tool_calls": result.tool_calls,
            "tokens": {"in": result.tokens_in, "out": result.tokens_out},
            "duration": result.duration_seconds,
            "trajectory": result.trajectory,
            "session_summary": result.session_summary,
            "errors": result.errors,
        }, f, indent=2, default=str)
    print(f"\nFull result saved to {output_path}")


if __name__ == "__main__":
    main()
