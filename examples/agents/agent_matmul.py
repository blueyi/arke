#!/usr/bin/env python3
"""Arke Agent Demo — a live LLM optimizes a kernel via the 8-tool Façade.

This is the first **real** LLM integration demo for Arke (P0-A, 2026-06-24).
It puts a live model (via the configured provider — the yunwu.ai OpenAI-
compatible relay by default) in the driver's seat of the Arke optimization
loop: the model inspects the hardware + op, lists legal actions, applies
decisions with @rationale, verifies correctness, and profiles real Triton
kernels on the GPU.

Run:
    source ~/.venvs/arke/bin/activate && source ~/.env.rc
    python examples/agents/agent_matmul.py

Requires LLM credentials in the environment (ANTHROPIC_API_KEY / YUNWU_API_KEY
/ OPENAI_API_KEY). See arke/agent/llm_config.py for resolution order.
"""

import json
import logging
import sys

from arke.agent.llm_config import LLMConfigError, load_from_openclaw
from arke.agent.runner import LLMRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        config = load_from_openclaw()
    except LLMConfigError as e:
        logger.error("No LLM provider configured: %s", e)
        return 1
    logger.info("Primary provider: %s", config.primary)
    logger.info("Providers: %s", list(config.providers.keys()))

    # Optimize a matmul kernel. The runner builds the ArkeEnv, wires the
    # 8-tool Façade, and drives the model through the compile→profile→adjust
    # loop. (Multi-op fused kernels will be added once the multi-node ArkeEnv
    # lands — see docs/phase1/stage8-plan.md.)
    op_name = "matmul"
    shapes = {"A": [512, 512], "B": [512, 512]}

    with LLMRunner(config, timeout=150.0) as runner:
        result = runner.optimize(
            op_name=op_name,
            shapes=shapes,
            target_hw="nvidia_ampere",
            max_turns=20,
            # bare model name → primary provider's default model; or pass
            # e.g. "yunwu/claude-sonnet-4-6" to pin one explicitly.
            model_spec=None,
        )

    print("\n" + "=" * 60)
    print(f"Model:      {result.model_used}")
    print(f"Stop:       {result.stop_reason}")
    print(f"Decisions:  {result.decisions}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Tokens:     {result.tokens_in} in / {result.tokens_out} out")
    print(f"Duration:   {result.duration_seconds}s")
    print(f"Errors:     {result.errors}")
    print("=" * 60)

    summary = result.session_summary
    print(f"\nSession state:    {summary['state']}")
    print(f"Budget:           {summary['budget']}")
    print(f"Best performance: {summary.get('best_performance')}")

    print("\nTrajectory:")
    for entry in result.trajectory:
        if entry["type"] == "action":
            params = json.dumps(entry["params"], ensure_ascii=False)[:70]
            print(f"  [{entry['step']}] {entry['tool']}({params})")

    print("\nDecisions (with @rationale):")
    for d in summary.get("decision_log", []):
        rat = (d.get("rationale") or "")[:70]
        print(f"  {d['kind']}({json.dumps(d['params'], ensure_ascii=False)[:40]}) :: {rat}")

    output_path = "examples/agents/agent_matmul_result.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nFull result saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
