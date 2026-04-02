# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Strategy IR (Layer 2).

The Strategy IR describes HOW to optimize, decoupled from what to compute.
Each decision carries an optional @rationale for explainability and learning.

Terminology: "Strategy" (not "Schedule") — emphasizes that the LLM is making
strategic optimization decisions, not writing execution schedules.
See docs/design/naming-system.md for rationale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Rationale:
    """Natural language explanation for an optimization decision."""
    text: str
    lang: str = "en"


@dataclass
class Decision:
    """A single optimization decision.

    Kinds: tile | reorder | fuse | parallel | place | vectorize | unroll
           | algorithm | launch_config | autotune
    Levels: 1=strategy (Phase 1), 2=loop (Phase 2), 3=hardware (Phase 3)

    ``launch_config`` params schema::

        {
            "num_warps": 4,        # int or list[int] for autotune candidates
            "num_stages": 3,       # int or list[int] for autotune candidates
            "block_sizes": {       # optional: override default BLOCK_SIZE values
                "BLOCK_M": 64,     # int or list[int]
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            }
        }

    ``autotune`` params schema::

        {
            "configs": [
                {"num_warps": 2, "num_stages": 3, "block_sizes": {"BLOCK_SIZE": 1024}},
                {"num_warps": 4, "num_stages": 3, "block_sizes": {"BLOCK_SIZE": 4096}},
                {"num_warps": 8, "num_stages": 3, "block_sizes": {"BLOCK_SIZE": 8192}},
            ],
            "key": ["n_elements"],  # autotune key dimensions
        }
    """
    kind: str
    params: dict
    rationale: Rationale | None = None
    step: int = 0  # Auto-assigned by StrategyIR
    level: int = 1  # Decision abstraction level (1=strategy, 2=loop, 3=hardware)


@dataclass
class HardwareConstraints:
    """Hardware resource constraints."""
    shared_memory_limit: int = 0
    register_limit: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32


@dataclass
class StrategyIR:
    """Layer 2 of Arke IR — the Strategy IR.

    Describes optimization decisions, decoupled from computation semantics.
    """

    version: str = "0.2.0"
    kernel_id: str = ""          # Reference to SemanticIR
    target_hw: str = ""          # e.g., "nvidia_ampere", "ascend_a3"
    decisions: list[Decision] = field(default_factory=list)
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)

    @property
    def decision_count(self) -> int:
        """Return the number of optimization decisions."""
        return len(self.decisions)

    def add_decision(self, decision: Decision) -> Decision:
        """Add an optimization decision, auto-assigning step number."""
        decision.step = len(self.decisions) + 1
        self.decisions.append(decision)
        return decision

    def pop_decisions(self, n: int = 1) -> list[Decision]:
        """Remove and return the last N decisions (for rollback)."""
        removed = []
        for _ in range(min(n, len(self.decisions))):
            removed.append(self.decisions.pop())
        return removed

    def tile(self, loop: str, factors: list[int],
             rationale: str | None = None) -> Decision:
        """Add a tile decision for the given loop with specified factors."""
        return self.add_decision(Decision(
            kind="tile",
            params={"loop": loop, "factors": factors},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def reorder(self, order: list[str],
                rationale: str | None = None) -> Decision:
        """Add a reorder decision with the specified loop order."""
        return self.add_decision(Decision(
            kind="reorder",
            params={"order": order},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def fuse(self, ops: list[str], fusion_type: str = "epilogue",
             rationale: str | None = None) -> Decision:
        """Add a fusion decision for the specified operators."""
        return self.add_decision(Decision(
            kind="fuse",
            params={"ops": ops, "type": fusion_type},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def parallel(self, loops: list[str], mapping: dict[str, str],
                 rationale: str | None = None) -> Decision:
        """Add a parallelism decision mapping loops to hardware dimensions."""
        return self.add_decision(Decision(
            kind="parallel",
            params={"loops": loops, "mapping": mapping},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def place(self, tensor: str, memory: str,
              rationale: str | None = None) -> Decision:
        """Add a memory placement decision for a tensor."""
        return self.add_decision(Decision(
            kind="place",
            params={"tensor": tensor, "memory": memory},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def summary(self) -> str:
        """Human-readable summary for compact context."""
        lines = [f"Strategy for {self.kernel_id} on {self.target_hw}:"]
        for d in self.decisions:
            r = f" — {d.rationale.text}" if d.rationale else ""
            lines.append(f"  #{d.step} {d.kind}({d.params}){r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the strategy IR to a plain dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the strategy IR to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str) -> None:
        """Save the strategy IR to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> StrategyIR:
        """Deserialize a StrategyIR from a plain dict."""
        ir = cls(
            version=data.get("version", "0.1.0"),
            kernel_id=data.get("kernel_id", ""),
            target_hw=data.get("target_hw", ""),
        )
        for d in data.get("decisions", []):
            rationale = None
            if d.get("rationale"):
                rationale = Rationale(**d["rationale"])
            ir.add_decision(Decision(
                kind=d["kind"],
                params=d["params"],
                rationale=rationale,
            ))
        return ir

    @classmethod
    def from_json(cls, json_str: str) -> StrategyIR:
        """Deserialize a StrategyIR from a JSON string."""
        return cls.from_dict(json.loads(json_str))
