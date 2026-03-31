# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Schedule Tree (Layer 2).

The Schedule Tree describes HOW to optimize, decoupled from what to compute.
Each decision carries an optional @rationale for AI learning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Rationale:
    """Natural language explanation for an optimization decision."""
    text: str
    lang: str = "en"


@dataclass
class ScheduleDecision:
    """A single optimization decision."""
    kind: str  # tile | reorder | fuse | parallel | place | vectorize | unroll
    params: dict  # Decision-specific parameters
    rationale: Optional[Rationale] = None


@dataclass
class HardwareConstraints:
    """Hardware resource constraints."""
    shared_memory_limit: int = 0
    register_limit: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32


@dataclass
class SearchSpace:
    """Defines the legal search space for AI optimization."""
    dimensions: dict[str, dict] = field(default_factory=dict)
    # e.g., {"tiling_i": {"type": "power_of_2", "range": [16, 256]}}


@dataclass
class ScheduleTree:
    """Layer 2 of Arke IR — the Schedule Tree.

    Describes optimization decisions, decoupled from computation semantics.
    """

    version: str = "0.1.0"
    target_graph: str = ""  # Reference to SemanticGraph
    target_hw: str = ""  # e.g., "nvidia_ampere", "ascend_a3"
    decisions: list[ScheduleDecision] = field(default_factory=list)
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    search_space: SearchSpace = field(default_factory=SearchSpace)

    def add_decision(self, decision: ScheduleDecision) -> None:
        """Add an optimization decision."""
        self.decisions.append(decision)

    def tile(
        self,
        loop: str,
        factors: list[int],
        rationale: Optional[str] = None,
    ) -> ScheduleDecision:
        """Add a tiling decision."""
        d = ScheduleDecision(
            kind="tile",
            params={"loop": loop, "factors": factors},
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.add_decision(d)
        return d

    def reorder(
        self,
        order: list[str],
        rationale: Optional[str] = None,
    ) -> ScheduleDecision:
        """Add a loop reorder decision."""
        d = ScheduleDecision(
            kind="reorder",
            params={"order": order},
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.add_decision(d)
        return d

    def fuse(
        self,
        ops: list[str],
        fusion_type: str = "epilogue",
        rationale: Optional[str] = None,
    ) -> ScheduleDecision:
        """Add an operator fusion decision."""
        d = ScheduleDecision(
            kind="fuse",
            params={"ops": ops, "type": fusion_type},
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.add_decision(d)
        return d

    def parallel(
        self,
        loops: list[str],
        mapping: dict[str, str],
        rationale: Optional[str] = None,
    ) -> ScheduleDecision:
        """Add a parallel mapping decision."""
        d = ScheduleDecision(
            kind="parallel",
            params={"loops": loops, "mapping": mapping},
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.add_decision(d)
        return d

    def place(
        self,
        tensor: str,
        memory: str,
        rationale: Optional[str] = None,
    ) -> ScheduleDecision:
        """Add a memory placement decision."""
        d = ScheduleDecision(
            kind="place",
            params={"tensor": tensor, "memory": memory},
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.add_decision(d)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=indent)

    def to_file(self, path: str) -> None:
        """Save to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
