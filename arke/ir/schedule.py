# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — ScheduleIR (Layer 2).

Compiler-generated scheduling / hardware-near mapping layer.
Bridges StrategyIR decisions into an explicit schedule structure that can later
lower into InstructionIR and backend emission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from arke.ir.strategy import Decision, HardwareConstraints, Rationale


@dataclass
class LoopNest:
    """A scheduled loop nest dimension."""
    loop: str
    tile_factors: list[int] = field(default_factory=list)
    order: int | None = None
    mapping: str | None = None  # e.g. block_x / block_y / warp / thread_x
    vector_width: int | None = None
    unroll_factor: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"loop": self.loop}
        if self.tile_factors:
            d["tile_factors"] = list(self.tile_factors)
        if self.order is not None:
            d["order"] = self.order
        if self.mapping is not None:
            d["mapping"] = self.mapping
        if self.vector_width is not None:
            d["vector_width"] = self.vector_width
        if self.unroll_factor is not None:
            d["unroll_factor"] = self.unroll_factor
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopNest":
        return cls(
            loop=d["loop"],
            tile_factors=list(d.get("tile_factors", [])),
            order=d.get("order"),
            mapping=d.get("mapping"),
            vector_width=d.get("vector_width"),
            unroll_factor=d.get("unroll_factor"),
        )


@dataclass
class MemoryPlacement:
    """Scheduled memory placement for a tensor or tile."""
    tensor: str
    memory: str

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "memory": self.memory}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryPlacement":
        return cls(tensor=d["tensor"], memory=d["memory"])


@dataclass
class ResourceBinding:
    """Concrete resource binding derived from compute/resource decisions."""
    warps: int | None = None
    num_stages: int | None = None
    shared_memory: int | None = None
    pipeline_depth: int | None = None
    threads_per_block: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.warps is not None:
            d["warps"] = self.warps
        if self.num_stages is not None:
            d["num_stages"] = self.num_stages
        if self.shared_memory is not None:
            d["shared_memory"] = self.shared_memory
        if self.pipeline_depth is not None:
            d["pipeline_depth"] = self.pipeline_depth
        if self.threads_per_block is not None:
            d["threads_per_block"] = self.threads_per_block
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResourceBinding":
        return cls(
            warps=d.get("warps"),
            num_stages=d.get("num_stages"),
            shared_memory=d.get("shared_memory"),
            pipeline_depth=d.get("pipeline_depth"),
            threads_per_block=d.get("threads_per_block"),
        )


@dataclass
class FusionGroup:
    """A fused op group carried into scheduling."""
    ops: list[str]
    fusion_type: str = "epilogue"

    def to_dict(self) -> dict[str, Any]:
        return {"ops": list(self.ops), "type": self.fusion_type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FusionGroup":
        return cls(ops=list(d.get("ops", [])), fusion_type=d.get("type", "epilogue"))


@dataclass
class ScheduleDecisionRecord:
    """Traceability record from StrategyIR decision to ScheduleIR effect."""
    source_kind: str
    source_step: int
    effect: str
    rationale: Rationale | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_kind": self.source_kind,
            "source_step": self.source_step,
            "effect": self.effect,
        }
        if self.rationale is not None:
            d["rationale"] = {"text": self.rationale.text, "lang": self.rationale.lang}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScheduleDecisionRecord":
        rat = None
        if d.get("rationale"):
            rat = Rationale(**d["rationale"])
        return cls(
            source_kind=d["source_kind"],
            source_step=d.get("source_step", 0),
            effect=d["effect"],
            rationale=rat,
        )


@dataclass
class ScheduleIR:
    """Layer 2 schedule representation."""
    version: str = "2.0.0"
    kernel_id: str = ""
    target_hw: str = ""
    loop_nests: list[LoopNest] = field(default_factory=list)
    placements: list[MemoryPlacement] = field(default_factory=list)
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    resources: ResourceBinding = field(default_factory=ResourceBinding)
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    provenance: list[ScheduleDecisionRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_loop(self, loop: str) -> LoopNest:
        existing = self.get_loop(loop)
        if existing is not None:
            return existing
        ln = LoopNest(loop=loop)
        self.loop_nests.append(ln)
        return ln

    def get_loop(self, loop: str) -> LoopNest | None:
        for ln in self.loop_nests:
            if ln.loop == loop:
                return ln
        return None

    def apply_decision(self, decision: Decision) -> None:
        kind = decision.kind
        params = decision.params

        if kind == "tile":
            ln = self.add_loop(params["loop"])
            ln.tile_factors = list(params.get("factors", []))
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, f"tile:{ln.loop}", decision.rationale))
            return

        if kind == "reorder":
            for idx, loop in enumerate(params.get("order", [])):
                ln = self.add_loop(loop)
                ln.order = idx
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, "reorder", decision.rationale))
            return

        if kind == "parallel":
            mapping = params.get("mapping", {})
            for loop_name, mapped in mapping.items():
                ln = self.add_loop(loop_name)
                ln.mapping = mapped
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, "parallel-map", decision.rationale))
            return

        if kind == "vectorize":
            ln = self.add_loop(params["loop"])
            ln.vector_width = params.get("width")
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, f"vectorize:{ln.loop}", decision.rationale))
            return

        if kind == "unroll":
            ln = self.add_loop(params["loop"])
            ln.unroll_factor = params.get("factor")
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, f"unroll:{ln.loop}", decision.rationale))
            return

        if kind == "place":
            self.placements.append(MemoryPlacement(
                tensor=params["tensor"],
                memory=params["memory"],
            ))
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, f"place:{params['tensor']}", decision.rationale))
            return

        if kind == "fuse":
            self.fusion_groups.append(FusionGroup(
                ops=list(params.get("ops", [])),
                fusion_type=params.get("type", "epilogue"),
            ))
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, "fuse", decision.rationale))
            return

        if kind == "compute":
            if "warps" in params:
                self.resources.warps = params["warps"]
                self.resources.threads_per_block = params["warps"] * self.constraints.warp_size
            if "num_stages" in params:
                self.resources.num_stages = params["num_stages"]
                self.resources.pipeline_depth = params["num_stages"]
            if "shared_memory" in params:
                self.resources.shared_memory = params["shared_memory"]
            self.provenance.append(ScheduleDecisionRecord(kind, decision.step, "compute", decision.rationale))
            return

        self.provenance.append(ScheduleDecisionRecord(kind, decision.step, f"ignored:{kind}", decision.rationale))

    def summary(self) -> str:
        lines = [f"Schedule for {self.kernel_id} on {self.target_hw}:"]
        if self.loop_nests:
            for ln in self.loop_nests:
                lines.append(f"  loop {ln.loop}: tile={ln.tile_factors} order={ln.order} mapping={ln.mapping}")
        if self.placements:
            for p in self.placements:
                lines.append(f"  place {p.tensor} -> {p.memory}")
        if self.resources.to_dict():
            lines.append(f"  resources: {self.resources.to_dict()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "target_hw": self.target_hw,
            "loop_nests": [x.to_dict() for x in self.loop_nests],
            "placements": [x.to_dict() for x in self.placements],
            "fusion_groups": [x.to_dict() for x in self.fusion_groups],
            "resources": self.resources.to_dict(),
            "provenance": [x.to_dict() for x in self.provenance],
        }
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        c = self.constraints
        constraints = {
            "shared_memory_limit": c.shared_memory_limit,
            "register_limit": c.register_limit,
            "max_threads_per_block": c.max_threads_per_block,
            "warp_size": c.warp_size,
        }
        if any(v not in (0, 32) for v in constraints.values()):
            d["constraints"] = constraints
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleIR":
        constraints = HardwareConstraints(**data.get("constraints", {}))
        return cls(
            version=data.get("version", "2.0.0"),
            kernel_id=data.get("kernel_id", ""),
            target_hw=data.get("target_hw", ""),
            loop_nests=[LoopNest.from_dict(x) for x in data.get("loop_nests", [])],
            placements=[MemoryPlacement.from_dict(x) for x in data.get("placements", [])],
            fusion_groups=[FusionGroup.from_dict(x) for x in data.get("fusion_groups", [])],
            resources=ResourceBinding.from_dict(data.get("resources", {})),
            constraints=constraints,
            provenance=[ScheduleDecisionRecord.from_dict(x) for x in data.get("provenance", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ScheduleIR":
        return cls.from_dict(json.loads(json_str))
