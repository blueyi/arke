# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — StrategyIR v1.0 (Layer 3).

Optimization decisions: "how to optimize."
L1 (backend-agnostic) and L2 (resource / backend-bound) decision levels.

See docs/spec/arke-ir-spec-design.md §7.3 for the complete schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union


@dataclass
class Rationale:
    """Natural language explanation for an optimization decision."""
    text: str
    lang: str = "en"


# ─── Decision Types ────────────────────────────────────────────────────────

@dataclass
class Decision:
    """A single L1 (backend-agnostic) optimization decision.

    Kinds (L1):
        tile         - Tile a loop: {loop, factors}
        reorder      - Reorder loop nest: {order}
        fuse         - Fuse operators: {ops, type}
        parallel     - Map loops to hardware dims: {loops, mapping}
        place        - Tensor memory placement: {tensor, memory}
        vectorize    - Vectorize a loop: {loop, width}
        unroll       - Unroll a loop: {loop, factor}
        algorithm    - Algorithm variant: {name, params}

    Kinds (L2 — resource / backend-bound, set by specialization passes):
        compute       - {warps, num_stages, shared_memory, pipeline_depth}
        cache_config  - {l1_size, l2_hint}
        memory_fence  - {scope}
    """
    kind: str
    params: dict[str, Any]
    rationale: Rationale | None = None
    step: int = 0          # auto-assigned by StrategyIR
    level: int = 1         # 1 = L1 backend-agnostic, 2 = L2 backend-specific


@dataclass
class ConditionalDecision:
    """A decision that applies only when a shape predicate holds.

    Example:
        when dim("S") <= 512 { tile(K=32) } otherwise { tile(K=64) }
    """
    predicate: str
    true_decisions: list[Decision]
    false_decisions: list[Decision]
    rationale: Rationale | None = None
    step: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "kind": "__conditional__",
            "predicate": self.predicate,
            "true_decisions": [_decision_to_dict(x) for x in self.true_decisions],
            "false_decisions": [_decision_to_dict(x) for x in self.false_decisions],
            "step": self.step,
        }
        if self.rationale:
            d["rationale"] = {"text": self.rationale.text, "lang": self.rationale.lang}
        return d


AnyDecision = Union[Decision, ConditionalDecision]


def _decision_to_dict(d: AnyDecision) -> dict:
    if isinstance(d, ConditionalDecision):
        return d.to_dict()
    result: dict = {"kind": d.kind, "params": d.params, "step": d.step, "level": d.level}
    if d.rationale:
        result["rationale"] = {"text": d.rationale.text, "lang": d.rationale.lang}
    return result


def _parse_decision(d: dict) -> Decision:
    """Parse a single decision dict in the current v2 format."""
    rat = None
    if d.get("rationale"):
        rat_data = d["rationale"]
        rat = Rationale(
            text=rat_data if isinstance(rat_data, str) else rat_data.get("text", ""),
            lang=rat_data.get("lang", "en") if isinstance(rat_data, dict) else "en",
        )
    return Decision(
        kind=d["kind"],
        params=dict(d["params"]),
        rationale=rat,
        step=d.get("step", 0),
        level=d.get("level", 1),
    )


@dataclass
class ShapeRegime:
    """A named strategy regime for a specific shape range."""
    name: str
    predicate: str
    decisions: list[AnyDecision] = field(default_factory=list)


@dataclass
class HardwareConstraints:
    """Hardware resource constraints."""
    shared_memory_limit: int = 0
    register_limit: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32


@dataclass
class StrategyIR:
    """Optimization strategy IR v1.0.

    Current v2-oriented structure:
    - decisions list accepts AnyDecision (Decision | ConditionalDecision)
    - shape_regimes: named profiles for shape-based dispatch
    - level field on each Decision (L1 vs L2)
    - compute is the canonical Layer-2 resource decision
    """
    version: str = "1.0.0"
    kernel_id: str = ""
    target_hw: str = ""
    decisions: list[AnyDecision] = field(default_factory=list)
    shape_regimes: list[ShapeRegime] = field(default_factory=list)
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    def add_decision(self, decision: AnyDecision) -> AnyDecision:
        decision.step = len(self.decisions) + 1
        self.decisions.append(decision)
        return decision

    def pop_decisions(self, n: int = 1) -> list[AnyDecision]:
        removed = []
        for _ in range(min(n, len(self.decisions))):
            removed.append(self.decisions.pop())
        return removed

    def tile(self, loop: str, factors: list[int],
             rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="tile", params={"loop": loop, "factors": factors},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def reorder(self, order: list[str], rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="reorder", params={"order": order},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def fuse(self, ops: list[str], fusion_type: str = "epilogue",
             rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="fuse", params={"ops": ops, "type": fusion_type},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def parallel(self, loops: list[str], mapping: dict[str, str],
                 rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="parallel", params={"loops": loops, "mapping": mapping},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def place(self, tensor: str, memory: str,
              rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="place", params={"tensor": tensor, "memory": memory},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def compute(self, warps: int | None = None,
                num_stages: int | None = None,
                shared_memory: int | None = None,
                rationale: str | None = None) -> Decision:
        """L2 decision: resource configuration in canonical v2 form."""
        params: dict = {}
        if warps is not None:
            params["warps"] = warps
        if num_stages is not None:
            params["num_stages"] = num_stages
        if shared_memory is not None:
            params["shared_memory"] = shared_memory
        return self.add_decision(Decision(
            kind="compute", params=params,
            rationale=Rationale(text=rationale) if rationale else None,
            level=2,
        ))

    def when(self, predicate: str, true_decisions: list[Decision],
             false_decisions: list[Decision] | None = None,
             rationale: str | None = None) -> ConditionalDecision:
        cd = ConditionalDecision(
            predicate=predicate,
            true_decisions=true_decisions,
            false_decisions=false_decisions or [],
            rationale=Rationale(text=rationale) if rationale else None,
        )
        return self.add_decision(cd)

    def summary(self) -> str:
        lines = [f"Strategy for {self.kernel_id} on {self.target_hw}:"]
        for d in self.decisions:
            if isinstance(d, ConditionalDecision):
                r = f" -- {d.rationale.text}" if d.rationale else ""
                lines.append(f"  #{d.step} when({d.predicate}):{r}")
            else:
                r = f" -- {d.rationale.text}" if d.rationale else ""
                lines.append(f"  #{d.step} {d.kind}({d.params}){r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "target_hw": self.target_hw,
            "decisions": [_decision_to_dict(dec) for dec in self.decisions],
        }
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        if self.shape_regimes:
            d["shape_regimes"] = [
                {"name": r.name, "predicate": r.predicate,
                 "decisions": [_decision_to_dict(dec) for dec in r.decisions]}
                for r in self.shape_regimes
            ]
        from dataclasses import asdict as _asdict
        c = _asdict(self.constraints)
        if any(v not in (0, 32) for v in c.values()):
            d["constraints"] = c
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> StrategyIR:
        """Deserialize StrategyIR from the current format."""
        ir = cls(
            version=data.get("version", "1.0.0"),
            kernel_id=data.get("kernel_id", ""),
            target_hw=data.get("target_hw", ""),
            metadata=dict(data.get("metadata", {})),
        )
        for d in data.get("decisions", []):
            if d.get("kind") == "__conditional__":
                true_ds = [_parse_decision(td) for td in d.get("true_decisions", [])]
                false_ds = [_parse_decision(fd) for fd in d.get("false_decisions", [])]
                rat = Rationale(**d["rationale"]) if d.get("rationale") else None
                ir.decisions.append(ConditionalDecision(
                    predicate=d["predicate"],
                    true_decisions=true_ds,
                    false_decisions=false_ds,
                    rationale=rat,
                    step=d.get("step", 0),
                ))
            else:
                ir.decisions.append(_parse_decision(d))
        return ir

    @classmethod
    def from_json(cls, json_str: str) -> StrategyIR:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> StrategyIR:
        with open(path) as f:
            return cls.from_json(f.read())
