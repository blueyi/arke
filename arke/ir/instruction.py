# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — InstructionIR (Layer 1).

Backend-near low-level instruction form produced from ScheduleIR.
This is still a structured Python IR, but close to backend emission boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instruction:
    """A backend-near instruction-like operation."""
    opcode: str
    operands: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "opcode": self.opcode,
            "operands": list(self.operands),
        }
        if self.attrs:
            d["attrs"] = dict(self.attrs)
        if self.comment:
            d["comment"] = self.comment
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instruction":
        return cls(
            opcode=d["opcode"],
            operands=list(d.get("operands", [])),
            attrs=dict(d.get("attrs", {})),
            comment=d.get("comment", ""),
        )


@dataclass
class InstructionBlock:
    """A named instruction region."""
    name: str
    instructions: list[Instruction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instructions": [x.to_dict() for x in self.instructions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstructionBlock":
        return cls(
            name=d["name"],
            instructions=[Instruction.from_dict(x) for x in d.get("instructions", [])],
        )


@dataclass
class InstructionIR:
    """Layer 1 instruction representation."""
    version: str = "2.0.0"
    kernel_id: str = ""
    target_hw: str = ""
    blocks: list[InstructionBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_block(self, name: str) -> InstructionBlock:
        block = InstructionBlock(name=name)
        self.blocks.append(block)
        return block

    def summary(self) -> str:
        count = sum(len(b.instructions) for b in self.blocks)
        return (
            f"InstructionIR for {self.kernel_id} on {self.target_hw}: "
            f"{len(self.blocks)} blocks, {count} instructions"
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "target_hw": self.target_hw,
            "blocks": [x.to_dict() for x in self.blocks],
        }
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionIR":
        return cls(
            version=data.get("version", "2.0.0"),
            kernel_id=data.get("kernel_id", ""),
            target_hw=data.get("target_hw", ""),
            blocks=[InstructionBlock.from_dict(x) for x in data.get("blocks", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionIR":
        return cls.from_dict(json.loads(json_str))
