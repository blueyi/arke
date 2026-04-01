# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Learn — Trajectory recording and export.

Records optimization trajectories as JSONL for analysis and learning.
Each line is a JSON object with state/action/result triplets.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryRecord:
    """A single record in an optimization trajectory."""
    step: int
    timestamp: float
    event_type: str  # "action" | "result" | "observation" | "decision"

    # Action fields
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # Result fields
    success: bool | None = None
    result: dict[str, Any] = field(default_factory=dict)

    # State snapshot (optional, for key transitions)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "step": self.step,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
        }
        if self.tool:
            d["tool"] = self.tool
        if self.params:
            d["params"] = self.params
        if self.success is not None:
            d["success"] = self.success
        if self.result:
            d["result"] = self.result
        if self.state:
            d["state"] = self.state
        return d


class TrajectoryWriter:
    """Writes optimization trajectory to JSONL file.

    Usage:
        writer = TrajectoryWriter("trajectory.jsonl")
        writer.write_action(1, "apply_decision", {"kind": "tile", ...})
        writer.write_result(1, True, {"validation": {"pass": True}})
        writer.close()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._step = 0

    def write_header(self, metadata: dict[str, Any]) -> None:
        """Write a metadata header line."""
        record = {
            "event_type": "header",
            "timestamp": time.time(),
            **metadata,
        }
        self._file.write(json.dumps(record, default=str) + "\n")

    def write_action(
        self, step: int, tool: str, params: dict[str, Any]
    ) -> None:
        """Record a tool call action."""
        record = TrajectoryRecord(
            step=step,
            timestamp=time.time(),
            event_type="action",
            tool=tool,
            params=params,
        )
        self._file.write(json.dumps(record.to_dict(), default=str) + "\n")

    def write_result(
        self, step: int, success: bool, result: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> None:
        """Record a tool call result."""
        record = TrajectoryRecord(
            step=step,
            timestamp=time.time(),
            event_type="result",
            success=success,
            result=result,
            state=state or {},
        )
        self._file.write(json.dumps(record.to_dict(), default=str) + "\n")

    def write_observation(
        self, step: int, observation: dict[str, Any]
    ) -> None:
        """Record an observation/state snapshot."""
        record = TrajectoryRecord(
            step=step,
            timestamp=time.time(),
            event_type="observation",
            result=observation,
        )
        self._file.write(json.dumps(record.to_dict(), default=str) + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def export_session_trajectory(
    trajectory: list[dict[str, Any]],
    session_summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export a completed session's trajectory to JSONL.

    Args:
        trajectory: List of trajectory entries from OptimizationSession.export_trajectory()
        session_summary: Session summary dict
        output_path: Path to write JSONL file
    """
    with TrajectoryWriter(output_path) as writer:
        # Header
        writer.write_header({
            "kernel_id": session_summary.get("kernel_id", ""),
            "target_hw": session_summary.get("target_hw", ""),
            "total_decisions": session_summary.get("decisions", 0),
            "budget": session_summary.get("budget", {}),
            "best_performance": session_summary.get("best_performance"),
            "duration_seconds": session_summary.get("duration_seconds", 0),
        })

        # Trajectory entries
        for entry in trajectory:
            if entry["type"] == "action":
                writer.write_action(
                    entry["step"], entry["tool"], entry["params"]
                )
            elif entry["type"] == "result":
                writer.write_result(
                    entry["step"],
                    entry.get("result", {}).get("success", True),
                    entry.get("result", {}),
                )
