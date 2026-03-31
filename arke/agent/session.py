# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Session lifecycle management.

An optimization session encapsulates:
- SemanticIR (immutable computation)
- ArkeEnv (mutable optimization state)
- LLM conversation history
- Budget tracking
- Optimization trajectory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from arke.agent.prompts import build_initial_user_message, build_system_prompt
from arke.agent.tools_schema import TOOL_METADATA, get_tool_schemas
from arke.engine.env import ArkeEnv
from arke.ir.semantic import SemanticIR


class SessionState(str, Enum):
    """Session lifecycle states."""
    CREATED = "created"          # Session created, kernel not yet defined
    ANALYZING = "analyzing"      # Kernel defined, LLM analyzing
    OPTIMIZING = "optimizing"    # LLM making decisions
    VERIFYING = "verifying"      # Compiling and verifying
    FINALIZED = "finalized"      # Optimization complete
    FAILED = "failed"            # Unrecoverable error


@dataclass
class OptimizationBudget:
    """Tracks optimization resource usage."""
    max_decisions: int = 50
    max_compiles: int = 10
    target_performance: float = 0.7  # ratio of vendor baseline
    warning_threshold: int = 40      # warn LLM at this step

    decisions_used: int = 0
    compiles_used: int = 0

    @property
    def decisions_remaining(self) -> int:
        return max(0, self.max_decisions - self.decisions_used)

    @property
    def compiles_remaining(self) -> int:
        return max(0, self.max_compiles - self.compiles_used)

    @property
    def exhausted(self) -> bool:
        return self.decisions_remaining == 0

    @property
    def should_warn(self) -> bool:
        return self.decisions_used >= self.warning_threshold and not self.exhausted

    def use_decision(self) -> None:
        self.decisions_used += 1

    def use_compile(self) -> None:
        self.compiles_used += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions_used": self.decisions_used,
            "decisions_remaining": self.decisions_remaining,
            "compiles_used": self.compiles_used,
            "compiles_remaining": self.compiles_remaining,
            "target_performance": self.target_performance,
        }


@dataclass
class TrajectoryEntry:
    """A single entry in the optimization trajectory."""
    step: int
    type: str           # "action" | "result" | "compile" | "observe"
    tool: str = ""
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationSession:
    """A complete LLM optimization session.

    Lifecycle:
        1. Create session with SemanticIR + target
        2. Session builds ArkeEnv, system prompt, tools
        3. LLM interacts via tool-use (run_tool dispatches to ArkeEnv)
        4. Session tracks budget, trajectory, state
        5. Finalize when done
    """

    semantic_ir: SemanticIR
    target_hw: str
    env: ArkeEnv = field(init=False)
    state: SessionState = SessionState.CREATED
    budget: OptimizationBudget = field(default_factory=OptimizationBudget)
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    best_performance: dict[str, Any] | None = None
    _step: int = 0
    _created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.env = ArkeEnv(self.semantic_ir, self.target_hw)
        self._init_messages()

    def _init_messages(self) -> None:
        """Initialize conversation with system prompt."""
        system_prompt = build_system_prompt(
            hw_profile=self.env.hw_profile,
            budget_decisions=self.budget.max_decisions,
            budget_compiles=self.budget.max_compiles,
            target_performance=self.budget.target_performance,
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

    # ─── Tool dispatch ───

    def run_tool(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Execute a tool call and return the result.

        Handles budget tracking, trajectory recording, and state transitions.
        """
        self._step += 1

        # Budget check
        meta = TOOL_METADATA.get(tool_name, {})
        budget_type = meta.get("budget_type", "free")

        if budget_type == "decision":
            if self.budget.exhausted:
                return self._budget_exhausted_response()
            self.budget.use_decision()
        elif budget_type == "compile":
            if self.budget.compiles_remaining <= 0:
                return {"success": False, "error": "Compile budget exhausted"}
            self.budget.use_compile()

        # Record trajectory
        self.trajectory.append(TrajectoryEntry(
            step=self._step, type="action", tool=tool_name, params=params,
        ))

        # Dispatch to ArkeEnv
        result = self._dispatch(tool_name, params)

        # Record result
        self.trajectory.append(TrajectoryEntry(
            step=self._step, type="result", tool=tool_name, result=result,
        ))

        # Inject budget info into every response
        result["budget"] = self.budget.to_dict()

        # Budget warning
        if self.budget.should_warn:
            result["budget_warning"] = (
                f"You have used {self.budget.decisions_used}/{self.budget.max_decisions} decisions. "
                f"Consider finalizing soon."
            )

        # State transitions
        self._update_state(tool_name, result)

        return result

    def _dispatch(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate ArkeEnv method."""
        dispatch_map: dict[str, Any] = {
            "create_kernel": lambda p: self._handle_create_kernel(p),
            "get_hw_profile": lambda p: self.env.get_hw_profile(),
            "analyze_compute": lambda p: self.env.analyze_compute(),
            "list_legal_actions": lambda p: self._handle_list_legal(p),
            "apply_decision": lambda p: self.env.apply_decision(
                p["kind"], p["params"], p.get("rationale", ""),
            ),
            "verify_correctness": lambda p: {"status": "not_implemented_yet"},
            "compile_and_profile": lambda p: {"status": "not_implemented_yet"},
            "rollback": lambda p: self.env.rollback(p.get("steps", 1)),
            "checkpoint": lambda p: self.env.checkpoint(p.get("name")),
            "restore": lambda p: self.env.restore(p["checkpoint_id"]),
        }

        handler = dispatch_map.get(tool_name)
        if handler is None:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        return handler(params)

    def _handle_create_kernel(self, params: dict) -> dict[str, Any]:
        """Handle create_kernel (for agent-mode where LLM defines the kernel)."""
        # In most flows, the kernel is already defined in __init__.
        # This tool is for agent-initiated kernel creation.
        return {
            "success": True,
            "kernel_id": self.semantic_ir.kernel_id,
            "semantic_ir": self.env.get_semantic_ir(),
            "auto_analysis": self.env.analyze_compute(),
        }

    def _handle_list_legal(self, params: dict) -> dict[str, Any]:
        """Handle list_legal_actions with filtering and limits."""
        # TODO: Implement real legal action enumeration
        kind = params.get("kind")
        limit = params.get("limit", 10)
        return {
            "legal_actions": [],
            "search_space_size": 0,
            "filter": kind,
            "note": "Legal action enumeration not yet implemented",
        }

    def _budget_exhausted_response(self) -> dict[str, Any]:
        """Response when budget is exhausted."""
        return {
            "success": False,
            "error": "Decision budget exhausted",
            "budget": self.budget.to_dict(),
            "suggestion": (
                "Call compile_and_profile() to evaluate your current strategy, "
                "then finalize. Or rollback to a checkpoint and try a different approach."
            ),
        }

    def _update_state(self, tool_name: str, result: dict) -> None:
        """Update session state based on tool call."""
        if tool_name == "create_kernel" and result.get("success"):
            self.state = SessionState.ANALYZING
        elif tool_name in ("analyze_compute", "list_legal_actions"):
            if self.state == SessionState.CREATED:
                self.state = SessionState.ANALYZING
        elif tool_name == "apply_decision":
            self.state = SessionState.OPTIMIZING
        elif tool_name in ("verify_correctness", "compile_and_profile"):
            self.state = SessionState.VERIFYING
            # Track best performance
            perf = result.get("performance")
            if perf and (self.best_performance is None or
                         perf.get("latency_us", float("inf")) <
                         self.best_performance.get("latency_us", float("inf"))):
                self.best_performance = perf

    # ─── Properties ───

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return get_tool_schemas()

    @property
    def system_prompt(self) -> str:
        return self.messages[0]["content"] if self.messages else ""

    @property
    def duration_seconds(self) -> float:
        return time.time() - self._created_at

    def summary(self) -> dict[str, Any]:
        """Session summary for reporting."""
        return {
            "kernel_id": self.semantic_ir.kernel_id,
            "target_hw": self.target_hw,
            "state": self.state.value,
            "decisions": self.env.strategy.decision_count,
            "budget": self.budget.to_dict(),
            "best_performance": self.best_performance,
            "trajectory_steps": len(self.trajectory),
            "duration_seconds": round(self.duration_seconds, 1),
        }

    def export_trajectory(self) -> list[dict[str, Any]]:
        """Export trajectory as list of dicts (for JSONL)."""
        return [
            {
                "step": e.step,
                "type": e.type,
                "tool": e.tool,
                "params": e.params,
                "result": e.result,
                "timestamp": e.timestamp,
            }
            for e in self.trajectory
        ]
