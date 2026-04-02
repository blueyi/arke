# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — ArkeEnv core.

The central environment that LLM agents interact with via tool-use.
Modeled after Gymnasium Env: agent sends actions, env returns observations.
"""

from __future__ import annotations

import json
from pathlib import Path

from arke.engine.legal_actions import LegalActionsEngine
from arke.engine.validator import StaticValidator
from arke.ir.ops.catalog import OP_CATALOG
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import Decision, Rationale, StrategyIR


class ArkeEnv:
    """LLM Agent's interaction environment.

    Manages:
    - Semantic IR (what to compute — immutable after creation)
    - Strategy IR (how to optimize — modified by apply_decision)
    - Hardware profile
    - Decision history + checkpoints
    - Validation
    """

    def __init__(self, semantic: SemanticIR, target_hw: str):
        """Initialize the environment with a semantic IR and hardware target."""
        self.semantic = semantic
        self.target_hw = target_hw
        self.strategy = StrategyIR(
            kernel_id=semantic.kernel_id,
            target_hw=target_hw,
        )
        self.hw_profile = self._load_hw_profile(target_hw)
        self.validator = StaticValidator()
        self.legal_actions_engine = LegalActionsEngine()
        self.checkpoints: dict[str, StrategyIR] = {}
        self._step = 0

    # ─── Tool implementations ───

    def get_semantic_ir(self) -> dict:
        """Tool: get_semantic_ir — return the computation definition."""
        return self.semantic.to_dict()

    def get_hw_profile(self) -> dict:
        """Tool: get_hw_profile — return hardware parameters."""
        return self.hw_profile

    def get_current_strategy(self) -> dict:
        """Tool: get_current_strategy — return current optimization state."""
        return self.strategy.to_dict()

    def analyze_compute(self) -> dict:
        """Tool: analyze_compute — analyze computation characteristics."""
        nodes_list: list[dict] = []
        fusions_list: list[dict] = []
        analysis: dict = {
            "kernel": self.semantic.kernel_id,
            "nodes": nodes_list,
            "fusion_opportunities": fusions_list,
        }

        for node in self.semantic.nodes:
            op_def = OP_CATALOG.get(node.op)
            category = op_def.category if op_def else "unknown"
            nodes_list.append({
                "id": node.id,
                "op": node.op,
                "category": category,
                "properties": node.semantics.properties,
            })

        for fg in self.semantic.fusion_groups:
            fusions_list.append({
                "nodes": fg.nodes,
                "type": fg.fusion_type,
                "reason": fg.reason,
            })

        return analysis

    def list_legal_actions(
        self, kind: str | None = None, limit: int = 10
    ) -> dict:
        """Tool: list_legal_actions — enumerate legal optimization moves."""
        result = self.legal_actions_engine.enumerate(
            self.semantic, self.strategy, self.hw_profile,
            kind=kind, limit=limit,
        )
        return {
            "legal_actions": [
                {
                    "id": a.id,
                    "kind": a.kind,
                    "params": a.params,
                    "estimated_impact": a.estimated_impact,
                    "codegen_support": a.codegen_support,
                }
                for a in result.legal_actions
            ],
            "blocked_actions": [
                {
                    "id": b.id,
                    "kind": b.kind,
                    "params": b.params,
                    "blocked_reason": b.blocked_reason,
                }
                for b in result.blocked_actions
            ],
            "search_space_size": result.search_space_size,
            "hint": result.hint,
        }

    def apply_decision(
        self, kind: str, params: dict, rationale: str
    ) -> dict:
        """Tool: apply_decision — apply one optimization decision.

        Returns validation result. Auto-rollbacks on V0 failure.
        """
        decision = Decision(
            kind=kind,
            params=params,
            rationale=Rationale(text=rationale) if rationale else None,
        )
        self.strategy.add_decision(decision)
        self._step += 1

        # V0 validation
        result = self.validator.validate(
            self.semantic, self.strategy, self.hw_profile
        )

        if not result.passed:
            # Auto-rollback on V0 failure
            self.strategy.pop_decisions(1)
            return {
                "success": False,
                "step": self._step,
                "validation": {
                    "pass": False,
                    "violations": result.violations,
                },
                "auto_rollback": True,
            }

        return {
            "success": True,
            "step": self._step,
            "validation": {
                "pass": True,
                "resource_usage": result.resource_usage,
            },
            "decisions_so_far": self.strategy.decision_count,
        }

    def rollback(self, steps: int = 1) -> dict:
        """Tool: rollback — undo the last N decisions."""
        removed = self.strategy.pop_decisions(steps)
        return {
            "rolled_back": len(removed),
            "decisions_remaining": self.strategy.decision_count,
        }

    def checkpoint(self, name: str | None = None) -> dict:
        """Tool: checkpoint — save current state."""
        cp_id = name or f"cp_{self.strategy.decision_count}"
        self.checkpoints[cp_id] = StrategyIR.from_dict(self.strategy.to_dict())
        return {"checkpoint_id": cp_id}

    def restore(self, checkpoint_id: str) -> dict:
        """Tool: restore — restore a saved checkpoint."""
        if checkpoint_id not in self.checkpoints:
            return {"success": False, "error": f"Checkpoint '{checkpoint_id}' not found"}
        self.strategy = StrategyIR.from_dict(self.checkpoints[checkpoint_id].to_dict())
        return {
            "success": True,
            "restored_to": checkpoint_id,
            "decisions": self.strategy.decision_count,
        }

    def observe(self) -> dict:
        """Tool: observe — get current state summary (delta-friendly)."""
        return {
            "kernel_id": self.semantic.kernel_id,
            "target_hw": self.target_hw,
            "decision_count": self.strategy.decision_count,
            "strategy_summary": self.strategy.summary(),
            "checkpoints": list(self.checkpoints.keys()),
        }

    # ─── Internal ───

    def _load_hw_profile(self, target: str) -> dict:
        """Load hardware profile from bundled JSON."""
        # Normalize target name
        target_map = {
            "nvidia_ampere": "nvidia_ampere",
            "ampere": "nvidia_ampere",
            "ascend_a3": "ascend_a3",
        }
        filename = target_map.get(target, target)
        path = Path(__file__).parent.parent / "ir" / "targets" / f"{filename}.json"
        if path.exists():
            result: dict = json.loads(path.read_text())
            return result
        return {"name": target, "error": f"Profile not found: {path}"}
