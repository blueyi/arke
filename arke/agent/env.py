# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — ArkeEnv: Façade-Substrate boundary container (D8-F1.1).

`ArkeEnv` packages everything a Façade tool needs to do its job:
  - `state`     : OptimizationState (mutated by tools 4/7/8)
  - `hw_profile`: HardwareProfile descriptor (read by tool 1)
  - `op_name` + `op_inputs`: kernel being optimized (read by tools 2/3/5/6)

It is the only object passed across the Façade-Substrate boundary.
External agents (Claude Code, MCP clients) NEVER touch ArkeEnv directly —
they call the 8 tools, which read/write through this container.

Design ref: docs/architecture/arke-harness.md §3 §8
Stage tracker: docs/phase1/stage8-plan.md D8-F1.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arke.agent.state import OptimizationBudget, OptimizationState
from arke.compiler.passes.base import HardwareProfile
from arke.ir.ops.registry import REGISTRY
from arke.ir.schedule import ScheduleIR
from arke.ir.strategy import Decision, Rationale


# ─── Legal-action candidate generator ─────────────────────────────────────

# Phase-1 initial implementation: enumerate a small fixed set of legal
# candidates per kind based on the op's index_vars + a few default factors.
# This is the "always-on floor" — a fuller legality calc lands in D8-F1.3.

_DEFAULT_TILE_FACTORS: tuple[tuple[int, ...], ...] = (
    (16,), (32,), (64,), (128,),
    (16, 16), (32, 32), (64, 64),
)

_DEFAULT_UNROLL_FACTORS: tuple[int, ...] = (2, 4, 8)
_DEFAULT_VECTORIZE_WIDTHS: tuple[int, ...] = (2, 4, 8)
_DEFAULT_PARALLEL_MAPPINGS: tuple[str, ...] = ("threadblock.x", "threadblock.y", "warp")
_DEFAULT_PLACE_MEMORIES: tuple[str, ...] = ("shared", "register")


def _enum_tile_candidates(loops: list[str]) -> list[Decision]:
    out: list[Decision] = []
    for loop in loops:
        for factors in _DEFAULT_TILE_FACTORS:
            out.append(Decision(
                kind="tile",
                params={"loop": loop, "factors": list(factors)},
                level=1,
            ))
    return out


def _enum_unroll_candidates(loops: list[str]) -> list[Decision]:
    return [
        Decision(kind="unroll", params={"loop": loop, "factor": f}, level=1)
        for loop in loops for f in _DEFAULT_UNROLL_FACTORS
    ]


def _enum_vectorize_candidates(loops: list[str]) -> list[Decision]:
    return [
        Decision(kind="vectorize", params={"loop": loop, "width": w}, level=1)
        for loop in loops for w in _DEFAULT_VECTORIZE_WIDTHS
    ]


def _enum_parallel_candidates(loops: list[str]) -> list[Decision]:
    return [
        Decision(kind="parallel", params={"mapping": {loop: m}}, level=1)
        for loop in loops for m in _DEFAULT_PARALLEL_MAPPINGS
    ]


def _enum_place_candidates(inputs: list[str]) -> list[Decision]:
    return [
        Decision(kind="place", params={"tensor": t, "memory": m}, level=1)
        for t in inputs for m in _DEFAULT_PLACE_MEMORIES
    ]


def _enum_fuse_candidates(op_name: str) -> list[Decision]:
    # Fusion candidates require a multi-op graph; for single-op env return [].
    return []


_LEGAL_KIND_GENERATORS = {
    "tile": _enum_tile_candidates,
    "unroll": _enum_unroll_candidates,
    "vectorize": _enum_vectorize_candidates,
    "parallel": _enum_parallel_candidates,
}


# ─── ArkeEnv ──────────────────────────────────────────────────────────────

@dataclass
class ArkeEnv:
    """Façade-Substrate boundary container.

    Construct via `ArkeEnv.from_op(op_name, shapes)` — sets up state,
    hw_profile, and kernel context in one call.
    """
    state: OptimizationState
    hw_profile: HardwareProfile
    op_name: str
    op_inputs: dict[str, list[int]] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def from_op(
        cls,
        op_name: str,
        shapes: dict[str, list[int]] | None = None,
        *,
        hw_profile: HardwareProfile | None = None,
        budget: OptimizationBudget | None = None,
        seed: int = 42,
    ) -> ArkeEnv:
        """Construct an ArkeEnv for optimizing `op_name`.

        Args:
            op_name: must be in `arke.ir.ops.registry.REGISTRY`.
            shapes: per-input shape map. Missing entries fall back to a default.
            hw_profile: optional hardware descriptor. Defaults to local probe.
            budget: optional budget caps. Defaults to OptimizationBudget defaults.
            seed: deterministic seed for V1/V2 input generation.

        Raises:
            KeyError: if op_name not registered.
        """
        if op_name not in REGISTRY:
            raise KeyError(
                f"Unknown op: {op_name!r}. Available: {sorted(REGISTRY.names())[:10]}..."
            )
        op = REGISTRY.get(op_name)
        # op.inputs is dict[name, type_string] — iterate keys
        input_names = list(op.inputs.keys()) if isinstance(op.inputs, dict) else list(op.inputs)
        merged_shapes = dict(shapes or {})
        for inp_name in input_names:
            if inp_name not in merged_shapes:
                merged_shapes[inp_name] = [4, 8]  # safe default

        hw = hw_profile if hw_profile is not None else HardwareProfile()
        strategy = ScheduleIR(kernel_id=op_name, target_hw=hw.name)
        state = OptimizationState(strategy=strategy, budget=budget)

        return cls(
            state=state,
            hw_profile=hw,
            op_name=op_name,
            op_inputs=merged_shapes,
            seed=seed,
        )

    # ── Façade tool 3: list_legal_actions ────────────────────────────────

    def list_legal_actions(
        self,
        *,
        top_n: int = 10,
        filter_kind: str | None = None,
    ) -> list[Decision]:
        """Return top-N legal next-decisions for current state.

        Args:
            top_n: max candidates to return (after filter).
            filter_kind: if set, restrict to this Decision.kind
                         (one of: tile / unroll / vectorize / parallel / place).

        Returns:
            List of Decision (level=1), generated from op index_vars +
            input tensors. Empty list if no legal candidates exist.

        This is the *generator-of-candidates* — not a ranker. Ranking
        belongs to the agent (LLM). Future work (D7-A1+) will add
        heuristic pre-filtering and shape-aware legality checks.
        """
        op = REGISTRY.get(self.op_name)
        loops = list(op.index_vars) if op.index_vars else ["i", "j"]
        inputs = list(op.inputs.keys()) if isinstance(op.inputs, dict) else list(op.inputs)

        candidates: list[Decision] = []
        kinds_to_gen = (
            [filter_kind]
            if filter_kind is not None
            else ["tile", "unroll", "vectorize", "parallel", "place"]
        )

        for kind in kinds_to_gen:
            if kind == "place":
                candidates.extend(_enum_place_candidates(inputs))
            elif kind in _LEGAL_KIND_GENERATORS:
                candidates.extend(_LEGAL_KIND_GENERATORS[kind](loops))
            elif kind == "fuse":
                candidates.extend(_enum_fuse_candidates(self.op_name))
            # else: unknown kind → skip silently (caller may pass any string)

        # Filter out decisions that are no-ops vs current strategy
        # (e.g. tile with same factors already applied to same loop)
        candidates = self._filter_redundant(candidates)

        return candidates[:top_n]

    def _filter_redundant(self, candidates: list[Decision]) -> list[Decision]:
        """Drop candidates that would duplicate an already-applied decision."""
        applied_tile = {
            d.params.get("loop"): tuple(d.params.get("factors", []))
            for d in self.state.decision_log if d.kind == "tile"
        }
        applied_unroll = {
            d.params.get("loop"): d.params.get("factor")
            for d in self.state.decision_log if d.kind == "unroll"
        }
        applied_vectorize = {
            d.params.get("loop"): d.params.get("width")
            for d in self.state.decision_log if d.kind == "vectorize"
        }
        out: list[Decision] = []
        for d in candidates:
            if d.kind == "tile":
                if applied_tile.get(d.params["loop"]) == tuple(d.params["factors"]):
                    continue
            elif d.kind == "unroll":
                if applied_unroll.get(d.params["loop"]) == d.params["factor"]:
                    continue
            elif d.kind == "vectorize":
                if applied_vectorize.get(d.params["loop"]) == d.params["width"]:
                    continue
            out.append(d)
        return out

    # ── Convenience accessors ────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"ArkeEnv(op={self.op_name}, hw={self.hw_profile.name}, "
            f"shapes={self.op_inputs}, {self.state.summary()})"
        )
