# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — Legal Actions Enumeration.

Enumerates all legal optimization actions from the current state.
This is the LLM's "move generator" — analogous to listing legal moves in chess.

Design principles:
- Return top-N actions with estimated impact (save LLM context)
- Include blocked actions with reasons (helps LLM learn constraints)
- Filter by kind for focused exploration
- Actions are deterministic given (SemanticIR, StrategyIR, HWProfile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class ActionCandidate:
    """A legal optimization action the LLM can take."""
    id: str
    kind: str
    params: dict[str, Any]
    estimated_impact: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0  # Higher = more promising
    codegen_support: bool = True  # Whether current codegen can handle this


@dataclass
class BlockedAction:
    """An action that would be illegal and why."""
    id: str
    kind: str
    params: dict[str, Any]
    blocked_reason: str


@dataclass
class LegalActionsResult:
    """Result of legal action enumeration."""
    legal_actions: list[ActionCandidate]
    blocked_actions: list[BlockedAction]
    search_space_size: int
    hint: str = ""


class LegalActionsEngine:
    """Enumerates legal optimization actions.

    Given (SemanticIR, StrategyIR, HWProfile), produces all valid
    next steps the LLM can take.
    """

    # Standard tile factors (powers of 2, aligned with warp size)
    TILE_FACTORS = [16, 32, 64, 128, 256]
    # Two-level tiling: (outer, inner)
    TILE_FACTOR_PAIRS = [
        [128, 32], [128, 16], [64, 32], [64, 16],
        [256, 16], [256, 32], [32, 16], [32, 8],
    ]

    def enumerate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict[str, Any],
        kind: str | None = None,
        limit: int = 10,
    ) -> LegalActionsResult:
        """Enumerate legal actions, optionally filtered by kind."""
        all_legal: list[ActionCandidate] = []
        all_blocked: list[BlockedAction] = []

        generators = {
            "tile": self._gen_tile_actions,
            "fuse": self._gen_fuse_actions,
            "reorder": self._gen_reorder_actions,
            "parallel": self._gen_parallel_actions,
            "place": self._gen_place_actions,
        }

        if kind:
            gen = generators.get(kind)
            if gen:
                legal, blocked = gen(semantic, strategy, hw_profile)
                all_legal.extend(legal)
                all_blocked.extend(blocked)
        else:
            for gen in generators.values():
                legal, blocked = gen(semantic, strategy, hw_profile)
                all_legal.extend(legal)
                all_blocked.extend(blocked)

        total = len(all_legal)

        # Sort by priority (highest first) and limit
        all_legal.sort(key=lambda a: -a.priority)
        top_actions = all_legal[:limit]

        hint = self._generate_hint(semantic, strategy, hw_profile)

        return LegalActionsResult(
            legal_actions=top_actions,
            blocked_actions=all_blocked[:5],  # Top 5 blocked
            search_space_size=total,
            hint=hint,
        )

    # ─── Tile Actions ───

    def _gen_tile_actions(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> tuple[list[ActionCandidate], list[BlockedAction]]:
        """Generate tiling actions for all tileable loops."""
        legal: list[ActionCandidate] = []
        blocked: list[BlockedAction] = []

        # Find all index vars from semantic IR nodes
        tileable_loops = self._get_tileable_loops(semantic, strategy)
        already_tiled = self._get_tiled_loops(strategy)
        shared_mem_limit = hw_profile.get("constraints", {}).get(
            "max_shared_memory_per_block", 49152
        )
        warp_size = hw_profile.get("constraints", {}).get("warp_size", 32)

        for loop_name, loop_bound in tileable_loops.items():
            if loop_name in already_tiled:
                blocked.append(BlockedAction(
                    id=f"tile_{loop_name}_blocked",
                    kind="tile",
                    params={"loop": loop_name},
                    blocked_reason=f"Loop '{loop_name}' is already tiled",
                ))
                continue

            for factors in self.TILE_FACTOR_PAIRS:
                # Check if factors divide the loop bound
                outer = factors[0]
                if loop_bound > 0 and loop_bound % outer != 0:
                    continue

                action_id = f"tile_{loop_name}_{'_'.join(str(f) for f in factors)}"

                # Estimate shared memory impact
                est_shared = self._estimate_shared_memory_for_tile(
                    loop_name, factors, semantic, strategy
                )

                # Check constraint
                current_shared = self._estimate_current_shared(strategy)
                if current_shared + est_shared > shared_mem_limit:
                    blocked.append(BlockedAction(
                        id=action_id,
                        kind="tile",
                        params={"loop": loop_name, "factors": factors},
                        blocked_reason=f"Would exceed shared memory limit ({current_shared + est_shared} > {shared_mem_limit})",
                    ))
                    continue

                # Priority: prefer tiles aligned with warp size and tensor core
                priority = self._tile_priority(
                    loop_name, factors, loop_bound, warp_size, semantic
                )

                legal.append(ActionCandidate(
                    id=action_id,
                    kind="tile",
                    params={"loop": loop_name, "factors": factors},
                    estimated_impact={
                        "shared_memory_delta": f"+{est_shared // 1024}KB" if est_shared > 0 else "+0KB",
                        "blocks_in_dim": loop_bound // outer if loop_bound > 0 else "unknown",
                    },
                    priority=priority,
                ))

        return legal, blocked

    # ─── Fuse Actions ───

    def _gen_fuse_actions(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> tuple[list[ActionCandidate], list[BlockedAction]]:
        """Generate fusion actions from detected fusion groups."""
        legal: list[ActionCandidate] = []
        blocked: list[BlockedAction] = []

        already_fused = self._get_fused_node_sets(strategy)

        for fg in semantic.fusion_groups:
            node_set = frozenset(fg.nodes)
            if node_set in already_fused:
                blocked.append(BlockedAction(
                    id=f"fuse_{fg.id}_blocked",
                    kind="fuse",
                    params={"nodes": fg.nodes, "type": fg.fusion_type},
                    blocked_reason=f"Nodes {fg.nodes} already fused",
                ))
                continue

            # Estimate benefit
            benefit = self._estimate_fusion_benefit(fg, semantic)

            legal.append(ActionCandidate(
                id=f"fuse_{fg.id}",
                kind="fuse",
                params={"nodes": fg.nodes, "type": fg.fusion_type},
                estimated_impact={
                    "type": fg.fusion_type,
                    "benefit": benefit,
                    "eliminates_intermediate": True,
                },
                priority=10.0,  # Fusion is usually high priority
            ))

        return legal, blocked

    # ─── Parallel Actions ───

    def _gen_parallel_actions(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> tuple[list[ActionCandidate], list[BlockedAction]]:
        """Generate parallelization mapping actions."""
        legal: list[ActionCandidate] = []
        blocked: list[BlockedAction] = []

        # Can only parallelize after tiling
        tiled_loops = self._get_tiled_loops(strategy)
        if not tiled_loops:
            return legal, blocked

        already_parallelized = self._get_parallel_loops(strategy)
        max_threads = hw_profile.get("constraints", {}).get("max_threads_per_block", 1024)
        compute_units = hw_profile.get("compute_units", 1)

        # Generate outer loop → block mapping candidates
        outer_loops = [f"{loop}_outer" for loop in tiled_loops]
        inner_loops = [f"{loop}_inner" for loop in tiled_loops]

        if outer_loops and not already_parallelized:
            # Map outer loops to blocks
            if len(outer_loops) >= 2:
                mapping = {
                    outer_loops[0]: "block.x",
                    outer_loops[1]: "block.y",
                }
                legal.append(ActionCandidate(
                    id="parallel_outer_2d",
                    kind="parallel",
                    params={"loops": outer_loops[:2], "mapping": mapping},
                    estimated_impact={
                        "grid_dims": 2,
                        "compute_units": compute_units,
                    },
                    priority=5.0,
                ))
            elif len(outer_loops) == 1:
                mapping = {outer_loops[0]: "block.x"}
                legal.append(ActionCandidate(
                    id="parallel_outer_1d",
                    kind="parallel",
                    params={"loops": outer_loops[:1], "mapping": mapping},
                    estimated_impact={"grid_dims": 1},
                    priority=4.0,
                ))

        return legal, blocked

    # ─── Place Actions ───

    def _gen_place_actions(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> tuple[list[ActionCandidate], list[BlockedAction]]:
        """Generate memory placement actions."""
        legal: list[ActionCandidate] = []
        blocked: list[BlockedAction] = []

        already_placed = self._get_placed_tensors(strategy)
        shared_mem_limit = hw_profile.get("constraints", {}).get(
            "max_shared_memory_per_block", 49152
        )

        # Candidate tensors: input params that are reused
        for param in semantic.params:
            tile_name = f"{param.name}_tile"
            if tile_name in already_placed or param.name in already_placed:
                continue

            # Estimate tile size (depends on tiling)
            tile_size = self._estimate_param_tile_size(param, strategy)
            current_shared = self._estimate_current_shared(strategy)

            if current_shared + tile_size > shared_mem_limit:
                blocked.append(BlockedAction(
                    id=f"place_{param.name}_shared_blocked",
                    kind="place",
                    params={"tensor": tile_name, "memory": "shared"},
                    blocked_reason=f"Would exceed shared memory ({current_shared + tile_size} > {shared_mem_limit})",
                ))
                continue

            # Estimate reuse factor
            reuse = self._estimate_reuse_factor(param.name, semantic)

            legal.append(ActionCandidate(
                id=f"place_{param.name}_shared",
                kind="place",
                params={"tensor": tile_name, "memory": "shared"},
                estimated_impact={
                    "memory_level": "shared",
                    "tile_size_bytes": tile_size,
                    "reuse_factor": f"{reuse}x",
                },
                priority=3.0 + min(reuse, 10),  # Higher reuse = higher priority
            ))

        return legal, blocked

    # ─── Reorder Actions ───

    def _gen_reorder_actions(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> tuple[list[ActionCandidate], list[BlockedAction]]:
        """Generate loop reorder actions (simplified: only if tiled)."""
        legal: list[ActionCandidate] = []
        blocked: list[BlockedAction] = []

        tiled_loops = self._get_tiled_loops(strategy)
        if len(tiled_loops) < 2:
            return legal, blocked

        # Standard reorder: outer parallel, inner reuse
        outer = [f"{l}_outer" for l in tiled_loops]
        inner = [f"{l}_inner" for l in tiled_loops]
        reduction_loops = self._get_reduction_loops(semantic)
        reduction_outer = [f"{l}_outer" for l in reduction_loops if l in tiled_loops]
        reduction_inner = [f"{l}_inner" for l in reduction_loops if l in tiled_loops]

        standard_order = outer + reduction_outer + inner + reduction_inner
        if standard_order:
            legal.append(ActionCandidate(
                id="reorder_standard",
                kind="reorder",
                params={"order": standard_order},
                estimated_impact={"pattern": "outer_parallel_inner_reuse"},
                priority=2.0,
            ))

        return legal, blocked

    # ─── Helper Methods ───

    def _get_tileable_loops(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> dict[str, int]:
        """Get tileable loop names and their bounds."""
        loops: dict[str, int] = {}
        for node in semantic.nodes:
            for var in node.semantics.index_vars:
                if var not in loops:
                    # Infer bound from shapes (simplified)
                    bound = self._infer_loop_bound(var, node, semantic)
                    loops[var] = bound
        return loops

    def _infer_loop_bound(
        self, var: str, node: Any, semantic: SemanticIR
    ) -> int:
        """Infer a loop variable's bound from tensor shapes."""
        # For matmul: i→M, j→N, k→K
        # Map index vars to output/input dims based on position
        idx = node.semantics.index_vars
        if not idx:
            return 0

        try:
            pos = idx.index(var)
        except ValueError:
            return 0

        # If it's a reduction axis, look at the input shape
        if var in node.semantics.reduction_axes:
            # Reduction var — bound comes from input dim
            # For matmul k: it's the shared dim (last dim of A / first dim of B)
            for param in semantic.params:
                if len(param.shape) > 1:
                    return param.shape[-1]  # Last dim is often the reduction dim
            return 0

        # Non-reduction — bound comes from output shape
        if pos < len(node.output.shape):
            return node.output.shape[pos]
        return 0

    def _get_tiled_loops(self, strategy: StrategyIR) -> set[str]:
        """Get set of already-tiled loop names."""
        tiled = set()
        for d in strategy.decisions:
            if d.kind == "tile":
                tiled.add(d.params.get("loop", ""))
        return tiled

    def _get_fused_node_sets(self, strategy: StrategyIR) -> set[frozenset[str]]:
        """Get already-fused node sets."""
        fused = set()
        for d in strategy.decisions:
            if d.kind == "fuse":
                nodes = d.params.get("ops", d.params.get("nodes", []))
                fused.add(frozenset(nodes))
        return fused

    def _get_parallel_loops(self, strategy: StrategyIR) -> set[str]:
        """Get already-parallelized loops."""
        parallel = set()
        for d in strategy.decisions:
            if d.kind == "parallel":
                for loop in d.params.get("loops", []):
                    parallel.add(loop)
        return parallel

    def _get_placed_tensors(self, strategy: StrategyIR) -> set[str]:
        """Get tensors already placed in specific memory."""
        placed = set()
        for d in strategy.decisions:
            if d.kind == "place":
                placed.add(d.params.get("tensor", ""))
        return placed

    def _get_reduction_loops(self, semantic: SemanticIR) -> list[str]:
        """Get reduction loop variables."""
        reduction = []
        for node in semantic.nodes:
            for var in node.semantics.reduction_axes:
                if var not in reduction:
                    reduction.append(var)
        return reduction

    def _estimate_shared_memory_for_tile(
        self,
        loop_name: str,
        factors: list[int],
        semantic: SemanticIR,
        strategy: StrategyIR,
    ) -> int:
        """Estimate shared memory needed for a tile decision."""
        # Simplified: tile_size * dtype_bytes * num_affected_tensors
        tile_size = factors[0] if factors else 64
        dtype_bytes = 2  # f16 default
        return tile_size * tile_size * dtype_bytes  # Very rough estimate

    def _estimate_current_shared(self, strategy: StrategyIR) -> int:
        """Estimate current shared memory usage from decisions."""
        total = 0
        for d in strategy.decisions:
            if d.kind == "place" and d.params.get("memory") == "shared":
                total += 4096  # Placeholder per placement
        return total

    def _estimate_param_tile_size(self, param: Any, strategy: StrategyIR) -> int:
        """Estimate tile size for a parameter."""
        # Based on tiling decisions
        tile_dim = 64  # default
        dtype_bytes = 2 if param.dtype in ("f16", "bf16") else 4
        return tile_dim * tile_dim * dtype_bytes

    def _estimate_reuse_factor(self, param_name: str, semantic: SemanticIR) -> int:
        """Estimate how many times a parameter tile is reused."""
        # For matmul: A is reused across j (N times), B across i (M times)
        for node in semantic.nodes:
            if node.op == "matmul":
                if len(node.output.shape) >= 2:
                    M, N = node.output.shape[0], node.output.shape[1]
                    # A is used in the i,k loops, reused across j
                    if param_name in ("A",):
                        return N // 64  # Rough: N / tile_j
                    elif param_name in ("B",):
                        return M // 64  # Rough: M / tile_i
        return 1

    def _estimate_fusion_benefit(self, fg: Any, semantic: SemanticIR) -> str:
        """Estimate benefit of a fusion."""
        # Estimate eliminated memory traffic
        for node_id in fg.nodes:
            node = semantic.get_node(node_id)
            if node and node.output.shape:
                size = 1
                for dim in node.output.shape:
                    size *= dim
                dtype_bytes = 2  # f16
                bytes_saved = size * dtype_bytes
                return f"eliminates {bytes_saved // (1024*1024)}MB intermediate write"
        return "reduces memory traffic"

    def _tile_priority(
        self,
        loop_name: str,
        factors: list[int],
        loop_bound: int,
        warp_size: int,
        semantic: SemanticIR,
    ) -> float:
        """Score a tile candidate. Higher = better."""
        priority = 5.0

        outer = factors[0]
        inner = factors[1] if len(factors) > 1 else outer

        # Prefer power-of-2 factors
        if outer & (outer - 1) == 0:
            priority += 1.0

        # Prefer inner tile aligned with warp size
        if inner % warp_size == 0 or warp_size % inner == 0:
            priority += 1.0

        # Prefer tiles that give enough blocks for occupancy
        if loop_bound > 0:
            num_blocks = loop_bound // outer
            if 4 <= num_blocks <= 256:
                priority += 1.0
            elif num_blocks > 256:
                priority += 0.5

        # Prefer moderate tile sizes (not too small, not too large)
        if 32 <= outer <= 128:
            priority += 0.5

        return priority

    def _generate_hint(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> str:
        """Generate a natural language hint for the LLM."""
        tiled = self._get_tiled_loops(strategy)
        fused = self._get_fused_node_sets(strategy)
        step = strategy.decision_count

        if step == 0:
            if semantic.fusion_groups:
                return "Start with fusion — it's usually the highest-impact first move."
            return "Start with tiling the main computation loops."

        if not fused and semantic.fusion_groups:
            return "Consider fusing operators before tiling for better results."

        if tiled and not self._get_parallel_loops(strategy):
            return "Loops are tiled. Consider parallelization (map outer loops to GPU blocks)."

        if tiled and not self._get_placed_tensors(strategy):
            return "Consider placing frequently-reused tiles in shared memory."

        return ""
