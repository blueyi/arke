# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — V0 Static Validator.

Validates Strategy IR decisions against hardware constraints and IR invariants.
Runs after every apply_decision (<1ms), provides immediate feedback to LLM.

Checks:
1. Tile legality (positive factors, non-empty)
2. Hardware constraints (shared memory, threads)
3. No duplicate transforms (same loop tiled twice)
4. Fusion legality (nodes connected, type valid)
5. Resource estimation (shared memory budget tracking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR
from arke.lang.types import dtype_bits


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str = ""
    violations: list[str] = field(default_factory=list)


@dataclass
class ResourceEstimate:
    """Estimated resource usage from current strategy."""
    shared_memory_bytes: int = 0
    estimated_threads_per_block: int = 0
    estimated_blocks: int = 0
    register_pressure: str = "low"  # low | medium | high


@dataclass
class ValidationResult:
    """Aggregate result of all V0 checks."""
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> list[str]:
        """Return all violation messages from all checks."""
        v = []
        for c in self.checks:
            v.extend(c.violations)
        return v


class StaticValidator:
    """V0 Static Validator — runs after every apply_decision.

    Design: fast (<1ms), conservative (reject clearly bad, accept unclear).
    """

    def validate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> ValidationResult:
        """Validate strategy IR against hardware constraints and IR invariants."""
        checks = [
            self._check_tile_legality(semantic, strategy),
            self._check_hw_constraints(semantic, strategy, hw_profile),
            self._check_no_duplicate_transforms(strategy),
            self._check_fusion_legality(semantic, strategy),
            self._check_launch_config(strategy),
            self._check_autotune(strategy),
        ]

        resources = self._estimate_resources(semantic, strategy, hw_profile)

        return ValidationResult(
            passed=all(c.passed for c in checks),
            checks=checks,
            resource_usage={
                "shared_memory_bytes": resources.shared_memory_bytes,
                "estimated_threads_per_block": resources.estimated_threads_per_block,
                "estimated_blocks": resources.estimated_blocks,
                "register_pressure": resources.register_pressure,
            },
        )

    def _check_tile_legality(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> CheckResult:
        """Check that tile factors are valid."""
        violations = []
        for d in strategy.decisions:
            if d.kind == "tile":
                factors = d.params.get("factors", [])
                if not factors:
                    violations.append(f"Step #{d.step}: empty tile factors")
                for f in factors:
                    if not isinstance(f, int) or f <= 0:
                        violations.append(
                            f"Step #{d.step}: tile factor {f}"
                            " must be positive integer"
                        )
        return CheckResult(
            name="tile_legality",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _check_hw_constraints(
        self, semantic: SemanticIR, strategy: StrategyIR, hw_profile: dict
    ) -> CheckResult:
        """Check hardware resource limits."""
        violations = []
        constraints = hw_profile.get("constraints", {})

        resources = self._estimate_resources(semantic, strategy, hw_profile)

        # Shared memory check
        sm_limit = constraints.get("max_shared_memory_per_block", 49152)
        if resources.shared_memory_bytes > sm_limit:
            overshoot = resources.shared_memory_bytes - sm_limit
            violations.append(
                f"Shared memory {resources.shared_memory_bytes}B exceeds limit {sm_limit}B "
                f"(over by {overshoot}B). Consider: reduce tile sizes, place fewer tensors "
                f"in shared memory, or use smaller dtypes."
            )

        # Thread count check
        max_threads = constraints.get("max_threads_per_block", 1024)
        if resources.estimated_threads_per_block > max_threads:
            violations.append(
                f"Estimated threads per block"
                f" {resources.estimated_threads_per_block}"
                f" exceeds limit {max_threads}"
            )

        return CheckResult(
            name="hw_constraints",
            passed=len(violations) == 0,
            violations=violations,
            message=f"shared_memory: {resources.shared_memory_bytes}/{sm_limit}",
        )

    def _check_no_duplicate_transforms(self, strategy: StrategyIR) -> CheckResult:
        """Check no loop is tiled twice, no nodes fused twice."""
        violations = []

        tiled_loops: set[str] = set()
        fused_pairs: set[frozenset[str]] = set()

        for d in strategy.decisions:
            if d.kind == "tile":
                loop = d.params.get("loop", "")
                if loop in tiled_loops:
                    violations.append(f"Step #{d.step}: loop '{loop}' already tiled")
                tiled_loops.add(loop)

            elif d.kind == "fuse":
                nodes = frozenset(d.params.get("ops", d.params.get("nodes", [])))
                if nodes in fused_pairs:
                    violations.append(f"Step #{d.step}: nodes {sorted(nodes)} already fused")
                fused_pairs.add(nodes)

        return CheckResult(
            name="no_duplicate_transforms",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _check_fusion_legality(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> CheckResult:
        """Check fusion decisions reference valid nodes with valid types."""
        violations = []
        valid_node_ids = {n.id for n in semantic.nodes}
        valid_fusion_types = {"epilogue", "prologue", "horizontal", "vertical"}

        for d in strategy.decisions:
            if d.kind == "fuse":
                nodes = d.params.get("ops", d.params.get("nodes", []))
                ftype = d.params.get("type", "")

                for nid in nodes:
                    if nid not in valid_node_ids:
                        violations.append(
                            f"Step #{d.step}: fusion references unknown node '{nid}'"
                        )

                if ftype and ftype not in valid_fusion_types:
                    violations.append(
                        f"Step #{d.step}: invalid fusion type '{ftype}'"
                    )

        return CheckResult(
            name="fusion_legality",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _check_launch_config(self, strategy: StrategyIR) -> CheckResult:
        """Validate launch_config decisions."""
        violations: list[str] = []
        valid_warps = (1, 2, 4, 8, 16, 32)

        for d in strategy.decisions:
            if d.kind != "launch_config":
                continue

            # Validate num_warps
            nw = d.params.get("num_warps")
            if nw is not None:
                if isinstance(nw, int):
                    if nw not in valid_warps:
                        violations.append(
                            f"Step #{d.step}: invalid num_warps {nw},"
                            f" must be one of {valid_warps}"
                        )
                elif isinstance(nw, list):
                    for w in nw:
                        if w not in valid_warps:
                            violations.append(
                                f"Step #{d.step}: invalid num_warps candidate {w},"
                                f" must be one of {valid_warps}"
                            )

            # Validate num_stages (must be positive int or list of positive ints)
            ns = d.params.get("num_stages")
            if ns is not None:
                if isinstance(ns, int):
                    if ns < 1:
                        violations.append(
                            f"Step #{d.step}: num_stages must be >= 1, got {ns}"
                        )
                elif isinstance(ns, list):
                    for s in ns:
                        if not isinstance(s, int) or s < 1:
                            violations.append(
                                f"Step #{d.step}: invalid num_stages candidate {s},"
                                " must be positive integer"
                            )

            # Validate block_sizes (values must be positive ints or lists of positive ints)
            bs = d.params.get("block_sizes")
            if bs is not None and isinstance(bs, dict):
                for key, val in bs.items():
                    if isinstance(val, int):
                        if val <= 0:
                            violations.append(
                                f"Step #{d.step}: block_sizes['{key}'] must be"
                                f" positive, got {val}"
                            )
                    elif isinstance(val, list):
                        for v in val:
                            if not isinstance(v, int) or v <= 0:
                                violations.append(
                                    f"Step #{d.step}: invalid block_sizes['{key}']"
                                    f" candidate {v}, must be positive integer"
                                )

        return CheckResult(
            name="launch_config",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _check_autotune(self, strategy: StrategyIR) -> CheckResult:
        """Validate autotune decisions."""
        violations: list[str] = []
        valid_warps = (1, 2, 4, 8, 16, 32)

        for d in strategy.decisions:
            if d.kind != "autotune":
                continue

            configs = d.params.get("configs")
            if configs is None or not isinstance(configs, list):
                violations.append(
                    f"Step #{d.step}: autotune decision must have a 'configs' list"
                )
                continue

            if len(configs) == 0:
                violations.append(
                    f"Step #{d.step}: autotune configs list must not be empty"
                )

            for i, cfg in enumerate(configs):
                if not isinstance(cfg, dict):
                    violations.append(
                        f"Step #{d.step}: autotune config[{i}] must be a dict"
                    )
                    continue

                nw = cfg.get("num_warps")
                if nw is not None and isinstance(nw, int) and nw not in valid_warps:
                    violations.append(
                        f"Step #{d.step}: autotune config[{i}] invalid"
                        f" num_warps {nw}, must be one of {valid_warps}"
                    )

                ns = cfg.get("num_stages")
                if ns is not None and isinstance(ns, int) and ns < 1:
                    violations.append(
                        f"Step #{d.step}: autotune config[{i}] num_stages"
                        f" must be >= 1, got {ns}"
                    )

            key = d.params.get("key")
            if key is not None and not isinstance(key, list):
                violations.append(
                    f"Step #{d.step}: autotune 'key' must be a list of strings"
                )

        return CheckResult(
            name="autotune",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _estimate_resources(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> ResourceEstimate:
        """Estimate resource usage from strategy decisions."""
        shared_mem = 0
        threads = 1
        blocks = 1

        # Get dtype size
        dtype_bytes = 2  # f16 default
        if semantic.params:
            bits = dtype_bits(semantic.params[0].dtype)
            dtype_bytes = max(bits // 8, 1)

        # Tile decisions affect shared memory and thread count
        tile_sizes: dict[str, int] = {}
        for d in strategy.decisions:
            if d.kind == "tile":
                loop = d.params.get("loop", "")
                factors = d.params.get("factors", [])
                if factors:
                    tile_sizes[loop] = factors[0]

            elif d.kind == "place" and d.params.get("memory") == "shared":
                # Estimate memory for placed tensor
                # Use tile sizes to estimate tile dimensions
                tile_dim = 64  # default
                if tile_sizes:
                    tile_dim = max(tile_sizes.values())
                shared_mem += tile_dim * tile_dim * dtype_bytes

        # Estimate thread count from innermost tile sizes
        if tile_sizes:
            inner_sizes = []
            for d in strategy.decisions:
                if d.kind == "tile":
                    factors = d.params.get("factors", [])
                    if len(factors) >= 2:
                        inner_sizes.append(factors[-1])
                    elif factors:
                        inner_sizes.append(factors[0])

            if inner_sizes:
                threads = 1
                for s in inner_sizes[:2]:  # First 2 dimensions
                    threads *= s
                threads = min(threads, 1024)

        # Estimate block count from outer tile sizes
        if tile_sizes and semantic.nodes:
            output_shape = semantic.nodes[0].output.shape
            for i, (loop, tile) in enumerate(tile_sizes.items()):
                if i < len(output_shape) and tile > 0:
                    blocks *= max(1, output_shape[i] // tile)

        register_pressure = "low"
        if shared_mem > 16384:
            register_pressure = "medium"
        if shared_mem > 32768:
            register_pressure = "high"

        return ResourceEstimate(
            shared_memory_bytes=shared_mem,
            estimated_threads_per_block=threads,
            estimated_blocks=blocks,
            register_pressure=register_pressure,
        )
