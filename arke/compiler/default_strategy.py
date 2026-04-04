# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Default Strategy Generator.

When a .ak file has no `strategy` block, the Arke compiler calls
DefaultStrategyGenerator to produce a reasonable baseline StrategyIR
based on operator type and hardware profile.

Design:
  - Rule-based heuristics derived from the hardware profile
  - Covers all 13 ops in OP_CATALOG (Cat A-E)
  - Each decision includes a @rationale explaining WHY the value was chosen
  - Output is a valid StrategyIR that can be further refined by the LLM Agent

Usage:
    from arke.compiler.default_strategy import DefaultStrategyGenerator

    gen = DefaultStrategyGenerator(hw_profile)
    strategy = gen.generate(semantic_ir)
"""

from __future__ import annotations

import math
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


# ─── Heuristic helpers ────────────────────────────────────────────────────────

def _shared_mem_bytes(hw: dict) -> int:
    """Return shared memory per CU in bytes from hw profile."""
    for level in hw.get("memory_hierarchy", []):
        if level.get("name") == "shared":
            return int(level.get("size_per_cu", 49152))
    return 49152  # Ampere default


def _warp_size(hw: dict) -> int:
    return int(hw.get("constraints", {}).get("warp_size", 32))


def _max_threads(hw: dict) -> int:
    return int(hw.get("constraints", {}).get("max_threads_per_block", 1024))


def _tensor_core_shape(hw: dict) -> tuple[int, int, int] | None:
    """Return (M, N, K) tensor core tile shape if available."""
    shapes = hw.get("matrix_unit", {}).get("shapes", [])
    if shapes:
        s = shapes[0]
        return (s[0], s[1], s[2])
    return None


def _dtype_bytes(dtype: str) -> int:
    return {"f16": 2, "bf16": 2, "f32": 4, "f64": 8,
            "i8": 1, "i16": 2, "i32": 4, "i64": 8, "u8": 1}.get(dtype, 2)


def _good_block_size(n: int, warp: int = 32, max_threads: int = 1024) -> int:
    """Round n up to next power-of-2 multiple of warp, capped at max_threads."""
    candidates = [warp * i for i in [1, 2, 4, 8, 16, 32] if warp * i <= max_threads]
    for c in reversed(candidates):
        if c <= n:
            return c
    return warp


# ─── Op-specific strategy builders ───────────────────────────────────────────

def _strategy_matmul(
    ir: SemanticIR,
    strategy: StrategyIR,
    hw: dict,
    op_node,
) -> None:
    """Generate default strategy for matmul / batch_matmul."""
    tc = _tensor_core_shape(hw)
    smem = _shared_mem_bytes(hw)
    dtype = ir.params[0].dtype if ir.params else "f16"
    elem_bytes = _dtype_bytes(dtype)

    # Tile sizes: fill ~half of shared memory with A + B tiles
    # A_tile: BLOCK_M × BLOCK_K,  B_tile: BLOCK_K × BLOCK_N
    # BLOCK_M * BLOCK_K * 2 * elem_bytes ≤ smem / 2
    if tc:
        # Align to tensor core shapes
        block_m = max(tc[0] * 4, 64)   # e.g. 64 for [16,8,16]
        block_n = max(tc[1] * 8, 64)
        block_k = tc[2]                  # e.g. 16 — inner K tile
        block_m = min(block_m, 128)
        block_n = min(block_n, 128)
    else:
        block_m = 64
        block_n = 64
        block_k = 32

    # Ensure tiles fit in shared memory
    tile_bytes = (block_m * block_k + block_k * block_n) * elem_bytes
    while tile_bytes > smem // 2 and block_k > 16:
        block_k //= 2
        tile_bytes = (block_m * block_k + block_k * block_n) * elem_bytes

    is_batch = (op_node.op == "batch_matmul")
    tc_note = f"aligned to tensor core {tc[0]}×{tc[1]}×{tc[2]}" if tc else "no tensor cores"

    if is_batch:
        strategy.tile("M", [block_m],
            rationale=f"M-tile={block_m} for thread-block row coverage ({tc_note})")
        strategy.tile("N", [block_n],
            rationale=f"N-tile={block_n} for thread-block col coverage ({tc_note})")
        strategy.tile("K", [block_k],
            rationale=f"K-tile={block_k}: A+B tiles = {tile_bytes}B ≤ smem/2 ({smem//2}B)")
        strategy.reorder(["batch", "M", "N", "K"],
            rationale="batch/M/N outer for parallelism, K inner for reduction")
        strategy.parallel(["batch", "M", "N"],
            {"batch": "blockIdx.z", "M": "blockIdx.x", "N": "blockIdx.y"},
            rationale="batch→z, M→x, N→y: each thread block owns one output tile per batch")
    else:
        strategy.tile("M", [block_m],
            rationale=f"M-tile={block_m} for thread-block row coverage ({tc_note})")
        strategy.tile("N", [block_n],
            rationale=f"N-tile={block_n} for thread-block col coverage ({tc_note})")
        strategy.tile("K", [block_k],
            rationale=f"K-tile={block_k}: A+B tiles = {tile_bytes}B ≤ smem/2 ({smem//2}B)")
        strategy.reorder(["M", "N", "K"],
            rationale="outer M/N for parallelism, inner K for reduction")
        strategy.parallel(["M", "N"],
            {"M": "blockIdx.x", "N": "blockIdx.y"},
            rationale="M→blockIdx.x, N→blockIdx.y: each block computes one output tile")

    strategy.place("A_tile", "shared",
        rationale=f"A tile ({block_m}×{block_k}×{elem_bytes}B={block_m*block_k*elem_bytes}B) "
                  f"in shared memory for K-loop reuse")
    strategy.place("B_tile", "shared",
        rationale=f"B tile ({block_k}×{block_n}×{elem_bytes}B={block_k*block_n*elem_bytes}B) "
                  f"in shared memory for K-loop reuse")


def _strategy_elementwise(
    ir: SemanticIR,
    strategy: StrategyIR,
    hw: dict,
    op_node,
) -> None:
    """Generate default strategy for elementwise ops (relu/gelu/silu/add/mul)."""
    warp = _warp_size(hw)
    max_t = _max_threads(hw)
    dtype = ir.params[0].dtype if ir.params else "f16"
    elem_bytes = _dtype_bytes(dtype)

    # For elementwise: one thread per element, vectorized loads
    # block_size = min(1024, next power-of-2 of N)
    shape = ir.params[0].shape if ir.params else [1024, 1024]
    n_cols = shape[-1] if len(shape) >= 2 else shape[0]

    block_size = _good_block_size(n_cols, warp, max_t)
    vec_width = min(8, 128 // (elem_bytes * 8))  # vector load width in elements
    vec_width = max(1, vec_width)

    strategy.tile("N", [block_size],
        rationale=f"block_size={block_size}: one thread block per row, "
                  f"covers N={n_cols} elements with coalesced access")
    strategy.parallel(["M"],
        {"M": "blockIdx.x"},
        rationale="each thread block processes one row (M dimension mapped to blockIdx.x)")
    strategy.add_decision(
        __import__("arke.ir.strategy", fromlist=["Decision"]).Decision(
            kind="vectorize",
            params={"width": vec_width, "loop": "N"},
            rationale=__import__("arke.ir.strategy", fromlist=["Rationale"]).Rationale(
                text=f"vectorize N with width={vec_width}: "
                     f"{vec_width*elem_bytes}B load = {'optimal' if vec_width*elem_bytes>=8 else 'partial'} "
                     f"128-bit transaction for {dtype}"
            ),
        )
    )


def _strategy_reduce(
    ir: SemanticIR,
    strategy: StrategyIR,
    hw: dict,
    op_node,
) -> None:
    """Generate default strategy for row-wise reduce ops (softmax/layernorm/rmsnorm)."""
    warp = _warp_size(hw)
    max_t = _max_threads(hw)
    smem = _shared_mem_bytes(hw)
    dtype = ir.params[0].dtype if ir.params else "f16"
    elem_bytes = _dtype_bytes(dtype)

    shape = ir.params[0].shape if ir.params else [1024, 1024]
    n_cols = shape[-1] if len(shape) >= 2 else shape[0]

    # For row-wise ops: one block per row, threads reduce across columns
    # block_size = min(max_threads, next power-of-2 covering N)
    block_size = warp
    while block_size < n_cols and block_size < max_t:
        block_size *= 2
    block_size = min(block_size, max_t)

    # Check if row fits in shared memory
    row_bytes = n_cols * elem_bytes
    use_shared = row_bytes <= smem // 2

    strategy.tile("N", [block_size],
        rationale=f"block_size={block_size} threads per row: "
                  f"{'covers entire row in shared mem' if use_shared else 'partial rows, iterative reduction'} "
                  f"(N={n_cols}, smem/2={smem//2}B)")
    strategy.parallel(["M"],
        {"M": "blockIdx.x"},
        rationale="one thread block per row (M→blockIdx.x); rows are independent reduction tasks")

    if use_shared:
        strategy.place("row_buf", "shared",
            rationale=f"row buffer ({row_bytes}B) fits in shared memory ({smem}B); "
                      f"load once, compute in registers")
    else:
        strategy.place("partial_sum", "shared",
            rationale=f"partial reduction results ({block_size*elem_bytes}B) in shared memory "
                      f"for warp-level reduction; row too large for full load")


def _strategy_transpose(
    ir: SemanticIR,
    strategy: StrategyIR,
    hw: dict,
    op_node,
) -> None:
    """Generate default strategy for transpose."""
    warp = _warp_size(hw)
    smem = _shared_mem_bytes(hw)
    dtype = ir.params[0].dtype if ir.params else "f16"
    elem_bytes = _dtype_bytes(dtype)

    # Classic tiled transpose to avoid non-coalesced writes
    # tile size = 32×32 avoids shared memory bank conflicts with padding
    tile = 32
    tile_bytes = tile * (tile + 1) * elem_bytes  # +1 col padding for bank conflicts

    while tile_bytes > smem // 2 and tile > 8:
        tile //= 2
        tile_bytes = tile * (tile + 1) * elem_bytes

    strategy.tile("M", [tile],
        rationale=f"tile_M={tile}: tiled transpose reads coalesced from global memory")
    strategy.tile("N", [tile],
        rationale=f"tile_N={tile}: tiled transpose writes coalesced to global memory")
    strategy.parallel(["M", "N"],
        {"M": "blockIdx.y", "N": "blockIdx.x"},
        rationale="M→blockIdx.y, N→blockIdx.x: standard 2D grid for transpose")
    strategy.place("tile_buf", "shared",
        rationale=f"32×33 tile buffer ({tile_bytes}B) in shared memory: "
                  f"+1 column padding eliminates bank conflicts")


# ─── Dispatch table ───────────────────────────────────────────────────────────

_OP_CATEGORIES = {
    # compute (matmul-like)
    "matmul":       "compute",
    "batch_matmul": "compute",
    # elementwise
    "relu":   "elementwise",
    "gelu":   "elementwise",
    "silu":   "elementwise",
    "add":    "elementwise",
    "mul":    "elementwise",
    # row-wise reduce
    "softmax":    "reduce",
    "layernorm":  "reduce",
    "rmsnorm":    "reduce",
    "reduce_max": "reduce",
    "reduce_sum": "reduce",
    # move
    "transpose": "move",
}

_STRATEGY_FN = {
    "compute":     _strategy_matmul,
    "elementwise": _strategy_elementwise,
    "reduce":      _strategy_reduce,
    "move":        _strategy_transpose,
}


# ─── Public API ───────────────────────────────────────────────────────────────

class DefaultStrategyGenerator:
    """Generate a baseline StrategyIR from SemanticIR + hardware profile.

    Called by the Arke compiler when a .ak file has no `strategy` block.

    Example::

        gen = DefaultStrategyGenerator(hw_profile)
        strategy = gen.generate(semantic_ir)
        # → StrategyIR with heuristic tile/parallel/place decisions

    The generated strategy is a starting point. The LLM Agent can then
    refine it using the normal tool-use loop.
    """

    def __init__(self, hw_profile: dict[str, Any]) -> None:
        self.hw = hw_profile

    def generate(self, ir: SemanticIR) -> StrategyIR:
        """Generate a default StrategyIR for the given SemanticIR.

        Dispatches per-operator based on the first (or only) node's op type.
        For fused kernels with multiple nodes, generates decisions for the
        dominant op and adds a fuse decision for the remaining ops.
        """
        hw_name = self.hw.get("name", "unknown")
        strategy = StrategyIR(
            kernel_id=ir.kernel_id,
            target_hw=hw_name,
        )

        if not ir.nodes:
            return strategy

        # Identify dominant op (first non-elementwise node, or first node)
        dominant_node = ir.nodes[0]
        for node in ir.nodes:
            if _OP_CATEGORIES.get(node.op, "elementwise") in ("compute", "reduce", "move"):
                dominant_node = node
                break

        # Dispatch to op-specific strategy builder
        cat = _OP_CATEGORIES.get(dominant_node.op, "elementwise")
        builder_fn = _STRATEGY_FN.get(cat, _strategy_elementwise)
        builder_fn(ir, strategy, self.hw, dominant_node)

        # If there are additional epilogue ops (e.g. matmul → relu/gelu),
        # add a fuse decision for them
        if len(ir.nodes) > 1:
            epilogue_ops = [
                n.id for n in ir.nodes[1:]
                if _OP_CATEGORIES.get(n.op, "elementwise") == "elementwise"
            ]
            if epilogue_ops:
                all_ops = [dominant_node.id] + epilogue_ops
                strategy.fuse(
                    all_ops,
                    fusion_type="epilogue",
                    rationale=f"fuse {dominant_node.op} + "
                              f"{', '.join(n.op for n in ir.nodes[1:])} as epilogue: "
                              f"eliminates intermediate global memory write",
                )

        return strategy
