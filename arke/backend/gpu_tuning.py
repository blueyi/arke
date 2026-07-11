# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU kernel tuning policy — shape-aware launch configuration selection.

This module is the SINGLE SOURCE OF TRUTH for GPU kernel launch parameters
(block size, grid layout, tile sizes). All emitter functions import their
launch configs from here, ensuring:
  1. Tuning decisions are centralized (no magic numbers scattered in emitters).
  2. Policies are independently testable (pure functions, no GPU dependency).
  3. The module is extensible (future: auto-tuning hooks, multi-chip dispatch,
     StrategyIR L2 integration).

Architecture principle: the emitters (mlir_emitter.py) describe WHAT the kernel
computes; this module decides HOW it should be launched (parallelization params).
This separation lets us add new tuning heuristics without touching kernel IR.

Usage in emitters:
    from arke.backend.gpu_tuning import rowwise_block_size, elementwise_grid

Hardware assumptions (current): RTX 3060 Laptop, SM 8.6, 30 SMs, 192 GB/s BW.
When multi-chip support is added, these become parameters to the policy functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ── Hardware profile (future: loaded from ArkeBackend.hw_profile()) ──────────

@dataclass(frozen=True)
class GPUProfile:
    """Static hardware characteristics for tuning decisions."""
    chip: str = "sm_86"
    num_sm: int = 30
    max_threads_per_sm: int = 1536
    max_threads_per_block: int = 1024
    warp_size: int = 32
    l2_cache_bytes: int = 3 * 1024 * 1024  # 3 MB
    bandwidth_gbps: float = 192.0

    @property
    def max_warps_per_block(self) -> int:
        return self.max_threads_per_block // self.warp_size


# Default hardware profile (RTX 3060 Laptop)
DEFAULT_GPU = GPUProfile()


# ── Rowwise kernel tuning ────────────────────────────────────────────────────

# Empirical threshold: block=512 wins over block=256 for row dimension D >= this.
# At D=4096, block=512 gives +16% softmax, +52% layernorm, +23% rmsnorm.
# At D<=1024, block=512 regresses (threads under-utilized, 1 extra barrier level).
# At D=2048, results are mixed (slightly positive for layernorm, neutral for others).
_ROWWISE_BLOCK_LARGE_THRESHOLD = 4096
_ROWWISE_BLOCK_DEFAULT = 256
_ROWWISE_BLOCK_LARGE = 512


def rowwise_block_size(D: int, *, hw: GPUProfile = DEFAULT_GPU) -> int:
    """Select optimal block size for a row-wise reduction/norm kernel.

    Args:
        D: Row dimension (number of columns). Each block processes one row,
           with threads cooperating via shared-memory tree-reduce.
        hw: Target hardware profile (for future multi-chip dispatch).

    Returns:
        Block size (number of threads per block). Must be a power of 2 ≤ 1024.

    Design rationale:
        - Larger blocks → more threads per row → fewer elements per thread →
          better instruction-level parallelism and latency hiding.
        - Larger blocks → deeper tree-reduce (log2(block) barriers) → more
          sync overhead.
        - Sweet spot: block=512 amortizes the extra barrier level when D is
          large enough that the per-element savings exceed the barrier cost.
    """
    if D >= _ROWWISE_BLOCK_LARGE_THRESHOLD:
        return _ROWWISE_BLOCK_LARGE
    return _ROWWISE_BLOCK_DEFAULT


# ── Elementwise kernel tuning ────────────────────────────────────────────────

_ELEMENTWISE_BLOCK_DEFAULT = 256


def elementwise_block_size(total_elements: int, *,
                           hw: GPUProfile = DEFAULT_GPU) -> int:
    """Select block size for flat elementwise kernels.

    Current policy: fixed 256 (1 element/thread, flat grid). This saturates
    memory bandwidth at total_elements >= ~64K. Future policy might use larger
    blocks with elements-per-thread for very large tensors on high-BW GPUs.
    """
    return _ELEMENTWISE_BLOCK_DEFAULT


# ── Matmul kernel tuning ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MMAConfig:
    """Tensor-core matmul tile configuration.

    BM = WM * WTM * 16   (M-dim tile per block)
    BN = WN * WTN * 16   (N-dim tile per block)
    BK = BK              (K-dim tile per iteration)
    Threads/block = WM * WN * warp_size
    """
    WM: int = 2
    WN: int = 2
    WTM: int = 2
    WTN: int = 4
    BK: int = 16

    @property
    def BM(self) -> int:
        return self.WM * self.WTM * 16

    @property
    def BN(self) -> int:
        return self.WN * self.WTN * 16

    @property
    def threads_per_block(self) -> int:
        return self.WM * self.WN * 32  # 32 = warp_size


# Production config: BM=64, BN=128, BK=16. Verified 0.91× cuBLAS geomean.
MMA_DEFAULT = MMAConfig(WM=2, WN=2, WTM=2, WTN=4, BK=16)


def matmul_mma_config(M: int, N: int, K: int, *,
                      hw: GPUProfile = DEFAULT_GPU) -> MMAConfig | None:
    """Select MMA tile configuration, or None if MMA is not applicable.

    Returns None when the shape cannot tile evenly with ANY available config,
    in which case the caller falls back to the scalar regblock/tiled ladder.

    Future: shape-adaptive configs (smaller tiles for small shapes to increase
    grid occupancy). Currently uses a single default config.
    """
    cfg = MMA_DEFAULT
    if M % cfg.BM != 0 or N % cfg.BN != 0 or K % cfg.BK != 0:
        return None
    return cfg


# ── Kernel family dispatch (future: StrategyIR L2 integration) ───────────────

KernelFamily = Literal["elementwise", "rowwise", "matmul_mma", "matmul_regblock",
                       "matmul_tiled", "movement", "gated", "index",
                       "attention", "fused"]


def select_kernel_family(op: str, shape: list[int]) -> KernelFamily:
    """Classify an op+shape into its kernel family for dispatch.

    This is informational — the actual dispatch happens in MLIRGPUBackend.lower().
    Useful for logging, auto-tuning, and StrategyIR L2 decision metadata.
    """
    from arke.backend.mlir_emitter import (
        GPU_ELEMENTWISE_OPS, GPU_ROWWISE_OPS, GPU_MOVEMENT_OPS,
        GPU_GATED_OPS, GPU_INDEX_OPS,
    )
    if op in GPU_ELEMENTWISE_OPS:
        return "elementwise"
    if op in GPU_ROWWISE_OPS:
        return "rowwise"
    if op in GPU_MOVEMENT_OPS:
        return "movement"
    if op in GPU_GATED_OPS:
        return "gated"
    if op in GPU_INDEX_OPS:
        return "index"
    if op == "matmul":
        return "matmul_mma"  # default; lower() decides fallback
    if op in ("flash_attention", "cross_attention", "grouped_query_attention",
              "multi_latent_attention", "paged_attention"):
        return "attention"
    return "fused"
