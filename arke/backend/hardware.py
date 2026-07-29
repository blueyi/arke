# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — explicit HardwareModel abstraction (K-H2).

Before K-H2 the notion of "target hardware" was scattered across several
partial descriptors, each serving one consumer:

  - ``compiler/passes/base.py::HardwareProfile``  (pass pipeline)
  - ``backend/gpu_tuning.py::GPUProfile``          (launch tuning)
  - per-backend ``chip`` strings ("sm_86")          (codegen)

None of them described the *structure* an optimizing agent needs to reason
about legal moves: the memory hierarchy (register → shared → L2 → global),
the synchronization domains (warp / block / device), the compute-unit
descriptors (SIMT cores, tensor cores), and the alignment constraints those
impose on tiling. ``HardwareModel`` is that structured, backend-agnostic
description.

It is deliberately a *data* model — no codegen, no Triton/MLIR/CUDA
specifics. A backend maps its concrete target onto a ``HardwareModel``
(``TritonBackend`` / ``MLIRGPUBackend`` / ``CudaCBackend`` / ``LLVMBackend``
all target the same RTX 3060 SM 8.6 model today). The StrategyIR legal-action
generator consumes it to bound tile factors, pipeline stages, and tensor-core
availability; the pass pipeline reads it in place of the old flat
``HardwareProfile``.

Extensibility seam (Ascend PAUSED but must stay pluggable): a future
Ascend / AMD / other accelerator supplies its own ``HardwareModel`` instance
(different memory tree, sync domains, compute units) with no change to the
model schema — the same way ``ArkeBackend`` + ``BackendRegistry`` keep the
backend surface pluggable. See docs/architecture/arke-compiler-infrastructure.md §7.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─── Memory hierarchy ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryLevel:
    """One level of the memory hierarchy.

    ``scope`` names the sharing domain the level lives in:
      - "thread"  : private to a lane (registers)
      - "block"   : shared across a thread block / workgroup (shared memory)
      - "device"  : global to the whole device (L2, HBM/global)
    """
    name: str                 # "register", "shared", "l2", "global"
    scope: str                # "thread" | "block" | "device"
    size_bytes: int           # capacity per scope instance (0 = effectively unbounded)
    bandwidth_gbps: float = 0.0   # peak bandwidth (0 = unknown / not modeled)
    latency_cycles: int = 0       # approx access latency (0 = unknown)


# ─── Synchronization domains ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SyncDomain:
    """A synchronization scope and the width of the group it synchronizes.

    Examples (NVIDIA):
      SyncDomain("warp", width=32, barrier_free=True)   # implicit lockstep
      SyncDomain("block", width=1024)                    # __syncthreads()
      SyncDomain("device", width=0)                      # grid-level (kernel boundary)
    """
    name: str                 # "warp" | "block" | "device"
    width: int                # lanes/threads in the group (0 = unbounded/grid)
    barrier_free: bool = False  # True when sync is implicit (e.g. warp lockstep)


# ─── Compute units ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComputeUnit:
    """A class of execution resource on the device.

    kind: "simt"        — scalar/vector SIMT lanes
          "tensor_core"  — matrix-multiply-accumulate units (MMA / WMMA)
    """
    kind: str
    count: int = 0                     # units per SM (0 = unknown)
    supported_dtypes: tuple[str, ...] = ()   # e.g. ("f16", "bf16", "tf32")
    peak_tflops: float = 0.0


# ─── Alignment constraints ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AlignmentConstraints:
    """Alignment / granularity rules a tiling strategy must respect."""
    warp_size: int = 32
    # Tensor-core MMA tile granularity (m, n, k). () when no TC.
    mma_tile: tuple[int, ...] = ()
    # Preferred vector width (elements) for coalesced global loads.
    vector_width: int = 4
    # Shared-memory bank width in bytes (for conflict-free tiling).
    shared_bank_bytes: int = 4


# ─── Top-level hardware model ────────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareModel:
    """Structured, backend-agnostic description of a compute target.

    An optimizing agent reads this to bound its legal action space (tile
    factors ≤ what shared memory holds, pipeline stages ≤ what latency
    hiding needs, tensor-core moves only when a TC ComputeUnit exists, …).
    Backends map their concrete target onto one instance of this model.
    """
    name: str
    compute_capability: tuple[int, int]
    num_sms: int
    max_threads_per_block: int
    max_threads_per_sm: int
    memory_levels: tuple[MemoryLevel, ...]
    sync_domains: tuple[SyncDomain, ...]
    compute_units: tuple[ComputeUnit, ...]
    alignment: AlignmentConstraints = field(default_factory=AlignmentConstraints)

    # ── Convenience queries (agent / tuning consume these) ──────────────

    def memory_level(self, name: str) -> MemoryLevel | None:
        for m in self.memory_levels:
            if m.name == name:
                return m
        return None

    def shared_memory_bytes(self) -> int:
        m = self.memory_level("shared")
        return m.size_bytes if m else 0

    def has_tensor_core(self) -> bool:
        return any(cu.kind == "tensor_core" for cu in self.compute_units)

    def tensor_core(self) -> ComputeUnit | None:
        for cu in self.compute_units:
            if cu.kind == "tensor_core":
                return cu
        return None

    def sync_domain(self, name: str) -> SyncDomain | None:
        for s in self.sync_domains:
            if s.name == name:
                return s
        return None

    @property
    def warp_size(self) -> int:
        return self.alignment.warp_size


# ─── Concrete instances ──────────────────────────────────────────────────────

def nvidia_sm86(name: str = "nvidia_ampere") -> HardwareModel:
    """RTX 3060 Laptop (SM 8.6) — the primary Phase-1..5 NVIDIA target.

    Numbers reconciled with the two legacy descriptors this replaces
    (``compiler/passes/base.py::HardwareProfile`` and
    ``backend/gpu_tuning.py::GPUProfile``).
    """
    return HardwareModel(
        name=name,
        compute_capability=(8, 6),
        num_sms=30,
        max_threads_per_block=1024,
        max_threads_per_sm=1536,
        memory_levels=(
            MemoryLevel("register", "thread", size_bytes=256 * 1024,  # 64K 32-bit regs/SM
                        bandwidth_gbps=0.0, latency_cycles=1),
            MemoryLevel("shared", "block", size_bytes=49152,
                        bandwidth_gbps=0.0, latency_cycles=30),
            MemoryLevel("l2", "device", size_bytes=3 * 1024 * 1024,
                        bandwidth_gbps=0.0, latency_cycles=200),
            MemoryLevel("global", "device", size_bytes=6 * 1024 * 1024 * 1024,
                        bandwidth_gbps=192.0, latency_cycles=400),
        ),
        sync_domains=(
            SyncDomain("warp", width=32, barrier_free=True),
            SyncDomain("block", width=1024, barrier_free=False),
            SyncDomain("device", width=0, barrier_free=False),
        ),
        compute_units=(
            ComputeUnit("simt", count=128, supported_dtypes=("f16", "bf16", "f32", "i32"),
                        peak_tflops=12.74),
            # Ampere 3rd-gen tensor cores: fp16/bf16/tf32 MMA, 16x8x16 granularity.
            ComputeUnit("tensor_core", count=4, supported_dtypes=("f16", "bf16", "tf32"),
                        peak_tflops=101.0),
        ),
        alignment=AlignmentConstraints(
            warp_size=32,
            mma_tile=(16, 8, 16),
            vector_width=4,
            shared_bank_bytes=4,
        ),
    )


# Default model for the current dev target.
DEFAULT_HARDWARE = nvidia_sm86()
