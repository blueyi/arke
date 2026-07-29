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


def ascend_910b(name: str = "ascend_910b") -> HardwareModel:
    """⚠️ PAPER EXERCISE — Huawei Ascend 910B (DaVinci arch), NOT validated.

    Audit R4 (docs/audit/2026-07-29-architecture-audit.md): the HardwareModel
    schema has only ever been instantiated for one target (nvidia_sm86), so its
    "泛化力" was never stress-tested. This instance fills the schema from public
    Ascend 910B specs to surface — *at design time* — which fields the current
    NVIDIA-shaped schema cannot express for a SIMD/Cube accelerator. Nothing in
    the codebase runs on it; it exists to drive the dry-run gap analysis in
    ``tests/backend/test_hardware_model_ascend_dryrun.py`` and the gap list in
    docs/architecture/arke-compiler-infrastructure.md §7.7.

    KNOWN SCHEMA MISFITS discovered by filling this in (the point of R4):

    1. **No SIMT/warp concept.** DaVinci is SIMD: a Cube unit does a fixed
       16x16x16 MMA and Vector units do wide SIMD. There is no 32-lane warp and
       no implicit lockstep. We model the Cube as a ``tensor_core`` ComputeUnit
       and Vector as ``simt`` (a lie of convenience) and set warp_size=1, but
       ``SyncDomain("warp", barrier_free=True)`` has no Ascend analog — the
       schema conflates "SIMT lane group" with "sync scope".
    2. **Richer on-chip memory tree.** Ascend exposes L1 (unified buffer /
       "UB"), plus L0A / L0B / L0C feeding the Cube (input/input/accum). The
       flat ``memory_levels`` list with scope ∈ {thread,block,device} can list
       them but cannot express that L0A/L0B are *operand-specific* Cube feeders
       (not general scratch) nor the GM→L1→L0 staging DMA the compiler must
       schedule explicitly. ``scope`` has no value for "cube-operand".
    3. **Explicit DMA / no cache coherence.** NVIDIA L2 is a transparent cache;
       Ascend movement between GM/L1/L0 is explicit engine-scheduled DMA. The
       schema has no field for "software-managed vs hardware-cached", so a
       StrategyIR legal-action generator reading this model can't tell it must
       *emit copies* rather than rely on a cache.
    4. **mma_tile is a single (m,n,k).** Ascend Cube is fixed 16x16x16 for fp16
       but the fractal/zZ data layout it requires (nZ/zN) is unrepresentable —
       ``AlignmentConstraints`` has no "operand memory layout" field.

    These four are the concrete refactor list for when Ascend is un-paused;
    R4's value is having them *before* writing a line of Ascend codegen.
    """
    return HardwareModel(
        name=name,
        compute_capability=(0, 0),   # SCHEMA MISFIT #1: CUDA-ism, no Ascend analog
        num_sms=32,                  # ~32 AI Cores (DaVinci cores), approx public spec
        max_threads_per_block=0,     # SCHEMA MISFIT #1: no thread-block model on SIMD
        max_threads_per_sm=0,        # SCHEMA MISFIT #1: same
        memory_levels=(
            # SCHEMA MISFIT #2/#3: L0A/L0B/L0C are Cube-operand feeders, not
            # general "thread"/"block" scratch; GM<->L1<->L0 is explicit DMA.
            MemoryLevel("l0c", "block", size_bytes=256 * 1024, latency_cycles=1),   # accumulator
            MemoryLevel("l0a", "block", size_bytes=64 * 1024, latency_cycles=1),    # Cube lhs feeder
            MemoryLevel("l0b", "block", size_bytes=64 * 1024, latency_cycles=1),    # Cube rhs feeder
            MemoryLevel("l1", "block", size_bytes=1024 * 1024, latency_cycles=30),  # unified buffer (UB)
            MemoryLevel("global", "device", size_bytes=64 * 1024 * 1024 * 1024,
                        bandwidth_gbps=400.0, latency_cycles=400),                  # HBM
        ),
        sync_domains=(
            # SCHEMA MISFIT #1: "aicore" is a compute-engine group, not a
            # warp/block sync scope; barrier_free is meaningless here.
            SyncDomain("aicore", width=1, barrier_free=False),
            SyncDomain("device", width=0, barrier_free=False),
        ),
        compute_units=(
            # SCHEMA MISFIT #1: Vector unit modeled as "simt" is a convenience lie.
            ComputeUnit("simt", count=1, supported_dtypes=("f16", "f32"), peak_tflops=0.0),
            # Cube MMA unit, fixed 16x16x16 fp16.
            ComputeUnit("tensor_core", count=1, supported_dtypes=("f16",), peak_tflops=376.0),
        ),
        alignment=AlignmentConstraints(
            warp_size=1,             # SCHEMA MISFIT #1: no warp
            mma_tile=(16, 16, 16),   # SCHEMA MISFIT #4: fractal layout unexpressed
            vector_width=16,
            shared_bank_bytes=32,
        ),
    )
