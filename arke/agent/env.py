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
from typing import TYPE_CHECKING, Any

from arke.agent.state import OptimizationBudget, OptimizationState
from arke.compiler.passes.base import HardwareProfile
from arke.ir.ops.registry import REGISTRY
from arke.ir.schedule import ScheduleIR
from arke.ir.strategy import Decision, Rationale

if TYPE_CHECKING:
    from arke.backend.hardware import HardwareModel


# ─── Legal-action candidate generator ─────────────────────────────────────
#
# P1-b (2026-06-24): candidates are now SHAPE- and HARDWARE-aware. Tile /
# unroll / vectorize factors are derived from the operator's real input
# dimensions and filtered against the HardwareProfile (warp size, max
# threads/block, shared memory), instead of a fixed module-level Cartesian
# product. This makes `list_legal_actions` an honest *legality* surface for
# the agent rather than a static menu. Shape-unaware fallbacks are kept for
# robustness when a loop maps to no known dimension.

# Fallback factors when a loop dimension is unknown (shape not resolvable).
_FALLBACK_TILE_FACTORS: tuple[int, ...] = (16, 32, 64, 128)
_FALLBACK_UNROLL_FACTORS: tuple[int, ...] = (2, 4, 8)
_FALLBACK_VECTORIZE_WIDTHS: tuple[int, ...] = (2, 4, 8)
_DEFAULT_PARALLEL_MAPPINGS: tuple[str, ...] = ("threadblock.x", "threadblock.y", "warp")
_DEFAULT_PLACE_MEMORIES: tuple[str, ...] = ("shared", "register")

# Backward-compat alias (older imports/tests referenced this name).
_DEFAULT_TILE_FACTORS: tuple[tuple[int, ...], ...] = (
    (16,), (32,), (64,), (128,),
    (16, 16), (32, 32), (64, 64),
)
_DEFAULT_UNROLL_FACTORS = _FALLBACK_UNROLL_FACTORS
_DEFAULT_VECTORIZE_WIDTHS = _FALLBACK_VECTORIZE_WIDTHS


def _pow2_divisors_up_to(n: int, cap: int) -> list[int]:
    """Powers of two that evenly tile a dimension of size ``n`` (≤ cap).

    Tiles are the canonical GPU tiling granularity. We keep only powers of
    two that don't exceed the dimension (a tile larger than the dim is a
    no-op) and don't exceed ``cap`` (a hardware/threads bound).
    """
    out: list[int] = []
    f = 16
    while f <= min(n, cap):
        out.append(f)
        f *= 2
    if not out:
        # tiny dimension → at least offer the dim itself (or 16 floor)
        out.append(min(max(n, 1), 16))
    return out


def _loop_dim_map(loops: list[str], shapes: dict[str, list[int]]) -> dict[str, int]:
    """Best-effort map loop index var → concrete dimension size.

    Heuristic: flatten all input dims and align by position to the loop
    list (i→first reduced/output axis, j→second, k→contraction). When the
    op exposes a clean 2-D/3-D shape (matmul A[M,K] B[K,N]) this recovers
    M/N/K; otherwise it falls back to the largest available dim.
    """
    dims: list[int] = []
    for shp in shapes.values():
        dims.extend(int(d) for d in shp if isinstance(d, int) and d > 0)
    dim_map: dict[str, int] = {}
    if not dims:
        return dim_map
    biggest = max(dims)
    for idx, loop in enumerate(loops):
        dim_map[loop] = dims[idx] if idx < len(dims) else biggest
    return dim_map


def _enum_tile_candidates(
    loops: list[str],
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    cap = hw.max_threads_per_block if hw else 1024
    dim_map = _loop_dim_map(loops, shapes or {})
    out: list[Decision] = []
    for loop in loops:
        if loop in dim_map:
            factors = _pow2_divisors_up_to(dim_map[loop], cap)
        else:
            factors = list(_FALLBACK_TILE_FACTORS)
        for f in factors:
            out.append(Decision(kind="tile", params={"loop": loop, "factors": [f]}, level=1))
    return out


def _enum_unroll_candidates(
    loops: list[str],
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    dim_map = _loop_dim_map(loops, shapes or {})
    out: list[Decision] = []
    for loop in loops:
        dim = dim_map.get(loop)
        for f in _FALLBACK_UNROLL_FACTORS:
            # Don't offer an unroll factor larger than the loop trip count.
            if dim is not None and f > dim:
                continue
            out.append(Decision(kind="unroll", params={"loop": loop, "factor": f}, level=1))
    return out


def _enum_vectorize_candidates(
    loops: list[str],
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    dim_map = _loop_dim_map(loops, shapes or {})
    out: list[Decision] = []
    for loop in loops:
        dim = dim_map.get(loop)
        for w in _FALLBACK_VECTORIZE_WIDTHS:
            # Vector width must evenly divide the (innermost) dimension.
            if dim is not None and dim % w != 0:
                continue
            out.append(Decision(kind="vectorize", params={"loop": loop, "width": w}, level=1))
    return out


def _enum_parallel_candidates(
    loops: list[str],
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    return [
        Decision(kind="parallel", params={"mapping": {loop: m}}, level=1)
        for loop in loops for m in _DEFAULT_PARALLEL_MAPPINGS
    ]


# Bytes-per-element by dtype tag (conservative; defaults to fp16=2B which is
# the Phase-1 workhorse precision). Used to estimate a tensor's shared-memory
# footprint for the S1 capacity legality check.
_DTYPE_BYTES: dict[str, int] = {
    "f16": 2, "fp16": 2, "bf16": 2, "f32": 4, "fp32": 4,
    "f64": 8, "fp64": 8, "i8": 1, "int8": 1, "i32": 4, "int32": 4,
}
_DEFAULT_DTYPE_BYTES = 2  # fp16


def _tensor_bytes(shape: list[int] | None, dtype_bytes: int = _DEFAULT_DTYPE_BYTES) -> int:
    """Estimate the byte footprint of a tensor from its shape.

    Returns 0 for an unknown/empty shape (no capacity claim made → the
    candidate is allowed, since we cannot prove it illegal).
    """
    if not shape:
        return 0
    n = 1
    for d in shape:
        if isinstance(d, int) and d > 0:
            n *= d
    return n * dtype_bytes


def _enum_place_candidates(
    inputs: list[str],
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    """Enumerate legal `place(tensor, memory)` decisions.

    S1 legality (2026-06-26): a `place(shared)` candidate is emitted only if
    the tensor's estimated footprint fits the hardware shared-memory budget
    (`hw.shared_memory_bytes`, e.g. 48 KiB on Ampere SM 8.6). Tensors too
    large to live in shared memory are still offered a `register` placement.
    When the shape is unknown we make no capacity claim and emit both (we
    cannot prove illegality). This turns `place` from a static menu into an
    honest compiler/HW-computed legal set — the core AI-Native differentiator.
    """
    shapes = shapes or {}
    smem_cap = hw.shared_memory_bytes if hw else 49152
    out: list[Decision] = []
    for t in inputs:
        nbytes = _tensor_bytes(shapes.get(t))
        for m in _DEFAULT_PLACE_MEMORIES:
            if m == "shared" and nbytes > 0 and nbytes > smem_cap:
                # Provably exceeds shared-memory capacity → not a legal move.
                continue
            out.append(Decision(kind="place", params={"tensor": t, "memory": m}, level=1))
    return out


def _enum_fuse_candidates(op_name: str) -> list[Decision]:
    # Fusion candidates require a multi-op graph; for single-op env return [].
    return []


# ─── L3 instruction-level enumerators (P5-S5) ──────────────────────────────
#
# These encode the P5-S3 hand-tuning domain knowledge as *legality filters*
# so the LLM can only pick configurations that are statically sound on the
# target hardware. Ranking among the legal set is the LLM's job.

# Ops whose LLVM emitter consumes each L3 kind (mirrors llvm_emitter.L3_AWARE_OPS,
# split per kind; kept local to avoid an agent->backend import edge).
_L3_WMMA_OPS = frozenset({"matmul"})
_L3_BLOCK_THREADS_OPS = frozenset({"softmax", "layernorm", "rmsnorm"})

# Register budget heuristic for wmma candidates: accumulators are
# WTM*WTN*8 fp32 regs/thread; beyond ~64 acc regs the total per-thread
# usage (acc + frag + address arithmetic) exceeds ~168 regs and occupancy
# collapses (P5-S3: the naive 1x4-strip widening failure). Cap acc at 64.
_WMMA_MAX_ACC_REGS = 64


def _enum_wmma_tile_candidates(
    op_name: str,
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    """Enumerate legal L3 `wmma_tile` warp-grid configs for a matmul op.

    Legality (all statically checkable, mirrors the P5-S3 sweep filters):
      - threads = WM*WN*32 <= hw.max_threads_per_block
      - accumulator regs/thread = WTM*WTN*8 <= _WMMA_MAX_ACC_REGS
      - double-buffered fp16 staging smem = 2*(BM*BK + BK*BN)*2B fits budget
      - M % BM == 0 and N % BN == 0 for the actual input shapes
      - shape must be TC-eligible at all (M,N >= 1024, K % 16 == 0)
    """
    if op_name not in _L3_WMMA_OPS:
        return []
    shapes = shapes or {}
    dims = [s for s in shapes.values() if s and len(s) >= 2]
    if len(dims) < 2:
        return []
    M, K = dims[0][0], dims[0][1]
    N = dims[1][1]
    if not (isinstance(M, int) and isinstance(N, int) and isinstance(K, int)):
        return []
    if M < 1024 or N < 1024 or K % 16 != 0:
        return []  # not TC-eligible: no wmma decisions are legal

    hw = hw or HardwareProfile()
    max_threads = hw.max_threads_per_block
    smem_cap = hw.shared_memory_bytes
    BK = 16  # fixed by the m16n16k16 wmma shape

    out: list[Decision] = []
    for WM in (1, 2, 4):
        for WN in (1, 2, 4):
            threads = WM * WN * 32
            if threads > max_threads or threads < 64:
                continue
            for WTM in (1, 2, 4):
                for WTN in (1, 2, 4):
                    if WTM * WTN * 8 > _WMMA_MAX_ACC_REGS:
                        continue
                    BM, BN = WM * WTM * 16, WN * WTN * 16
                    if BM < 32 or BN < 32:
                        continue  # degenerate tile: staging overhead dominates
                    # double-buffered fp16 A/B staging
                    smem = 2 * (BM * BK + BK * BN) * 2
                    if smem > smem_cap:
                        continue
                    if M % BM != 0 or N % BN != 0:
                        continue
                    out.append(Decision(
                        kind="wmma_tile",
                        params={"WM": WM, "WN": WN, "WTM": WTM, "WTN": WTN},
                        level=3,
                    ))
    # Order by descending block-tile area, then thread count: larger tiles
    # amortize global loads better and are where the strong configs live
    # (P5-S3). This is a *presentation* order, not a legality claim — but it
    # matters because callers truncate to top_n, and generation order (nested
    # WM/WN loops) would otherwise fill the window with tiny WM=1 tiles and
    # hide the large-tile end of the range entirely.
    out.sort(key=lambda d: (
        -(d.params["WM"] * d.params["WTM"] * d.params["WN"] * d.params["WTN"]),
        -(d.params["WM"] * d.params["WN"]),
    ))
    return out


def _enum_block_threads_candidates(
    op_name: str,
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    """Enumerate legal L3 `block_threads` for rowwise reduction ops.

    Legality: power-of-2 warp count (the cross-warp smem tree requires it),
    within [64, hw.max_threads_per_block], and no more threads than would
    leave every thread idle (n <= next_pow2(N) is not required — extra
    threads just stride-skip — but n far above N wastes occupancy, so we
    cap candidates at 1024 and let the LLM rank).
    """
    if op_name not in _L3_BLOCK_THREADS_OPS:
        return []
    hw = hw or HardwareProfile()
    out: list[Decision] = []
    for n in (128, 256, 512, 1024):
        if n > hw.max_threads_per_block:
            continue
        out.append(Decision(kind="block_threads", params={"n": n}, level=3))
    return out


def _enum_pipeline_stages_candidates(
    op_name: str,
    shapes: dict[str, list[int]] | None = None,
    hw: HardwareProfile | None = None,
) -> list[Decision]:
    """Enumerate legal L3 `pipeline_stages` (staging ring depth) for matmul.

    STAGING-level pipeline only (global->shared plain load/store overlap).
    Fragment-level buffering of wmma sideeffect asm is known-harmful
    (P5-S3: 31% slower) and must NEVER be enumerated — depth here always
    refers to the shared-memory staging ring, which the emitter guarantees.

    Legality (statically checkable):
      - op is matmul and the shape is TC-eligible (the scalar path has its
        own fixed double-buffer; depth is only a knob on the wmma kernel)
      - depth in {2, 3}; 4 is offered only if the default 128x128 tile's
        fp16 staging smem depth*(BM*BK + BK*BN)*2B fits the hw budget.
        (The emitter re-validates against the *actual* tile at consumption
        and falls back to 2 if a larger wmma_tile pushes smem over budget.)
    """
    if op_name not in _L3_WMMA_OPS:
        return []
    shapes = shapes or {}
    dims = [s for s in shapes.values() if s and len(s) >= 2]
    if len(dims) < 2:
        return []
    M, K = dims[0][0], dims[0][1]
    N = dims[1][1]
    if not (isinstance(M, int) and isinstance(N, int) and isinstance(K, int)):
        return []
    if M < 1024 or N < 1024 or K % 16 != 0:
        return []  # not TC-eligible: staging-depth knob does not exist

    hw = hw or HardwareProfile()
    smem_cap = hw.shared_memory_bytes
    BM, BN, BK = 128, 128, 16   # default wmma tile (2,4,4,2)
    out: list[Decision] = []
    for depth in (2, 3, 4):
        smem = depth * (BM * BK + BK * BN) * 2
        if smem > smem_cap:
            continue
        out.append(Decision(
            kind="pipeline_stages", params={"depth": depth}, level=3,
        ))
    return out


_LEGAL_KIND_GENERATORS = {
    "tile": _enum_tile_candidates,
    "unroll": _enum_unroll_candidates,
    "vectorize": _enum_vectorize_candidates,
    "parallel": _enum_parallel_candidates,
}

# L3 generators receive (op_name, shapes, hw) — a different signature from
# the L1 loop-based generators, so they live in their own registry.
_L3_KIND_GENERATORS = {
    "wmma_tile": _enum_wmma_tile_candidates,
    "block_threads": _enum_block_threads_candidates,
    # Step 4b: staging-pipeline depth (ring-buffer NSTAGE) on the wmma
    # matmul kernel. STAGING-level only — fragment-level buffering of
    # sideeffect asm must NEVER appear here (P5-S3: known-harmful, cannot
    # overlap). The enumerator + emitter both enforce this by construction.
    "pipeline_stages": _enum_pipeline_stages_candidates,
    # NOTE: `fma_contract` is not enumerated: it is strictly-better-or-equal
    # on every op we emit (never a real tradeoff on sm_86), so offering
    # {on, off} would only waste the agent's decision budget. It remains
    # expressible via StrategyIR.fma_contract() for explicit experiments.
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
    # K-H2: structured hardware model consumed by the action-space generator
    # to gate tensor-core / pipeline moves. Defaults to the NVIDIA SM 8.6 model.
    hw_model: "HardwareModel | None" = None

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

        # K-H2: derive the structured HardwareModel for the action-space
        # generator (TC / pipeline gating). Defaults to the NVIDIA SM 8.6 model.
        from arke.backend.hardware import nvidia_sm86
        hw_model = nvidia_sm86(name=hw.name)

        return cls(
            state=state,
            hw_profile=hw,
            op_name=op_name,
            op_inputs=merged_shapes,
            seed=seed,
            hw_model=hw_model,
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
                         (one of: tile / unroll / vectorize / parallel / place
                          / wmma_tile / block_threads / pipeline_stages).

        Returns:
            List of Decision (level=1 for classic kinds, level=3 for
            instruction-level kinds), generated from op index_vars +
            input tensors. Empty list if no legal candidates exist.

        This is the *generator-of-candidates* — not a ranker. Ranking
        belongs to the agent (LLM). Candidates are **shape- and
        hardware-aware** (S1, 2026-06-26): tile/unroll/vectorize factors are
        derived from real input dims and filtered against the HardwareProfile
        (threads/block, divisibility); `place(shared)` is filtered against the
        hardware shared-memory budget. P5-S5 adds L3 instruction-level kinds
        (`wmma_tile`, `block_threads`) filtered by register/smem/divisibility
        legality — encoding the P5-S3 tuning domain knowledge so known-bad
        configs (occupancy-collapsing tiles) are not even offered. This makes
        the returned set an honest *legality* surface, not a static menu — the
        core AI-Native bounded-action-space guarantee.
        """
        op = REGISTRY.get(self.op_name)
        loops = list(op.index_vars) if op.index_vars else ["i", "j"]
        inputs = list(op.inputs.keys()) if isinstance(op.inputs, dict) else list(op.inputs)

        candidates: list[Decision] = []
        kinds_to_gen = (
            [filter_kind]
            if filter_kind is not None
            else ["tile", "unroll", "vectorize", "parallel", "place",
                  "wmma_tile", "block_threads", "pipeline_stages"]
        )

        for kind in kinds_to_gen:
            if kind == "place":
                candidates.extend(_enum_place_candidates(inputs, self.op_inputs, self.hw_profile))
            elif kind in _LEGAL_KIND_GENERATORS:
                candidates.extend(
                    _LEGAL_KIND_GENERATORS[kind](loops, self.op_inputs, self.hw_profile)
                )
            elif kind in _L3_KIND_GENERATORS:
                # K-H2: gate tensor-core-only actions on hardware capability.
                # wmma_tile is only legal when the target HardwareModel has a
                # tensor-core compute unit; skip it entirely on non-TC hardware
                # instead of relying solely on the shape heuristic downstream.
                if kind == "wmma_tile" and self.hw_model is not None \
                        and not self.hw_model.has_tensor_core():
                    continue
                candidates.extend(
                    _L3_KIND_GENERATORS[kind](self.op_name, self.op_inputs, self.hw_profile)
                )
            elif kind == "fuse":
                candidates.extend(_enum_fuse_candidates(self.op_name))
            # else: unknown kind → skip silently (caller may pass any string)

        # Filter out decisions that are no-ops vs current strategy
        # (e.g. tile with same factors already applied to same loop)
        candidates = self._filter_redundant(candidates)

        # Kind-balanced sampling: ensure the returned set represents ALL
        # available kinds, not just the first kind that happens to generate
        # the most raw candidates. Without balancing, tile (which typically
        # produces more candidates due to many loop×factor combos) would
        # crowd out unroll/vectorize/parallel at low top_n.
        if top_n < len(candidates):
            candidates = self._kind_balanced_sample(candidates, top_n)
        else:
            candidates = candidates[:top_n]

        return candidates

    @staticmethod
    def _kind_balanced_sample(candidates: list[Decision], n: int) -> list[Decision]:
        """Round-robin across kinds to fill n slots fairly."""
        from collections import defaultdict
        by_kind: dict[str, list[Decision]] = defaultdict(list)
        for c in candidates:
            by_kind[c.kind].append(c)
        # Ensure stable kind ordering
        kind_order = list(by_kind.keys())
        result: list[Decision] = []
        idx = {k: 0 for k in kind_order}
        while len(result) < n:
            added = False
            for k in kind_order:
                if idx[k] < len(by_kind[k]):
                    result.append(by_kind[k][idx[k]])
                    idx[k] += 1
                    added = True
                    if len(result) >= n:
                        break
            if not added:
                break
        return result

    def _filter_redundant(self, candidates: list[Decision]) -> list[Decision]:
        """Drop candidates that would duplicate an already-applied decision.

        Accumulates ALL previously applied factors/params per loop so that
        applying N tile decisions on the same loop cumulatively shrinks the
        candidate set (not just last-write-wins).
        """
        from collections import defaultdict

        # Accumulate ALL applied factors per loop (set, not overwrite)
        applied_tile: dict[str, set[tuple]] = defaultdict(set)
        for d in self.state.decision_log:
            if d.kind == "tile":
                loop = d.params.get("loop")
                factors = tuple(d.params.get("factors", []))
                applied_tile[loop].add(factors)

        applied_unroll: dict[str, set] = defaultdict(set)
        for d in self.state.decision_log:
            if d.kind == "unroll":
                loop = d.params.get("loop")
                factor = d.params.get("factor")
                applied_unroll[loop].add(factor)

        applied_vectorize: dict[str, set] = defaultdict(set)
        for d in self.state.decision_log:
            if d.kind == "vectorize":
                loop = d.params.get("loop")
                width = d.params.get("width")
                applied_vectorize[loop].add(width)

        out: list[Decision] = []
        for d in candidates:
            if d.kind == "tile":
                if tuple(d.params["factors"]) in applied_tile.get(d.params["loop"], set()):
                    continue
            elif d.kind == "unroll":
                if d.params["factor"] in applied_unroll.get(d.params["loop"], set()):
                    continue
            elif d.kind == "vectorize":
                if d.params["width"] in applied_vectorize.get(d.params["loop"], set()):
                    continue
            out.append(d)
        return out

    # ── Convenience accessors ────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"ArkeEnv(op={self.op_name}, hw={self.hw_profile.name}, "
            f"shapes={self.op_inputs}, {self.state.summary()})"
        )
