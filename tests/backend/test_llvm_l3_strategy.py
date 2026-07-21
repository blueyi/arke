# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""P5-S5 Step 1-3: StrategyIR L3 (instruction-level) -> LLVM IR consumption.

Covers:
  - L3 builder methods on StrategyIR (wmma_tile / block_threads /
    fma_contract / pipeline_stages) and level=3 tagging
  - extract_l3_params() collection semantics (None / StrategyIR / bare list,
    last-decision-wins, non-L3 kinds ignored)
  - LLVMBackend.lower(graph, strategy) consumption:
      * strategy=None -> exact pre-P5-S5 default behavior (zero regression)
      * wmma_tile reconfigures the TC warp grid (grid/block change)
      * block_threads overrides the rowwise reduction block size
      * invalid L3 params fall back safely to tuned defaults
  - JSON round-trip preserves L3 decisions (level + params + rationale)

GPU-dependent correctness of L3-configured kernels lives in
test_llvm_wmma_correctness.py-style suites; here we only assert emitted
kernel *configuration* so the tests run without a GPU.
"""

from __future__ import annotations

import pytest

from arke.backend.llvm_backend import LLVMBackend
from arke.ir.graph import IRGraph, IRNode
from arke.ir.strategy import (
    Decision,
    Rationale,
    StrategyIR,
    extract_l3_params,
    L3_KINDS,
)


def _matmul_graph(M: int = 1024, N: int = 1024, K: int = 1024) -> IRGraph:
    g = IRGraph(name="t_mm")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul",
                      inputs={"A": "A", "B": "B"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _softmax_graph(M: int = 64, N: int = 4096) -> IRGraph:
    g = IRGraph(name="t_sm")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op="softmax",
                      inputs={"X": "X"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


@pytest.fixture(scope="module")
def backend() -> LLVMBackend:
    return LLVMBackend(chip="sm_86")


# ─── L3 builders & schema ───────────────────────────────────────────────────

class TestL3Builders:
    def test_wmma_tile_builder(self):
        s = StrategyIR(kernel_id="k", target_hw="sm_86")
        d = s.wmma_tile(2, 4, 4, 2, rationale="sweep winner")
        assert d.kind == "wmma_tile"
        assert d.level == 3
        assert d.params == {"WM": 2, "WN": 4, "WTM": 4, "WTN": 2}
        assert d.rationale is not None and "sweep" in d.rationale.text

    def test_block_threads_builder(self):
        s = StrategyIR()
        d = s.block_threads(512, rationale="match CUDA-C rowwise_block_for_n")
        assert d.kind == "block_threads" and d.level == 3
        assert d.params == {"n": 512}

    def test_fma_contract_builder(self):
        s = StrategyIR()
        d = s.fma_contract(True)
        assert d.kind == "fma_contract" and d.level == 3
        assert d.params == {"enabled": True}

    def test_pipeline_stages_builder(self):
        s = StrategyIR()
        d = s.pipeline_stages(2)
        assert d.kind == "pipeline_stages" and d.level == 3
        assert d.params == {"depth": 2}

    def test_l3_kinds_registry_covers_builders(self):
        assert set(L3_KINDS) == {
            "wmma_tile", "block_threads", "fma_contract", "pipeline_stages",
        }

    def test_json_roundtrip_preserves_l3(self):
        s = StrategyIR(kernel_id="mm", target_hw="sm_86")
        s.wmma_tile(2, 4, 4, 2, rationale="occupancy 0.25->0.33")
        s2 = StrategyIR.from_json(s.to_json())
        assert len(s2.decisions) == 1
        d = s2.decisions[0]
        assert isinstance(d, Decision)
        assert d.kind == "wmma_tile" and d.level == 3
        assert d.params == {"WM": 2, "WN": 4, "WTM": 4, "WTN": 2}
        assert d.rationale is not None and "occupancy" in d.rationale.text


# ─── extract_l3_params ──────────────────────────────────────────────────────

class TestExtractL3Params:
    def test_none_returns_empty(self):
        assert extract_l3_params(None) == {}

    def test_strategy_ir_extraction(self):
        s = StrategyIR()
        s.tile("i", [64], rationale="L1 noise, must be ignored")
        s.wmma_tile(2, 2, 2, 4)
        out = extract_l3_params(s)
        assert out == {"wmma_tile": {"WM": 2, "WN": 2, "WTM": 2, "WTN": 4}}

    def test_bare_list_accepted(self):
        ds = [Decision(kind="block_threads", params={"n": 256}, level=3)]
        assert extract_l3_params(ds) == {"block_threads": {"n": 256}}

    def test_last_decision_wins(self):
        s = StrategyIR()
        s.wmma_tile(2, 2, 2, 4)
        s.wmma_tile(2, 4, 4, 2, rationale="refined after profile")
        assert extract_l3_params(s)["wmma_tile"]["WN"] == 4

    def test_l3_kind_without_level3_ignored(self):
        # kind collision at wrong level must not leak into L3 params
        ds = [Decision(kind="wmma_tile", params={"WM": 9}, level=1)]
        assert extract_l3_params(ds) == {}

    def test_garbage_strategy_object(self):
        assert extract_l3_params(42) == {}
        assert extract_l3_params("nope") == {}


# ─── LLVMBackend.lower consumption ─────────────────────────────────────────

class TestLowerConsumesL3:
    def test_strategy_none_default_preserved(self, backend):
        """Zero-regression guard: no strategy -> P5-S3 tuned default."""
        art = backend.lower(_matmul_graph())
        e = art.metadata["emitted"]
        assert e.block == (256, 1, 1)          # 8 warps (2x4 grid)
        assert e.grid == (8, 8, 1)             # 1024/128 x 1024/128
        assert art.metadata["l3_params"] == {}

    def test_wmma_tile_reconfigures_warp_grid(self, backend):
        s = StrategyIR()
        s.wmma_tile(2, 2, 2, 4, rationale="64x128 2x2 grid")
        art = backend.lower(_matmul_graph(), strategy=s)
        e = art.metadata["emitted"]
        assert e.block == (128, 1, 1)          # 4 warps
        assert e.grid == (8, 16, 1)            # N/128 x M/64
        assert art.metadata["l3_params"]["wmma_tile"]["WTN"] == 4
        # emitted IR must actually carry the reconfigured smem tile
        assert "[64 x [16 x half]]" in art.source_code   # BM=64 A-tile

    def test_invalid_wmma_tile_falls_back(self, backend):
        s = StrategyIR()
        s.wmma_tile(7, 9, 3, 5)               # 336x720 tile: no divisibility
        art = backend.lower(_matmul_graph(), strategy=s)
        e = art.metadata["emitted"]
        assert e.block == (256, 1, 1)          # tuned default retained
        assert e.grid == (8, 8, 1)

    def test_oversized_threads_falls_back(self, backend):
        s = StrategyIR()
        s.wmma_tile(8, 8, 1, 1)               # 2048 threads > 1024 cap
        art = backend.lower(_matmul_graph(), strategy=s)
        assert art.metadata["emitted"].block == (256, 1, 1)

    def test_block_threads_overrides_softmax(self, backend):
        default = backend.lower(_softmax_graph())
        assert default.metadata["emitted"].block[0] == 512   # N=4096 heuristic
        s = StrategyIR()
        s.block_threads(256, rationale="small-batch occupancy probe")
        art = backend.lower(_softmax_graph(), strategy=s)
        assert art.metadata["emitted"].block[0] == 256

    def test_block_threads_invalid_falls_back(self, backend):
        s = StrategyIR()
        s.block_threads(96)                    # 3 warps: not power-of-2 warps
        art = backend.lower(_softmax_graph(), strategy=s)
        assert art.metadata["emitted"].block[0] == 512

    def test_l3_on_unaware_op_is_noop(self, backend):
        """Ops outside L3_AWARE_OPS must ignore strategy without error."""
        g = IRGraph(name="t_relu")
        g.add_input("X", dtype="float32", shape=[128, 128])
        g.add_node(IRNode(id="n0", op="relu", inputs={"X": "X"},
                          outputs=["out"]))
        g.set_outputs(["out"])
        s = StrategyIR()
        s.wmma_tile(2, 4, 4, 2)
        art = backend.lower(g, strategy=s)     # must not raise
        assert art.op_name == "relu"


# ─── pipeline_stages consumption (Step 4b) ─────────────────────────────────

class TestPipelineStagesConsumption:
    """P5-S5 Step 4b: pipeline_stages{depth} -> wmma staging ring NSTAGE."""

    def test_default_is_two_stage_ring(self, backend):
        art = backend.lower(_matmul_graph())
        # default: classic double buffer, [2 x ...] shmem ring
        assert "[2 x [128 x [16 x half]]]" in art.source_code
        assert "[3 x [" not in art.source_code

    def test_depth3_emits_three_stage_ring(self, backend):
        s = StrategyIR()
        s.pipeline_stages(3, rationale="deeper staging overlap probe")
        art = backend.lower(_matmul_graph(), strategy=s)
        # 3-slot ring for both A and B staging buffers
        assert "[3 x [128 x [16 x half]]]" in art.source_code
        assert "[3 x [16 x [128 x half]]]" in art.source_code
        # non-pow2 ring -> urem indexing; prologue stages tiles 0 and 1
        assert "urem i32 %t, 3" in art.source_code
        assert art.metadata["l3_params"]["pipeline_stages"]["depth"] == 3
        # launch config unchanged: depth only affects smem/staging
        assert art.metadata["emitted"].block == (256, 1, 1)
        assert art.metadata["emitted"].grid == (8, 8, 1)

    def test_depth4_emits_four_stage_ring(self, backend):
        s = StrategyIR()
        s.pipeline_stages(4)
        art = backend.lower(_matmul_graph(), strategy=s)
        assert "[4 x [128 x [16 x half]]]" in art.source_code
        # pow2 ring -> and-mask indexing
        assert "and i32 %t, 3" in art.source_code

    def test_invalid_depth_falls_back_to_two(self, backend):
        for bad in (0, 1, 5, -3, "three", None):
            s = StrategyIR()
            s.pipeline_stages(bad)  # type: ignore[arg-type]
            art = backend.lower(_matmul_graph(), strategy=s)
            assert "[2 x [128 x [16 x half]]]" in art.source_code, bad

    def test_smem_boundary_and_overflow(self, backend):
        """depth*tile smem <= 48KB is honored; beyond 48KB falls back to 2."""
        # Boundary-legal: BM=128, BN=256 -> 4*(128*16+16*256)*2 = 48KB exactly.
        s = StrategyIR()
        s.wmma_tile(2, 4, 4, 4)          # BM=128, BN=256 (needs N%256==0)
        s.pipeline_stages(4)
        art = backend.lower(_matmul_graph(M=1024, N=2048, K=1024), strategy=s)
        assert "[4 x [128 x [16 x half]]]" in art.source_code
        # Overflow: BM=128, BN=512 -> 3*(128*16+16*512)*2 = 60KB > 48KB
        # -> depth falls back to 2 while the wmma_tile override sticks.
        s = StrategyIR()
        s.wmma_tile(2, 8, 4, 4)          # BM=128, BN=512 (needs N%512==0)
        s.pipeline_stages(3)
        art = backend.lower(_matmul_graph(M=1024, N=2048, K=1024), strategy=s)
        assert "[2 x [128 x [16 x half]]]" in art.source_code
        assert "[3 x [" not in art.source_code

    def test_depth3_correct_ring_with_custom_tile(self, backend):
        """pipeline_stages composes with wmma_tile."""
        s = StrategyIR()
        s.wmma_tile(2, 2, 2, 4)          # BM=64, BN=128
        s.pipeline_stages(3)
        art = backend.lower(_matmul_graph(), strategy=s)
        assert "[3 x [64 x [16 x half]]]" in art.source_code
        assert art.metadata["emitted"].block == (128, 1, 1)

    def test_non_tc_shape_ignores_depth(self, backend):
        """Scalar-path matmul has a fixed structural double-buffer; the
        pipeline_stages decision must be silently inert (no wmma ring)."""
        s = StrategyIR()
        s.pipeline_stages(3)
        art = backend.lower(_matmul_graph(512, 512, 512), strategy=s)
        assert "wmma" not in art.source_code
        assert "[3 x [" not in art.source_code


# ─── L3 bounded action space (Step 4a) ─────────────────────────────────────

class TestL3ActionSpace:
    """P5-S5 Step 4a: list_legal_actions offers L3 instruction-level kinds."""

    def _matmul_env(self, M=1024, N=1024, K=1024):
        from arke.agent.env import ArkeEnv
        return ArkeEnv.from_op("matmul", {"A": [M, K], "B": [K, N]})

    def test_wmma_tile_candidates_present_and_legal(self):
        env = self._matmul_env()
        acts = env.list_legal_actions(top_n=100, filter_kind="wmma_tile")
        assert acts, "TC-eligible matmul must offer wmma_tile actions"
        tuples = set()
        for a in acts:
            assert a.kind == "wmma_tile" and a.level == 3
            p = a.params
            # every offered config must satisfy the static legality filters
            assert p["WM"] * p["WN"] * 32 <= 1024
            assert p["WTM"] * p["WTN"] * 8 <= 64      # occupancy guard
            BM, BN = p["WM"] * p["WTM"] * 16, p["WN"] * p["WTN"] * 16
            assert 1024 % BM == 0 and 1024 % BN == 0
            assert 2 * (BM * 16 + 16 * BN) * 2 <= 49152
            tuples.add((p["WM"], p["WN"], p["WTM"], p["WTN"]))
        assert (2, 4, 4, 2) in tuples      # the P5-S3 sweep winner
        assert (2, 2, 2, 4) in tuples      # the 64x128 predecessor

    def test_non_tc_shape_offers_no_wmma(self):
        env = self._matmul_env(512, 512, 512)
        assert env.list_legal_actions(top_n=100, filter_kind="wmma_tile") == []

    def test_pipeline_stages_candidates_for_tc_matmul(self):
        """Step 4b: TC-eligible matmul offers staging depths {2,3,4}
        (4 allowed on sm_86: 4*(128*16+16*128)*2 = 32KB <= 48KB)."""
        env = self._matmul_env()
        acts = env.list_legal_actions(top_n=100, filter_kind="pipeline_stages")
        depths = sorted(a.params["depth"] for a in acts)
        assert depths == [2, 3, 4]
        assert all(a.kind == "pipeline_stages" and a.level == 3 for a in acts)

    def test_pipeline_stages_absent_for_non_tc(self):
        env = self._matmul_env(512, 512, 512)
        assert env.list_legal_actions(
            top_n=100, filter_kind="pipeline_stages") == []
        from arke.agent.env import ArkeEnv
        env2 = ArkeEnv.from_op("softmax", {"X": [64, 4096]})
        assert env2.list_legal_actions(
            top_n=100, filter_kind="pipeline_stages") == []

    def test_pipeline_stages_to_emitter_roundtrip(self):
        """An enumerated depth-3 action applied as a decision must produce
        a 3-slot staging ring in the emitted LLVM IR."""
        from arke.backend.llvm_backend import LLVMBackend
        env = self._matmul_env()
        acts = env.list_legal_actions(top_n=100, filter_kind="pipeline_stages")
        pick = next(a for a in acts if a.params["depth"] == 3)
        s = StrategyIR(kernel_id="mm", target_hw="sm_86")
        s.add_decision(pick)
        art = LLVMBackend(chip="sm_86").lower(_matmul_graph(), strategy=s)
        assert "[3 x [128 x [16 x half]]]" in art.source_code

    def test_block_threads_for_reduction_ops(self):
        from arke.agent.env import ArkeEnv
        env = ArkeEnv.from_op("softmax", {"X": [64, 4096]})
        acts = env.list_legal_actions(top_n=100, filter_kind="block_threads")
        assert [a.params["n"] for a in acts] == [128, 256, 512, 1024]
        assert all(a.level == 3 for a in acts)

    def test_elementwise_offers_no_l3(self):
        from arke.agent.env import ArkeEnv
        env = ArkeEnv.from_op("relu", {"X": [128, 128]})
        assert env.list_legal_actions(top_n=100, filter_kind="wmma_tile") == []
        assert env.list_legal_actions(top_n=100, filter_kind="block_threads") == []

    def test_unfiltered_mix_includes_l3(self):
        env = self._matmul_env()
        acts = env.list_legal_actions(top_n=30)
        kinds = {a.kind for a in acts}
        assert "wmma_tile" in kinds, "kind-balanced sample must include L3"

    def test_action_space_to_emitter_roundtrip(self):
        """An enumerated L3 action, applied as a decision, must configure the
        emitter — closing the action-space -> StrategyIR -> LLVM IR loop."""
        from arke.backend.llvm_backend import LLVMBackend
        env = self._matmul_env()
        acts = env.list_legal_actions(top_n=100, filter_kind="wmma_tile")
        pick = next(a for a in acts
                    if (a.params["WM"], a.params["WN"],
                        a.params["WTM"], a.params["WTN"]) == (2, 2, 2, 4))
        s = StrategyIR(kernel_id="mm", target_hw="sm_86")
        s.add_decision(pick)
        art = LLVMBackend(chip="sm_86").lower(_matmul_graph(), strategy=s)
        assert art.metadata["emitted"].block == (128, 1, 1)
        assert art.metadata["emitted"].grid == (8, 16, 1)


# ─── Facade compile_and_profile wiring (Step 5a) ───────────────────────────

import torch as _torch  # noqa: E402


def _llvm_toolchain_ready() -> bool:
    if not _torch.cuda.is_available():
        return False
    b = LLVMBackend(chip="sm_86")
    return bool(b.llc and b.ptxas)


_GPU_LLVM = pytest.mark.skipif(
    not _llvm_toolchain_ready(),
    reason="requires CUDA + llc(LLVM20) + ptxas",
)


class TestFacadeSchemaSurfacesL3:
    """Runner system prompt (not the frozen Facade schema) is where the LLM
    learns about L3 kinds + the llvm backend. The Facade v1.0 tool
    schemas/descriptions are frozen byte-for-byte; we assert the system
    prompt teaches L3 so the live-LLM loop can reach the new action space.
    """

    def test_system_prompt_mentions_l3_and_llvm(self):
        from arke.agent.runner import _SYSTEM_PROMPT
        assert "wmma_tile" in _SYSTEM_PROMPT
        assert "block_threads" in _SYSTEM_PROMPT
        assert "pipeline_stages" in _SYSTEM_PROMPT
        assert 'backend="llvm"' in _SYSTEM_PROMPT

    def test_env_list_legal_actions_docstring_surfaces_l3(self):
        # The env-level filter_kind doc lists the L3 kinds (Step 4a); the
        # tool passthrough exposes them to the agent verbatim.
        from arke.agent.env import ArkeEnv
        doc = ArkeEnv.list_legal_actions.__doc__ or ""
        assert "wmma_tile" in doc
        assert "block_threads" in doc
        assert "pipeline_stages" in doc


@_GPU_LLVM
class TestFacadeLLVMStrategyInjection:
    """compile_and_profile(backend='llvm') consumes the env's decision_log
    (incl. L3 wmma_tile) to configure the emitted kernel."""

    def _matmul_env(self, M=1024, N=1024, K=1024):
        from arke.agent.env import ArkeEnv
        return ArkeEnv.from_op("matmul", {"A": [M, K], "B": [K, N]})

    def test_llvm_backend_runs_and_times(self):
        from arke.agent.tools import CompileAndProfileTool
        env = self._matmul_env()
        tool = CompileAndProfileTool(env=env)
        res = tool.execute({
            "op_name": "matmul",
            "shapes": {"A": [1024, 1024], "B": [1024, 1024]},
            "backend": "llvm",
        })
        assert res.success, res.error
        assert res.data["backend"] == "llvm"
        assert res.data["latency_ms"] is not None
        assert res.data["latency_ms"] > 0
        # No decisions applied yet -> strategy_decisions == 0
        assert res.data["strategy_decisions"] == 0

    def test_wmma_tile_decision_configures_kernel(self):
        """Apply a wmma_tile L3 decision via the env, then profile through
        the Facade tool. The strategy must reach the emitter (strategy_decisions
        > 0) and the run must succeed + measure a real latency."""
        from arke.agent.tools import ApplyDecisionTool, CompileAndProfileTool
        env = self._matmul_env()
        # enumerate + apply a wmma_tile decision through the real tool
        acts = env.list_legal_actions(top_n=100, filter_kind="wmma_tile")
        pick = next(a for a in acts
                    if (a.params["WM"], a.params["WN"],
                        a.params["WTM"], a.params["WTN"]) == (2, 2, 2, 4))
        ad = ApplyDecisionTool(env)
        r = ad.execute({
            "kind": "wmma_tile",
            "params": dict(pick.params),
            "level": 3,
            "rationale": "TC warp-grid (2,2,2,4) baseline config",
        })
        assert r.success, r.error

        tool = CompileAndProfileTool(env=env)
        res = tool.execute({
            "op_name": "matmul",
            "shapes": {"A": [1024, 1024], "B": [1024, 1024]},
            "backend": "llvm",
        })
        assert res.success, res.error
        assert res.data["backend"] == "llvm"
        assert res.data["strategy_decisions"] >= 1
        assert res.data["correct"] is True
        assert res.data["latency_ms"] > 0
