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
