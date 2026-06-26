# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shape/HW-aware legal-action generation (P1-b).

Verifies that list_legal_actions derives tile/unroll/vectorize candidates
from the operator's real input dimensions + HardwareProfile, instead of a
fixed module-level Cartesian product.
"""

from __future__ import annotations

from arke.agent.env import ArkeEnv


def _tile_factors(env, loop):
    return sorted(
        d.params["factors"][0]
        for d in env.list_legal_actions(top_n=50, filter_kind="tile")
        if d.params["loop"] == loop
    )


def test_tile_factors_scale_with_large_dim():
    """A 512-wide loop offers tiles up to 512; a 128-wide loop caps at 128."""
    env = ArkeEnv.from_op("matmul", {"A": [512, 128], "B": [128, 512]})
    i_factors = _tile_factors(env, "i")  # i ← 512
    assert 256 in i_factors and 512 in i_factors
    j_factors = _tile_factors(env, "j")  # j ← 128
    assert max(j_factors) <= 128


def test_tile_factors_respect_small_dim():
    """Tiny dims must not offer tiles larger than the dimension."""
    env = ArkeEnv.from_op("matmul", {"A": [32, 16], "B": [16, 64]})
    i_factors = _tile_factors(env, "i")  # i ← 32
    assert max(i_factors) <= 32


def test_vectorize_width_divides_dim():
    """Every offered vectorize width must evenly divide its loop dim."""
    env = ArkeEnv.from_op("matmul", {"A": [512, 128], "B": [128, 512]})
    for d in env.list_legal_actions(top_n=50, filter_kind="vectorize"):
        # all dims here are powers of two → widths 2/4/8 all divide
        assert d.params["width"] in (2, 4, 8)


def test_unroll_factor_not_exceeding_trip_count():
    """Unroll factor must not exceed the loop trip count."""
    env = ArkeEnv.from_op("matmul", {"A": [4, 4], "B": [4, 4]})
    for d in env.list_legal_actions(top_n=50, filter_kind="unroll"):
        # dims are 4 → only unroll factor 2,4 legal (8 dropped)
        assert d.params["factor"] <= 4


def test_unknown_shape_falls_back():
    """With default [4,8] shapes the generator still returns candidates."""
    env = ArkeEnv.from_op("relu")  # default shapes
    cands = env.list_legal_actions(top_n=10)
    assert len(cands) > 0


# ── S1 (2026-06-26): place() shared-memory capacity legality ──────────────


def _place_memories_for(env, tensor):
    return {
        d.params["memory"]
        for d in env.list_legal_actions(top_n=200, filter_kind="place")
        if d.params["tensor"] == tensor
    }


def test_place_shared_dropped_when_tensor_exceeds_smem():
    """A tensor larger than the HW shared-memory budget gets no place(shared).

    Ampere SM 8.6 shared_memory_bytes = 49152 (48 KiB). A 2048x2048 fp16
    tensor = 2048*2048*2 = 8 MiB ≫ 48 KiB → `place(shared)` must be absent,
    but `place(register)` is still offered. This is the S1 compiler/HW-computed
    legality guarantee: an illegal move never appears in the bounded set.
    """
    env = ArkeEnv.from_op("matmul", {"A": [2048, 2048], "B": [2048, 2048]})
    mems = _place_memories_for(env, "A")
    assert "shared" not in mems, "oversized tensor must not be offered shared placement"
    assert "register" in mems, "register placement should still be available"


def test_place_shared_allowed_when_tensor_fits_smem():
    """A small tensor that fits shared memory keeps its place(shared) option.

    A 32x32 fp16 tensor = 32*32*2 = 2048 B < 48 KiB → both placements legal.
    """
    env = ArkeEnv.from_op("matmul", {"A": [32, 32], "B": [32, 32]})
    mems = _place_memories_for(env, "A")
    assert "shared" in mems and "register" in mems


def test_place_unknown_shape_makes_no_capacity_claim():
    """When the shape is unknown we cannot prove illegality → emit both."""
    from arke.agent.env import _enum_place_candidates

    cands = _enum_place_candidates(["X"], shapes={}, hw=None)
    mems = {d.params["memory"] for d in cands}
    assert "shared" in mems and "register" in mems
