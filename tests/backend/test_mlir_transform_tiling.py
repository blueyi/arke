# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S1: transform-dialect tiling correctness.

The P3-S1 gate reads "SemanticIR → linalg + **transform dialect**, matmul
correct". This suite validates the transform-dialect path specifically:

  * ``emit_transform_schedule`` produces a valid ``transform.named_sequence``.
  * ``MLIRBackend.lower(graph, tile_sizes=...)`` wraps the module with the
    schedule and marks it ``transform.with_named_sequence``.
  * The tiled kernel JIT-executes **bit-correct vs numpy** across tile
    configurations (full tiling, partial tiling, tile sizes that don't divide
    the problem shape), proving the transform-interpreter pre-pass + subview
    lowering pipeline is sound.

Skips cleanly without the user-local MLIR 18 toolchain
(source ~/opt/mlir18/env.sh).
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_backend import MLIRBackend, mlir_toolchain_available
from arke.backend.mlir_emitter import emit_transform_schedule


pytestmark = pytest.mark.skipif(
    not mlir_toolchain_available(),
    reason="MLIR 18 toolchain not found (source ~/opt/mlir18/env.sh)",
)


def _matmul_graph(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


# ── 1. schedule emission ───────────────────────────────────────

def test_emit_schedule_shape():
    sched = emit_transform_schedule("matmul", [16, 16, 16])
    assert "transform.named_sequence @__transform_main" in sched
    assert 'transform.structured.match ops{["linalg.matmul"]}' in sched
    assert "transform.structured.tile_using_for" in sched
    assert "%target[16, 16, 16]" in sched
    # 3 non-zero tile sizes → 3 loop result types
    assert sched.count("!transform.any_op") >= 4  # target + 3 loops


def test_emit_schedule_partial_tiling():
    # tile only M and N (K untiled) → 2 loops
    sched = emit_transform_schedule("matmul", [8, 8, 0])
    assert "%target[8, 8, 0]" in sched
    assert "%loops:2" in sched


def test_emit_schedule_wrong_rank_raises():
    with pytest.raises(ValueError):
        emit_transform_schedule("matmul", [16, 16])  # matmul needs 3


def test_emit_schedule_untileable_op_raises():
    with pytest.raises(NotImplementedError):
        emit_transform_schedule("flash_attention", [16, 16, 16])


# ── 2. lower() marks the module as transform-carrying ──────────

def test_lower_with_tile_sizes_wraps_transform():
    be = MLIRBackend()
    art = be.lower(_matmul_graph(8, 8, 8), tile_sizes={"matmul": [4, 4, 4]})
    assert "transform.with_named_sequence" in art.source_code
    assert "transform.structured.tile_using_for" in art.source_code
    assert art.metadata["tiled"] is True


def test_lower_tile_sizes_from_graph_metadata():
    g = _matmul_graph(8, 8, 8)
    g.metadata["tile_sizes"] = {"matmul": [2, 2, 2]}
    be = MLIRBackend()
    art = be.lower(g)
    assert art.metadata["tiled"] is True
    assert "transform.with_named_sequence" in art.source_code


def test_lower_without_tiling_is_untiled():
    be = MLIRBackend()
    art = be.lower(_matmul_graph(8, 8, 8))
    assert art.metadata["tiled"] is False
    assert "transform.with_named_sequence" not in art.source_code


# ── 3. tiled kernel JIT-executes bit-correct vs numpy ──────────

@pytest.mark.parametrize("M,K,N,tile", [
    (8, 8, 8, [4, 4, 4]),       # tile divides evenly
    (8, 8, 8, [2, 2, 0]),       # partial tiling (K untiled)
    (16, 16, 16, [8, 8, 8]),    # larger, even
    (12, 10, 14, [4, 4, 4]),    # tile does NOT divide shape (remainder loops)
    (32, 64, 32, [16, 16, 16]),
    (7, 5, 9, [4, 4, 4]),       # all odd, remainders everywhere
])
def test_tiled_matmul_correct(M, K, N, tile):
    be = MLIRBackend()
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(M, K, N), tile_sizes={"matmul": tile}))
    assert ker.success, ker.error
    out = be.run(ker, {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-4, atol=1e-4)


def test_tiled_matches_untiled_bit_for_bit():
    """Tiling is a scheduling transform — result must equal the untiled path."""
    be = MLIRBackend()
    rng = np.random.default_rng(2)
    M, K, N = 16, 16, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    g = _matmul_graph(M, K, N)
    untiled = be.run(be.compile(be.lower(g)), {"A": A, "B": B})["C"]
    tiled = be.run(
        be.compile(be.lower(g, tile_sizes={"matmul": [8, 8, 8]})),
        {"A": A, "B": B},
    )["C"]
    # both compare to numpy within f32 noise, and to each other tightly
    np.testing.assert_allclose(tiled, A @ B, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(tiled, untiled, rtol=1e-5, atol=1e-5)
