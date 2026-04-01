# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Legal Actions Engine."""


from arke.engine.env import ArkeEnv
from arke.ir.builder import KernelBuilder


def _make_matmul_relu() -> tuple:
    """Build matmul+relu IR and ArkeEnv."""
    b = KernelBuilder("fused_matmul_relu")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    ir = b.build()
    env = ArkeEnv(ir, "nvidia_ampere")
    return ir, env


# ============================================================
# Basic Enumeration Tests
# ============================================================

def test_initial_actions_include_tile():
    """Fresh kernel should have tile actions."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions()
    tile_actions = [a for a in result["legal_actions"] if a["kind"] == "tile"]
    assert len(tile_actions) > 0


def test_initial_actions_include_fuse():
    """matmul+relu should have fusion opportunity."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions()
    fuse_actions = [a for a in result["legal_actions"] if a["kind"] == "fuse"]
    assert len(fuse_actions) > 0
    assert fuse_actions[0]["params"]["type"] == "epilogue"


def test_initial_hint_suggests_fusion():
    """When fusion is available, hint should suggest it."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions()
    assert "fus" in result["hint"].lower()


def test_filter_by_kind():
    """Filtering by kind returns only that kind."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions(kind="tile")
    for action in result["legal_actions"]:
        assert action["kind"] == "tile"


def test_limit_respected():
    """Limit parameter caps results."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions(limit=3)
    assert len(result["legal_actions"]) <= 3


def test_search_space_size():
    """Total search space should be reported."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions()
    assert result["search_space_size"] >= len(result["legal_actions"])


# ============================================================
# State-Dependent Tests
# ============================================================

def test_tiled_loop_blocked():
    """Tiling the same loop twice should be blocked."""
    ir, env = _make_matmul_relu()

    # First tile succeeds
    env.apply_decision("tile", {"loop": "i", "factors": [64, 16]}, "test")

    # list_legal_actions should show 'i' as blocked
    result = env.list_legal_actions(kind="tile")
    tile_i_actions = [a for a in result["legal_actions"]
                      if a["params"].get("loop") == "i"]
    assert len(tile_i_actions) == 0  # No more tile-i actions

    blocked_i = [b for b in result["blocked_actions"]
                 if b["params"].get("loop") == "i"]
    assert len(blocked_i) > 0
    assert "already tiled" in blocked_i[0]["blocked_reason"].lower()


def test_fused_nodes_blocked():
    """Fusing already-fused nodes should be blocked."""
    ir, env = _make_matmul_relu()

    # Get fusion groups from semantic IR
    fuse_actions = [a for a in env.list_legal_actions()["legal_actions"]
                    if a["kind"] == "fuse"]
    assert len(fuse_actions) > 0

    # Apply fusion
    nodes = fuse_actions[0]["params"]["nodes"]
    env.apply_decision("fuse", {"ops": nodes, "type": "epilogue"}, "test fusion")

    # Now check — should be blocked
    result = env.list_legal_actions(kind="fuse")
    assert len(result["legal_actions"]) == 0
    assert len(result["blocked_actions"]) > 0


def test_parallel_after_tiling():
    """Parallel actions appear after tiling."""
    ir, env = _make_matmul_relu()

    # No parallel before tiling
    result_before = env.list_legal_actions(kind="parallel")
    assert len(result_before["legal_actions"]) == 0

    # Tile two loops
    env.apply_decision("tile", {"loop": "i", "factors": [64, 16]}, "tile i")
    env.apply_decision("tile", {"loop": "j", "factors": [128, 16]}, "tile j")

    # Now parallel should be available
    result_after = env.list_legal_actions(kind="parallel")
    assert len(result_after["legal_actions"]) > 0


def test_place_after_params_exist():
    """Place actions should be available for input params."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions(kind="place")
    place_actions = result["legal_actions"]
    assert len(place_actions) >= 1  # At least one param can be placed


def test_hint_changes_with_state():
    """Hint should adapt to current optimization state."""
    ir, env = _make_matmul_relu()

    # Initial: suggest fusion
    hint_initial = env.list_legal_actions()["hint"]

    # After fusion: suggest tiling
    env.apply_decision("fuse", {"ops": ["matmul_0", "relu_1"], "type": "epilogue"}, "fuse")

    # After tiling: suggest parallel
    env.apply_decision("tile", {"loop": "i", "factors": [64, 16]}, "tile i")
    env.apply_decision("tile", {"loop": "j", "factors": [128, 16]}, "tile j")
    hint_after_tile = env.list_legal_actions()["hint"]
    assert "parallel" in hint_after_tile.lower()


# ============================================================
# Action Quality Tests
# ============================================================

def test_actions_have_estimated_impact():
    """Each action should have estimated_impact."""
    ir, env = _make_matmul_relu()
    result = env.list_legal_actions()
    for action in result["legal_actions"]:
        assert "estimated_impact" in action


def test_actions_sorted_by_priority():
    """Actions should be sorted by priority (highest first)."""
    ir, env = _make_matmul_relu()
    # Fusion should come before tiling (higher priority)
    result = env.list_legal_actions()
    actions = result["legal_actions"]
    if len(actions) >= 2:
        fuse_idx = next((i for i, a in enumerate(actions) if a["kind"] == "fuse"), None)
        tile_idx = next((i for i, a in enumerate(actions) if a["kind"] == "tile"), None)
        if fuse_idx is not None and tile_idx is not None:
            assert fuse_idx < tile_idx  # Fusion before tiling


# ============================================================
# Session Integration Tests
# ============================================================

def test_session_list_legal_actions():
    """list_legal_actions works through session."""
    from arke.agent.session import OptimizationSession
    b = KernelBuilder("test_mm")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    ir = b.build()

    session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")
    result = session.run_tool("list_legal_actions", {})
    assert "legal_actions" in result
    assert "search_space_size" in result
    assert result["search_space_size"] > 0
