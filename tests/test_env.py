# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArkeEnv and V0 validator."""

from arke.ir.builder import KernelBuilder
from arke.engine.env import ArkeEnv
from arke.engine.validator import StaticValidator


def _make_matmul_env() -> ArkeEnv:
    """Helper: build a matmul+relu env for testing."""
    b = KernelBuilder("test_matmul_relu")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    ir = b.build()
    return ArkeEnv(ir, "nvidia_ampere")


def test_env_creation():
    """Test ArkeEnv initializes correctly."""
    env = _make_matmul_env()
    assert env.semantic.graph_id == "test_matmul_relu"
    assert env.target_hw == "nvidia_ampere"
    assert env.strategy.decision_count == 0
    assert env.hw_profile.get("name") == "nvidia_ampere"


def test_env_get_semantic_ir():
    """Test get_semantic_ir returns valid dict."""
    env = _make_matmul_env()
    sir = env.get_semantic_ir()
    assert sir["graph_id"] == "test_matmul_relu"
    assert len(sir["nodes"]) == 2


def test_env_get_hw_profile():
    """Test get_hw_profile returns loaded profile."""
    env = _make_matmul_env()
    hw = env.get_hw_profile()
    assert hw["name"] == "nvidia_ampere"
    assert hw["constraints"]["warp_size"] == 32


def test_env_analyze_compute():
    """Test analyze_compute returns analysis."""
    env = _make_matmul_env()
    analysis = env.analyze_compute()
    assert analysis["kernel"] == "test_matmul_relu"
    assert len(analysis["nodes"]) == 2
    assert analysis["nodes"][0]["category"] == "compute"
    assert analysis["nodes"][1]["category"] == "elementwise"
    assert len(analysis["fusion_opportunities"]) >= 1


def test_env_apply_decision_success():
    """Test applying a valid decision."""
    env = _make_matmul_env()
    result = env.apply_decision("tile", {"loop": "i", "factors": [64, 16]}, "test")
    assert result["success"] is True
    assert result["step"] == 1
    assert env.strategy.decision_count == 1


def test_env_apply_decision_invalid():
    """Test applying an invalid decision (empty factors) triggers auto-rollback."""
    env = _make_matmul_env()
    result = env.apply_decision("tile", {"loop": "i", "factors": []}, "bad")
    assert result["success"] is False
    assert result["auto_rollback"] is True
    assert env.strategy.decision_count == 0


def test_env_apply_duplicate_tile():
    """Test tiling the same loop twice is rejected."""
    env = _make_matmul_env()
    r1 = env.apply_decision("tile", {"loop": "i", "factors": [64]}, "first")
    assert r1["success"] is True
    r2 = env.apply_decision("tile", {"loop": "i", "factors": [32]}, "duplicate")
    assert r2["success"] is False
    assert env.strategy.decision_count == 1  # Only first survived


def test_env_rollback():
    """Test rollback removes decisions."""
    env = _make_matmul_env()
    env.apply_decision("tile", {"loop": "i", "factors": [64]}, "a")
    env.apply_decision("tile", {"loop": "j", "factors": [128]}, "b")
    assert env.strategy.decision_count == 2

    result = env.rollback(1)
    assert result["rolled_back"] == 1
    assert env.strategy.decision_count == 1


def test_env_checkpoint_restore():
    """Test checkpoint and restore."""
    env = _make_matmul_env()
    env.apply_decision("tile", {"loop": "i", "factors": [64]}, "a")
    cp = env.checkpoint("before_fuse")
    assert cp["checkpoint_id"] == "before_fuse"

    env.apply_decision("fuse", {"ops": ["matmul", "relu"], "type": "epilogue"}, "b")
    assert env.strategy.decision_count == 2

    result = env.restore("before_fuse")
    assert result["success"] is True
    assert env.strategy.decision_count == 1


def test_env_observe():
    """Test observe returns current state."""
    env = _make_matmul_env()
    env.apply_decision("tile", {"loop": "i", "factors": [64]}, "test")
    obs = env.observe()
    assert obs["decision_count"] == 1
    assert "tile" in obs["strategy_summary"]


def test_validator_positive_factors():
    """V0 rejects non-positive tile factors."""
    env = _make_matmul_env()
    result = env.apply_decision("tile", {"loop": "i", "factors": [-1]}, "bad")
    assert result["success"] is False
    assert any("positive" in v for v in result["validation"]["violations"])
