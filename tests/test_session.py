# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arke Agent tools schema and session."""

from arke.agent.session import OptimizationSession, SessionState
from arke.agent.tools_schema import (
    TOOL_METADATA,
    TOOLS,
    get_tool_names,
    get_tool_schema,
)
from arke.ir.builder import KernelBuilder


def _make_session() -> OptimizationSession:
    """Helper: create a test session."""
    b = KernelBuilder("test_matmul_relu")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    ir = b.build()
    return OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")


# ============================================================
# Tool Schema Tests
# ============================================================

def test_tool_count():
    """10 tools defined."""
    assert len(TOOLS) == 10
    assert len(get_tool_names()) == 10


def test_tool_names():
    """All expected tools present."""
    names = get_tool_names()
    expected = [
        "create_kernel", "get_hw_profile", "analyze_compute",
        "list_legal_actions", "apply_decision", "verify_correctness",
        "compile_and_profile", "rollback", "checkpoint", "restore",
    ]
    assert names == expected


def test_tool_schema_format():
    """All tools follow OpenAI function calling format."""
    for tool in TOOLS:
        assert tool["type"] == "function"
        assert "function" in tool
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_tool_metadata_complete():
    """Every tool has metadata."""
    names = get_tool_names()
    for name in names:
        assert name in TOOL_METADATA, f"Missing metadata for {name}"
        meta = TOOL_METADATA[name]
        assert "concurrent_safe" in meta
        assert "budget_type" in meta
        assert "cost" in meta


def test_get_tool_schema():
    """Lookup individual tool schema."""
    schema = get_tool_schema("apply_decision")
    assert schema is not None
    assert schema["function"]["name"] == "apply_decision"
    assert get_tool_schema("nonexistent") is None


def test_apply_decision_requires_rationale():
    """apply_decision has rationale as required parameter."""
    schema = get_tool_schema("apply_decision")
    required = schema["function"]["parameters"]["required"]
    assert "rationale" in required


def test_concurrent_safe_tools():
    """Read-only tools should be concurrent_safe."""
    safe = [n for n, m in TOOL_METADATA.items() if m["concurrent_safe"]]
    assert "get_hw_profile" in safe
    assert "analyze_compute" in safe
    assert "list_legal_actions" in safe
    # Mutating tools should NOT be concurrent_safe
    assert "apply_decision" not in safe
    assert "rollback" not in safe


# ============================================================
# Session Tests
# ============================================================

def test_session_creation():
    """Session initializes correctly."""
    session = _make_session()
    assert session.state == SessionState.CREATED
    assert session.budget.decisions_remaining == 50
    assert session.budget.compiles_remaining == 10
    assert len(session.messages) == 1  # system prompt
    assert "optimizer" in session.system_prompt.lower()


def test_session_system_prompt_contains_hw():
    """System prompt includes hardware info."""
    session = _make_session()
    prompt = session.system_prompt
    assert "nvidia" in prompt.lower() or "ampere" in prompt.lower()
    assert "49152" in prompt or "48" in prompt  # shared memory


def test_session_get_hw_profile():
    """Running get_hw_profile tool."""
    session = _make_session()
    result = session.run_tool("get_hw_profile", {})
    assert result["name"] == "nvidia_ampere"
    assert "budget" in result  # budget injected


def test_session_analyze_compute():
    """Running analyze_compute tool."""
    session = _make_session()
    result = session.run_tool("analyze_compute", {})
    assert result["kernel"] == "test_matmul_relu"
    assert len(result["nodes"]) == 2
    assert session.state == SessionState.ANALYZING


def test_session_apply_decision():
    """Running apply_decision tool."""
    session = _make_session()
    result = session.run_tool("apply_decision", {
        "kind": "tile",
        "params": {"loop": "i", "factors": [64, 16]},
        "rationale": "align with L2 cache line",
    })
    assert result["success"] is True
    assert session.state == SessionState.OPTIMIZING
    assert session.budget.decisions_used == 1
    assert session.budget.decisions_remaining == 49


def test_session_budget_tracking():
    """Budget decrements correctly."""
    session = _make_session()
    session.budget.max_decisions = 3

    for i in range(3):
        result = session.run_tool("apply_decision", {
            "kind": "tile",
            "params": {"loop": f"loop_{i}", "factors": [64]},
            "rationale": f"test {i}",
        })
        assert result["success"] is True

    # 4th should fail
    result = session.run_tool("apply_decision", {
        "kind": "tile",
        "params": {"loop": "loop_3", "factors": [64]},
        "rationale": "over budget",
    })
    assert result["success"] is False
    assert "budget" in result.get("error", "").lower()


def test_session_rollback():
    """Rollback via session."""
    session = _make_session()
    session.run_tool("apply_decision", {
        "kind": "tile", "params": {"loop": "i", "factors": [64]}, "rationale": "test",
    })
    assert session.env.strategy.decision_count == 1

    result = session.run_tool("rollback", {"steps": 1})
    assert result["rolled_back"] == 1
    assert session.env.strategy.decision_count == 0


def test_session_checkpoint_restore():
    """Checkpoint and restore via session."""
    session = _make_session()
    session.run_tool("apply_decision", {
        "kind": "tile", "params": {"loop": "i", "factors": [64]}, "rationale": "a",
    })
    session.run_tool("checkpoint", {"name": "cp1"})
    session.run_tool("apply_decision", {
        "kind": "tile", "params": {"loop": "j", "factors": [128]}, "rationale": "b",
    })
    assert session.env.strategy.decision_count == 2

    result = session.run_tool("restore", {"checkpoint_id": "cp1"})
    assert result["success"] is True
    assert session.env.strategy.decision_count == 1


def test_session_trajectory():
    """Trajectory records all actions."""
    session = _make_session()
    session.run_tool("analyze_compute", {})
    session.run_tool("apply_decision", {
        "kind": "tile", "params": {"loop": "i", "factors": [64]}, "rationale": "test",
    })

    trajectory = session.export_trajectory()
    assert len(trajectory) >= 4  # 2 actions + 2 results
    assert trajectory[0]["tool"] == "analyze_compute"


def test_session_summary():
    """Summary contains key info."""
    session = _make_session()
    session.run_tool("apply_decision", {
        "kind": "tile", "params": {"loop": "i", "factors": [64]}, "rationale": "test",
    })
    summary = session.summary()
    assert summary["kernel_id"] == "test_matmul_relu"
    assert summary["decisions"] == 1
    assert summary["state"] == "optimizing"
