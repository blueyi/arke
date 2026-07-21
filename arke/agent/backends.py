# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Harness — agent-backend abstraction.

The Harness exposes ONE frozen Façade (8 tools). *Who drives* those tools is a
pluggable **agent backend**:

- ``builtin``   — Arke's in-tree :class:`LLMRunner` drives the loop with a live
  LLM (BYOK via :func:`arke.agent.llm_config.load_config`). Fully self-contained.
- ``heuristic`` — the deterministic, no-LLM :func:`arke.agent.optimize.optimize`
  flow (dry-run strategy generation). Zero credentials, zero network.
- ``hermes`` / ``openclaw`` / ``<mcp>`` — an EXTERNAL agent runtime drives the
  Façade over MCP. Arke runs as the tool server (``arke mcp serve``); the
  external agent is the client. This backend prints the exact MCP launch
  contract the external runtime should connect to (Arke does not embed those
  runtimes — per the D1=a build-vs-reuse decision, Arke owns the domain server,
  external frameworks own the agent loop).

This keeps the Façade vendor-agnostic while giving a single CLI a uniform
``--backend`` selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Backends that drive the Façade via an external MCP client rather than in-tree.
EXTERNAL_MCP_BACKENDS = ("hermes", "openclaw", "cline", "continue", "claude-desktop", "mcp")
BUILTIN_BACKENDS = ("builtin", "heuristic")
REGISTRY_BACKENDS = ("triton", "mlir_gpu", "mlir", "cuda_c", "cuda-c", "cuda", "llvm", "llvm_ir")
ALL_BACKENDS = BUILTIN_BACKENDS + REGISTRY_BACKENDS + EXTERNAL_MCP_BACKENDS


@dataclass
class BackendResult:
    """Uniform result across agent backends."""
    backend: str
    op_name: str
    success: bool
    mode: str                       # "live" | "heuristic" | "mcp-server"
    detail: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "op_name": self.op_name,
            "success": self.success,
            "mode": self.mode,
            "detail": self.detail,
            "message": self.message,
        }


def run_backend(
    backend: str,
    *,
    op_name: str,
    shapes: dict[str, list[int]],
    target_hw: str = "nvidia_ampere",
    max_turns: int = 15,
    model_spec: str | None = None,
    output_dir: str | None = None,
    timeout: float = 180.0,
) -> BackendResult:
    """Dispatch an optimization run to the selected agent backend.

    Returns a :class:`BackendResult`. For external MCP backends, this does not
    block on the external agent — it returns the launch contract (the server
    command + tool list) the external runtime should connect to.
    """
    backend = (backend or "builtin").lower()

    if backend == "builtin":
        return _run_builtin(
            op_name=op_name, shapes=shapes, target_hw=target_hw,
            max_turns=max_turns, model_spec=model_spec,
            output_dir=output_dir, timeout=timeout,
        )
    if backend == "heuristic":
        return _run_heuristic(
            op_name=op_name, shapes=shapes, target_hw=target_hw,
            output_dir=output_dir,
        )
    if backend in EXTERNAL_MCP_BACKENDS:
        return _mcp_contract(
            backend=backend, op_name=op_name, shapes=shapes, target_hw=target_hw,
        )
    if backend in REGISTRY_BACKENDS:
        return _run_registry_backend(
            backend=backend, op_name=op_name, shapes=shapes, target_hw=target_hw,
            output_dir=output_dir,
        )
    raise ValueError(
        f"Unknown agent backend {backend!r}. "
        f"Choose from: {', '.join(ALL_BACKENDS)}"
    )


def _run_builtin(
    *, op_name, shapes, target_hw, max_turns, model_spec, output_dir, timeout,
) -> BackendResult:
    """Drive the Façade with the in-tree live-LLM runner (BYOK)."""
    from arke.agent.llm_config import LLMConfigError, load_config
    from arke.agent.runner import LLMRunner

    try:
        config = load_config()
    except LLMConfigError as e:
        return BackendResult(
            backend="builtin", op_name=op_name, success=False, mode="live",
            message=f"No LLM provider configured (BYOK): {e}",
        )

    with LLMRunner(config, timeout=timeout) as runner:
        result = runner.optimize(
            op_name=op_name, shapes=shapes, target_hw=target_hw,
            max_turns=max_turns, model_spec=model_spec,
            state_out=output_dir,
        )
    # P5-S5 Step 5b: persist the full turn-by-turn trajectory next to
    # state.json so live-run behavior is auditable after the fact (CLI
    # stdout gets truncated; state.json only holds the final state).
    if output_dir:
        import json as _json
        import os as _os
        _os.makedirs(output_dir, exist_ok=True)
        traj_path = _os.path.join(output_dir, "trajectory.json")
        try:
            with open(traj_path, "w", encoding="utf-8") as fh:
                _json.dump(result.to_dict(), fh, default=str, indent=2)
        except Exception:  # trajectory dump must never fail the run
            pass
    return BackendResult(
        backend="builtin", op_name=op_name,
        success=not result.errors, mode="live",
        detail=result.to_dict(),
        message=(f"live run: {result.decisions} decisions, "
                 f"{result.tool_calls} tool calls, model={result.model_used}"),
    )


def _run_heuristic(*, op_name, shapes, target_hw, output_dir) -> BackendResult:
    """Deterministic no-LLM strategy generation (dry-run)."""
    from arke.agent.optimize import optimize

    shape_csv = None
    if shapes:
        first = next(iter(shapes.values()))
        shape_csv = ",".join(str(d) for d in first)
    result = optimize(
        None, kernel=op_name, shape=shape_csv,
        output_dir=output_dir or "benchmarks/results/optimize",
        target_hw=target_hw, dry_run=True,
    )
    return BackendResult(
        backend="heuristic", op_name=op_name,
        success=result.success, mode="heuristic",
        detail=result.to_dict(),
        message=(f"heuristic dry-run: {result.decision_count} decisions, "
                 f"{result.cycles_completed} cycles"),
    )


def _mcp_contract(*, backend, op_name, shapes, target_hw) -> BackendResult:
    """Return the MCP launch contract for an external agent runtime."""
    shape_csv = ""
    if shapes:
        first = next(iter(shapes.values()))
        shape_csv = ",".join(str(d) for d in first)
    server_cmd = (
        f"arke mcp serve --kernel {op_name}"
        + (f" --shape {shape_csv}" if shape_csv else "")
        + f" --target {target_hw}"
    )
    hints = {
        "hermes": "Add to Hermes MCP config: command='arke', args=['mcp','serve',...]. "
                  "Hermes then drives the 8 Façade tools as an MCP client.",
        "openclaw": "Register the server command in OpenClaw's MCP servers list; "
                    "OpenClaw connects over stdio and calls tools/list then tools/call.",
    }
    return BackendResult(
        backend=backend, op_name=op_name, success=True, mode="mcp-server",
        detail={
            "server_command": server_cmd,
            "protocol": "MCP (JSON-RPC 2.0 over stdio)",
            "facade_tools": 8,
            "integration_hint": hints.get(backend,
                "Connect your MCP client to the server command over stdio."),
        },
        message=(f"Arke runs as an MCP server; drive it from {backend} via:\n"
                 f"  {server_cmd}"),
    )


__all__ = ["BackendResult", "run_backend", "ALL_BACKENDS",
           "BUILTIN_BACKENDS", "REGISTRY_BACKENDS", "EXTERNAL_MCP_BACKENDS"]


def _run_registry_backend(
    *,
    backend: str,
    op_name: str,
    shapes: dict[str, list[int]],
    target_hw: str = "nvidia_ampere",
    output_dir: str | None = None,
) -> BackendResult:
    """Run a single compile-and-execute pass through a registered backend.

    This is NOT an LLM-driven loop — it compiles the op with default/heuristic
    StrategyIR and reports the result. Useful for sanity-checking that a backend
    can handle a given op.
    """
    from arke.backend.protocol import get_default_registry
    from arke.ir.graph import IRGraph, IRNode
    from arke.ir.ops.registry import REGISTRY as OP_REGISTRY

    reg = get_default_registry()
    try:
        be = reg.get(backend)
    except KeyError as e:
        return BackendResult(backend=backend, op_name=op_name, success=False,
                             mode="registry", message=str(e))

    # Build a minimal IRGraph for the op.
    #
    # IRNode.inputs maps {schema_input_name: value_name_in_graph}.
    # _shapes_for() returns keys that match OpSchema.inputs (e.g. "A", "B"
    # for matmul, "X" for relu, "Q", "K", "V" for flash_attention).
    # We look up the OpSchema to get the canonical input ordering and pair
    # each schema input with the corresponding shapes-dict key by position.
    # Fallback: identity mapping when the schema is unavailable.
    shapes_keys = list(shapes.keys())
    g = IRGraph(name=op_name)
    for inp_name, shape in shapes.items():
        g.add_input(inp_name, dtype="float32", shape=shape)
    outputs = [f"{op_name}_out"]

    try:
        op_schema = OP_REGISTRY.get(op_name)
        schema_input_names = list(op_schema.inputs.keys())
    except KeyError:
        schema_input_names = None

    if schema_input_names and len(schema_input_names) == len(shapes_keys):
        # Pair schema inputs with graph value names positionally.
        node_inputs = dict(zip(schema_input_names, shapes_keys))
    else:
        # Fallback: identity mapping (shapes keys ARE the input names).
        node_inputs = {k: k for k in shapes_keys}

    g.add_node(IRNode(
        id="n0", op=op_name,
        inputs=node_inputs,
        outputs=outputs,
    ))
    g.set_outputs(outputs)

    if not be.supports_op(op_name):
        return BackendResult(
            backend=backend, op_name=op_name, success=False, mode="registry",
            message=f"Backend {backend!r} does not support op {op_name!r}",
        )

    try:
        art = be.lower(g)
        kernel = be.compile(art)
    except Exception as e:
        return BackendResult(backend=backend, op_name=op_name, success=False,
                             mode="registry", message=f"compile error: {e}")

    # Build test inputs as torch.Tensor on device — backends (Triton, CUDA-C,
    # MLIR-GPU) all expect torch tensors, not numpy arrays.
    import torch as _torch
    test_inputs = {}
    for inp_name, shape in shapes.items():
        test_inputs[inp_name] = _torch.randn(*shape, dtype=_torch.float32,
                                              device="cuda" if _torch.cuda.is_available() else "cpu")

    try:
        out = be.run(kernel, test_inputs)
    except Exception as e:
        return BackendResult(backend=backend, op_name=op_name, success=False,
                             mode="registry", message=f"run error: {e}")

    return BackendResult(
        backend=backend, op_name=op_name, success=True, mode="registry",
        message=f"Backend {backend!r} compiled and ran {op_name} successfully.",
        detail={"output_keys": list(out.keys()) if isinstance(out, dict) else ["result"]},
    )
