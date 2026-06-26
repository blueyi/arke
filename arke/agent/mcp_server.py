# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Arke Harness — MCP server (Mode C, N3).

A **zero-dependency** JSON-RPC-over-stdio server that exposes the locked
Façade v1.0 8-tool surface to any MCP-compatible client (Hermes, Cline,
Continue, Claude Desktop, …). We implement the MCP wire protocol directly
(JSON-RPC 2.0 line-delimited over stdio) rather than depending on the
``mcp`` SDK, keeping Arke's runtime dependency-free.

Supported MCP methods:
  - ``initialize``        → server capabilities + protocol version
  - ``tools/list``        → the 8 Façade tools (name + description + inputSchema)
  - ``tools/call``        → execute one tool, return its ToolResult JSON
  - ``ping``              → liveness
  - ``notifications/initialized`` (notification, no response)

The 8 tools are env-bound: the server is constructed for a single
``(op_name, shapes, target_hw)`` optimization context — the client drives
the compile→profile→adjust loop by calling the tools in sequence. This
mirrors how an MCP client drives any tool server, and the Substrate sees no
mode distinction (Mode A/B/C all share the same Façade).

Design ref: docs/architecture/arke-harness.md §3.1 (Mode C), §14 (MCP).
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from arke.agent.env import ArkeEnv
from arke.agent.facade import FACADE_CONTRACT_ID, FACADE_VERSION
from arke.agent.tools import ToolRegistry

MCP_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "arke-harness"


class ArkeMCPServer:
    """A minimal MCP server exposing the 8 Façade tools over JSON-RPC/stdio."""

    def __init__(self, op_name: str, shapes: dict[str, list[int]], target_hw: str = "nvidia_ampere"):
        self.env = ArkeEnv.from_op(op_name, shapes or {})
        self.registry = ToolRegistry.with_env(self.env)
        self.op_name = op_name
        self.target_hw = target_hw

    # ── MCP method handlers ───────────────────────────────────────────
    def _tools_list(self) -> dict[str, Any]:
        tools = []
        for name in self.registry.names():
            tool = self.registry.get(name)
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters_schema(),
            })
        return {"tools": tools}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if name not in self.registry.names():
            return {
                "content": [{"type": "text", "text": f"unknown tool: {name!r}"}],
                "isError": True,
            }
        tool = self.registry.get(name)
        result = tool.execute(arguments)
        payload = json.loads(result.to_json())
        is_error = not payload.get("success", True)
        return {
            "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
            "isError": is_error,
        }

    def _initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": _SERVER_NAME,
                "version": FACADE_VERSION,
                "facadeContractId": FACADE_CONTRACT_ID,
                "op": self.op_name,
                "target_hw": self.target_hw,
            },
        }

    # ── JSON-RPC dispatch ─────────────────────────────────────────────
    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC request. Returns a response dict, or None for
        notifications (no ``id``)."""
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {}) or {}

        # Notifications (no id) → no response.
        if req_id is None and method and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = self._initialize()
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = self._tools_list()
            elif method == "tools/call":
                result = self._tools_call(params)
            else:
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as e:  # noqa: BLE001 — JSON-RPC error envelope
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    # ── stdio serve loop ──────────────────────────────────────────────
    def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Run the line-delimited JSON-RPC loop until stdin closes."""
        rd = stdin or sys.stdin
        wr = stdout or sys.stdout
        for line in rd:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                wr.write(json.dumps({"jsonrpc": "2.0", "id": None,
                         "error": {"code": -32700, "message": "parse error"}}) + "\n")
                wr.flush()
                continue
            response = self.handle(request)
            if response is not None:
                wr.write(json.dumps(response, default=str) + "\n")
                wr.flush()


def serve(op_name: str, shapes: dict[str, list[int]], target_hw: str = "nvidia_ampere") -> None:
    """Entry point for ``arke mcp serve``."""
    ArkeMCPServer(op_name, shapes, target_hw).serve_stdio()
