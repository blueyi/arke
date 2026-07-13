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
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": _SERVER_NAME,
                "version": FACADE_VERSION,
                "facadeContractId": FACADE_CONTRACT_ID,
                "op": self.op_name,
                "target_hw": self.target_hw,
            },
        }

    # ── resources primitive ───────────────────────────────────────────
    def _resources_list(self) -> dict[str, Any]:
        """Expose Arke optimization context as readable MCP resources."""
        return {"resources": [
            {"uri": "arke://context/op",
             "name": "Optimization context",
             "description": "Current op, shapes, and target hardware.",
             "mimeType": "application/json"},
            {"uri": "arke://hw/profile",
             "name": "Hardware profile",
             "description": "Target HW constraints (SM, smem, regs, warp size).",
             "mimeType": "application/json"},
            {"uri": "arke://strategy/current",
             "name": "Current StrategyIR",
             "description": "The StrategyIR + decision log accumulated so far.",
             "mimeType": "application/json"},
            {"uri": "arke://actions/legal",
             "name": "Legal actions",
             "description": "The compiler-legal move set for the current state.",
             "mimeType": "application/json"},
        ]}

    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        if uri == "arke://context/op":
            body = {"op": self.op_name, "shapes": self.env.op_inputs,
                    "target_hw": self.target_hw}
        elif uri == "arke://hw/profile":
            hw = self.env.hw_profile
            body = getattr(hw, "to_dict", lambda: {"name": hw.name})()
        elif uri == "arke://strategy/current":
            body = self.env.state.to_dict()
        elif uri == "arke://actions/legal":
            payload = json.loads(
                self.registry.get("list_legal_actions").execute({}).to_json())
            body = payload.get("data", payload)
        else:
            raise ValueError(f"unknown resource uri: {uri!r}")
        return {"contents": [{
            "uri": uri, "mimeType": "application/json",
            "text": json.dumps(body, default=str, indent=2),
        }]}

    # ── prompts primitive ─────────────────────────────────────────────
    def _prompts_list(self) -> dict[str, Any]:
        """Expose reusable optimization-workflow prompt templates."""
        return {"prompts": [
            {"name": "optimize_kernel",
             "description": "Guide the agent through the bounded compile→verify→"
                            "profile→adjust loop for the current op.",
             "arguments": [
                 {"name": "target_ratio", "description":
                  "Target speedup vs baseline (e.g. 1.0 = parity)", "required": False},
             ]},
            {"name": "explain_strategy",
             "description": "Summarize the current StrategyIR + @rationale trail.",
             "arguments": []},
        ]}

    def _prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name == "optimize_kernel":
            target = args.get("target_ratio", "1.0")
            text = (
                f"You are optimizing the '{self.op_name}' kernel "
                f"(shapes={self.env.op_inputs}) for {self.target_hw}.\n"
                "Loop: (1) get_hw_profile + analyze_compute; (2) list_legal_actions; "
                "(3) apply_decision with a concrete @rationale for WHY; "
                "(4) verify_correctness (V1); (5) compile_and_profile (V2). "
                "checkpoint before risky exploration, rollback if a branch regresses. "
                f"Only apply moves from list_legal_actions. Target baseline_ratio >= {target}. "
                "Every decision MUST carry a non-empty rationale."
            )
        elif name == "explain_strategy":
            snap = self.env.state.to_dict()
            text = (
                "Summarize the current optimization strategy. Decision log:\n"
                + json.dumps(snap.get("decision_log", snap), default=str, indent=2)
            )
        else:
            raise ValueError(f"unknown prompt: {name!r}")
        return {
            "description": f"Arke prompt: {name}",
            "messages": [{"role": "user",
                          "content": {"type": "text", "text": text}}],
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
            elif method == "resources/list":
                result = self._resources_list()
            elif method == "resources/read":
                result = self._resources_read(params)
            elif method == "prompts/list":
                result = self._prompts_list()
            elif method == "prompts/get":
                result = self._prompts_get(params)
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


    # ── Streamable HTTP transport (remote) ────────────────────────────
    def build_http_server(self, host: str = "127.0.0.1", port: int = 8765):
        """Construct (but do not start) an ``http.server.HTTPServer`` bound to
        ``host:port`` that serves this MCP env over Streamable HTTP.

        Returns the ``HTTPServer`` instance; call ``.serve_forever()`` to run
        or ``.shutdown()`` to stop. :meth:`serve_http` uses this internally.
        Passing ``port=0`` binds an ephemeral port (read back via
        ``httpd.server_address[1]``) — handy for tests.
        """
        import http.server

        server_self = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, code: int, body: dict, ctype: str = "application/json"):
                data = json.dumps(body, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("/mcp", "/health", ""):
                    self._send(200, {
                        "server": _SERVER_NAME, "version": FACADE_VERSION,
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "transport": "streamable-http", "op": server_self.op_name,
                    })
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if self.path.rstrip("/") != "/mcp":
                    self._send(404, {"error": "not found; POST to /mcp"})
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                try:
                    request = json.loads(raw)
                except json.JSONDecodeError:
                    self._send(200, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}})
                    return

                # If client wants SSE streaming (Accept: text/event-stream),
                # respond with Server-Sent Events. For tools/call on expensive
                # tools, emit progress events before the final result.
                accept = self.headers.get("Accept", "")
                wants_sse = "text/event-stream" in accept

                if wants_sse:
                    self._handle_sse(request)
                    return

                response = server_self.handle(request)
                if response is None:  # notification → 202 Accepted
                    self.send_response(202)
                    self.end_headers()
                    return
                self._send(200, response)

            def _sse_frame(self, event: str, data: dict) -> bytes:
                """Format one SSE frame: event + data + blank line."""
                payload = json.dumps(data, default=str)
                return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")

            def _handle_sse(self, request: dict):
                """Stream progress + final result as SSE frames.

                After writing all frames the handler returns, which closes the
                connection (signals EOF to the client so it can finish reading).
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # Omit Connection: keep-alive — let the handler return close
                # the connection (EOF), so clients read() finishes.
                self.end_headers()

                method = request.get("method", "")
                params = request.get("params", {}) or {}

                is_tools_call = method == "tools/call"
                tool_name = params.get("name", "") if is_tools_call else ""
                expensive = tool_name in ("compile_and_profile", "verify_correctness")

                buf: list[bytes] = []

                if expensive:
                    buf.append(self._sse_frame("progress", {
                        "stage": "starting", "tool": tool_name,
                        "message": f"Starting {tool_name}...",
                    }))
                    buf.append(self._sse_frame("progress", {
                        "stage": "compiling", "tool": tool_name,
                        "message": f"Compiling kernel ({tool_name})...",
                    }))

                response = server_self.handle(request)

                if expensive:
                    buf.append(self._sse_frame("progress", {
                        "stage": "complete", "tool": tool_name,
                        "message": f"{tool_name} completed.",
                    }))

                if response is None:
                    buf.append(self._sse_frame("notification_ack", {
                        "message": "notification accepted"}))
                else:
                    buf.append(self._sse_frame("result", response))

                # Write all frames at once and return (closes connection).
                self.wfile.write(b"".join(buf))
                self.wfile.flush()

            def log_message(self, format, *args):  # noqa: A002 — silence logging
                pass

        return http.server.HTTPServer((host, port), _Handler)

    def serve_http(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Serve MCP over Streamable HTTP (remote transport).

        Implements the MCP Streamable-HTTP shape with the Python stdlib only
        (zero runtime deps, matching the stdio server): a single ``POST /mcp``
        endpoint accepting a JSON-RPC request body and returning a JSON-RPC
        response; ``GET /mcp`` (or ``/health``) returns a liveness document.
        This lets a remote MCP client (another host/container) drive the same
        8-tool Façade + resources + prompts.
        """
        httpd = self.build_http_server(host=host, port=port)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()


def serve(op_name: str, shapes: dict[str, list[int]], target_hw: str = "nvidia_ampere") -> None:
    """Entry point for ``arke mcp serve`` (stdio)."""
    ArkeMCPServer(op_name, shapes, target_hw).serve_stdio()


def serve_http(op_name: str, shapes: dict[str, list[int]],
               target_hw: str = "nvidia_ampere",
               host: str = "127.0.0.1", port: int = 8765) -> None:
    """Entry point for ``arke mcp serve --http`` (Streamable HTTP, remote)."""
    ArkeMCPServer(op_name, shapes, target_hw).serve_http(host=host, port=port)
