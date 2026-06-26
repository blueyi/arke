# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for C4 — Arke MCP server (Mode C, N3).

Drives the JSON-RPC handler in-process (no subprocess / no real stdio) to
verify the MCP wire contract: initialize, tools/list (exactly the 8 frozen
Façade tools), tools/call, notifications (no response), and error envelopes.
"""

from __future__ import annotations

import io
import json

from arke.agent.facade import FACADE_V1_TOOLS
from arke.agent.mcp_server import ArkeMCPServer, MCP_PROTOCOL_VERSION


def _server():
    return ArkeMCPServer("matmul", {"A": [128, 64], "B": [64, 128]})


def _req(method, params=None, req_id=1):
    r = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        r["params"] = params
    return r


def test_initialize_reports_protocol_and_facade():
    resp = _server().handle(_req("initialize"))
    assert resp["jsonrpc"] == "2.0" and resp["id"] == 1
    res = resp["result"]
    assert res["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "tools" in res["capabilities"]
    assert res["serverInfo"]["name"] == "arke-harness"
    assert res["serverInfo"]["op"] == "matmul"


def test_tools_list_exposes_exactly_the_8_facade_tools():
    resp = _server().handle(_req("tools/list"))
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert set(names) == set(FACADE_V1_TOOLS)  # exactly the 8 frozen tools
    assert len(names) == 8
    for t in tools:
        assert "description" in t and "inputSchema" in t


def test_tools_call_get_hw_profile():
    resp = _server().handle(_req("tools/call", {"name": "get_hw_profile", "arguments": {}}))
    out = resp["result"]
    assert out["isError"] is False
    payload = json.loads(out["content"][0]["text"])
    assert payload["success"] is True


def test_tools_call_list_legal_actions():
    resp = _server().handle(_req("tools/call",
                                 {"name": "list_legal_actions", "arguments": {"top_n": 5}}))
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["success"] is True
    assert payload["data"]["count"] >= 1


def test_tools_call_unknown_tool_is_error():
    resp = _server().handle(_req("tools/call", {"name": "nope", "arguments": {}}))
    assert resp["result"]["isError"] is True


def test_unknown_method_returns_jsonrpc_error():
    resp = _server().handle(_req("frobnicate"))
    assert resp["error"]["code"] == -32601


def test_notification_returns_none():
    # notifications/initialized has no id → no response
    resp = _server().handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert resp is None


def test_serve_stdio_round_trip():
    """End-to-end over fake stdio streams: initialize then tools/list."""
    srv = _server()
    inp = io.StringIO(
        json.dumps(_req("initialize", req_id=1)) + "\n"
        + json.dumps(_req("tools/list", req_id=2)) + "\n"
    )
    out = io.StringIO()
    srv.serve_stdio(stdin=inp, stdout=out)
    lines = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert len(lines[1]["result"]["tools"]) == 8
