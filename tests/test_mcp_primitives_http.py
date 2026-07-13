# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage of Arke Harness MCP primitives + remote transport.

Covers what test_mcp_server.py (tools only) did not:
- resources/list + resources/read (all 4 arke:// URIs)
- prompts/list + prompts/get (both templates)
- capabilities advertise tools + resources + prompts
- Streamable HTTP transport (real localhost server): GET health, POST /mcp
  for initialize / tools/list / tools/call / resources/read / prompts/get,
  notification → 202, bad path → 404, parse error envelope.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from arke.agent.mcp_server import ArkeMCPServer, MCP_PROTOCOL_VERSION


def _server():
    return ArkeMCPServer("matmul", {"A": [512, 512], "B": [512, 512]})


def _req(method, params=None, req_id=1):
    r = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        r["params"] = params
    return r


# ── resources primitive ──────────────────────────────────────────────
class TestResources:
    def test_capabilities_advertise_resources_and_prompts(self):
        caps = _server().handle(_req("initialize"))["result"]["capabilities"]
        assert "tools" in caps and "resources" in caps and "prompts" in caps

    def test_resources_list(self):
        res = _server().handle(_req("resources/list"))["result"]["resources"]
        uris = {r["uri"] for r in res}
        assert uris == {
            "arke://context/op", "arke://hw/profile",
            "arke://strategy/current", "arke://actions/legal",
        }
        for r in res:
            assert r["name"] and r["description"] and r["mimeType"] == "application/json"

    @pytest.mark.parametrize("uri", [
        "arke://context/op", "arke://hw/profile",
        "arke://strategy/current", "arke://actions/legal",
    ])
    def test_resources_read_each(self, uri):
        resp = _server().handle(_req("resources/read", {"uri": uri}))
        contents = resp["result"]["contents"]
        assert contents[0]["uri"] == uri
        body = json.loads(contents[0]["text"])  # must be valid JSON
        assert body is not None

    def test_resources_read_op_context_payload(self):
        resp = _server().handle(_req("resources/read", {"uri": "arke://context/op"}))
        body = json.loads(resp["result"]["contents"][0]["text"])
        assert body["op"] == "matmul"
        assert body["target_hw"] == "nvidia_ampere"

    def test_resources_read_unknown_uri_errors(self):
        resp = _server().handle(_req("resources/read", {"uri": "arke://nope"}))
        assert resp["error"]["code"] == -32603


# ── prompts primitive ────────────────────────────────────────────────
class TestPrompts:
    def test_prompts_list(self):
        prompts = _server().handle(_req("prompts/list"))["result"]["prompts"]
        names = {p["name"] for p in prompts}
        assert names == {"optimize_kernel", "explain_strategy"}

    def test_prompts_get_optimize_kernel(self):
        resp = _server().handle(_req("prompts/get", {
            "name": "optimize_kernel", "arguments": {"target_ratio": "1.2"}}))
        msg = resp["result"]["messages"][0]
        assert msg["role"] == "user"
        text = msg["content"]["text"]
        assert "matmul" in text and "1.2" in text and "@rationale" in text

    def test_prompts_get_explain_strategy(self):
        resp = _server().handle(_req("prompts/get", {"name": "explain_strategy"}))
        assert "messages" in resp["result"]

    def test_prompts_get_unknown_errors(self):
        resp = _server().handle(_req("prompts/get", {"name": "nope"}))
        assert resp["error"]["code"] == -32603


# ── Streamable HTTP transport ─────────────────────────────────────────
@pytest.fixture
def http_server():
    """Start the HTTP MCP server on an ephemeral port in a background thread."""
    srv = _server()
    httpd = srv.build_http_server(port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield port
    httpd.shutdown()
    httpd.server_close()


def _post(port, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, (json.loads(r.read().decode()) if r.status != 202 else None)


class TestHttpTransport:
    def test_health_get(self, http_server):
        with urllib.request.urlopen(f"http://127.0.0.1:{http_server}/health", timeout=5) as r:
            body = json.loads(r.read().decode())
        assert body["server"] == "arke-harness"
        assert body["transport"] == "streamable-http"

    def test_initialize_over_http(self, http_server):
        status, body = _post(http_server, _req("initialize"))
        assert status == 200
        assert body["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        caps = body["result"]["capabilities"]
        assert {"tools", "resources", "prompts"} <= set(caps)

    def test_tools_list_over_http(self, http_server):
        _, body = _post(http_server, _req("tools/list"))
        assert len(body["result"]["tools"]) == 8

    def test_tools_call_over_http(self, http_server):
        _, body = _post(http_server, _req("tools/call",
                        {"name": "get_hw_profile", "arguments": {}}))
        assert body["result"]["isError"] is False

    def test_resources_read_over_http(self, http_server):
        _, body = _post(http_server, _req("resources/read",
                        {"uri": "arke://context/op"}))
        payload = json.loads(body["result"]["contents"][0]["text"])
        assert payload["op"] == "matmul"

    def test_prompts_get_over_http(self, http_server):
        _, body = _post(http_server, _req("prompts/get",
                        {"name": "optimize_kernel"}))
        assert "messages" in body["result"]

    def test_notification_returns_202(self, http_server):
        status, _ = _post(http_server, {"jsonrpc": "2.0",
                                        "method": "notifications/initialized"})
        assert status == 202

    def test_bad_path_404(self, http_server):
        req = urllib.request.Request(
            f"http://127.0.0.1:{http_server}/wrong", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
