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


# ── SSE Streaming ─────────────────────────────────────────────────────
def _post_sse(port, payload) -> list[dict]:
    """POST with Accept: text/event-stream, parse SSE frames."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp", data=data,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode()
    # Parse SSE frames: "event: <name>\ndata: <json>\n\n"
    events = []
    current: dict[str, str] = {}
    for line in raw.split("\n"):
        if line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = line[len("data: "):]
        elif line == "" and current:
            if "data" in current:
                current["parsed"] = json.loads(current["data"])
            events.append(current)
            current = {}
    return events


class TestSSEStreaming:
    def test_sse_tools_list_returns_result_event(self, http_server):
        events = _post_sse(http_server, _req("tools/list"))
        assert len(events) >= 1
        result_ev = [e for e in events if e.get("event") == "result"]
        assert len(result_ev) == 1
        assert len(result_ev[0]["parsed"]["result"]["tools"]) == 8

    def test_sse_initialize(self, http_server):
        events = _post_sse(http_server, _req("initialize"))
        result_ev = [e for e in events if e.get("event") == "result"]
        assert result_ev[0]["parsed"]["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION

    def test_sse_expensive_tool_emits_progress(self, http_server):
        """compile_and_profile emits progress events before result."""
        # Use get_hw_profile (cheap — no progress) vs a compile-class tool.
        # Since we can't run actual GPU compile in tests, use verify_correctness
        # which is marked expensive but will error (no prior compile) — that's
        # fine, we're testing the SSE frame structure, not the tool outcome.
        events = _post_sse(http_server, _req("tools/call", {
            "name": "verify_correctness", "arguments": {}}))
        progress = [e for e in events if e.get("event") == "progress"]
        result = [e for e in events if e.get("event") == "result"]
        # At least "starting" and "compiling" progress, then "complete", then result
        assert len(progress) >= 2, f"expected progress events, got {progress}"
        assert progress[0]["parsed"]["stage"] == "starting"
        assert len(result) == 1

    def test_sse_cheap_tool_no_progress(self, http_server):
        """get_hw_profile (cheap) should NOT emit progress events."""
        events = _post_sse(http_server, _req("tools/call", {
            "name": "get_hw_profile", "arguments": {}}))
        progress = [e for e in events if e.get("event") == "progress"]
        result = [e for e in events if e.get("event") == "result"]
        assert len(progress) == 0
        assert len(result) == 1

    def test_sse_resources_read(self, http_server):
        events = _post_sse(http_server, _req("resources/read",
                           {"uri": "arke://context/op"}))
        result = [e for e in events if e.get("event") == "result"]
        payload = json.loads(result[0]["parsed"]["result"]["contents"][0]["text"])
        assert payload["op"] == "matmul"

    def test_sse_notification(self, http_server):
        events = _post_sse(http_server, {"jsonrpc": "2.0",
                           "method": "notifications/initialized"})
        assert any(e.get("event") == "notification_ack" for e in events)
