# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for C1 LLM-backed soft-verify (llm_soft_verify)."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import pytest

from arke.agent.verification import llm_soft_verify, soft_verify, SoftVerifyResult


class _FakeLLMHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /v1/chat/completions mock."""

    # Class-level response control
    response_json = {"approved": True, "reason": "looks correct"}
    status_code = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        # Store last request for assertions
        _FakeLLMHandler.last_request = body
        resp = json.dumps({
            "choices": [{"message": {"content": json.dumps(self.response_json)}}]
        }).encode()
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):
        pass  # suppress output


@pytest.fixture
def llm_server():
    """Start a fake LLM server on an ephemeral port."""
    _FakeLLMHandler.response_json = {"approved": True, "reason": "looks correct"}
    _FakeLLMHandler.status_code = 200
    _FakeLLMHandler.last_request = None
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLMHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()


class TestLLMSoftVerify:
    def test_rules_fail_short_circuits_llm(self, llm_server):
        """If rule-based check fails, LLM is never called."""
        decisions = [{"kind": "tile", "params": {"factors": [-1]}}]
        r = llm_soft_verify(
            decisions, op_name="matmul",
            base_url=f"http://127.0.0.1:{llm_server}/v1",
            api_key="test-key",
        )
        assert not r.approved
        assert any("R2" in v for v in r.rule_violations)
        assert _FakeLLMHandler.last_request is None  # LLM never called

    def test_llm_approves(self, llm_server):
        _FakeLLMHandler.response_json = {"approved": True, "reason": "strategy is valid"}
        decisions = [{"kind": "tile", "params": {"loop": "i", "factors": [32]}}]
        r = llm_soft_verify(
            decisions, op_name="matmul",
            shapes={"A": [512, 512], "B": [512, 512]},
            base_url=f"http://127.0.0.1:{llm_server}/v1",
            api_key="test-key",
        )
        assert r.approved
        assert "valid" in r.reason
        assert _FakeLLMHandler.last_request is not None

    def test_llm_rejects(self, llm_server):
        _FakeLLMHandler.response_json = {"approved": False, "reason": "tile too large for shared mem"}
        decisions = [{"kind": "tile", "params": {"loop": "i", "factors": [64]}}]
        r = llm_soft_verify(
            decisions, op_name="softmax",
            base_url=f"http://127.0.0.1:{llm_server}/v1",
            api_key="test-key",
        )
        assert not r.approved
        assert "shared mem" in r.reason
        assert any("LLM:" in v for v in r.rule_violations)

    def test_llm_failure_fail_open(self, llm_server):
        """LLM error should NOT block — fail open (approved=True)."""
        _FakeLLMHandler.status_code = 500
        decisions = [{"kind": "tile", "params": {"loop": "i", "factors": [32]}}]
        r = llm_soft_verify(
            decisions, op_name="matmul",
            base_url=f"http://127.0.0.1:{llm_server}/v1",
            api_key="test-key",
        )
        assert r.approved
        assert "fail-open" in r.reason

    def test_no_api_key_skips(self):
        """No API key = skip LLM, return approved."""
        decisions = [{"kind": "tile", "params": {"loop": "i", "factors": [32]}}]
        r = llm_soft_verify(
            decisions, op_name="matmul",
            api_key="",
            base_url="http://localhost:1",  # unreachable
        )
        assert r.approved
        assert "no API key" in r.reason

    def test_prompt_structure(self, llm_server):
        """Verify the prompt sent to LLM contains op/shapes/decisions."""
        _FakeLLMHandler.response_json = {"approved": True, "reason": "ok"}
        llm_soft_verify(
            [{"kind": "tile", "params": {"loop": "k", "factors": [16]}, "rationale": "reduce contraction"}],
            op_name="matmul",
            shapes={"A": [256, 256], "B": [256, 256]},
            hw_name="nvidia_ampere",
            base_url=f"http://127.0.0.1:{llm_server}/v1",
            api_key="test-key",
        )
        req = _FakeLLMHandler.last_request
        user_msg = req["messages"][1]["content"]
        assert "matmul" in user_msg
        assert "256" in user_msg
        assert "reduce contraction" in user_msg
        assert "nvidia_ampere" in user_msg

    def test_llm_returns_markdown_wrapped_json(self, llm_server):
        """Handle LLM wrapping JSON in markdown code block."""
        # Simulate LLM returning ```json\n{...}\n```
        class _MarkdownHandler(_FakeLLMHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                content = '```json\n{"approved": false, "reason": "bad tile alignment"}\n```'
                resp = json.dumps({
                    "choices": [{"message": {"content": content}}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            def log_message(self, *a): pass

        md_srv = HTTPServer(("127.0.0.1", 0), _MarkdownHandler)
        port = md_srv.server_address[1]
        t = threading.Thread(target=md_srv.serve_forever, daemon=True)
        t.start()
        try:
            r = llm_soft_verify(
                [{"kind": "tile", "params": {"loop": "i", "factors": [7]}}],
                op_name="softmax",
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="test-key",
            )
            assert not r.approved
            assert "bad tile alignment" in r.reason
        finally:
            md_srv.shutdown()
