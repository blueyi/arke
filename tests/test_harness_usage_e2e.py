# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end verification of the 3 documented Harness usage modes.

Backs the examples in docs/architecture/arke-harness-usage.md §3 so the doc
cannot silently rot. Live-LLM (builtin) is exercised only in credential-free
form (asserts a clean "no provider" failure) — the real live path needs BYOK
keys and is covered by benchmarks/live/. The heuristic + MCP-contract + MCP
stdio-server paths are fully exercised here with no network.
"""

import json
import subprocess
import sys

import pytest

from arke.agent.mcp_server import ArkeMCPServer


class TestModeA_HeuristicCLI:
    """Doc §3 example A — `arke run --backend heuristic`."""

    def test_cli_heuristic_matmul(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "arke.cli", "run",
             "--kernel", "matmul", "--shape", "512,512,512",
             "--backend", "heuristic", "-o", str(tmp_path / "out"), "--json"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["backend"] == "heuristic"
        assert payload["success"] is True
        assert payload["detail"]["decision_count"] >= 1


class TestModeB_McpServer:
    """Doc §3 example B — external agent over MCP (stdio JSON-RPC)."""

    def test_initialize_and_tools_list(self):
        server = ArkeMCPServer("matmul", {"A": [512, 512], "B": [512, 512]})
        init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init["result"]["serverInfo"]["name"] == "arke-harness"
        tl = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = tl["result"]["tools"]
        assert len(tools) == 8
        names = {t["name"] for t in tools}
        assert "list_legal_actions" in names and "compile_and_profile" in names

    def test_tools_call_free_tool(self):
        server = ArkeMCPServer("matmul", {"A": [512, 512], "B": [512, 512]})
        resp = server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_hw_profile", "arguments": {}},
        })
        assert "result" in resp

    def test_cli_run_hermes_prints_server_contract(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "arke.cli", "run",
             "--kernel", "softmax", "--shape", "64,4096",
             "--backend", "hermes", "--json"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["mode"] == "mcp-server"
        assert "arke mcp serve --kernel softmax" in payload["detail"]["server_command"]


class TestModeC_Programmatic:
    """Doc §3bis 3b — programmatic LLMRunner (credential-free failure path)."""

    def test_builtin_without_creds_clean_failure(self, monkeypatch, tmp_path):
        from arke.agent.backends import run_backend
        for var in ("ARKE_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "YUNWU_API_KEY", "ARKE_LLM_CONFIG"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        r = run_backend("builtin", op_name="matmul",
                        shapes={"A": [8, 8], "B": [8, 8]})
        assert not r.success
        assert r.mode == "live"
