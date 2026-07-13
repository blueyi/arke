# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent-backend abstraction + unified `arke run` CLI."""

import json

import pytest

from arke.agent.backends import (
    run_backend, BackendResult, ALL_BACKENDS,
    BUILTIN_BACKENDS, EXTERNAL_MCP_BACKENDS,
)


class TestBackendRegistry:
    def test_backend_sets(self):
        assert "builtin" in BUILTIN_BACKENDS
        assert "heuristic" in BUILTIN_BACKENDS
        assert "hermes" in EXTERNAL_MCP_BACKENDS
        assert "openclaw" in EXTERNAL_MCP_BACKENDS
        assert set(ALL_BACKENDS) >= {"builtin", "heuristic", "hermes", "openclaw"}

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown agent backend"):
            run_backend("nonsense", op_name="matmul", shapes={"A": [8, 8], "B": [8, 8]})


class TestHeuristicBackend:
    def test_heuristic_runs_no_creds(self, tmp_path):
        r = run_backend(
            "heuristic", op_name="matmul",
            shapes={"A": [64, 64], "B": [64, 64]},
            output_dir=str(tmp_path / "out"),
        )
        assert isinstance(r, BackendResult)
        assert r.backend == "heuristic"
        assert r.mode == "heuristic"
        assert r.success
        assert r.detail["decision_count"] >= 1


class TestMcpContractBackends:
    @pytest.mark.parametrize("backend", ["hermes", "openclaw", "cline", "mcp"])
    def test_mcp_contract(self, backend):
        r = run_backend(
            backend, op_name="softmax", shapes={"X": [64, 4096]},
        )
        assert r.success
        assert r.mode == "mcp-server"
        assert "arke mcp serve --kernel softmax" in r.detail["server_command"]
        assert r.detail["facade_tools"] == 8

    def test_hermes_integration_hint(self):
        r = run_backend("hermes", op_name="matmul", shapes={"A": [8, 8], "B": [8, 8]})
        assert "Hermes" in r.detail["integration_hint"]


class TestBuiltinBackendNoCreds:
    def test_builtin_without_creds_returns_failure(self, monkeypatch, tmp_path):
        # Strip all provider creds + config so BYOK resolution fails cleanly.
        for var in ("ARKE_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "YUNWU_API_KEY", "ARKE_LLM_CONFIG"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        r = run_backend(
            "builtin", op_name="matmul",
            shapes={"A": [8, 8], "B": [8, 8]},
        )
        assert not r.success
        assert r.mode == "live"
        assert "BYOK" in r.message or "provider" in r.message.lower()


class TestCliRun:
    def test_cli_run_heuristic(self, tmp_path, capsys):
        from arke.cli import _build_parser, _cmd_run
        parser = _build_parser()
        args = parser.parse_args([
            "run", "--kernel", "matmul", "--shape", "64,64,64",
            "--backend", "heuristic", "-o", str(tmp_path / "out"), "--json",
        ])
        rc = _cmd_run(args)
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["backend"] == "heuristic"
        assert payload["success"] is True

    def test_cli_run_hermes_contract(self, tmp_path, capsys):
        from arke.cli import _build_parser, _cmd_run
        parser = _build_parser()
        args = parser.parse_args([
            "run", "--kernel", "matmul", "--shape", "512,512,512",
            "--backend", "hermes", "--json",
        ])
        rc = _cmd_run(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "mcp-server"
        assert "arke mcp serve" in payload["detail"]["server_command"]
