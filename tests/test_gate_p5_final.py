# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the P5-S_FINAL acceptance gate (no GPU)."""

from __future__ import annotations

import json

import pytest

from benchmarks import gate_p5_final as g


class TestCheckKB:
    def test_pass_when_enough_and_multisource(self, tmp_path, monkeypatch):
        kb = tmp_path / "kb.jsonl"
        rows = [{"op": "matmul", "phase": 1} for _ in range(200)]
        rows += [{"op": "matmul", "phase": 5, "backend": "llvm"} for _ in range(5)]
        kb.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setattr(g, "KB_PATH", kb)
        r = g.check_kb()
        assert r["pass"] is True
        assert r["count"] == 205
        assert r["live_llvm_p5_entries"] == 5

    def test_fail_when_under_min(self, tmp_path, monkeypatch):
        kb = tmp_path / "kb.jsonl"
        rows = [{"op": "x", "phase": 5, "backend": "llvm"} for _ in range(199)]
        kb.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setattr(g, "KB_PATH", kb)
        assert g.check_kb()["pass"] is False

    def test_fail_when_no_live_llvm(self, tmp_path, monkeypatch):
        kb = tmp_path / "kb.jsonl"
        rows = [{"op": "x", "phase": 1} for _ in range(250)]
        kb.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setattr(g, "KB_PATH", kb)
        # enough entries but no live LLVM phase-5 rationales -> not release-ready
        assert g.check_kb()["pass"] is False

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(g, "KB_PATH", tmp_path / "nope.jsonl")
        assert g.check_kb()["pass"] is False


class TestCheckS5TGate:
    def _write(self, tmp_path, monkeypatch, overall, crit):
        j = tmp_path / "gate.json"
        j.write_text(json.dumps({"overall": overall,
                                 "criteria": {k: {"pass": v} for k, v in crit.items()}}))
        monkeypatch.setattr(g, "S5_GATE_JSON", j)

    def test_pass_all_five(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, True,
                    {"C1": True, "C2": True, "C3": True, "C4": True, "C5": True})
        assert g.check_s5t_gate()["pass"] is True

    def test_fail_if_any_criterion_false(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, False,
                    {"C1": True, "C2": False, "C3": True, "C4": True, "C5": True})
        assert g.check_s5t_gate()["pass"] is False

    def test_missing_evidence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(g, "S5_GATE_JSON", tmp_path / "nope.json")
        assert g.check_s5t_gate()["pass"] is False


class TestBackendSeam:
    def test_seam_present(self):
        r = g.check_backend_seam()
        assert r["pass"] is True
        assert "register" in r["registry_methods"]


class TestS3Perf:
    def test_recorded_pass(self):
        r = g.check_s3_perf()
        assert r["pass"] is True
        assert r["recorded_geomean_median"] <= r["threshold"]
