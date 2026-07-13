# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for D3/M3 RL corpus quality gates (arke/learn/rl_quality.py)."""

import json
import pytest
from pathlib import Path

from arke.learn.rl_quality import (
    quality_gate, QualityReport, deduplicate, _validate_sample,
)


def _write_corpus(path: Path, samples: list[dict]) -> Path:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


def _step(op="matmul", reward=1):
    return {"type": "step", "op": op, "action": {"kind": "tile"}, "reward": reward,
            "shape": {"M": 512}, "prior_decisions": []}


def _traj(op="matmul", reward=3, decisions=None):
    return {"type": "trajectory", "op": op, "decisions": decisions or [{"kind": "tile"}],
            "final_reward": reward, "shape": {"M": 512}}


# ── Schema validation ───────────────────────────────────────────────
class TestSchemaValidation:
    def test_valid_step(self):
        assert _validate_sample(_step()) is None

    def test_valid_trajectory(self):
        assert _validate_sample(_traj()) is None

    def test_step_missing_field(self):
        s = _step()
        del s["reward"]
        assert "missing" in _validate_sample(s)

    def test_trajectory_missing_decisions(self):
        t = _traj()
        del t["decisions"]
        assert "missing" in _validate_sample(t)

    def test_bad_reward_type(self):
        s = _step()
        s["reward"] = "high"
        assert "not numeric" in _validate_sample(s)

    def test_unknown_type(self):
        assert "unknown" in _validate_sample({"type": "magic"})


# ── Deduplication ───────────────────────────────────────────────────
class TestDeduplication:
    def test_exact_duplicates_removed(self):
        t1 = _traj(op="matmul", reward=3)
        t2 = _traj(op="matmul", reward=3)  # identical
        out, removed = deduplicate([t1, t2])
        assert removed == 1
        assert len(out) == 1

    def test_different_decisions_kept(self):
        t1 = _traj(decisions=[{"kind": "tile", "params": {"BM": 32}}])
        t2 = _traj(decisions=[{"kind": "tile", "params": {"BM": 64}}])
        out, removed = deduplicate([t1, t2])
        assert removed == 0
        assert len(out) == 2

    def test_steps_not_deduped(self):
        s1 = _step()
        s2 = _step()  # identical steps are kept
        out, removed = deduplicate([s1, s2])
        assert removed == 0 and len(out) == 2

    def test_different_ops_kept(self):
        t1 = _traj(op="matmul")
        t2 = _traj(op="softmax")
        out, removed = deduplicate([t1, t2])
        assert removed == 0


# ── Full quality gate ──────────────────────────────────────────────
class TestQualityGate:
    def test_healthy_corpus_passes(self, tmp_path):
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _step("matmul", 1), _step("matmul", 3),
            _traj("matmul", 3), _traj("softmax", 2),
        ])
        report = quality_gate(corpus, min_beat=1)
        assert report.passed
        assert report.total_samples == 4
        assert report.total_steps == 2
        assert report.total_trajectories == 2
        assert report.dedup_removed == 0

    def test_no_beat_baseline_fails(self, tmp_path):
        """Corpus with all reward=1 → beat distribution check fails."""
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _step("matmul", 1), _traj("matmul", 1),
        ])
        report = quality_gate(corpus, min_beat=1)
        assert not report.passed
        beat_check = next(c for c in report.checks if c.name == "reward_distribution")
        assert not beat_check.passed

    def test_tier_coverage_fails(self, tmp_path):
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _traj("matmul", 3),
        ])
        report = quality_gate(corpus, min_per_tier=2, tiers=["matmul", "softmax"])
        assert not report.passed
        tier_check = next(c for c in report.checks if c.name == "tier_coverage")
        assert not tier_check.passed
        assert "softmax" in tier_check.detail

    def test_dedup_reflected(self, tmp_path):
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _traj("matmul", 3), _traj("matmul", 3),  # duplicate
        ])
        report = quality_gate(corpus, min_beat=1)
        assert report.dedup_removed == 1
        assert report.total_trajectories == 1

    def test_schema_error_fails(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text('{"type":"step","op":"x"}\n')  # missing 'action' and 'reward'
        report = quality_gate(p)
        schema_check = next(c for c in report.checks if c.name == "schema_sanity")
        assert not schema_check.passed

    def test_missing_file_fails(self, tmp_path):
        report = quality_gate(tmp_path / "nope.jsonl")
        assert not report.passed

    def test_report_summary(self, tmp_path):
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _step("matmul", 3), _traj("matmul", 3),
        ])
        report = quality_gate(corpus)
        s = report.summary()
        assert "PASS" in s
        assert "matmul" in s

    def test_report_to_dict(self, tmp_path):
        corpus = _write_corpus(tmp_path / "c.jsonl", [
            _traj("matmul", 3),
        ])
        report = quality_gate(corpus)
        d = report.to_dict()
        assert d["passed"] is True
        assert len(d["checks"]) == 4


class TestOnRealCorpus:
    """Run quality gate on the actual accumulated corpus (if present)."""

    @pytest.fixture
    def corpus_path(self):
        p = Path("benchmarks/results/phase4/live/rl_corpus.jsonl")
        if not p.exists():
            pytest.skip("No live corpus available")
        return p

    def test_real_corpus_schema_valid(self, corpus_path):
        report = quality_gate(corpus_path, min_beat=0)
        schema_check = next(c for c in report.checks if c.name == "schema_sanity")
        assert schema_check.passed, schema_check.detail

    def test_real_corpus_has_beat_baseline(self, corpus_path):
        report = quality_gate(corpus_path, min_beat=1)
        beat_check = next(c for c in report.checks if c.name == "reward_distribution")
        assert beat_check.passed, f"No beat-baseline trajectory: {report.reward_histogram}"
