# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for D3 RL dataset extraction (arke/learn/rl_dataset.py)."""

import json
from pathlib import Path

from arke.agent.verification import RobustReward
from arke.learn.rl_dataset import (
    extract_rl_samples, build_rl_dataset, reward_histogram,
    RLStepSample,
)


def _write_traj(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sample_trajectory(tmp_path: Path) -> Path:
    """A synthetic matmul optimization trajectory: 2 decisions, improving."""
    p = tmp_path / "traj.jsonl"
    _write_traj(p, [
        {"t": 0.0, "kind": "header", "data": {"op": "matmul", "shape": {"M": 512, "N": 512, "K": 512}}},
        {"t": 0.1, "kind": "decision", "data": {"kind": "tile", "params": {"BM": 32}, "rationale": "start small"}},
        {"t": 0.2, "kind": "profile", "data": {"op": "matmul", "correct": True, "baseline_ratio": 0.9, "latency_ms": 0.2, "backend": "cuda_c"}},
        {"t": 0.3, "kind": "decision", "data": {"kind": "tile", "params": {"BM": 64}, "rationale": "bigger tile amortizes"}},
        {"t": 0.4, "kind": "profile", "data": {"op": "matmul", "correct": True, "baseline_ratio": 1.2, "latency_ms": 0.15, "backend": "cuda_c", "robust_reward": 2}},
        {"t": 0.5, "kind": "done", "data": {"final_score": 1.2}},
    ])
    return p


class TestExtractRLSamples:
    def test_step_and_trajectory_extraction(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        steps, traj = extract_rl_samples(p)
        assert len(steps) == 2
        assert traj is not None
        assert traj.op == "matmul"
        assert len(traj.decisions) == 2

    def test_reward_assignment(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        steps, traj = extract_rl_samples(p)
        # First step: correct + ratio 0.9 (no speedup) → CORRECT tier (1)
        assert steps[0].reward == int(RobustReward.CORRECT)
        # Second step: pre-recorded robust_reward=2 → BEATS_EAGER
        assert steps[1].reward == 2
        # Trajectory final reward = last profile's reward
        assert traj.final_reward == 2

    def test_prior_decisions_accumulate(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        steps, _ = extract_rl_samples(p)
        assert steps[0].prior_decisions == []
        assert len(steps[1].prior_decisions) == 1
        assert steps[1].prior_decisions[0]["params"] == {"BM": 32}

    def test_shape_context_propagated(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        steps, _ = extract_rl_samples(p)
        assert steps[0].shape == {"M": 512, "N": 512, "K": 512}

    def test_empty_trajectory(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        steps, traj = extract_rl_samples(p)
        assert steps == []
        assert traj is None

    def test_incorrect_kernel_scores_negative(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        _write_traj(p, [
            {"t": 0.0, "kind": "header", "data": {"op": "matmul", "shape": {}}},
            {"t": 0.1, "kind": "decision", "data": {"kind": "tile", "params": {}, "rationale": "x"}},
            {"t": 0.2, "kind": "profile", "data": {"correct": False, "baseline_ratio": 5.0}},
        ])
        steps, _ = extract_rl_samples(p)
        # Fast but WRONG → -1 (anti-reward-hacking)
        assert steps[0].reward == int(RobustReward.INCORRECT)


class TestBuildRLDataset:
    def test_build_writes_jsonl(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        out = tmp_path / "rl_dataset.jsonl"
        counts = build_rl_dataset([p], out)
        assert counts["steps"] == 2
        assert counts["trajectories"] == 1
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3  # 2 steps + 1 trajectory
        # Verify each line is valid JSON with a 'type' tag
        types = [json.loads(ln)["type"] for ln in lines]
        assert types.count("step") == 2
        assert types.count("trajectory") == 1

    def test_reward_histogram(self, tmp_path):
        p = _sample_trajectory(tmp_path)
        steps, _ = extract_rl_samples(p)
        hist = reward_histogram(steps)
        assert hist[int(RobustReward.CORRECT)] == 1
        assert hist[2] == 1
