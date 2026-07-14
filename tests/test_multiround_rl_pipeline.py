# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 5 multi-round RL pipeline deepening.

Covers:
  * arke.learn.session_recorder — multi-round session → v1.0 trajectory.
  * arke.learn.rl_dataset — step-wise reward_delta + discounted return-to-go.
  * end-to-end: recorded session mines into a quality-gate-passing corpus.
"""

import json
from pathlib import Path

import pytest

from arke.agent.verification import RobustReward
from arke.learn.rl_dataset import (
    DEFAULT_DISCOUNT,
    build_rl_dataset,
    extract_rl_samples,
)
from arke.learn.rl_quality import quality_gate
from arke.learn.session_recorder import RoundOutcome, SessionRecorder
from arke.learn.trajectory import audit_decision_rationales
from arke.learn.trajectory_schema import (
    RECORD_KINDS_V1,
    TrajectoryRecord,
    validate_payload,
)


def _record_matmul_session(path: Path) -> dict:
    rec = SessionRecorder(path, op="matmul", shape={"M": 512, "N": 512, "K": 512})
    rec.start()
    rec.record_round(
        kind="tile", params={"loop": "M", "factors": [32]},
        rationale="seed small tile for correctness",
        outcome=RoundOutcome(correct=True, eager_ratio=0.9, strong_ratio=0.6,
                             latency_ms=0.30, bottleneck="memory_bandwidth"),
    )
    rec.record_round(
        kind="tile", params={"loop": "M", "factors": [64]},
        rationale="bigger tile beats eager",
        outcome=RoundOutcome(correct=True, eager_ratio=1.25, strong_ratio=0.95,
                             latency_ms=0.20, bottleneck="shared_memory_pressure"),
    )
    rec.record_round(
        kind="compute", params={"warps": 8, "num_stages": 3},
        rationale="deep pipeline beats cuBLAS",
        outcome=RoundOutcome(correct=True, eager_ratio=1.6, strong_ratio=1.12,
                             latency_ms=0.15, bottleneck="none"),
    )
    return rec.finish()


class TestRoundOutcome:
    def test_reward_tiers(self):
        # correct, no speedup → CORRECT
        assert RoundOutcome(correct=True).reward() == int(RobustReward.CORRECT)
        # correct + beats eager only → BEATS_EAGER
        assert RoundOutcome(correct=True, eager_ratio=1.2, strong_ratio=0.9).reward() \
            == int(RobustReward.BEATS_EAGER)
        # correct + beats both → BEATS_BOTH
        assert RoundOutcome(correct=True, eager_ratio=1.5, strong_ratio=1.1).reward() \
            == int(RobustReward.BEATS_BOTH)
        # incorrect (even if fast) → INCORRECT
        assert RoundOutcome(correct=False, eager_ratio=9.0, strong_ratio=9.0).reward() \
            == int(RobustReward.INCORRECT)

    def test_baseline_ratio_prefers_strong(self):
        o = RoundOutcome(correct=True, eager_ratio=1.5, strong_ratio=1.1)
        assert o.baseline_ratio == 1.1
        o2 = RoundOutcome(correct=True, eager_ratio=1.5)
        assert o2.baseline_ratio == 1.5


class TestSessionRecorder:
    def test_emits_valid_v1_trajectory(self, tmp_path):
        p = tmp_path / "matmul.trajectory.jsonl"
        summary = _record_matmul_session(p)
        assert summary["rounds"] == 3
        assert summary["rewards"] == [1, 2, 3]
        assert summary["best_reward"] == 3

        # Every line parses as a valid v1.0 record with a known kind.
        lines = p.read_text().strip().splitlines()
        kinds = []
        for ln in lines:
            d = json.loads(ln)
            rec = TrajectoryRecord.from_dict(d)  # raises on bad envelope
            assert rec.kind in RECORD_KINDS_V1
            kinds.append(rec.kind)
        # header first, done last, and a decision/compile/profile/adjust burst per round.
        assert kinds[0] == "header"
        assert kinds[-1] == "done"
        assert kinds.count("decision") == 3
        assert kinds.count("profile") == 3
        assert kinds.count("adjust") == 3

    def test_header_payload_valid(self, tmp_path):
        p = tmp_path / "t.jsonl"
        _record_matmul_session(p)
        header = json.loads(p.read_text().splitlines()[0])
        assert validate_payload("header", header["data"]) == []

    def test_rationale_required(self, tmp_path):
        rec = SessionRecorder(tmp_path / "t.jsonl", op="matmul")
        rec.start()
        with pytest.raises(ValueError):
            rec.record_round(kind="tile", params={}, rationale="  ",
                             outcome=RoundOutcome(correct=True))
        rec.finish()

    def test_every_decision_has_rationale(self, tmp_path):
        p = tmp_path / "t.jsonl"
        _record_matmul_session(p)
        assert audit_decision_rationales(p) == []

    def test_context_manager(self, tmp_path):
        p = tmp_path / "cm.jsonl"
        with SessionRecorder(p, op="softmax", shape={"rows": 64}) as rec:
            rec.record_round(kind="tile", params={"loop": "N", "factors": [128]},
                             rationale="row block",
                             outcome=RoundOutcome(correct=True, eager_ratio=1.3,
                                                  strong_ratio=1.05))
        assert p.exists()
        assert json.loads(p.read_text().splitlines()[-1])["kind"] == "done"


class TestStepwiseReturns:
    def test_reward_delta_and_return_to_go(self, tmp_path):
        p = tmp_path / "matmul.trajectory.jsonl"
        _record_matmul_session(p)
        steps, traj = extract_rl_samples(p, discount=0.95)

        assert [s.reward for s in steps] == [1, 2, 3]
        # reward_delta: first vs 0, then +1 each round.
        assert [s.reward_delta for s in steps] == [1, 1, 1]
        # step_index is 0-based and dense.
        assert [s.step_index for s in steps] == [0, 1, 2]
        # episode_len shared across steps.
        assert all(s.episode_len == 3 for s in steps)

        # return_to_go = discounted MC return computed by reverse scan.
        g = 0.95
        assert steps[2].return_to_go == pytest.approx(3.0)
        assert steps[1].return_to_go == pytest.approx(2 + g * 3)
        assert steps[0].return_to_go == pytest.approx(1 + g * (2 + g * 3))

    def test_trajectory_aggregates(self, tmp_path):
        p = tmp_path / "matmul.trajectory.jsonl"
        _record_matmul_session(p)
        _, traj = extract_rl_samples(p, discount=0.95)
        assert traj is not None
        assert traj.step_rewards == [1, 2, 3]
        assert traj.num_steps == 3
        assert traj.final_reward == 3
        # G0 equals the first step's return_to_go.
        assert traj.discounted_return == pytest.approx(1 + 0.95 * (2 + 0.95 * 3))

    def test_discount_zero_is_myopic(self, tmp_path):
        p = tmp_path / "m.jsonl"
        _record_matmul_session(p)
        steps, _ = extract_rl_samples(p, discount=0.0)
        # With γ=0 the return-to-go collapses to the instantaneous reward.
        assert [s.return_to_go for s in steps] == [1.0, 2.0, 3.0]

    def test_discount_one_is_undiscounted_sum(self, tmp_path):
        p = tmp_path / "m.jsonl"
        _record_matmul_session(p)
        steps, traj = extract_rl_samples(p, discount=1.0)
        assert traj is not None
        assert steps[0].return_to_go == pytest.approx(6.0)  # 1+2+3
        assert traj.discounted_return == pytest.approx(6.0)

    def test_default_discount_used(self, tmp_path):
        p = tmp_path / "m.jsonl"
        _record_matmul_session(p)
        steps, _ = extract_rl_samples(p)
        assert all(s.discount == DEFAULT_DISCOUNT for s in steps)

    def test_shape_context_from_header(self, tmp_path):
        p = tmp_path / "m.jsonl"
        _record_matmul_session(p)
        steps, _ = extract_rl_samples(p)
        assert steps[0].shape == {"M": 512, "N": 512, "K": 512}
        assert steps[0].op == "matmul"


class TestEndToEnd:
    def test_multiround_corpus_passes_quality_gate(self, tmp_path):
        # Record two multi-round episodes.
        p1 = tmp_path / "matmul.trajectory.jsonl"
        _record_matmul_session(p1)

        p2 = tmp_path / "softmax.trajectory.jsonl"
        with SessionRecorder(p2, op="softmax", shape={"rows": 64, "cols": 4096}) as rec:
            rec.record_round(kind="tile", params={"loop": "N", "factors": [128]},
                             rationale="row block for coalesced loads",
                             outcome=RoundOutcome(correct=True, eager_ratio=0.95,
                                                  strong_ratio=0.8))
            rec.record_round(kind="vectorize", params={"loop": "N", "width": 4},
                             rationale="vectorized loads beat eager",
                             outcome=RoundOutcome(correct=True, eager_ratio=1.3,
                                                  strong_ratio=1.05))

        corpus = tmp_path / "rl_corpus.jsonl"
        counts = build_rl_dataset([p1, p2], corpus)
        assert counts["steps"] == 5   # 3 + 2
        assert counts["trajectories"] == 2

        report = quality_gate(corpus, min_beat=1)
        assert report.passed, report.summary()
        assert report.total_steps == 5
        assert report.total_trajectories == 2

    def test_corpus_lines_carry_return_fields(self, tmp_path):
        p = tmp_path / "m.jsonl"
        _record_matmul_session(p)
        corpus = tmp_path / "c.jsonl"
        build_rl_dataset([p], corpus)
        for ln in corpus.read_text().strip().splitlines():
            d = json.loads(ln)
            if d["type"] == "step":
                assert "return_to_go" in d
                assert "reward_delta" in d
                assert "step_index" in d
            else:
                assert "discounted_return" in d
                assert "step_rewards" in d
