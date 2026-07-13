# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for D2 verification-layer enhancements (robust reward + staged gate)."""

import numpy as np

from arke.agent.verification import (
    RobustReward, robust_reward,
    GateStage, GateReport, staged_correctness_gate,
    classify_failure, reflexion_feedback,
    soft_verify, SoftVerifyResult,
)


class TestReflexion:
    def test_classify_compile_failure(self):
        assert classify_failure("compile", "") == "compile"
        assert classify_failure("correctness", "") == "correctness"
        assert classify_failure("performance", "") == "performance"

    def test_classify_by_message(self):
        assert classify_failure("?", "nvcc compilation failed") == "compile"
        assert classify_failure("?", "redefinition of SSA value") == "compile"
        assert classify_failure("?", "max_diff mismatch in allclose") == "correctness"
        assert classify_failure("?", "kernel timed out after 120s") == "timeout"

    def test_feedback_contains_category_and_hint(self):
        fb = reflexion_feedback(stage="compile", error_message="nvcc: redefinition of %e",
                                attempt=2, max_attempts=3)
        assert "attempt 2/3" in fb
        assert "category=compile" in fb
        assert "COMPILE" in fb
        assert "redefinition" in fb

    def test_feedback_trims_long_traces(self):
        long_err = "x" * 2000
        fb = reflexion_feedback(stage="compile", error_message=long_err)
        assert "trimmed" in fb
        assert len(fb) < 1500

    def test_correctness_hint_mentions_barrier(self):
        fb = reflexion_feedback(stage="correctness", error_message="wrong output")
        assert "__syncthreads" in fb or "barrier" in fb


class TestRobustReward:
    def test_incorrect_dominates(self):
        # Wrong kernel scores -1 regardless of speed (anti-hacking).
        assert robust_reward(correct=False, eager_ratio=10.0, strong_ratio=10.0) == RobustReward.INCORRECT
        assert robust_reward(correct=None, eager_ratio=5.0, strong_ratio=5.0) == RobustReward.INCORRECT

    def test_correct_no_speedup(self):
        assert robust_reward(correct=True, eager_ratio=0.9, strong_ratio=0.5) == RobustReward.CORRECT
        # exactly at threshold = not beating
        assert robust_reward(correct=True, eager_ratio=1.05, strong_ratio=0.5) == RobustReward.CORRECT

    def test_beats_eager_only(self):
        assert robust_reward(correct=True, eager_ratio=1.2, strong_ratio=0.5) == RobustReward.BEATS_EAGER

    def test_beats_both(self):
        assert robust_reward(correct=True, eager_ratio=1.5, strong_ratio=1.1) == RobustReward.BEATS_BOTH

    def test_none_ratios(self):
        # Correct but no perf data → CORRECT tier
        assert robust_reward(correct=True, eager_ratio=None, strong_ratio=None) == RobustReward.CORRECT

    def test_threshold_respected(self):
        # 3% speedup with 5% threshold → not beating
        assert robust_reward(correct=True, eager_ratio=1.03, strong_ratio=1.03, threshold=0.05) == RobustReward.CORRECT
        # 3% speedup with 1% threshold → beats both
        assert robust_reward(correct=True, eager_ratio=1.03, strong_ratio=1.03, threshold=0.01) == RobustReward.BEATS_BOTH


class TestStagedGate:
    """Use numpy relu as candidate/reference to exercise the 5-stage gate."""

    def _make_callbacks(self, break_stage=None):
        def run_candidate(inp):
            x = inp["x"]
            out = np.maximum(0, x)
            # Optionally inject a bug at a specific stage's inputs
            if break_stage is not None and inp.get("_stage") == break_stage:
                out = out + 1.0  # wrong
            return out

        def run_reference(inp):
            return np.maximum(0, inp["x"])

        def make_inputs(stage, variant):
            rng = np.random.default_rng(variant + int(stage) * 100)
            if stage == GateStage.SHAPE_SWEEP:
                shape = (8 * (variant + 1), 16)
            elif stage == GateStage.STABILITY:
                shape = (32, 32)
            elif stage == GateStage.EDGE:
                shape = (33, 47)  # non-power-of-2
            else:
                shape = (16, 16)
            x = rng.standard_normal(shape).astype(np.float32)
            if stage == GateStage.STABILITY:
                x = x * 1e4  # large magnitude
            return {"x": x, "_stage": stage}

        def allclose(cand, ref):
            ok = bool(np.allclose(cand, ref, atol=1e-5))
            diff = float(np.max(np.abs(cand - ref)))
            return ok, diff

        def is_finite(out):
            return bool(np.all(np.isfinite(out)))

        def equal(a, b):
            return bool(np.array_equal(a, b))

        return dict(run_candidate=run_candidate, run_reference=run_reference,
                    make_inputs=make_inputs, allclose=allclose,
                    is_finite=is_finite, equal=equal)

    def test_all_stages_pass(self):
        report = staged_correctness_gate(**self._make_callbacks())
        assert report.all_passed, report.summary()
        assert len(report.stages) == 5
        assert report.first_failure is None

    def test_smoke_failure_short_circuits(self):
        report = staged_correctness_gate(**self._make_callbacks(break_stage=GateStage.SMOKE))
        assert not report.all_passed
        assert report.first_failure == GateStage.SMOKE
        # short-circuit: only 1 stage attempted
        assert len(report.stages) == 1

    def test_edge_failure(self):
        report = staged_correctness_gate(**self._make_callbacks(break_stage=GateStage.EDGE))
        assert not report.all_passed
        assert report.first_failure == GateStage.EDGE
        # all 5 attempted (edge is last)
        assert len(report.stages) == 5


# ── Soft-verify (Sakana D2) ─────────────────────────────────────────
class TestSoftVerify:
    def test_clean_decisions_approved(self):
        decisions = [
            {"kind": "tile", "params": {"loop": "i", "factors": [32]}},
            {"kind": "parallel", "params": {"loop": "j", "width": 4}},
        ]
        r = soft_verify(decisions)
        assert r.approved
        assert len(r.rule_violations) == 0

    def test_budget_exceeded(self):
        decisions = [{"kind": "tile", "params": {}} for _ in range(60)]
        r = soft_verify(decisions, budget_max_decisions=50)
        assert not r.approved
        assert any("R1" in v for v in r.rule_violations)

    def test_bad_tile_factor(self):
        r = soft_verify([{"kind": "tile", "params": {"factors": [-1]}}])
        assert not r.approved
        assert any("R2" in v for v in r.rule_violations)

    def test_duplicate_decision(self):
        d = {"kind": "tile", "params": {"loop": "i", "factors": [32]}}
        r = soft_verify([d, d])
        assert not r.approved
        assert any("R3" in v for v in r.rule_violations)

    def test_invalid_vectorize_width(self):
        r = soft_verify([{"kind": "vectorize", "params": {"width": 0}}])
        assert not r.approved
        assert any("R4" in v for v in r.rule_violations)

    def test_matmul_tile_alignment(self):
        r = soft_verify(
            [{"kind": "tile", "params": {"loop": "i", "factors": [7]}}],
            op_name="matmul",
            shapes={"A": [512, 512], "B": [512, 512]},
        )
        assert not r.approved
        assert any("R5" in v for v in r.rule_violations)

    def test_matmul_aligned_passes(self):
        r = soft_verify(
            [{"kind": "tile", "params": {"loop": "i", "factors": [64]}}],
            op_name="matmul",
            shapes={"A": [512, 512], "B": [512, 512]},
        )
        assert r.approved

    def test_to_dict(self):
        r = soft_verify([{"kind": "tile", "params": {"factors": [-1]}}])
        d = r.to_dict()
        assert d["approved"] is False
        assert len(d["rule_violations"]) >= 1

    def test_empty_decisions_approved(self):
        r = soft_verify([])
        assert r.approved
