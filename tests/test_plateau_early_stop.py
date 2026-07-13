# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for C6 PlateauEarlyStop hook."""

import pytest

from arke.agent.extensions import PlateauEarlyStop, HookRegistry


class TestPlateauEarlyStop:
    def test_improving_does_not_trigger(self):
        hook = PlateauEarlyStop(patience=3, min_improvement=0.01)
        for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
            hook({"result": {"data": {"baseline_ratio": ratio}}})
        assert not hook.should_stop
        assert hook.best_ratio == pytest.approx(0.9)

    def test_plateau_triggers_after_patience(self):
        hook = PlateauEarlyStop(patience=3, min_improvement=0.01)
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        assert not hook.should_stop
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        assert not hook.should_stop  # first was improvement (0→1.0)
        hook({"result": {"data": {"baseline_ratio": 1.005}}})
        hook({"result": {"data": {"baseline_ratio": 1.003}}})
        hook({"result": {"data": {"baseline_ratio": 1.001}}})
        assert hook.should_stop  # 3 stale compiles after peak

    def test_reset(self):
        hook = PlateauEarlyStop(patience=2)
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        hook({"result": {"data": {"baseline_ratio": 1.0}}})
        assert hook.should_stop
        hook.reset()
        assert not hook.should_stop
        assert hook.best_ratio == 0.0
        assert len(hook.history) == 0

    def test_integrates_with_hook_registry(self):
        reg = HookRegistry()
        hook = PlateauEarlyStop(patience=2)
        reg.register("PostProfile", hook)
        # fire through the registry
        reg.fire("PostProfile", {"result": {"data": {"baseline_ratio": 0.5}}})
        reg.fire("PostProfile", {"result": {"data": {"baseline_ratio": 0.5}}})
        reg.fire("PostProfile", {"result": {"data": {"baseline_ratio": 0.5}}})
        assert hook.should_stop

    def test_missing_ratio_is_noop(self):
        hook = PlateauEarlyStop(patience=1)
        hook({"result": {"data": {}}})
        hook({"result": {"data": {}}})
        hook({"result": {"data": {}}})
        assert not hook.should_stop

    def test_history_tracked(self):
        hook = PlateauEarlyStop()
        hook({"result": {"data": {"baseline_ratio": 0.5}}})
        hook({"result": {"data": {"baseline_ratio": 1.2}}})
        assert hook.history == [0.5, 1.2]
