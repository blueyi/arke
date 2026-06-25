# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Demo B (D8-X2) — MaxAutotuneRunner onboarding tests.

Verifies the new BaselineRunner subclass satisfies the BaselineRunner ABC
contract and registers into the ladder. GPU-dependent correctness/timing is
covered by benchmarks/results/phase1/stage8/extensibility BL1 evidence; these
unit tests stay CPU-safe (contract shape only) so they run in CI.
"""

from __future__ import annotations

import pytest

import benchmarks.baselines.max_autotune  # noqa: F401  registers the runner
from benchmarks.baselines.base import BaselineRunner, get_all_runners
from benchmarks.baselines.max_autotune import MaxAutotuneRunner


def test_is_baseline_runner_subclass():
    assert issubclass(MaxAutotuneRunner, BaselineRunner)


def test_contract_members_present():
    r = MaxAutotuneRunner()
    # 6 ABC members resolvable without raising
    assert isinstance(r.name, str) and r.name == "torch.compile-max-autotune"
    assert isinstance(r.priority, int) and r.priority == 4
    assert "max-autotune" in r.source
    assert isinstance(r.available, bool)
    assert r.supports("matmul") is True
    assert r.supports("relu") is True
    assert r.supports("rmsnorm") is True
    assert r.supports("flash_attention") is False


def test_registered_in_ladder_when_available():
    """If CUDA is present the runner must surface in the ladder registry."""
    r = MaxAutotuneRunner()
    if not r.available:
        pytest.skip("CUDA/torch.compile not available on this host")
    names = [run.name for run in get_all_runners()]
    assert "torch.compile-max-autotune" in names


def test_get_fn_none_for_unsupported_op():
    r = MaxAutotuneRunner()
    # unsupported op returns None regardless of host (no compile attempted)
    assert r.get_fn("flash_attention", 16, 16, 16) is None
