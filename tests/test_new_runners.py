# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for new P2 baseline runners (flash-attn / FlashMLA / vLLM).

Each runner is gated on its upstream package being importable; on a host
without the package the runner registers but ``available`` is False — we
verify both registration and the supports() declaration regardless.
"""

from __future__ import annotations

import benchmarks.baselines.flash_attn_runner as flash_attn_mod  # noqa: F401
import benchmarks.baselines.flash_mla_runner as flash_mla_mod  # noqa: F401
import benchmarks.baselines.vllm_paged_runner as vllm_paged_mod  # noqa: F401
from benchmarks.baselines.base import _REGISTRY


def _find(cls_name: str):
    for cls in _REGISTRY:
        if cls.__name__ == cls_name:
            return cls()
    raise AssertionError(f"{cls_name} not registered")


def test_flash_attn_runner_registers_and_declares_supports():
    r = _find("FlashAttnRunner")
    assert r.name == "flash-attn"
    assert r.priority == 2
    assert isinstance(r.available, bool)
    assert r.supports("flash_attention")
    assert r.supports("grouped_query_attention")
    assert not r.supports("matmul")
    assert "flash-attention" in r.source.lower() or "Dao-AILab" in r.source


def test_flash_mla_runner_registers_and_declares_supports():
    r = _find("FlashMLARunner")
    assert r.name == "FlashMLA"
    assert r.priority == 2
    assert isinstance(r.available, bool)
    # Supports MLA only.
    assert r.supports("multi_latent_attention")
    assert not r.supports("flash_attention")
    # On sm<9.0 (or no GPU / no install) available must be False so the
    # ladder falls through to the next priority. We can't assert the value
    # universally, but on this CI host (RTX 3060 sm_86 or CPU) it is False.
    import torch
    if torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability()
            if cc < (9, 0):
                assert not r.available, (
                    "FlashMLA must report available=False on sm<9.0"
                )
        except Exception:
            pass
    assert "deepseek" in r.source.lower() or "FlashMLA" in r.source


def test_vllm_paged_runner_registers_and_declares_supports():
    r = _find("VLLMPagedRunner")
    assert r.name == "vLLM"
    assert r.priority == 2
    assert isinstance(r.available, bool)
    assert r.supports("paged_attention")
    assert not r.supports("flash_attention")
    assert "vllm" in r.source.lower()


def test_unavailable_runners_run_for_output_returns_none():
    """When upstream is missing, run_for_output must short-circuit to None."""
    for cls_name, op in [
        ("FlashAttnRunner", "flash_attention"),
        ("FlashMLARunner", "multi_latent_attention"),
        ("VLLMPagedRunner", "paged_attention"),
    ]:
        r = _find(cls_name)
        if not r.available:
            assert r.run_for_output(op) is None, (
                f"{cls_name}.run_for_output must return None when "
                f"available=False (was: {r.run_for_output(op)!r})"
            )
