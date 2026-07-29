# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""K-H2 — HardwareModel abstraction + unified lower() signature + capabilities().

Covers:
1. HardwareModel structure & convenience queries (memory levels, sync domains,
   compute units, alignment, tensor-core detection).
2. Every backend exposes a uniform ``lower(graph, hw=None)`` signature and a
   ``capabilities()`` that reports its real feature set on a HardwareModel.
3. The engine's action-space generator gates tensor-core-only moves
   (``wmma_tile``) on ``HardwareModel.has_tensor_core()``.
"""

from __future__ import annotations

import inspect

import pytest

from arke.backend.hardware import (
    AlignmentConstraints,
    ComputeUnit,
    DEFAULT_HARDWARE,
    HardwareModel,
    MemoryLevel,
    SyncDomain,
    nvidia_sm86,
)
from arke.backend.protocol import (
    BackendCapabilities,
    default_capabilities,
    get_default_registry,
)


# ─── HardwareModel structure ─────────────────────────────────────────────────

class TestHardwareModel:
    def test_sm86_shape(self):
        hw = nvidia_sm86()
        assert hw.compute_capability == (8, 6)
        assert hw.num_sms == 30
        assert hw.warp_size == 32
        assert hw.max_threads_per_block == 1024

    def test_memory_hierarchy(self):
        hw = nvidia_sm86()
        assert hw.shared_memory_bytes() == 49152
        assert hw.memory_level("global").size_bytes == 6 * 1024 ** 3
        assert hw.memory_level("shared").scope == "block"
        assert hw.memory_level("register").scope == "thread"
        assert hw.memory_level("nonexistent") is None

    def test_sync_domains(self):
        hw = nvidia_sm86()
        warp = hw.sync_domain("warp")
        assert warp.width == 32
        assert warp.barrier_free is True
        assert hw.sync_domain("block").barrier_free is False

    def test_tensor_core(self):
        hw = nvidia_sm86()
        assert hw.has_tensor_core()
        tc = hw.tensor_core()
        assert tc.kind == "tensor_core"
        assert "f16" in tc.supported_dtypes
        assert hw.alignment.mma_tile == (16, 8, 16)

    def test_non_tc_model(self):
        hw = HardwareModel(
            name="no_tc", compute_capability=(7, 0), num_sms=1,
            max_threads_per_block=1024, max_threads_per_sm=1536,
            memory_levels=(MemoryLevel("shared", "block", 49152),),
            sync_domains=(SyncDomain("warp", 32, True),),
            compute_units=(ComputeUnit("simt", 128),),
            alignment=AlignmentConstraints(),
        )
        assert not hw.has_tensor_core()
        assert hw.tensor_core() is None

    def test_default_is_sm86(self):
        assert DEFAULT_HARDWARE.compute_capability == (8, 6)


# ─── Unified backend surface ─────────────────────────────────────────────────

class TestBackendUniformity:
    def test_all_backends_lower_accept_hw(self):
        """Every registered backend's lower() accepts an ``hw`` parameter."""
        reg = get_default_registry()
        for name in reg.list_backends():
            be = reg.get(name)
            sig = inspect.signature(be.lower)
            assert "hw" in sig.parameters, f"{name}.lower() missing hw param"

    def test_all_backends_report_capabilities(self):
        reg = get_default_registry()
        for name in reg.list_backends():
            be = reg.get(name)
            cap = be.capabilities() if hasattr(be, "capabilities") \
                else default_capabilities(be)
            assert isinstance(cap, BackendCapabilities)
            assert cap.backend_name
            assert isinstance(cap.hardware, HardwareModel)
            assert cap.max_pipeline_stages >= 1

    def test_capabilities_reflect_backend_features(self):
        """Capabilities honestly differ per backend (not a flat default)."""
        reg = get_default_registry()
        caps = {}
        for name in reg.list_backends():
            be = reg.get(name)
            if hasattr(be, "capabilities"):
                caps[name] = be.capabilities()
        # Triton emits tensor-core + multi-stage pipelines.
        if "triton" in caps:
            assert caps["triton"].tensor_core is True
            assert caps["triton"].max_pipeline_stages >= 2
        # LLVM backend does NOT emit tensor cores (scalar/vector PTX only).
        if "llvm" in caps:
            assert caps["llvm"].tensor_core is False
            assert caps["llvm"].max_pipeline_stages == 1

    def test_default_capabilities_probes_supported_ops(self):
        reg = get_default_registry()
        # mock backend (if present) or any backend: default_capabilities builds
        # a BackendCapabilities without raising.
        for name in reg.list_backends():
            be = reg.get(name)
            cap = default_capabilities(be)
            assert isinstance(cap.supported_ops, frozenset)
            break


# ─── Action-space TC gating (engine consumes HardwareModel) ──────────────────

class TestActionSpaceTCGate:
    def _tc_eligible_matmul_env(self):
        from arke.agent.env import ArkeEnv
        return ArkeEnv.from_op("matmul", {"A": [2048, 1024], "B": [1024, 2048]})

    def test_wmma_offered_on_tc_hardware(self):
        env = self._tc_eligible_matmul_env()
        assert env.hw_model is not None and env.hw_model.has_tensor_core()
        wmma = env.list_legal_actions(filter_kind="wmma_tile", top_n=64)
        assert len(wmma) > 0

    def test_wmma_gated_off_on_non_tc_hardware(self):
        env = self._tc_eligible_matmul_env()
        env.hw_model = HardwareModel(
            name="no_tc", compute_capability=(7, 0), num_sms=30,
            max_threads_per_block=1024, max_threads_per_sm=1536,
            memory_levels=(MemoryLevel("shared", "block", 49152),),
            sync_domains=(SyncDomain("warp", 32, True),),
            compute_units=(ComputeUnit("simt", 128),),
            alignment=AlignmentConstraints(),
        )
        wmma = env.list_legal_actions(filter_kind="wmma_tile", top_n=64)
        assert len(wmma) == 0
