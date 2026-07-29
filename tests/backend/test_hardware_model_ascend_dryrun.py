# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""R4 dry-run: stress-test the HardwareModel schema against a second, non-NVIDIA
target (paper Ascend 910B) to surface schema generalization gaps at design time.

Audit R4 (docs/audit/2026-07-29-architecture-audit.md): HardwareModel had only
ever been instantiated for nvidia_sm86, so its claim of backend-agnostic
generalization was untested. `ascend_910b()` fills the schema from public specs
of a SIMD/Cube accelerator. This test drives it through the SAME query API the
StrategyIR legal-action generator and gpu_tuning consume, verifying:

  (a) the schema is at least structurally *consumable* for a non-NVIDIA target
      (no crash, queries return sane values), AND
  (b) the KNOWN MISFITS are explicitly asserted as a living gap list — if a
      future schema change fixes one, this test breaks and forces the doc/gap
      list to be updated (so the gaps can't silently rot).

Nothing here runs an Ascend kernel; it's a design-time contract check.
"""

from __future__ import annotations

from arke.backend.hardware import ascend_910b, nvidia_sm86


def test_ascend_model_is_structurally_consumable() -> None:
    """The agent's query API must not crash on the non-NVIDIA model."""
    hw = ascend_910b()
    assert hw.name == "ascend_910b"
    # Tensor-core query works (Cube modeled as tensor_core).
    assert hw.has_tensor_core()
    tc = hw.tensor_core()
    assert tc is not None and tc.supported_dtypes == ("f16",)
    assert tc.count == 1
    # Memory queries return the on-chip levels.
    assert hw.memory_level("l1") is not None
    assert hw.memory_level("l0c") is not None
    assert hw.memory_level("global").size_bytes > 0
    # shared_memory_bytes() looks for a level named "shared" — Ascend has none.
    # This is MISFIT #2 surfacing: the agent's "how big is scratch" query
    # returns 0 on Ascend because the schema hardcodes the NVIDIA name.
    assert hw.shared_memory_bytes() == 0


def test_ascend_known_schema_misfits_are_present() -> None:
    """Living record of the 4 R4 gaps. If the schema is refactored to fix one,
    this test breaks on purpose — update it AND the gap list in
    docs/architecture/arke-compiler-infrastructure.md §7.7 together.
    """
    hw = ascend_910b()

    # MISFIT #1: no SIMT/warp concept — warp_size forced to 1, no "warp" domain,
    # thread-block counts are meaningless (0).
    assert hw.warp_size == 1
    assert hw.sync_domain("warp") is None
    assert hw.max_threads_per_block == 0
    assert hw.compute_capability == (0, 0)

    # MISFIT #2: on-chip memory has no NVIDIA-style "shared" level; the Cube
    # operand feeders L0A/L0B/L0C are present but tagged with the generic
    # "block" scope (schema has no "cube-operand" scope value).
    assert hw.memory_level("shared") is None
    for lvl in ("l0a", "l0b", "l0c"):
        m = hw.memory_level(lvl)
        assert m is not None and m.scope == "block"  # <- the imprecise scope

    # MISFIT #3: no field distinguishes software-managed DMA (Ascend GM<->L1<->L0)
    # from a hardware cache (NVIDIA L2). Assert the schema simply lacks it, so
    # the gap is documented in code.
    assert not hasattr(hw.memory_levels[0], "software_managed")

    # MISFIT #4: mma_tile is a bare (m,n,k) with no operand-layout (fractal) field.
    assert hw.alignment.mma_tile == (16, 16, 16)
    assert not hasattr(hw.alignment, "operand_layout")


def test_nvidia_still_the_only_production_model() -> None:
    """Guard against accidentally wiring the paper model into a real backend.
    ascend_910b must remain a paper exercise (peak_tflops=0 on its Vector unit
    is one tell it was never measured)."""
    nv = nvidia_sm86()
    asc = ascend_910b()
    # NVIDIA model has measured SIMT tflops; the paper model does not.
    nv_simt = next(c for c in nv.compute_units if c.kind == "simt")
    asc_simt = next(c for c in asc.compute_units if c.kind == "simt")
    assert nv_simt.peak_tflops > 0.0
    assert asc_simt.peak_tflops == 0.0   # unmeasured paper value
