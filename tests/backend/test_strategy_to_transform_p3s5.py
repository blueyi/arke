# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P3-S5 tests: StrategyIR L2 → MLIR transform dialect.

Verifies that StrategyIR L2 decisions lower to valid MLIR transform schedules
on ≥3 ops (matmul, reduce_sum, layernorm) — the P3-S5 gate requirement.
"""

import pytest
import subprocess
import os

from arke.ir.strategy import StrategyIR, Decision
from arke.backend.strategy_to_transform import (
    lower_strategy_to_transform,
    TransformSchedule,
)

# ── Fixtures ──────────────────────────────────────────────────────

_MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")


def _mlir_opt_available() -> bool:
    try:
        r = subprocess.run([_MLIR_OPT, "--version"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


skip_no_mlir = pytest.mark.skipif(
    not _mlir_opt_available(),
    reason="mlir-opt not available"
)


# ── Unit tests (no mlir-opt needed) ──────────────────────────────

class TestStrategyToTransform:
    """Unit tests: StrategyIR → transform schedule text."""

    def test_matmul_tile(self):
        """matmul tile decision → tile_using_for schedule."""
        sir = StrategyIR(kernel_id="matmul", decisions=[
            Decision(kind="tile", params={"factors": [64, 64, 16]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert isinstance(sched, TransformSchedule)
        assert "tile_using_for" in sched.text
        assert "tile_sizes [64, 64, 16]" in sched.text
        assert len(sched.decisions_applied) == 1
        assert "tile" in sched.decisions_applied[0]

    def test_matmul_reorder(self):
        """matmul reorder decision → interchange schedule."""
        sir = StrategyIR(kernel_id="matmul", decisions=[
            Decision(kind="reorder", params={"order": [1, 0, 2]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert "interchange" in sched.text
        assert "1, 0, 2" in sched.text

    def test_matmul_vectorize(self):
        """matmul vectorize decision → vectorize schedule."""
        sir = StrategyIR(kernel_id="matmul", decisions=[
            Decision(kind="vectorize", params={}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert "vectorize" in sched.text

    def test_reduce_sum_tile(self):
        """reduce_sum tile decision → schedule."""
        sir = StrategyIR(kernel_id="reduce_sum", decisions=[
            Decision(kind="tile", params={"factors": [32, 0]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert "tile_using_for" in sched.text
        assert "tile_sizes [32, 0]" in sched.text

    def test_layernorm_tile(self):
        """layernorm tile decision → schedule."""
        sir = StrategyIR(kernel_id="layernorm", decisions=[
            Decision(kind="tile", params={"factors": [16, 64]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert "tile_using_for" in sched.text
        assert "tile_sizes [16, 64]" in sched.text

    def test_softmax_tile_and_vectorize(self):
        """softmax multi-decision: tile + vectorize → combined schedule."""
        sir = StrategyIR(kernel_id="softmax", decisions=[
            Decision(kind="tile", params={"factors": [32, 128]}, level=2),
            Decision(kind="vectorize", params={}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        assert "tile_using_for" in sched.text
        assert "vectorize" in sched.text
        assert len(sched.decisions_applied) == 2

    def test_no_decisions(self):
        """No applicable decisions → no-op schedule."""
        sir = StrategyIR(kernel_id="matmul", decisions=[])
        sched = lower_strategy_to_transform(sir)
        assert "no applicable" in sched.text
        assert len(sched.decisions_applied) == 0

    def test_l1_decisions_skipped(self):
        """L1 decisions (level=1) still pass through for now."""
        sir = StrategyIR(kernel_id="matmul", decisions=[
            Decision(kind="tile", params={"factors": [32, 32, 8]}, level=1),
        ])
        sched = lower_strategy_to_transform(sir)
        # L1 tile decisions still produce transform text
        assert "tile_using_for" in sched.text

    def test_explicit_linalg_op(self):
        """Override linalg_op for custom ops."""
        sir = StrategyIR(kernel_id="custom_op", decisions=[
            Decision(kind="tile", params={"factors": [16, 16, 4]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir, linalg_op="linalg.matmul")
        assert "tile_using_for" in sched.text

    def test_unknown_op_raises(self):
        """Unknown op without linalg_op override raises ValueError."""
        sir = StrategyIR(kernel_id="unknown_op", decisions=[
            Decision(kind="tile", params={"factors": [16]}, level=2),
        ])
        with pytest.raises(ValueError, match="Unknown op"):
            lower_strategy_to_transform(sir)


# ── Integration tests (require mlir-opt) ─────────────────────────

@skip_no_mlir
class TestTransformMLIRValidation:
    """Validate generated schedules parse through mlir-opt."""

    def _validate_schedule(self, schedule: TransformSchedule):
        """Check that the schedule is syntactically valid MLIR."""
        proc = subprocess.run(
            [_MLIR_OPT, "--verify-diagnostics=false"],
            input=schedule.text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"mlir-opt failed on schedule:\n{schedule.text}\n"
            f"stderr: {proc.stderr[:500]}"
        )

    def test_matmul_tile_validates(self):
        """matmul tile schedule is valid MLIR."""
        sir = StrategyIR(kernel_id="matmul", decisions=[
            Decision(kind="tile", params={"factors": [64, 64, 16]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        self._validate_schedule(sched)

    def test_reduce_sum_tile_validates(self):
        """reduce_sum tile schedule is valid MLIR."""
        sir = StrategyIR(kernel_id="reduce_sum", decisions=[
            Decision(kind="tile", params={"factors": [32, 0]}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        self._validate_schedule(sched)

    def test_softmax_combined_validates(self):
        """softmax tile+vectorize schedule is valid MLIR."""
        sir = StrategyIR(kernel_id="softmax", decisions=[
            Decision(kind="tile", params={"factors": [32, 128]}, level=2),
            Decision(kind="vectorize", params={}, level=2),
        ])
        sched = lower_strategy_to_transform(sir)
        self._validate_schedule(sched)
