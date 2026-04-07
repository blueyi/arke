# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for StrategyIR v1.0 (arke/ir/strategy.py)."""

import json
import pytest

from arke.ir.strategy import (
    AnyDecision,
    ConditionalDecision,
    Decision,
    HardwareConstraints,
    Rationale,
    ShapeRegime,
    StrategyIR,
    _decision_to_dict,
    _parse_decision,
)


# ============================================================
# Rationale
# ============================================================

class TestRationale:
    def test_basic(self):
        r = Rationale(text="good tile size for f16")
        assert r.text == "good tile size for f16"
        assert r.lang == "en"

    def test_lang(self):
        r = Rationale(text="example", lang="zh")
        assert r.lang == "zh"


# ============================================================
# Decision
# ============================================================

class TestDecision:
    def test_tile(self):
        d = Decision(kind="tile", params={"loop": "M", "factors": [64]})
        assert d.kind == "tile"
        assert d.level == 1
        assert d.step == 0

    def test_level_l2(self):
        d = Decision(
            kind="compute_resource",
            params={"warps": 4, "stages": 2},
            level=2,
        )
        assert d.level == 2

    def test_with_rationale(self):
        d = Decision(
            kind="fuse",
            params={"ops": ["matmul", "gelu"], "type": "epilogue"},
            rationale=Rationale(text="eliminate memory round-trip"),
        )
        assert d.rationale.text == "eliminate memory round-trip"

    def test_to_dict(self):
        d = Decision(
            kind="tile",
            params={"loop": "K", "factors": [32]},
            rationale=Rationale(text="fits in shared"),
            step=3,
            level=1,
        )
        result = _decision_to_dict(d)
        assert result["kind"] == "tile"
        assert result["params"]["loop"] == "K"
        assert result["rationale"]["text"] == "fits in shared"
        assert result["step"] == 3
        assert result["level"] == 1


# ============================================================
# ConditionalDecision
# ============================================================

class TestConditionalDecision:
    def test_basic(self):
        cd = ConditionalDecision(
            predicate='dim("S") <= 512',
            true_decisions=[
                Decision(kind="tile", params={"loop": "S", "factors": [256]})
            ],
            false_decisions=[
                Decision(kind="tile", params={"loop": "S", "factors": [512]})
            ],
        )
        assert cd.predicate == 'dim("S") <= 512'
        assert len(cd.true_decisions) == 1
        assert len(cd.false_decisions) == 1

    def test_to_dict(self):
        cd = ConditionalDecision(
            predicate='S <= 512',
            true_decisions=[
                Decision(kind="tile", params={"loop": "S", "factors": [256]})
            ],
            false_decisions=[],
            rationale=Rationale(text="small vs large sequence"),
            step=1,
        )
        d = cd.to_dict()
        assert d["kind"] == "__conditional__"
        assert d["predicate"] == "S <= 512"
        assert len(d["true_decisions"]) == 1
        assert d["rationale"]["text"] == "small vs large sequence"


# ============================================================
# _parse_decision (v0.2.0 compat)
# ============================================================

class TestParseDecision:
    def test_normal(self):
        d = _parse_decision({
            "kind": "tile",
            "params": {"loop": "M", "factors": [64]},
            "step": 1,
            "level": 1,
        })
        assert d.kind == "tile"
        assert d.level == 1

    def test_launch_config_migration(self):
        """v0.2.0 launch_config should be migrated to compute_resource L2."""
        d = _parse_decision({
            "kind": "launch_config",
            "params": {"num_warps": 4, "num_stages": 3},
            "step": 5,
        })
        assert d.kind == "compute_resource"
        assert d.level == 2
        assert d.params["warps"] == 4
        assert d.params["stages"] == 3

    def test_launch_config_with_warps_key(self):
        """v0.2.0 launch_config with 'warps' key (already migrated name)."""
        d = _parse_decision({
            "kind": "launch_config",
            "params": {"warps": 8, "stages": 2},
        })
        assert d.kind == "compute_resource"
        assert d.params["warps"] == 8
        assert d.params["stages"] == 2

    def test_rationale_string(self):
        """v0.2.0 might have rationale as plain string."""
        d = _parse_decision({
            "kind": "tile",
            "params": {"loop": "M", "factors": [64]},
            "rationale": "good tile size",
        })
        assert d.rationale is not None
        assert d.rationale.text == "good tile size"

    def test_rationale_dict(self):
        d = _parse_decision({
            "kind": "tile",
            "params": {"loop": "M", "factors": [64]},
            "rationale": {"text": "good tile size", "lang": "en"},
        })
        assert d.rationale.text == "good tile size"
        assert d.rationale.lang == "en"


# ============================================================
# StrategyIR convenience methods
# ============================================================

class TestStrategyIRConvenience:
    def test_tile(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.tile("M", [64], rationale="cache aligned")
        assert d.kind == "tile"
        assert d.step == 1
        assert d.params == {"loop": "M", "factors": [64]}
        assert d.rationale.text == "cache aligned"
        assert ir.decision_count == 1

    def test_reorder(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.reorder(["M", "N", "K"])
        assert d.kind == "reorder"
        assert d.params == {"order": ["M", "N", "K"]}

    def test_fuse(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.fuse(["matmul", "gelu"], "epilogue", "eliminate round-trip")
        assert d.kind == "fuse"
        assert d.params == {"ops": ["matmul", "gelu"], "type": "epilogue"}
        assert d.rationale.text == "eliminate round-trip"

    def test_parallel(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.parallel(
            ["M", "N"],
            {"M": "blockIdx.x", "N": "blockIdx.y"},
        )
        assert d.kind == "parallel"
        assert d.params["loops"] == ["M", "N"]

    def test_place(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.place("A_tile", "shared")
        assert d.kind == "place"
        assert d.params == {"tensor": "A_tile", "memory": "shared"}

    def test_compute_resource(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.compute_resource(warps=4, stages=3, rationale="good for Ampere")
        assert d.kind == "compute_resource"
        assert d.level == 2
        assert d.params == {"warps": 4, "stages": 3}

    def test_when(self):
        ir = StrategyIR(kernel_id="test")
        cd = ir.when(
            'S <= 512',
            [Decision(kind="tile", params={"loop": "S", "factors": [256]})],
            [Decision(kind="tile", params={"loop": "S", "factors": [512]})],
            rationale="shape-dependent tiling",
        )
        assert isinstance(cd, ConditionalDecision)
        assert cd.step == 1
        assert cd.predicate == "S <= 512"

    def test_pop_decisions(self):
        ir = StrategyIR(kernel_id="test")
        ir.tile("M", [64])
        ir.tile("N", [128])
        ir.tile("K", [32])
        assert ir.decision_count == 3
        removed = ir.pop_decisions(2)
        assert len(removed) == 2
        assert ir.decision_count == 1

    def test_step_auto_numbering(self):
        ir = StrategyIR(kernel_id="test")
        d1 = ir.tile("M", [64])
        d2 = ir.tile("N", [128])
        d3 = ir.compute_resource(warps=4)
        assert d1.step == 1
        assert d2.step == 2
        assert d3.step == 3

    def test_summary(self):
        ir = StrategyIR(kernel_id="matmul", target_hw="nvidia_ampere")
        ir.tile("M", [64], rationale="aligned")
        s = ir.summary()
        assert "matmul" in s
        assert "nvidia_ampere" in s
        assert "tile" in s


# ============================================================
# StrategyIR JSON round-trip
# ============================================================

class TestStrategyIRSerialization:
    def test_basic_round_trip(self):
        ir = StrategyIR(kernel_id="matmul", target_hw="nvidia_ampere")
        ir.tile("M", [64], rationale="cache line")
        ir.tile("N", [128])
        ir.compute_resource(warps=4, stages=3)

        j1 = ir.to_json()
        ir2 = StrategyIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2

    def test_round_trip_with_conditional(self):
        ir = StrategyIR(kernel_id="softmax", target_hw="nvidia_ampere")
        ir.when(
            'S <= 512',
            [Decision(kind="tile", params={"loop": "S", "factors": [256]})],
            [Decision(kind="tile", params={"loop": "S", "factors": [512]})],
            rationale="shape-dependent",
        )
        ir.compute_resource(warps=4, stages=2)

        j1 = ir.to_json()
        ir2 = StrategyIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2

        # Check conditional deserialized correctly
        assert isinstance(ir2.decisions[0], ConditionalDecision)
        assert ir2.decisions[0].predicate == "S <= 512"

    def test_v020_compat_load(self):
        """Loading a v0.2.0-style JSON with launch_config."""
        v020_json = {
            "version": "0.2.0",
            "kernel_id": "old_kernel",
            "target_hw": "nvidia_v100",
            "decisions": [
                {"kind": "tile", "params": {"loop": "M", "factors": [64]}, "step": 1},
                {"kind": "launch_config", "params": {"num_warps": 4, "num_stages": 2}, "step": 2},
            ],
        }
        ir = StrategyIR.from_dict(v020_json)
        assert ir.version == "0.2.0"
        assert ir.decision_count == 2
        # launch_config should be migrated
        d = ir.decisions[1]
        assert d.kind == "compute_resource"
        assert d.level == 2
        assert d.params["warps"] == 4
        assert d.params["stages"] == 2

    def test_l1_vs_l2_tagging(self):
        ir = StrategyIR(kernel_id="test")
        d1 = ir.tile("M", [64])
        d2 = ir.fuse(["a", "b"])
        d3 = ir.compute_resource(warps=8)

        assert d1.level == 1
        assert d2.level == 1
        assert d3.level == 2

        # Verify preserved in JSON
        data = ir.to_dict()
        assert data["decisions"][0]["level"] == 1
        assert data["decisions"][1]["level"] == 1
        assert data["decisions"][2]["level"] == 2

    def test_constraints_omitted_when_default(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.to_dict()
        assert "constraints" not in d

    def test_constraints_included_when_set(self):
        ir = StrategyIR(kernel_id="test")
        ir.constraints = HardwareConstraints(
            shared_memory_limit=49152,
            max_threads_per_block=1024,
        )
        d = ir.to_dict()
        assert "constraints" in d
        assert d["constraints"]["shared_memory_limit"] == 49152

    def test_shape_regimes_omitted_when_empty(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.to_dict()
        assert "shape_regimes" not in d

    def test_shape_regimes_round_trip(self):
        ir = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        ir.shape_regimes = [
            ShapeRegime(
                name="small",
                predicate="S <= 512",
                decisions=[Decision(kind="tile", params={"loop": "S", "factors": [256]})],
            ),
        ]
        d = ir.to_dict()
        assert "shape_regimes" in d
        assert d["shape_regimes"][0]["name"] == "small"
