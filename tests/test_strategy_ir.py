# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for StrategyIR (arke/ir/strategy.py)."""

import json

from arke.ir.schedule import ScheduleIR
from arke.ir.strategy import (
    ConditionalDecision,
    Decision,
    HardwareConstraints,
    Rationale,
    ShapeRegime,
    StrategyIR,
    _decision_to_dict,
    _parse_decision,
)


class TestRationale:
    def test_basic(self):
        r = Rationale(text="good tile size for f16")
        assert r.text == "good tile size for f16"
        assert r.lang == "en"

    def test_lang(self):
        r = Rationale(text="example", lang="zh")
        assert r.lang == "zh"


class TestDecision:
    def test_tile(self):
        d = Decision(kind="tile", params={"loop": "M", "factors": [64]})
        assert d.kind == "tile"
        assert d.level == 1
        assert d.step == 0

    def test_level_l2(self):
        d = Decision(
            kind="compute",
            params={"warps": 4, "num_stages": 2},
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
            predicate="S <= 512",
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

    def test_rationale_string(self):
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

    def test_schedule_ir_accepts_type_and_fusion_type_keys(self):
        schedule = ScheduleIR(kernel_id="test", target_hw="nvidia_ampere")
        schedule.apply_decision(Decision(kind="fuse", params={"ops": ["a", "b"], "type": "epilogue"}, step=1))
        schedule.apply_decision(
            Decision(
                kind="fuse",
                params={"ops": ["matmul", "cross_entropy"], "fusion_type": "producer_consumer"},
                step=2,
            )
        )
        assert schedule.fusion_groups[0].fusion_type == "epilogue"
        assert schedule.fusion_groups[1].fusion_type == "producer_consumer"

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

    def test_compute(self):
        ir = StrategyIR(kernel_id="test")
        d = ir.compute(warps=4, num_stages=3, shared_memory=49152, rationale="good for Ampere")
        assert d.kind == "compute"
        assert d.level == 2
        assert d.params == {"warps": 4, "num_stages": 3, "shared_memory": 49152}

    def test_when(self):
        ir = StrategyIR(kernel_id="test")
        cd = ir.when(
            "S <= 512",
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
        d3 = ir.compute(warps=4)
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


class TestStrategyIRSerialization:
    def test_basic_round_trip(self):
        ir = StrategyIR(kernel_id="matmul", target_hw="nvidia_ampere")
        ir.tile("M", [64], rationale="cache line")
        ir.tile("N", [128])
        ir.compute(warps=4, num_stages=3)

        j1 = ir.to_json()
        ir2 = StrategyIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2

    def test_round_trip_preserves_metadata(self):
        ir = StrategyIR(
            kernel_id="flash_attention",
            target_hw="nvidia_ampere",
            metadata={"compile_advice": {"allow_compile": False, "reason": "oom risk"}},
        )
        j1 = ir.to_json()
        ir2 = StrategyIR.from_json(j1)
        assert ir2.metadata["compile_advice"]["allow_compile"] is False

    def test_round_trip_with_conditional(self):
        ir = StrategyIR(kernel_id="softmax", target_hw="nvidia_ampere")
        ir.when(
            "S <= 512",
            [Decision(kind="tile", params={"loop": "S", "factors": [256]})],
            [Decision(kind="tile", params={"loop": "S", "factors": [512]})],
            rationale="shape-dependent",
        )
        ir.compute(warps=4, num_stages=2)

        j1 = ir.to_json()
        ir2 = StrategyIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2
        assert isinstance(ir2.decisions[0], ConditionalDecision)
        assert ir2.decisions[0].predicate == "S <= 512"

    def test_l1_vs_l2_tagging(self):
        ir = StrategyIR(kernel_id="test")
        d1 = ir.tile("M", [64])
        d2 = ir.fuse(["a", "b"])
        d3 = ir.compute(warps=8)

        assert d1.level == 1
        assert d2.level == 1
        assert d3.level == 2

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
                name="short_seq",
                predicate="S <= 512",
                decisions=[Decision(kind="tile", params={"loop": "S", "factors": [256]})],
            )
        ]
        payload = ir.to_dict()
        assert payload["shape_regimes"][0]["name"] == "short_seq"
        assert payload["shape_regimes"][0]["predicate"] == "S <= 512"
        assert payload["shape_regimes"][0]["decisions"][0]["kind"] == "tile"

    def test_json_is_valid(self):
        ir = StrategyIR(kernel_id="test")
        ir.tile("M", [64])
        json.loads(ir.to_json())
