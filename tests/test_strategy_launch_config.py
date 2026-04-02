# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for launch_config and autotune strategy decisions."""

from __future__ import annotations

import json

from arke.ir.strategy import Decision, StrategyIR


class TestLaunchConfig:
    """Test launch_config strategy decision."""

    def test_launch_config_decision_roundtrip(self) -> None:
        strategy = StrategyIR()
        strategy.decisions.append(
            Decision(
                kind="launch_config",
                params={"num_warps": 4, "num_stages": 3},
                rationale="Manual configuration for softmax kernel",
            )
        )
        j = strategy.to_json()
        data = json.loads(j)
        assert data["decisions"][0]["kind"] == "launch_config"
        assert data["decisions"][0]["params"]["num_warps"] == 4

    def test_autotune_decision_roundtrip(self) -> None:
        strategy = StrategyIR()
        strategy.decisions.append(
            Decision(
                kind="autotune",
                params={
                    "configs": [
                        {"num_warps": 2, "block_sizes": {"BLOCK_SIZE": 1024}},
                        {"num_warps": 4, "block_sizes": {"BLOCK_SIZE": 4096}},
                        {"num_warps": 8, "block_sizes": {"BLOCK_SIZE": 8192}},
                    ],
                    "key": ["n_elements"],
                },
                rationale="Autotune for elementwise kernel",
            )
        )
        j = strategy.to_json()
        data = json.loads(j)
        assert len(data["decisions"][0]["params"]["configs"]) == 3

    def test_template_engine_extracts_launch_config(self) -> None:
        from arke.backend.triton_template_engine import TritonTemplateEngine
        from arke.ir.builder import KernelBuilder

        engine = TritonTemplateEngine()
        b = KernelBuilder("test_softmax")
        b.param("X", [12, 128], "f16")
        node = b.op("softmax", X="X")
        b.returns(node, [12, 128], "f16")
        ir = b.build()

        strategy = StrategyIR()
        strategy.decisions.append(
            Decision(
                kind="launch_config",
                params={"num_warps": 8, "num_stages": 2},
                rationale="Optimized for wide softmax",
            )
        )

        source = engine.translate(ir, strategy)
        assert "def test_softmax" in source

    def test_template_engine_extracts_autotune(self) -> None:
        from arke.backend.triton_template_engine import TritonTemplateEngine
        from arke.ir.builder import KernelBuilder

        engine = TritonTemplateEngine()
        b = KernelBuilder("test_gelu")
        b.param("X", [1024], "f16")
        node = b.op("gelu", X="X")
        b.returns(node, [1024], "f16")
        ir = b.build()

        strategy = StrategyIR()
        strategy.decisions.append(
            Decision(
                kind="autotune",
                params={
                    "configs": [
                        {"num_warps": 2, "block_sizes": {"BLOCK_SIZE": 1024}},
                        {"num_warps": 4, "block_sizes": {"BLOCK_SIZE": 4096}},
                    ],
                    "key": ["n_elements"],
                },
            )
        )

        source = engine.translate(ir, strategy)
        assert "def test_gelu" in source

    def test_launch_config_with_list_warps(self) -> None:
        """Test launch_config with list of num_warps candidates."""
        d = Decision(
            kind="launch_config",
            params={"num_warps": [2, 4, 8], "num_stages": [2, 3]},
        )
        assert d.kind == "launch_config"
        assert d.params["num_warps"] == [2, 4, 8]

    def test_empty_strategy_no_launch_config(self) -> None:
        """Template engine handles empty strategy gracefully."""
        from arke.backend.triton_template_engine import TritonTemplateEngine
        from arke.ir.builder import KernelBuilder

        engine = TritonTemplateEngine()
        b = KernelBuilder("test_relu")
        b.param("X", [1024], "f16")
        node = b.op("relu", X="X")
        b.returns(node, [1024], "f16")
        ir = b.build()

        strategy = StrategyIR()  # no decisions
        source = engine.translate(ir, strategy)
        assert "def test_relu" in source
