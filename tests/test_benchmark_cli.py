# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from benchmarks.bench_l1 import run_op
from benchmarks.bench_l2 import FUSED_SHAPES
from benchmarks.cli import resolve_config


class TestBenchmarkCliConfig:
    def test_resolve_config_preserves_shape_filter(self):
        args = argparse.Namespace(
            all=False,
            bl=None,
            ot=None,
            st=None,
            layer="L1",
            op="relu",
            shapes="square-1k",
            baselines=None,
            model=None,
            warmup=1,
            reps=1,
            seq_len=None,
        )
        config = resolve_config(args)
        assert config["layers"] == ["L1"]
        assert config["ops"] == ["relu"]
        assert config["shapes"] == ["square-1k"]


class TestBenchmarkShapeFiltering:
    def test_l1_run_op_shape_filter_uses_registry_tags(self):
        results = run_op("relu", warmup=1, reps=1, tier=4, shape_tags=["square-1k"])
        assert results
        assert {r.shape_tag for r in results} == {"square-1k"}

    def test_l2_shape_filter_selects_only_requested_shape(self):
        filtered = [s for s in FUSED_SHAPES if s.tag == "square-1k"]
        assert len(filtered) == 1
        assert filtered[0].tag == "square-1k"
