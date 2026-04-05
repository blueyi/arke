# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark protocol: BL/OT/ST/Layer CLI, shapes coverage,
output structure, and arke bench entry point.

These tests guard the implementation against benchmark-protocol.md.

Convention: all CLI invocations use `arke bench` as the canonical entry
point. `python -m benchmarks` is only tested for backward compatibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ============================================================
# Fixtures / helpers
# ============================================================


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace matching cli.resolve_config() expectations."""
    defaults = dict(
        bl=None,
        ot=None,
        st=None,
        layer=None,
        op=None,
        shapes=None,
        baselines=None,
        model=None,
        all=False,
        warmup=200,
        reps=500,
        seq_len=None,
        verbose=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ============================================================
# 1. BL expansion table  (benchmark-protocol.md §CLI)
# ============================================================


class TestBLExpansion:
    """Verify BL1-6 expand to the correct OT/ST/Layer defaults."""

    from benchmarks.cli import BL_DEFAULTS, resolve_config  # noqa: E402 (class-level)

    @pytest.mark.parametrize(
        "bl,expect_layers,expect_ot_range,expect_max_st",
        [
            (1, ["L1"], (0, 2), 1),
            (2, ["L1"], (0, 2), 2),
            (3, ["L1"], (0, 2), 3),
            (4, ["L1", "L2"], (0, 4), 2),
            (5, ["L1", "L2"], (0, 4), 4),
            (6, ["L1", "L2", "L3"], (0, 4), 4),
        ],
    )
    def test_bl_expands_correctly(self, bl, expect_layers, expect_ot_range, expect_max_st):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=bl))
        assert cfg["layers"] == expect_layers, f"BL{bl} layers mismatch"
        assert cfg["ot_tiers"] == list(range(expect_ot_range[0], expect_ot_range[1] + 1))
        assert cfg["max_st"] == expect_max_st, f"BL{bl} max_st mismatch"

    def test_default_is_bl2(self):
        """No args → BL2."""
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args())
        assert cfg["bl"] == 2
        assert cfg["layers"] == ["L1"]
        assert cfg["max_st"] == 2

    def test_all_flag_is_bl6(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(**{"all": True}))
        assert cfg["bl"] == 6
        assert "L3" in cfg["layers"]


# ============================================================
# 2. Override parameters  (--ot, --st, --layer, --op, --shapes)
# ============================================================


class TestCLIOverrides:

    def test_ot_override(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=2, ot="0,2"))
        assert cfg["ot_tiers"] == [0, 2]

    def test_st_override(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=3, st="2"))
        assert cfg["st_tiers"] == [2]
        assert cfg["max_st"] == 2

    def test_layer_l3_implies_bl6(self):
        """--layer L3 forces BL6 scope (protocol validation rule)."""
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(layer="L3"))
        assert cfg["bl"] == 6
        assert cfg["layers"] == ["L3"]

    def test_layer_l2_expands_ot(self):
        """--layer L2 must have OT max >= 3."""
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=2, layer="L2"))
        assert max(cfg["ot_tiers"]) >= 3

    def test_op_override_takes_priority(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=2, op="matmul,softmax"))
        assert cfg["ops"] == ["matmul", "softmax"]

    def test_shapes_override(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(shapes="square-1k,square-4k"))
        assert cfg["shapes"] == ["square-1k", "square-4k"]

    def test_baselines_override(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(baselines="cublas,flaggems"))
        assert cfg["baselines"] == ["cublas", "flaggems"]

    def test_model_override(self):
        from benchmarks.cli import resolve_config

        cfg = resolve_config(_make_args(bl=6, model="gpt2"))
        assert cfg["model"] == "gpt2"


# ============================================================
# 3. OT → Operators mapping
# ============================================================


class TestOTOpsMapping:

    def test_ot0_elementwise(self):
        from benchmarks.cli import OT_OPS

        assert set(OT_OPS[0]) == {"relu", "gelu", "silu", "add", "mul"}

    def test_ot1_reduction(self):
        from benchmarks.cli import OT_OPS

        assert "softmax" in OT_OPS[1]
        assert "layernorm" in OT_OPS[1]
        assert "rmsnorm" in OT_OPS[1]
        assert "reduce_sum" in OT_OPS[1]

    def test_ot2_dense(self):
        from benchmarks.cli import OT_OPS

        assert "matmul" in OT_OPS[2]
        assert "batch_matmul" in OT_OPS[2]

    def test_ot3_gated(self):
        from benchmarks.cli import OT_OPS

        assert set(OT_OPS[3]) == {"swiglu", "geglu"}

    def test_ot4_attention(self):
        from benchmarks.cli import OT_OPS

        expected = {"flash_attention", "grouped_query_attention", "multi_latent_attention"}
        assert set(OT_OPS[4]) == expected

    def test_bl2_ops_are_ot0_to_ot2(self):
        """BL2 should only include OT0-OT2 operators."""
        from benchmarks.cli import OT_OPS, resolve_config

        cfg = resolve_config(_make_args(bl=2))
        expected_ops = OT_OPS[0] + OT_OPS[1] + OT_OPS[2]
        assert set(cfg["ops"]) == set(expected_ops)

    def test_bl5_ops_include_attention(self):
        """BL5 includes all OT tiers including OT4."""
        from benchmarks.cli import OT_OPS, resolve_config

        cfg = resolve_config(_make_args(bl=5))
        assert all(op in cfg["ops"] for op in OT_OPS[4])


# ============================================================
# 4. Shapes module: all 20 ops covered
# ============================================================


class TestShapesCoverage:

    ALL_OPS = [
        "relu", "gelu", "silu", "add", "mul",
        "softmax", "layernorm", "rmsnorm", "rmsnorm_residual",
        "reduce_sum", "reduce_max",
        "matmul", "batch_matmul", "grouped_matmul", "transpose",
        "swiglu", "geglu",
        "flash_attention", "grouped_query_attention", "multi_latent_attention",
    ]

    def test_all_ops_have_shapes(self):
        """Every OP_CATALOG operator must have at least 1 shape."""
        from benchmarks.shapes import get_shapes

        for op in self.ALL_OPS:
            shapes = get_shapes(op)
            assert len(shapes) > 0, f"No shapes for op '{op}'"

    def test_st_tier_filter(self):
        """tier=1 returns only ST1 shapes; tier=2 returns ST1+ST2."""
        from benchmarks.shapes import get_shapes

        t1 = get_shapes("matmul", tier=1)
        t2 = get_shapes("matmul", tier=2)
        t_all = get_shapes("matmul")

        assert len(t1) < len(t2) <= len(t_all)
        assert all(s.tier == 1 for s in t1)
        assert all(s.tier <= 2 for s in t2)

    def test_new_dataclasses_exist(self):
        """New dataclasses are importable."""
        from benchmarks.shapes import (
            AttentionShape,
            BatchMatmulShape,
            GatedShape,
            GroupedMatmulShape,
        )
        assert AttentionShape is not None
        assert BatchMatmulShape is not None
        assert GatedShape is not None
        assert GroupedMatmulShape is not None

    def test_attention_shapes_are_st4(self):
        """Attention shapes should be ST4 (production-only)."""
        from benchmarks.shapes import get_shapes

        fa = get_shapes("flash_attention")
        assert all(s.tier == 4 for s in fa)

    def test_op_tier_map_covers_all_ops(self):
        """OP_TIER dict covers all 20 ops."""
        from benchmarks.shapes import OP_TIER

        for op in self.ALL_OPS:
            assert op in OP_TIER, f"OP_TIER missing '{op}'"

    def test_invalid_op_raises(self):
        from benchmarks.shapes import get_shapes

        with pytest.raises(ValueError, match="No shape set"):
            get_shapes("nonexistent_op_xyz")

    @pytest.mark.parametrize("op", ["matmul", "softmax", "layernorm",
                                     "relu", "batch_matmul"])
    def test_core_ops_have_tier1_shapes(self, op):
        """Core ops must have at least 2 ST1 shapes."""
        from benchmarks.shapes import get_shapes

        t1 = get_shapes(op, tier=1)
        assert len(t1) >= 2, f"'{op}' has fewer than 2 ST1 shapes"


# ============================================================
# 5. Output structure: config.json / hardware.json / summary.json
# ============================================================


class TestOutputStructure:

    def test_write_config(self, tmp_path):
        from benchmarks.cli import _write_config

        config = {
            "run_id": "2026-04-05_120000",
            "bl": 2,
            "ot_tiers": [0, 1, 2],
            "st_tiers": [1, 2],
            "layers": ["L1"],
            "ops": ["matmul"],
            "shapes": None,
            "baselines": None,
            "warmup": 200,
            "reps": 500,
            "timestamp": "2026-04-05T12:00:00",
        }
        _write_config(tmp_path, config)

        config_file = tmp_path / "config.json"
        assert config_file.exists()
        loaded = json.loads(config_file.read_text())
        assert loaded["bl"] == 2
        assert loaded["layers"] == ["L1"]
        assert loaded["run_id"] == "2026-04-05_120000"

    def test_write_summary(self, tmp_path):
        """_write_summary reads PERF_ALL.csv and writes summary.json."""
        from benchmarks.cli import _write_summary

        # Write a minimal PERF_ALL.csv
        perf_all = tmp_path / "PERF_ALL.csv"
        rows = [
            {"operator": "matmul", "ratio_vs_baseline": "1.1"},
            {"operator": "matmul", "ratio_vs_baseline": "1.2"},
            {"operator": "softmax", "ratio_vs_baseline": "0.9"},
            {"operator": "softmax", "ratio_vs_baseline": "1.0"},
        ]
        with open(perf_all, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator", "ratio_vs_baseline"])
            writer.writeheader()
            writer.writerows(rows)

        _write_summary(tmp_path)

        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()
        s = json.loads(summary_file.read_text())
        assert "overall_geomean" in s
        assert "op_scores" in s
        assert "matmul" in s["op_scores"]
        assert "softmax" in s["op_scores"]
        assert s["total_shapes"] == 4
        # Verify geomean > 0
        assert s["overall_geomean"] > 0

    def test_merge_perf_all(self, tmp_path):
        """_merge_perf_all collects all perf_*.csv into PERF_ALL.csv."""
        from benchmarks.cli import _merge_perf_all

        # Create two sub-dirs with perf CSVs
        d1 = tmp_path / "L1" / "OT0"
        d1.mkdir(parents=True)
        d2 = tmp_path / "L1" / "OT2"
        d2.mkdir(parents=True)

        cols = ["operator", "latency_us", "ratio_vs_baseline"]
        for d, op in [(d1, "relu"), (d2, "matmul")]:
            with open(d / f"perf_{op}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerow({"operator": op, "latency_us": "10.0", "ratio_vs_baseline": "1.0"})

        _merge_perf_all(tmp_path)

        perf_all = tmp_path / "PERF_ALL.csv"
        assert perf_all.exists()
        with open(perf_all) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        ops = {r["operator"] for r in rows}
        assert ops == {"relu", "matmul"}

    def test_run_id_format(self):
        """Run ID should be YYYY-MM-DD_HHMMSS."""
        import re
        from benchmarks.cli import _generate_run_id

        run_id = _generate_run_id()
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{6}$", run_id), (
            f"run_id format wrong: {run_id}"
        )

    def test_summary_geomean_correctness(self, tmp_path):
        """Geomean of [1.0, 1.0, 1.0] == 1.0."""
        import math
        from benchmarks.cli import _write_summary

        perf_all = tmp_path / "PERF_ALL.csv"
        rows = [{"operator": "matmul", "ratio_vs_baseline": "1.0"} for _ in range(4)]
        with open(perf_all, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator", "ratio_vs_baseline"])
            writer.writeheader()
            writer.writerows(rows)

        _write_summary(tmp_path)
        s = json.loads((tmp_path / "summary.json").read_text())
        assert abs(s["overall_geomean"] - 1.0) < 1e-6

    def test_summary_skips_invalid_ratio(self, tmp_path):
        """N/A and empty ratio values should be ignored gracefully."""
        from benchmarks.cli import _write_summary

        perf_all = tmp_path / "PERF_ALL.csv"
        rows = [
            {"operator": "matmul", "ratio_vs_baseline": "N/A"},
            {"operator": "matmul", "ratio_vs_baseline": ""},
            {"operator": "matmul", "ratio_vs_baseline": "1.2"},
        ]
        with open(perf_all, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["operator", "ratio_vs_baseline"])
            writer.writeheader()
            writer.writerows(rows)

        _write_summary(tmp_path)  # Must not raise
        s = json.loads((tmp_path / "summary.json").read_text())
        assert s["total_shapes"] == 1


# ============================================================
# 6. arke bench CLI entry point
# All CLI invocations use `arke bench` as the canonical entry point.
# ============================================================


def _arke_bench(*args: str) -> subprocess.CompletedProcess:
    """Run `arke bench <args>` as a subprocess."""
    return subprocess.run(
        ["arke", "bench", *args],
        capture_output=True, text=True
    )


class TestArkeBenchCLI:

    def test_arke_bench_help(self):
        """arke bench --help must exit 0 and mention key parameters."""
        result = _arke_bench("--help")
        assert result.returncode == 0
        out = result.stdout
        assert "--bl" in out
        assert "--ot" in out
        assert "--st" in out
        assert "--layer" in out
        assert "--op" in out
        assert "--shapes" in out
        assert "--baselines" in out

    def test_arke_bench_subcommands_present(self):
        """report, diff, history subcommands must appear in help."""
        result = _arke_bench("--help")
        assert result.returncode == 0
        out = result.stdout
        assert "report" in out
        assert "diff" in out
        assert "history" in out

    def test_arke_bench_report_help(self):
        result = _arke_bench("report", "--help")
        assert result.returncode == 0
        assert "run_id" in result.stdout

    def test_arke_bench_diff_help(self):
        result = _arke_bench("diff", "--help")
        assert result.returncode == 0
        assert "run_id_1" in result.stdout
        assert "run_id_2" in result.stdout

    def test_arke_bench_history_help(self):
        result = _arke_bench("history", "--help")
        assert result.returncode == 0
        assert "--op" in result.stdout

    def test_python_m_benchmarks_compat(self):
        """`python -m benchmarks` is a compatibility alias for arke bench."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--bl" in result.stdout

    def test_arke_bench_diff_stub(self):
        """arke bench diff returns 0 and mentions 'Not yet implemented'."""
        result = _arke_bench("diff", "run_a", "run_b")
        assert result.returncode == 0
        assert "Not yet implemented" in result.stdout

    def test_arke_bench_history_stub(self):
        result = _arke_bench("history", "--op", "matmul")
        assert result.returncode == 0
        assert "Not yet implemented" in result.stdout


# ============================================================
# 7. Benchmark parser (_build_parser)
# ============================================================


class TestBenchmarkParser:

    def _parse(self, args: list[str]) -> argparse.Namespace:
        from benchmarks.cli import _build_parser

        parser = _build_parser()
        return parser.parse_args(args)

    def test_default_args(self):
        args = self._parse([])
        assert args.bl is None
        assert args.ot is None
        assert args.warmup == 200
        assert args.reps == 500

    def test_bl_arg(self):
        args = self._parse(["--bl", "3"])
        assert args.bl == 3

    def test_ot_arg(self):
        args = self._parse(["--ot", "0,2"])
        assert args.ot == "0,2"

    def test_st_arg(self):
        args = self._parse(["--st", "1,3"])
        assert args.st == "1,3"

    def test_layer_arg(self):
        args = self._parse(["--layer", "L1"])
        assert args.layer == "L1"

    def test_op_arg(self):
        args = self._parse(["--op", "matmul,softmax"])
        assert args.op == "matmul,softmax"

    def test_shapes_arg(self):
        args = self._parse(["--shapes", "square-1k,square-4k"])
        assert args.shapes == "square-1k,square-4k"

    def test_baselines_arg(self):
        args = self._parse(["--baselines", "cublas,flaggems"])
        assert args.baselines == "cublas,flaggems"

    def test_model_arg(self):
        args = self._parse(["--model", "gpt2"])
        assert args.model == "gpt2"

    def test_warmup_reps(self):
        args = self._parse(["--warmup", "50", "--reps", "100"])
        assert args.warmup == 50
        assert args.reps == 100

    def test_report_subcommand(self):
        args = self._parse(["report"])
        assert args.subcmd == "report"
        assert args.run_id is None

    def test_report_with_run_id(self):
        args = self._parse(["report", "2026-04-05_120000"])
        assert args.run_id == "2026-04-05_120000"

    def test_diff_subcommand(self):
        args = self._parse(["diff", "run_a", "run_b"])
        assert args.subcmd == "diff"
        assert args.run_id_1 == "run_a"
        assert args.run_id_2 == "run_b"

    def test_history_subcommand(self):
        args = self._parse(["history", "--op", "matmul"])
        assert args.subcmd == "history"
        assert args.op == "matmul"
