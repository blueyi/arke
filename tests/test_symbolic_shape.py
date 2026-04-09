# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from arke.compiler.pipeline import ArkePipeline
from arke.lang.grammar import parse_file
from benchmarks.shape_registry import get_registry_shapes_for_op


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATORS_DIR = REPO_ROOT / "examples" / "operators"


def _compile_example(name: str):
    pipeline = ArkePipeline()
    path = OPERATORS_DIR / name
    result = pipeline.compile_file(str(path))
    assert result.success, result.errors
    assert result.semantic_ir is not None
    return result.semantic_ir


class TestSymbolicShapeExamples:
    def test_matmul_example_preserves_symbolic_dims_and_constraints(self):
        ir = _compile_example("01_matmul.ak")
        sym_names = {sd.name for sd in ir.symbolic_dims}
        exprs = {sc.expr for sc in ir.shape_constraints}
        assert {"B", "S", "D"}.issubset(sym_names)
        assert "B <= 64" in exprs
        assert "S <= 8192" in exprs
        assert "D is static" in exprs

    def test_flash_attention_example_preserves_symbolic_dims_and_constraints(self):
        ir = _compile_example("15_flash_attention.ak")
        sym_names = {sd.name for sd in ir.symbolic_dims}
        exprs = {sc.expr for sc in ir.shape_constraints}
        assert {"B", "S", "D"}.issubset(sym_names)
        assert "B <= 64" in exprs
        assert "S <= 8192" in exprs
        assert "D is static" in exprs

    def test_paged_attention_example_preserves_symbolic_dims_and_constraints(self):
        ir = _compile_example("45_paged_attention.ak")
        sym_names = {sd.name for sd in ir.symbolic_dims}
        exprs = {sc.expr for sc in ir.shape_constraints}
        assert {"B", "S", "D"}.issubset(sym_names)
        assert "B <= 64" in exprs
        assert "S <= 8192" in exprs
        assert "D is static" in exprs


class TestBL5RegistryCoverage:
    def test_ot2_and_ot4_ops_have_registry_shapes(self):
        ops = [
            "matmul",
            "batch_matmul",
            "grouped_matmul",
            "flash_attention",
            "grouped_query_attention",
            "multi_latent_attention",
            "cross_attention",
            "paged_attention",
        ]
        for op in ops:
            rows = get_registry_shapes_for_op(op)
            assert rows, f"No benchmark shape rows found for {op}"

    def test_ot4_ops_include_st4_production_shapes(self):
        ops = [
            "flash_attention",
            "grouped_query_attention",
            "multi_latent_attention",
            "cross_attention",
            "paged_attention",
        ]
        for op in ops:
            rows = get_registry_shapes_for_op(op)
            tiers = {row.get("tier") for row in rows if row.get("tier") is not None}
            assert 4 in tiers, f"{op} missing ST4 coverage in benchmark registry"


class TestExampleSyntaxStillParses:
    def test_symbolic_examples_parse(self):
        files = [
            OPERATORS_DIR / "01_matmul.ak",
            OPERATORS_DIR / "15_flash_attention.ak",
            OPERATORS_DIR / "41_fused_linear_cross_entropy.ak",
            OPERATORS_DIR / "45_paged_attention.ak",
        ]
        for path in files:
            program = parse_file(str(path))
            assert program.kernels, f"No kernel parsed from {path.name}"
            assert program.kernels[0].where_clause is not None, f"Missing where clause in {path.name}"
