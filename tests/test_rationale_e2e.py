# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for @rationale preservation through the full pipeline.

Gate criterion G6-LI.3: @rationale annotations are preserved through:
    .ak parse → AST → ast_to_strategy() → StrategyIR → .akir save → .akir load

Tests verify that Decision.rationale is populated and survives round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from arke.compiler.pipeline import ArkePipeline
from arke.ir.akir import load_akir, save_akir
from arke.ir.converters import ast_to_strategy
from arke.ir.strategy import ConditionalDecision, Decision, Rationale, StrategyIR
from arke.lang.grammar import parse_file

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))

# At least 5 files known to have @rationale annotations
RATIONALE_FILES = [
    "00_relu.ak",
    "02_softmax.ak",
    "04_layernorm.ak",
    "15_flash_attention.ak",
    "45_paged_attention.ak",
    "39_rope.ak",
    "05_matmul_gelu.ak",
    "08_batch_matmul.ak",
]


def _collect_rationales(ir: StrategyIR) -> list[str]:
    """Collect all rationale texts from a StrategyIR."""
    texts: list[str] = []
    for d in ir.decisions:
        if isinstance(d, ConditionalDecision):
            if d.rationale:
                texts.append(d.rationale.text)
            for sub in d.true_decisions + d.false_decisions:
                if sub.rationale:
                    texts.append(sub.rationale.text)
        elif isinstance(d, Decision):
            if d.rationale:
                texts.append(d.rationale.text)
    return texts


class TestRationalePreservation:
    """Verify @rationale annotations are preserved through the full pipeline."""

    @pytest.fixture
    def pipeline(self):
        return ArkePipeline()

    @pytest.mark.parametrize("ak_file", RATIONALE_FILES)
    def test_rationale_in_strategy_ir(self, ak_file: str):
        """Parse .ak → ast_to_strategy() → verify rationale populated."""
        path = OPERATORS_DIR / ak_file
        if not path.exists():
            pytest.skip(f"{ak_file} not found")

        program = parse_file(str(path))
        assert program.strategies, f"No strategy in {ak_file}"

        strategy_def = program.strategies[0]
        ir = ast_to_strategy(strategy_def)

        rationales = _collect_rationales(ir)
        assert len(rationales) > 0, (
            f"{ak_file}: no rationale texts found in StrategyIR"
        )
        for text in rationales:
            assert len(text) > 0, "Rationale text should be non-empty"

    @pytest.mark.parametrize("ak_file", RATIONALE_FILES)
    def test_rationale_survives_akir_roundtrip(self, ak_file: str, pipeline, tmp_path):
        """Parse → StrategyIR → save .akir → load → rationale text intact."""
        path = OPERATORS_DIR / ak_file
        if not path.exists():
            pytest.skip(f"{ak_file} not found")

        result = pipeline.compile_file(str(path))
        assert result.success, f"Compilation failed for {ak_file}: {result.errors}"
        assert result.strategy_ir is not None, f"No StrategyIR for {ak_file}"

        # Collect rationales before round-trip
        before = _collect_rationales(result.strategy_ir)
        assert len(before) > 0, f"{ak_file}: no rationales before save"

        # Save → load .akir
        akir_path = str(tmp_path / f"{ak_file}.akir")
        save_akir(result.semantic_ir, result.strategy_ir, akir_path)
        _, loaded_strategy = load_akir(akir_path)

        assert loaded_strategy is not None, f"No StrategyIR after loading {akir_path}"

        # Collect rationales after round-trip
        after = _collect_rationales(loaded_strategy)
        assert len(after) == len(before), (
            f"{ak_file}: rationale count mismatch: {len(before)} → {len(after)}"
        )
        for b, a in zip(before, after):
            assert a == b, (
                f"{ak_file}: rationale text changed: {b!r} → {a!r}"
            )


class TestRationaleAllFiles:
    """Verify all .ak files with @rationale have it preserved."""

    @pytest.fixture
    def pipeline(self):
        return ArkePipeline()

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_rationale_roundtrip_all(self, ak_file: Path, pipeline, tmp_path):
        """Every .ak file: if it has @rationale, verify round-trip preservation."""
        # Read file to check if it has @rationale
        text = ak_file.read_text()
        if "@rationale" not in text:
            pytest.skip(f"{ak_file.name} has no @rationale")

        result = pipeline.compile_file(str(ak_file))
        assert result.success, f"Compilation failed: {result.errors}"

        if result.strategy_ir is None:
            pytest.skip(f"{ak_file.name} has no strategy block")

        rationales_before = _collect_rationales(result.strategy_ir)
        assert len(rationales_before) > 0, (
            f"{ak_file.name}: @rationale in source but none in StrategyIR"
        )

        # Round-trip through .akir
        akir_path = str(tmp_path / f"{ak_file.stem}.akir")
        save_akir(result.semantic_ir, result.strategy_ir, akir_path)
        _, loaded = load_akir(akir_path)

        assert loaded is not None
        rationales_after = _collect_rationales(loaded)
        assert rationales_before == rationales_after, (
            f"{ak_file.name}: rationale texts differ after .akir round-trip"
        )


class TestRationaleContent:
    """Verify rationale text content is meaningful (not empty placeholders)."""

    @pytest.mark.parametrize("ak_file", RATIONALE_FILES[:5])
    def test_rationale_is_descriptive(self, ak_file: str):
        """Check rationale text is meaningful (>10 chars, not placeholder)."""
        path = OPERATORS_DIR / ak_file
        if not path.exists():
            pytest.skip(f"{ak_file} not found")

        program = parse_file(str(path))
        assert program.strategies

        ir = ast_to_strategy(program.strategies[0])
        rationales = _collect_rationales(ir)

        for text in rationales:
            # Should be a meaningful sentence, not just "TODO" or empty
            assert len(text) >= 10, (
                f"Rationale too short to be meaningful: {text!r}"
            )

    def test_rationale_object_fields(self):
        """Verify Rationale dataclass has expected fields."""
        r = Rationale(text="test rationale", lang="en")
        assert r.text == "test rationale"
        assert r.lang == "en"

    def test_rationale_default_lang(self):
        """Verify Rationale defaults to English."""
        r = Rationale(text="some text")
        assert r.lang == "en"
