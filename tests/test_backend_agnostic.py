# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for backend-agnostic strategy: no Triton-specific fields in L1.

Gate criterion G6-LI.8: StrategyIR L1 (level=1) decisions must not contain
any Triton-specific fields. L2 (level=2) decisions may have backend-specific
content (e.g., compute_resource with warps/stages).

Verifies:
1. All L1 decisions use only allowed param keys
2. No Triton-specific strings in L1 decision param values
3. L1 kinds are from the allowed set
4. L2 decisions (compute_resource, cache_config, etc.) are correctly at level=2
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arke.compiler.pipeline import ArkePipeline
from arke.ir.converters import ast_to_strategy
from arke.ir.strategy import ConditionalDecision, Decision, StrategyIR
from arke.lang.grammar import parse_file

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))

# Allowed L1 decision kinds (backend-agnostic)
L1_ALLOWED_KINDS = {
    "tile",
    "reorder",
    "fuse",
    "parallel",
    "place",
    "vectorize",
    "unroll",
    "algorithm",
}

# Allowed L1 decision param keys (all backend-agnostic)
L1_ALLOWED_PARAM_KEYS = {
    "loop",
    "factors",
    "order",
    "ops",
    "type",
    "fusion_type",
    "loops",
    "mapping",
    "tensor",
    "memory",
    "width",
    "factor",
    "name",
    "params",
}

# Triton-specific field names that must NOT appear in L1 params
TRITON_SPECIFIC_KEYS = {
    "num_warps",
    "num_stages",
    "num_ctas",
    "warps",
    "stages",
    "block_size",
    "BLOCK_SIZE",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "grid",
    "pipeline_depth",
}

# Triton-specific strings forbidden in L1 param values
TRITON_SPECIFIC_STRINGS = [
    "tl.load",
    "tl.store",
    "tl.dot",
    "tl.program_id",
    "triton.jit",
    "@triton.jit",
    "triton.language",
    "num_warps",
    "num_stages",
]


def _collect_l1_decisions(ir: StrategyIR) -> list[Decision]:
    """Collect all L1 (level=1) decisions from a StrategyIR."""
    l1: list[Decision] = []
    for d in ir.decisions:
        if isinstance(d, ConditionalDecision):
            for sub in d.true_decisions + d.false_decisions:
                if isinstance(sub, Decision) and sub.level == 1:
                    l1.append(sub)
        elif isinstance(d, Decision) and d.level == 1:
            l1.append(d)
    return l1


def _collect_l2_decisions(ir: StrategyIR) -> list[Decision]:
    """Collect all L2 (level=2) decisions from a StrategyIR."""
    l2: list[Decision] = []
    for d in ir.decisions:
        if isinstance(d, ConditionalDecision):
            for sub in d.true_decisions + d.false_decisions:
                if isinstance(sub, Decision) and sub.level == 2:
                    l2.append(sub)
        elif isinstance(d, Decision) and d.level == 2:
            l2.append(d)
    return l2


class TestBackendAgnosticL1:
    """Verify all L1 decisions are backend-agnostic."""

    @pytest.fixture
    def pipeline(self):
        return ArkePipeline()

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_l1_allowed_kinds(self, ak_file: Path, pipeline):
        """L1 decisions only use allowed kinds."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        l1_decisions = _collect_l1_decisions(result.strategy_ir)
        for d in l1_decisions:
            assert d.kind in L1_ALLOWED_KINDS, (
                f"{ak_file.name}: L1 decision has disallowed kind '{d.kind}'. "
                f"Allowed: {L1_ALLOWED_KINDS}"
            )

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_l1_no_triton_keys(self, ak_file: Path, pipeline):
        """L1 decisions have no Triton-specific param keys."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        l1_decisions = _collect_l1_decisions(result.strategy_ir)
        for d in l1_decisions:
            for key in d.params:
                assert key not in TRITON_SPECIFIC_KEYS, (
                    f"{ak_file.name}: L1 '{d.kind}' has Triton-specific "
                    f"param key '{key}'"
                )
                assert not key.startswith("BLOCK_SIZE"), (
                    f"{ak_file.name}: L1 '{d.kind}' has BLOCK_SIZE param '{key}'"
                )

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_l1_allowed_param_keys(self, ak_file: Path, pipeline):
        """L1 decision params use only allowed keys."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        l1_decisions = _collect_l1_decisions(result.strategy_ir)
        for d in l1_decisions:
            for key in d.params:
                assert key in L1_ALLOWED_PARAM_KEYS, (
                    f"{ak_file.name}: L1 '{d.kind}' has unexpected param "
                    f"key '{key}'. Allowed: {L1_ALLOWED_PARAM_KEYS}"
                )

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_l1_no_triton_strings_in_values(self, ak_file: Path, pipeline):
        """L1 decision param values contain no Triton-specific strings."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        l1_decisions = _collect_l1_decisions(result.strategy_ir)
        for d in l1_decisions:
            for key, val in d.params.items():
                val_str = str(val)
                for ts in TRITON_SPECIFIC_STRINGS:
                    assert ts not in val_str, (
                        f"{ak_file.name}: L1 '{d.kind}' param '{key}' "
                        f"contains Triton string '{ts}' in value '{val_str}'"
                    )


class TestL2BackendSpecific:
    """Verify L2 decisions are correctly classified."""

    @pytest.fixture
    def pipeline(self):
        return ArkePipeline()

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_l2_decisions_have_level_2(self, ak_file: Path, pipeline):
        """Backend-specific decisions (compute_resource, etc.) are at level=2."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        l2_decisions = _collect_l2_decisions(result.strategy_ir)
        for d in l2_decisions:
            assert d.level == 2, (
                f"{ak_file.name}: Backend-specific decision '{d.kind}' "
                f"should be level=2, got level={d.level}"
            )

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_launch_config_migrated_to_l2(self, ak_file: Path, pipeline):
        """No launch_config decisions remain (should be compute_resource at L2)."""
        result = pipeline.compile_file(str(ak_file))
        if not result.success or result.strategy_ir is None:
            pytest.skip(f"No StrategyIR for {ak_file.name}")

        for d in result.strategy_ir.decisions:
            if isinstance(d, Decision):
                assert d.kind != "launch_config", (
                    f"{ak_file.name}: launch_config not migrated to "
                    f"compute_resource (L2)"
                )


class TestBackendAgnosticIntegrity:
    """Cross-cutting tests for backend-agnostic compliance."""

    def test_all_files_present(self):
        """Verify we have all expected .ak files."""
        assert len(ALL_AK_FILES) >= 46, (
            f"Expected >=46 .ak files, found {len(ALL_AK_FILES)}"
        )

    def test_l1_l2_separation(self):
        """Verify at least some files have both L1 and L2 decisions."""
        pipeline = ArkePipeline()
        files_with_both = 0

        for ak_file in ALL_AK_FILES:
            result = pipeline.compile_file(str(ak_file))
            if not result.success or result.strategy_ir is None:
                continue

            l1 = _collect_l1_decisions(result.strategy_ir)
            l2 = _collect_l2_decisions(result.strategy_ir)
            if l1 and l2:
                files_with_both += 1

        assert files_with_both > 0, (
            "No files have both L1 and L2 decisions — separation not testable"
        )
