# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Version consistency tests for package metadata and active IR schemas."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arke import __version__
from arke.ir.akir import AKIR_VERSION
from arke.ir.instruction import InstructionIR
from arke.ir.schedule import ScheduleIR
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

EXPECTED_PACKAGE_VERSION = "0.2.0.dev0"
EXPECTED_IR_VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None, "pyproject.toml must declare [project].version"
    return match.group(1)


class TestVersionConsistency:
    def test_package_version_matches_release_contract(self):
        assert __version__ == EXPECTED_PACKAGE_VERSION

    def test_pyproject_version_matches_package_version(self):
        assert _pyproject_version() == EXPECTED_PACKAGE_VERSION

    def test_ir_layer_defaults_match_active_schema_version(self):
        assert SemanticIR().version == EXPECTED_IR_VERSION
        assert StrategyIR().version == EXPECTED_IR_VERSION
        assert ScheduleIR().version == EXPECTED_IR_VERSION
        assert InstructionIR().version == EXPECTED_IR_VERSION

    def test_ir_layer_from_dict_defaults_match_active_schema_version(self):
        assert SemanticIR.from_dict({}).version == EXPECTED_IR_VERSION
        assert StrategyIR.from_dict({}).version == EXPECTED_IR_VERSION
        assert ScheduleIR.from_dict({}).version == EXPECTED_IR_VERSION
        assert InstructionIR.from_dict({}).version == EXPECTED_IR_VERSION

    def test_ir_layer_from_dict_rejects_legacy_schema_versions(self):
        with pytest.raises(ValueError, match="Unsupported SemanticIR version"):
            SemanticIR.from_dict({"version": "1.0.0"})
        with pytest.raises(ValueError, match="Unsupported StrategyIR version"):
            StrategyIR.from_dict({"version": "1.0.0"})
        with pytest.raises(ValueError, match="Unsupported ScheduleIR version"):
            ScheduleIR.from_dict({"version": "1.0.0"})
        with pytest.raises(ValueError, match="Unsupported InstructionIR version"):
            InstructionIR.from_dict({"version": "1.0.0"})

    def test_akir_version_matches_active_schema_version(self):
        assert AKIR_VERSION == EXPECTED_IR_VERSION
