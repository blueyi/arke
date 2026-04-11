# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE7_TRACK6_L1 = REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6" / "l1"
STAGE7_TRACK6_L2 = REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6" / "l2"


class TestStage7Track6Contract:
    def test_stage7_track6_standard_result_dirs_exist(self):
        assert STAGE7_TRACK6_L1.exists()
        assert STAGE7_TRACK6_L2.exists()

    def test_stage7_track6_l1_has_gate_artifacts(self):
        assert (STAGE7_TRACK6_L1 / "config.json").exists()
        assert (STAGE7_TRACK6_L1 / "hardware.json").exists()
        assert (STAGE7_TRACK6_L1 / "sources.json").exists()
        assert (STAGE7_TRACK6_L1 / "PERF_ALL.csv").exists()
        assert (STAGE7_TRACK6_L1 / "summary.json").exists()

    def test_stage7_track6_l2_has_gate_artifacts(self):
        assert (STAGE7_TRACK6_L2 / "config.json").exists()
        assert (STAGE7_TRACK6_L2 / "hardware.json").exists()
        assert (STAGE7_TRACK6_L2 / "sources.json").exists()
        assert (STAGE7_TRACK6_L2 / "PERF_ALL.csv").exists()
        assert (STAGE7_TRACK6_L2 / "summary.json").exists()

    def test_stage7_track6_summaries_expose_operator_scores(self):
        l1_summary = json.loads((STAGE7_TRACK6_L1 / "summary.json").read_text())
        l2_summary = json.loads((STAGE7_TRACK6_L2 / "summary.json").read_text())

        assert "operators" in l1_summary
        assert "op_scores" in l1_summary
        assert "operators" in l2_summary
        assert "op_scores" in l2_summary
        assert l1_summary["operators"]
        assert l2_summary["operators"]
