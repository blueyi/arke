# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from unittest import mock


class TestStage7PipelineNoTorch:
    def test_pipeline_module_imports_without_torch_for_compile_only_paths(self):
        sys.modules.pop("arke.compiler.pipeline", None)
        with mock.patch.dict(sys.modules, {"torch": None}):
            module = importlib.import_module("arke.compiler.pipeline")
            assert module.torch is None
            assert callable(module._synthesize_strategy_from_compile_advice)
