# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from benchmarks.status import classify_exception, normalize_status


class TestBenchmarkStatus:
    def test_classify_cuda_oom_exception(self):
        status = classify_exception(RuntimeError("CUDA out of memory while allocating tensor"))
        assert status.status == "oom"
        assert status.retryable is True
        assert "out of memory" in status.reason.lower()

    def test_classify_generic_exception(self):
        status = classify_exception(RuntimeError("kernel launch failed"))
        assert status.status == "error"
        assert status.retryable is False

    def test_normalize_status_accepts_known_values(self):
        status = normalize_status("skipped", reason="not supported on this host")
        assert status.status == "skipped"
        assert status.reason == "not supported on this host"

    def test_normalize_status_rejects_unknown_values(self):
        with pytest.raises(ValueError):
            normalize_status("mystery")
