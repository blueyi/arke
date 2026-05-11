# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Thread-local runtime context for baseline runners.

Used to thread the full shape dataclass to runners whose ``get_fn``
signature is fixed at ``(op, M, N, K, dtype)``. Cross-attention needs
both Sq and Skv, which don't fit cleanly into (M, N, K).
"""

from __future__ import annotations

import threading

_local = threading.local()


def set_current_shape(shape) -> None:
    _local.shape = shape


def clear_current_shape() -> None:
    _local.shape = None


def get_current_shape():
    return getattr(_local, "shape", None)
