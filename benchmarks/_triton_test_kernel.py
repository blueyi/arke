# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal Triton kernel for G0.2 gate verification.

Defined at module level so Triton's JIT can resolve ``tl`` references.
"""

import triton
import triton.language as tl


@triton.jit
def triton_add_kernel(x_ptr, n: tl.constexpr):
    """Add 1 to each element — simplest possible Triton kernel."""
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x + 1, mask=mask)
