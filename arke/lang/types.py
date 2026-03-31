# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Type System — scalar types, tensor types, layout types."""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Scalar Types
# ============================================================

# Floating point
f16 = "f16"
f32 = "f32"
f64 = "f64"
bf16 = "bf16"

# Integer
i8 = "i8"
i16 = "i16"
i32 = "i32"
i64 = "i64"

# Unsigned integer
u8 = "u8"
u16 = "u16"
u32 = "u32"
u64 = "u64"

# Special
bool_type = "bool"
index = "index"

SCALAR_TYPES = {
    f16, f32, f64, bf16,
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    bool_type, index,
}

FLOAT_TYPES = {f16, f32, f64, bf16}
INT_TYPES = {i8, i16, i32, i64}
UINT_TYPES = {u8, u16, u32, u64}

# ============================================================
# Layout Types
# ============================================================

ROW_MAJOR = "row_major"
COL_MAJOR = "col_major"

LAYOUT_TYPES = {ROW_MAJOR, COL_MAJOR}  # tiled and custom added later

# ============================================================
# Memory Levels
# ============================================================

GLOBAL = "global"
SHARED = "shared"
LOCAL = "local"
REGISTER = "register"

MEMORY_LEVELS = {GLOBAL, SHARED, LOCAL, REGISTER}


# ============================================================
# Type Checking Utilities
# ============================================================

def is_float(dtype: str) -> bool:
    """Check if a dtype is a floating point type."""
    return dtype in FLOAT_TYPES


def is_int(dtype: str) -> bool:
    """Check if a dtype is an integer type."""
    return dtype in INT_TYPES


def is_valid_scalar(dtype: str) -> bool:
    """Check if a dtype is a valid scalar type."""
    return dtype in SCALAR_TYPES


def dtype_bits(dtype: str) -> int:
    """Get the bit width of a scalar type."""
    bits_map = {
        "f16": 16, "f32": 32, "f64": 64, "bf16": 16,
        "i8": 8, "i16": 16, "i32": 32, "i64": 64,
        "u8": 8, "u16": 16, "u32": 32, "u64": 64,
        "bool": 1, "index": 64,
    }
    return bits_map.get(dtype, 0)
