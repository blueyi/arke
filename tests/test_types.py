# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arke type system."""

from arke.lang.types import (
    is_float, is_int, is_valid_scalar, dtype_bits,
    f16, f32, bf16, i32, u8, SCALAR_TYPES, MEMORY_LEVELS,
)


def test_scalar_type_classification():
    assert is_float(f16) is True
    assert is_float(f32) is True
    assert is_float(bf16) is True
    assert is_float(i32) is False

    assert is_int(i32) is True
    assert is_int(f16) is False

    assert is_valid_scalar(f16) is True
    assert is_valid_scalar(u8) is True
    assert is_valid_scalar("unknown") is False


def test_dtype_bits():
    assert dtype_bits(f16) == 16
    assert dtype_bits(f32) == 32
    assert dtype_bits(bf16) == 16
    assert dtype_bits(i32) == 32
    assert dtype_bits(u8) == 8
    assert dtype_bits("unknown") == 0


def test_type_sets():
    assert len(SCALAR_TYPES) == 14
    assert len(MEMORY_LEVELS) == 4
