# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-friendly syntax errors (P1, arke.lang.errors).

Verifies that parse errors surface as ArkeSyntaxError with line/column, a
source caret, back-translated expected literals, and a targeted fix
suggestion for the common agent error shapes the AI-Native review flagged.
"""

from __future__ import annotations

import pytest

from arke.lang.errors import ArkeSyntaxError
from arke.lang.grammar import parse_string

_BASE_OK = """kernel relu_kernel(X: Tensor<[128,3072],f16>) -> Tensor<[128,3072],f16>
where D: static
{
  let Y = relu(X=X);
  return Y;
}"""


def test_base_kernel_parses():
    """Sanity: the well-formed base kernel parses without error."""
    prog = parse_string(_BASE_OK)
    assert prog is not None


def test_missing_semicolon_friendly_error():
    src = _BASE_OK.replace("let Y = relu(X=X);", "let Y = relu(X=X)")
    with pytest.raises(ArkeSyntaxError) as ei:
        parse_string(src)
    e = ei.value
    assert e.line is not None and e.column is not None
    assert e.context  # has a source caret block
    assert "';'" in e.expected
    assert e.suggestion and "';'" in e.suggestion


def test_positional_arg_friendly_error():
    src = _BASE_OK.replace("relu(X=X)", "relu(X)")
    with pytest.raises(ArkeSyntaxError) as ei:
        parse_string(src)
    e = ei.value
    assert "'='" in e.expected
    assert e.suggestion and "named" in e.suggestion.lower()


def test_wrong_dtype_friendly_error():
    src = _BASE_OK.replace("Tensor<[128,3072],f16>", "Tensor<[128,3072],float16>", 1)
    with pytest.raises(ArkeSyntaxError) as ei:
        parse_string(src)
    e = ei.value
    assert e.got == "float16"
    assert e.suggestion and "f16" in e.suggestion


def test_error_to_dict_shape():
    src = _BASE_OK.replace("let Y = relu(X=X);", "let Y = relu(X=X)")
    with pytest.raises(ArkeSyntaxError) as ei:
        parse_string(src)
    d = ei.value.to_dict()
    assert d["error"] == "ArkeSyntaxError"
    for key in ("message", "line", "column", "context", "expected", "got", "suggestion"):
        assert key in d


def test_error_str_is_readable():
    src = _BASE_OK.replace("let Y = relu(X=X);", "let Y = relu(X=X)")
    with pytest.raises(ArkeSyntaxError) as ei:
        parse_string(src)
    s = str(ei.value)
    assert "Syntax error at line" in s
    assert "Fix:" in s
