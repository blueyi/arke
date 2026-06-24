# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Language — agent-friendly syntax errors (P1, 2026-06-24).

The 2026-06-24 AI-Native review flagged the Lang layer's #1 problem: the
parser raised raw Lark exceptions whose messages use *internal terminal
names* (``RPAR``, ``SEMICOLON``, ``F16``) with no source caret and no
fix suggestion. For an LLM driving a compile→fix self-correction loop,
that is close to unusable — it can't see *where* in its own generated
source the error is, nor *what* to write instead.

``ArkeSyntaxError`` wraps Lark's ``UnexpectedInput`` family into a
structured, actionable error:

  * ``line`` / ``column``     — 1-indexed source position
  * ``context``               — the offending source line(s) + a ``^`` caret
  * ``expected``              — the legal next tokens, *back-translated to
                                source literals* (``;`` not ``SEMICOLON``)
  * ``got``                   — the actual offending token text
  * ``suggestion``            — a targeted fix hint when we recognize the
                                error shape (missing ``;``, positional arg
                                where a named arg is required, ``float16``
                                → ``f16``, …)

The ``__str__`` renders all of this in a compact block an agent can read
and act on directly. ``to_dict()`` exposes the same fields for
programmatic consumers (the Harness, MCP clients).
"""

from __future__ import annotations

from typing import Any

# ── Terminal-name → source-literal back-translation ──────────────────
#
# Lark reports the *terminal names* it expected. Agents (and humans)
# think in source literals. This table translates the common ones; any
# terminal not listed falls back to its lowercased name.
_TERMINAL_TO_LITERAL: dict[str, str] = {
    "SEMICOLON": "';'",
    "COLON": "':'",
    "COMMA": "','",
    "EQUAL": "'='",
    "LPAR": "'('",
    "RPAR": "')'",
    "LBRACE": "'{'",
    "RBRACE": "'}'",
    "LSQB": "'['",
    "RSQB": "']'",
    "LESSTHAN": "'<'",
    "MORETHAN": "'>'",
    "ARROW": "'->'",
    "AT": "'@'",
    "TENSOR": "'Tensor<...>' type",
    "UNDERSCORE": "'_' (inferred type)",
    "IDENT": "an identifier",
    "KERNEL": "'kernel'",
    "STRATEGY": "'strategy'",
    "LET": "'let'",
    "RETURN": "'return'",
    "WHERE": "'where'",
    "FOR": "'for'",
    "RATIONALE": "'@rationale'",
    "NUMBER": "a number",
    "STRING": "a string literal",
}

# Known scalar dtype literals (for the float16→f16 style hint).
_DTYPE_ALIASES: dict[str, str] = {
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "bool": "b8",
    "half": "f16",
    "float": "f32",
    "double": "f64",
}


def _literal(terminal: str) -> str:
    return _TERMINAL_TO_LITERAL.get(terminal, terminal.lower())


class ArkeSyntaxError(SyntaxError):
    """Agent-friendly syntax error wrapping a Lark ``UnexpectedInput``."""

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        context: str = "",
        expected: list[str] | None = None,
        got: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        self.expected = expected or []
        self.got = got
        self.suggestion = suggestion

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "ArkeSyntaxError",
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "context": self.context,
            "expected": self.expected,
            "got": self.got,
            "suggestion": self.suggestion,
        }

    def __str__(self) -> str:
        lines = [f"Syntax error at line {self.line}, column {self.column}: {self.message}"]
        if self.context:
            lines.append("")
            lines.append(self.context.rstrip("\n"))
        if self.got is not None:
            lines.append(f"\nGot: {self.got!r}")
        if self.expected:
            shown = ", ".join(self.expected[:10])
            more = "" if len(self.expected) <= 10 else f" (+{len(self.expected) - 10} more)"
            lines.append(f"Expected one of: {shown}{more}")
        if self.suggestion:
            lines.append(f"\nFix: {self.suggestion}")
        return "\n".join(lines)


def _build_suggestion(
    expected_terminals: list[str],
    got_token: Any,
    source: str,
) -> str | None:
    """Recognize common agent error shapes → targeted fix hints."""
    exp = set(expected_terminals)
    got_text = str(getattr(got_token, "value", got_token) or "")
    got_type = str(getattr(got_token, "type", "") or "")

    # 1. Missing semicolon: parser wanted ';' but got the next statement
    #    keyword / identifier.
    if "SEMICOLON" in exp:
        return "Add a ';' to terminate the previous statement (every let/return statement ends with ';')."

    # 2. Positional argument where a named argument is required: op calls use
    #    named args (relu(X=X), matmul(A=A, B=B)) — got ')' or ',' where '='
    #    was expected.
    if "EQUAL" in exp and got_type in ("RPAR", "COMMA"):
        return ("Op-call arguments must be named: write 'op(PARAM=value)' not "
                "'op(value)'. e.g. relu(X=X), matmul(A=A, B=B).")

    # 3. Wrong dtype spelling: a bare identifier that looks like a long-form
    #    dtype name where a scalar dtype was expected.
    if got_text in _DTYPE_ALIASES:
        return f"Use Arke's short dtype name '{_DTYPE_ALIASES[got_text]}' instead of '{got_text}'."

    # 4. Tensor type expected.
    if "TENSOR" in exp and "UNDERSCORE" in exp:
        return ("Expected a type here: either 'Tensor<[dims],dtype>' (e.g. "
                "Tensor<[128,3072],f16>) or '_' to infer it.")

    return None


def wrap_lark_error(exc: Exception, source: str) -> ArkeSyntaxError:
    """Translate a Lark ``UnexpectedInput`` into an :class:`ArkeSyntaxError`.

    Falls back gracefully for any exception shape we don't fully recognize.
    """
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)

    # Source context with caret (Lark provides get_context on UnexpectedInput).
    context = ""
    get_ctx = getattr(exc, "get_context", None)
    if callable(get_ctx):
        try:
            context = str(get_ctx(source))
        except Exception:
            context = ""

    # Expected terminals (UnexpectedToken: .expected / .accepts).
    raw_expected = getattr(exc, "expected", None) or getattr(exc, "accepts", None) or []
    try:
        expected = sorted({_literal(t) for t in raw_expected})
    except TypeError:
        expected = []

    got_token = getattr(exc, "token", None)
    got = None
    if got_token is not None:
        got = str(getattr(got_token, "value", got_token))

    suggestion = None
    try:
        suggestion = _build_suggestion(list(raw_expected), got_token, source)
    except Exception:
        suggestion = None

    # Base message: keep it short — the structured fields carry the detail.
    kind = type(exc).__name__
    if "Characters" in kind:
        message = "unexpected character(s) in source"
    elif "Token" in kind:
        message = "unexpected token"
    elif "EOF" in kind:
        message = "unexpected end of input (unclosed block or missing token?)"
    else:
        message = "could not parse source"

    return ArkeSyntaxError(
        message,
        line=line,
        column=column,
        context=context,
        expected=expected,
        got=got,
        suggestion=suggestion,
    )
