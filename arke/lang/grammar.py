# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Language — Parser using Lark.

Parses .ak files into AST nodes defined in arke.lang.ast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, Tree, v_args

from arke.lang.ast import (
    Annotation,
    CompareCondition,
    BoolCondition,
    DimDecl,
    ImportStmt,
    InferType,
    KernelDef,
    LetStmt,
    OpCall,
    Parameter,
    Program,
    ReturnStmt,
    ScalarType,
    StrategyDef,
    StrategyStmt,
    TensorType,
    TupleType,
    WhenBlock,
    WhereClause,
)

# ============================================================
# Grammar file path
# ============================================================

_GRAMMAR_PATH = Path(__file__).parent / "arke.lark"

# ============================================================
# Lark parser instance (module-level singleton)
# ============================================================

_parser: Lark | None = None


def _get_parser() -> Lark:
    """Get or create the Lark parser singleton."""
    global _parser
    if _parser is None:
        grammar_text = _GRAMMAR_PATH.read_text()
        _parser = Lark(
            grammar_text,
            parser="lalr",
            start="start",
            propagate_positions=True,
        )
    return _parser


# ============================================================
# Helper: strip quotes from string tokens
# ============================================================

def _strip_quotes(s: str) -> str:
    """Remove surrounding quotes from a string literal token."""
    if isinstance(s, Token):
        s = str(s)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


# ============================================================
# Lark Transformer → AST Nodes
# ============================================================

@v_args(inline=True)
class ArkeTransformer(Transformer):
    """Transform Lark parse tree into Arke AST nodes."""

    # ── Program ──────────────────────────────────────────
    def start(self, *items):
        imports = []
        kernels = []
        strategies = []
        for item in items:
            if isinstance(item, ImportStmt):
                imports.append(item)
            elif isinstance(item, KernelDef):
                kernels.append(item)
            elif isinstance(item, StrategyDef):
                strategies.append(item)
        return Program(imports=imports, kernels=kernels, strategies=strategies)

    # ── Import ───────────────────────────────────────────
    def import_stmt(self, path, *rest):
        alias = str(rest[0]) if rest else None
        return ImportStmt(path=_strip_quotes(path), alias=alias)

    # ── Annotated Kernel ─────────────────────────────────
    def annotated_kernel_def(self, *items):
        annotations = []
        kernel = None
        for item in items:
            if isinstance(item, Annotation):
                annotations.append(item)
            elif isinstance(item, KernelDef):
                kernel = item
        if kernel is not None:
            kernel.annotations = annotations
        return kernel

    # ── Kernel Definition ────────────────────────────────
    def kernel_def(self, name, *rest):
        params = []
        return_type = None
        where = None
        body_stmts = []

        for item in rest:
            if isinstance(item, list):
                # param_list returns a list
                if item and isinstance(item[0], Parameter):
                    params = item
                else:
                    body_stmts = item
            elif isinstance(item, Parameter):
                params.append(item)
            elif isinstance(item, (TensorType, InferType, TupleType)):
                return_type = item
            elif isinstance(item, WhereClause):
                where = item
            elif isinstance(item, tuple) and len(item) == 2:
                # kernel_body returns (lets, return)
                body_stmts = list(item[0]) + [item[1]]

        return KernelDef(
            name=str(name),
            params=params,
            return_type=return_type,
            body=body_stmts,
            where_clause=where,
            annotations=[],
        )

    def param_list(self, *params):
        return list(params)

    def param(self, name, type_node):
        return Parameter(name=str(name), type=type_node)

    # ── Return Type ──────────────────────────────────────
    def tuple_return_type(self, *types):
        return TupleType(types=tuple(types))

    def tuple_infer_type(self, *types):
        return TupleType(types=tuple(types))

    def infer_type(self):
        return InferType()

    # ── Where Clause ─────────────────────────────────────
    def where_clause(self, *decls):
        return WhereClause(dims=list(decls))

    def dim_decl(self, name, kind):
        return kind._replace_name(str(name)) if hasattr(kind, '_replace_name') else kind

    def dim_static(self):
        return _DimKindBuilder("static", {})

    def dim_dynamic(self):
        return _DimKindBuilder("dynamic", {})

    def dim_dynamic_opts(self, opts):
        return _DimKindBuilder("dynamic", opts)

    def dynamic_opts(self, *opts):
        result = {}
        for k, v in opts:
            result[k] = v
        return result

    def dopt_max(self, val):
        return ("max", int(val))

    def dopt_min(self, val):
        return ("min", int(val))

    def dopt_multiple_of(self, val):
        return ("multiple_of", int(val))

    def dopt_default(self, val):
        return ("default", int(val))

    # ── Type System ──────────────────────────────────────
    def tensor_type(self, dims, dtype, *rest):
        layout = "row_major"
        if rest:
            layout = rest[0]
        return TensorType(shape=dims, dtype=dtype, layout=layout)

    def dim_list(self, *dims):
        return list(dims)

    def dim_int(self, val):
        return int(val)

    def dim_sym(self, name):
        return str(name)

    def layout_row_major(self):
        return "row_major"

    def layout_col_major(self):
        return "col_major"

    # Scalar types
    def scalar_f16(self): return ScalarType("f16")
    def scalar_bf16(self): return ScalarType("bf16")
    def scalar_f32(self): return ScalarType("f32")
    def scalar_f64(self): return ScalarType("f64")
    def scalar_i8(self): return ScalarType("i8")
    def scalar_i16(self): return ScalarType("i16")
    def scalar_i32(self): return ScalarType("i32")
    def scalar_i64(self): return ScalarType("i64")
    def scalar_u8(self): return ScalarType("u8")
    def scalar_u16(self): return ScalarType("u16")
    def scalar_u32(self): return ScalarType("u32")
    def scalar_u64(self): return ScalarType("u64")
    def scalar_bool(self): return ScalarType("bool")
    def scalar_index(self): return ScalarType("index")

    # ── Kernel Body ──────────────────────────────────────
    def kernel_body(self, *stmts):
        lets = []
        ret = None
        for s in stmts:
            if isinstance(s, LetStmt):
                lets.append(s)
            elif isinstance(s, ReturnStmt):
                ret = s
        return lets, ret

    def let_stmt(self, lhs, op_call):
        return LetStmt(lhs=lhs, op_call=op_call)

    def lhs_single(self, name):
        return str(name)

    def lhs_tuple(self, *names):
        return [str(n) for n in names]

    def return_stmt(self, expr):
        return ReturnStmt(value=expr)

    def return_single(self, name):
        return str(name)

    def return_tuple(self, *names):
        return [str(n) for n in names]

    # ── Operator Calls ───────────────────────────────────
    def op_call(self, name, *rest):
        args = []
        if rest and rest[0] is not None:
            args = rest[0]
        return OpCall(op=str(name), args=args)

    def arg_list(self, *args):
        return list(args)

    def named_arg(self, name, value):
        return (str(name), value)

    def arg_ident(self, val):
        return str(val)

    def arg_int(self, val):
        return int(val)

    def arg_neg_int(self, val):
        return -int(val)

    def arg_float(self, val):
        return float(val)

    def arg_neg_float(self, val):
        return -float(val)

    def arg_string(self, val):
        return _strip_quotes(val)

    def arg_bool(self, val):
        return str(val) == "true"

    def arg_list_val(self, *vals):
        return list(vals)

    # ── Strategy Definition ──────────────────────────────
    def strategy_def(self, name, target, body):
        return StrategyDef(
            name=str(name),
            target=_strip_quotes(target),
            body=body,
        )

    def strategy_body(self, *items):
        return list(items)

    def strategy_stmt(self, directive, *rest):
        kwargs = {}
        annotations = []
        for item in rest:
            if isinstance(item, dict):
                kwargs = item
            elif isinstance(item, Annotation):
                annotations.append(item)
            elif item is None:
                pass
        return StrategyStmt(
            directive=str(directive),
            kwargs=kwargs,
            annotations=annotations,
        )

    def strategy_kwargs(self, *kwargs):
        result = {}
        for k, v in kwargs:
            result[k] = v
        return result

    def strategy_kwarg(self, name, value):
        return (str(name), value)

    # Strategy values
    def sval_string(self, val):
        return _strip_quotes(val)

    def sval_int(self, val):
        return int(val)

    def sval_float(self, val):
        return float(val)

    def sval_bool(self, val):
        return str(val) == "true"

    def sval_ident(self, val):
        return str(val)

    def sval_list(self, *vals):
        return list(vals)

    def sval_map(self, *entries):
        return dict(entries)

    def strategy_map_entry(self, key, value):
        return (_strip_quotes(key), value)

    # ── When/Otherwise ───────────────────────────────────
    def when_block(self, *items):
        arms = []
        otherwise_body = None
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                arms.append(item)
            elif isinstance(item, list):
                otherwise_body = item
        return WhenBlock(arms=arms, otherwise_body=otherwise_body)

    def when_arm(self, condition, body):
        return (condition, body)

    def otherwise_arm(self, body):
        return body

    # ── Conditions ───────────────────────────────────────
    def cond_compare(self, ident, op, value):
        return CompareCondition(ident=str(ident), op=str(op), value=int(value))

    def cond_and(self, left, right):
        return BoolCondition(op="and", left=left, right=right)

    def cond_or(self, left, right):
        return BoolCondition(op="or", left=left, right=right)

    def cond_paren(self, cond):
        return cond

    # ── Annotations ──────────────────────────────────────
    def annotation(self, name, *rest):
        args = []
        if rest and rest[0] is not None:
            args = rest[0]
        return Annotation(name=str(name), args=args)

    def annotation_args(self, *args):
        return list(args)

    def ann_kwarg(self, name, value):
        return (str(name), value)

    def ann_positional(self, val):
        return _strip_quotes(val)

    # Annotation values
    def annval_string(self, val):
        return _strip_quotes(val)

    def annval_int(self, val):
        return int(val)

    def annval_float(self, val):
        return float(val)

    def annval_bool(self, val):
        return str(val) == "true"

    def annval_ident(self, val):
        return str(val)

    def annval_list(self, *vals):
        return list(vals)


# ============================================================
# DimKindBuilder helper
# ============================================================

class _DimKindBuilder:
    """Helper to build DimDecl from grammar rules.

    dim_decl receives (name, dim_kind_builder) and calls _replace_name.
    """
    def __init__(self, kind: str, opts: dict):
        self.kind = kind
        self.opts = opts

    def _replace_name(self, name: str) -> DimDecl:
        return DimDecl(name=name, kind=self.kind, opts=self.opts)


# ============================================================
# Transformer instance (module-level singleton)
# ============================================================

_transformer = ArkeTransformer()


# ============================================================
# Public API
# ============================================================

def parse_string(source: str) -> Program:
    """Parse an Arke source string and return a Program AST.

    Args:
        source: Arke language source code string

    Returns:
        Program AST node

    Raises:
        lark.exceptions.LarkError: on parse errors
    """
    parser = _get_parser()
    tree = parser.parse(source)
    return _transformer.transform(tree)


def parse_file(path: str | Path) -> Program:
    """Parse an Arke .ak file and return a Program AST.

    Args:
        path: Path to the .ak file

    Returns:
        Program AST node

    Raises:
        FileNotFoundError: if the file doesn't exist
        lark.exceptions.LarkError: on parse errors
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    return parse_string(source)
