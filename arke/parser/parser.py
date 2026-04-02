# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke language parser — .ak files → AST → Semantic IR.

Usage:
    from arke.parser import parse_file, parse_string
    program = parse_string(source_code)
    ir = program.kernels[0].to_ir()
"""

from __future__ import annotations

from pathlib import Path

from lark import Lark, Transformer, v_args

from arke.parser.ast_nodes import (
    Annotation,
    ImportStmt,
    KernelDef,
    LetStmt,
    OpCall,
    Param,
    Program,
    ReturnStmt,
    StrategyAction,
    StrategyDef,
    TensorType,
)

# Grammar file path
_GRAMMAR_PATH = Path(__file__).parent / "arke.lark"


def _get_parser() -> Lark:
    """Create Lark parser (cached at module level)."""
    return Lark(
        _GRAMMAR_PATH.read_text(),
        parser="earley",
        propagate_positions=True,
    )


_parser: Lark | None = None


def _parser_instance() -> Lark:
    global _parser
    if _parser is None:
        _parser = _get_parser()
    return _parser


@v_args(inline=True)
class ArkeTransformer(Transformer):
    """Transform Lark parse tree into Arke AST nodes."""

    # ─── Tokens ───

    def IDENT(self, token):
        """Transform an identifier token to a string."""
        return str(token)

    def INT(self, token):
        """Transform an integer token to an int value."""
        s = str(token)
        if s.startswith(("0x", "0X")):
            return int(s, 16)
        return int(s)

    def FLOAT(self, token):
        """Transform a float token to a float value."""
        return float(str(token))

    def STRING(self, token):
        """Transform a string token, stripping surrounding quotes."""
        # Strip quotes
        return str(token)[1:-1]

    # ─── Types ───

    def dim_list(self, *dims):
        """Collect dimension values into a list."""
        return list(dims)

    def SCALAR_TYPE(self, token):
        """Transform a scalar type token to a string."""
        return str(token)

    def scalar_type(self, val):
        """Return the scalar type value."""
        return val

    def LAYOUT(self, token):
        """Transform a layout token to a string."""
        return str(token)

    def layout(self, val):
        """Return the layout value."""
        return val

    def tensor_type(self, *args):
        """Construct a TensorType from shape, dtype, and optional layout."""
        dims = args[0]
        dtype = args[1]
        lay = args[2] if len(args) > 2 else "row_major"
        return TensorType(shape=dims, dtype=dtype, layout=lay)

    # ─── Kernel ───

    def param(self, name, typ):
        """Construct a Param AST node from name and type."""
        return Param(name=name, type=typ)

    def param_list(self, *params):
        """Collect parameters into a list."""
        return list(params)

    def named_arg(self, key, value):
        """Construct a named argument tuple."""
        return (key, value)

    def positional_arg(self, name):
        """Construct a positional argument tuple."""
        return (None, name)

    def arg_list(self, *args):
        """Collect arguments into a list."""
        return list(args)

    def op_call(self, op_name, *rest):
        """Construct an OpCall node from operator name and arguments."""
        args = rest[0] if rest else []
        # Convert to dict: named args use key, positional use op-specific names
        named = {}
        for i, (key, val) in enumerate(args):
            if key is not None:
                named[key] = val
            else:
                # Positional: use the value as both key and value for single-arg ops
                named[f"_pos_{i}"] = val
        return OpCall(op=op_name, args=named)

    def let_stmt(self, name, value):
        """Construct a LetStmt from variable name and value."""
        return LetStmt(name=name, value=value)

    def return_stmt(self, name):
        """Construct a ReturnStmt from variable name."""
        return ReturnStmt(name=name)

    def kernel_body(self, *stmts):
        """Collect kernel body statements into a list."""
        return list(stmts)

    def kernel_def(self, name, *rest):
        """Construct a KernelDef from name, params, return type, and body."""
        # rest could be: [params, return_type, body] or [return_type, body]
        if len(rest) == 3:
            params, ret_type, body = rest
        else:
            params = []
            ret_type, body = rest
        return KernelDef(
            name=name,
            params=params,
            return_type=ret_type,
            body=body,
        )

    # ─── Strategy ───

    def annotation(self, key, value):
        """Construct an Annotation from key and value."""
        return Annotation(key=key, value=value)

    def bool_true(self):
        """Return True for a boolean true literal."""
        return True

    def bool_false(self):
        """Return False for a boolean false literal."""
        return False

    def ident_value(self, name):
        """Return an identifier value."""
        return name

    def array(self, *items):
        """Collect array items into a list."""
        return list(items)

    def map_entry(self, key, value):
        """Construct a map entry tuple from key and value."""
        return (key, value)

    def map(self, *entries):
        """Construct a dict from map entries."""
        return dict(entries)

    def strategy_value(self, val):
        """Return a strategy value."""
        return val

    def strategy_kwarg(self, key, value):
        """Construct a strategy keyword argument tuple."""
        return (key, value)

    def strategy_args(self, *kwargs):
        """Collect strategy keyword arguments into a dict."""
        return dict(kwargs)

    def strategy_action(self, action_name, args):
        """Construct a strategy action tuple from name and arguments."""
        return (action_name, args)

    def strategy_stmt(self, *parts):
        """Construct a StrategyAction with optional annotation."""
        action_name, params = parts[0]
        ann = parts[1] if len(parts) > 1 else None
        return StrategyAction(
            action=action_name,
            params=params,
            annotation=ann,
        )

    def strategy_body(self, *stmts):
        """Collect strategy statements into a list."""
        return list(stmts)

    def strategy_def(self, name, target, body):
        """Construct a StrategyDef from name, target, and body."""
        return StrategyDef(name=name, target=target, actions=body)

    # ─── Import ───

    def import_stmt(self, path, alias):
        """Construct an ImportStmt from path and alias."""
        return ImportStmt(path=path, alias=alias)

    # ─── Program ───

    def start(self, *items):
        """Construct the top-level Program from parsed items."""
        prog = Program()
        for item in items:
            if isinstance(item, ImportStmt):
                prog.imports.append(item)
            elif isinstance(item, KernelDef):
                prog.kernels.append(item)
            elif isinstance(item, StrategyDef):
                prog.strategies.append(item)
        return prog


def parse_string(source: str) -> Program:
    """Parse Arke source code string into AST.

    Args:
        source: Arke language source code

    Returns:
        Program AST node
    """
    tree = _parser_instance().parse(source)
    return ArkeTransformer().transform(tree)


def parse_file(path: str | Path) -> Program:
    """Parse an .ak file into AST.

    Args:
        path: Path to .ak file

    Returns:
        Program AST node
    """
    source = Path(path).read_text()
    return parse_string(source)
