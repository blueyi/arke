#!/usr/bin/env python3
"""Add missing docstrings to arke source files."""

import re

# Each entry: (filepath, function/class name, line number, docstring)
# We'll use simple text replacement for reliability.

edits = {
    # ─── arke/parser/parser.py ───
    "arke/parser/parser.py": [
        ("    def IDENT(self, token):\n", '        """Transform an identifier token to a string."""\n'),
        ("    def INT(self, token):\n", '        """Transform an integer token to an int value."""\n'),
        ("    def FLOAT(self, token):\n", '        """Transform a float token to a float value."""\n'),
        ("    def STRING(self, token):\n", '        """Transform a string token, stripping surrounding quotes."""\n'),
        ("    def dim_list(self, *dims):\n", '        """Collect dimension values into a list."""\n'),
        ("    def SCALAR_TYPE(self, token):\n", '        """Transform a scalar type token to a string."""\n'),
        ("    def scalar_type(self, val):\n", '        """Return the scalar type value."""\n'),
        ("    def LAYOUT(self, token):\n", '        """Transform a layout token to a string."""\n'),
        ("    def layout(self, val):\n", '        """Return the layout value."""\n'),
        ("    def tensor_type(self, *args):\n", '        """Construct a TensorType from shape, dtype, and optional layout."""\n'),
        ("    def param(self, name, typ):\n", '        """Construct a Param from name and type."""\n'),
        ("    def param_list(self, *params):\n", '        """Collect parameters into a list."""\n'),
        ("    def named_arg(self, key, value):\n", '        """Construct a named argument tuple."""\n'),
        ("    def positional_arg(self, name):\n", '        """Construct a positional argument tuple."""\n'),
        ("    def arg_list(self, *args):\n", '        """Collect arguments into a list."""\n'),
        ("    def op_call(self, op_name, *rest):\n", '        """Construct an OpCall node from operator name and arguments."""\n'),
        ("    def let_stmt(self, name, value):\n", '        """Construct a LetStmt from variable name and value."""\n'),
        ("    def return_stmt(self, name):\n", '        """Construct a ReturnStmt from variable name."""\n'),
        ("    def kernel_body(self, *stmts):\n", '        """Collect kernel body statements into a list."""\n'),
        ("    def kernel_def(self, name, *rest):\n", '        """Construct a KernelDef from name, params, return type, and body."""\n'),
        ("    def annotation(self, key, value):\n", '        """Construct an Annotation from key and value."""\n'),
        ("    def bool_true(self):\n", '        """Return True for a boolean true literal."""\n'),
        ("    def bool_false(self):\n", '        """Return False for a boolean false literal."""\n'),
        ("    def ident_value(self, name):\n", '        """Return an identifier value."""\n'),
        ("    def array(self, *items):\n", '        """Collect array items into a list."""\n'),
        ("    def map_entry(self, key, value):\n", '        """Construct a map entry tuple from key and value."""\n'),
        ("    def map(self, *entries):\n", '        """Construct a dict from map entries."""\n'),
        ("    def strategy_value(self, val):\n", '        """Return a strategy value."""\n'),
        ("    def strategy_kwarg(self, key, value):\n", '        """Construct a strategy keyword argument tuple."""\n'),
        ("    def strategy_args(self, *kwargs):\n", '        """Collect strategy keyword arguments into a dict."""\n'),
        ("    def strategy_action(self, action_name, args):\n", '        """Construct a strategy action tuple from name and arguments."""\n'),
        ("    def strategy_stmt(self, *parts):\n", '        """Construct a StrategyAction from action parts and optional annotation."""\n'),
        ("    def strategy_body(self, *stmts):\n", '        """Collect strategy statements into a list."""\n'),
        ("    def strategy_def(self, name, target, body):\n", '        """Construct a StrategyDef from name, target, and body."""\n'),
        ("    def import_stmt(self, path, alias):\n", '        """Construct an ImportStmt from path and alias."""\n'),
        ("    def start(self, *items):\n", '        """Construct the top-level Program from parsed items."""\n'),
    ],

    # ─── arke/ir/strategy.py ───
    "arke/ir/strategy.py": [
        ("    def decision_count(self) -> int:\n", '        """Return the number of optimization decisions."""\n'),
        ("    def tile(self, loop: str, factors: list[int],\n", '        """Add a tile decision for the given loop with specified factors."""\n'),
        ("    def reorder(self, order: list[str],\n", '        """Add a reorder decision with the specified loop order."""\n'),
        ("    def fuse(self, ops: list[str], fusion_type: str = \"epilogue\",\n", '        """Add a fusion decision for the specified operators."""\n'),
        ("    def parallel(self, loops: list[str], mapping: dict[str, str],\n", '        """Add a parallelism decision mapping loops to hardware dimensions."""\n'),
        ("    def place(self, tensor: str, memory: str,\n", '        """Add a memory placement decision for a tensor."""\n'),
        ("    def to_dict(self) -> dict:\n", '        """Serialize the strategy IR to a plain dict."""\n'),
        ("    def to_json(self, indent: int = 2) -> str:\n", '        """Serialize the strategy IR to a JSON string."""\n'),
        ("    def to_file(self, path: str) -> None:\n", '        """Save the strategy IR to a JSON file."""\n'),
        ("    def from_dict(cls, data: dict) -> StrategyIR:\n", '        """Deserialize a StrategyIR from a plain dict."""\n'),
        ("    def from_json(cls, json_str: str) -> StrategyIR:\n", '        """Deserialize a StrategyIR from a JSON string."""\n'),
    ],

    # ─── arke/ir/semantic.py ───
    "arke/ir/semantic.py": [
        ("    def to_tensor_desc(self) -> TensorDesc:\n", '        """Convert this parameter to a TensorDesc."""\n'),
        # ParamRef.to_dict
        ("class ParamRef:\n    \"\"\"Reference to a kernel parameter.\"\"\"\n    name: str\n\n    def to_dict(self) -> dict:\n",
         None),  # skip - need special handling
        # ParamRef.from_dict
        # NodeRef.to_dict
        # NodeRef.from_dict
    ],

    # ─── arke/parser/ast_nodes.py ───
    "arke/parser/ast_nodes.py": [
        ("    def get_kernel(self, name: str) -> KernelDef | None:\n", '        """Get a kernel definition by name, or None if not found."""\n'),
        ("    def get_strategy(self, name: str) -> StrategyDef | None:\n", '        """Get a strategy definition by name, or None if not found."""\n'),
    ],

    # ─── arke/learn/trajectory.py ───
    "arke/learn/trajectory.py": [
        ("    def to_dict(self) -> dict[str, Any]:\n", '        """Serialize this trajectory record to a dict."""\n'),
        # TrajectoryWriter.__init__
        ("    def __init__(self, path: str | Path):\n", '        """Initialize a trajectory writer for the given file path."""\n'),
        ("    def flush(self) -> None:\n", '        """Flush buffered trajectory data to disk."""\n'),
        ("    def close(self) -> None:\n", '        """Close the trajectory file."""\n'),
    ],

    # ─── arke/engine/accuracy.py ───
    "arke/engine/accuracy.py": [
        # AccuracyMetrics.to_dict
        ("class AccuracyMetrics:\n",  None),  # skip
        # CompareConfig.to_dict  
        # CompareResult.to_dict
        # BenchmarkResult.to_dict
        # _aggregate_metrics helper functions
    ],

    # ─── arke/agent/session.py ───
    "arke/agent/session.py": [
        ("    def decisions_remaining(self) -> int:\n", '        """Return the number of decisions remaining in the budget."""\n'),
        ("    def compiles_remaining(self) -> int:\n", '        """Return the number of compiles remaining in the budget."""\n'),
        ("    def exhausted(self) -> bool:\n", '        """Return True if the decision budget is exhausted."""\n'),
        ("    def should_warn(self) -> bool:\n", '        """Return True if the budget warning threshold has been reached."""\n'),
        ("    def use_decision(self) -> None:\n", '        """Record usage of one decision from the budget."""\n'),
        ("    def use_compile(self) -> None:\n", '        """Record usage of one compile from the budget."""\n'),
        # OptimizationBudget.to_dict
        ("class OptimizationBudget:\n", None),  # need special
    ],
}

import sys

def add_docstring_after_def(filepath, search_line, docstring_line):
    """Add a docstring after the function/method definition line."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the search_line in content
    idx = content.find(search_line)
    if idx == -1:
        print(f"  WARNING: Could not find '{search_line.strip()}' in {filepath}")
        return False
    
    # Find the end of the def line (could be multi-line)
    # We need to find the colon that ends the def, then the newline
    def_start = idx
    # For the search line, find where the colon+newline is
    colon_idx = content.find(':\n', idx)
    if colon_idx == -1:
        print(f"  WARNING: Could not find ':' for '{search_line.strip()}' in {filepath}")
        return False
    
    insert_pos = colon_idx + 2  # after :\n
    
    # Check if there's already a docstring
    rest = content[insert_pos:insert_pos+50].lstrip()
    if rest.startswith('"""') or rest.startswith("'''"):
        # Already has docstring
        return False
    
    # Determine indentation
    # Find the indentation of the def line
    line_start = content.rfind('\n', 0, idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    indent = ''
    for ch in content[line_start:]:
        if ch in (' ', '\t'):
            indent += ch
        else:
            break
    # Docstring indent is one level deeper
    doc_indent = indent + '    '
    
    formatted_doc = doc_indent + docstring_line
    content = content[:insert_pos] + formatted_doc + content[insert_pos:]
    
    with open(filepath, 'w') as f:
        f.write(content)
    return True

# Process the simple edits
for filepath, edit_list in edits.items():
    print(f"\nProcessing {filepath}...")
    for item in edit_list:
        search_line, docstring = item
        if docstring is None:
            continue
        result = add_docstring_after_def(filepath, search_line, docstring)
        if result:
            name = search_line.strip().split('(')[0].replace('def ', '').replace('self, ', '')
            print(f"  Added docstring to {name}")

print("\nDone with simple edits. Now handling special cases...")
