#!/usr/bin/env python3
"""Add missing docstrings to arke source files.

Run from project root: python _add_docstrings.py
"""

import re

def insert_docstring(content: str, func_sig: str, docstring: str) -> str:
    """Insert docstring after a function signature in file content.
    
    func_sig: unique portion of the 'def ...:' line to search for
    docstring: the one-line docstring text (without triple quotes)
    """
    idx = content.find(func_sig)
    if idx == -1:
        print(f"  WARNING: not found: {func_sig[:60]}")
        return content
    
    # Find the colon-newline after this def
    colon_pos = content.find(":\n", idx)
    if colon_pos == -1:
        print(f"  WARNING: no colon-newline for: {func_sig[:60]}")
        return content
    
    insert_at = colon_pos + 2  # after :\n
    
    # Check not already docstringed
    after = content[insert_at:insert_at+80].lstrip()
    if after.startswith('"""') or after.startswith("'''"):
        return content
    
    # Detect indent
    line_start = content.rfind('\n', 0, idx)
    line_start = 0 if line_start == -1 else line_start + 1
    base_indent = ''
    for ch in content[line_start:idx]:
        if ch in ' \t':
            base_indent += ch
        else:
            break
    body_indent = base_indent + '    '
    
    doc_line = f'{body_indent}"""{docstring}"""\n'
    return content[:insert_at] + doc_line + content[insert_at:]


def process_file(filepath: str, edits: list[tuple[str, str]]) -> int:
    """Apply a list of (func_sig, docstring_text) edits to a file."""
    with open(filepath) as f:
        content = f.read()
    
    count = 0
    for func_sig, doc in edits:
        new_content = insert_docstring(content, func_sig, doc)
        if new_content != content:
            count += 1
            content = new_content
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  {filepath}: {count} docstrings added")
    return count


total = 0

# ─── arke/parser/parser.py ───
# Note: IDENT already done via edit tool
total += process_file("arke/parser/parser.py", [
    ("def INT(self, token):", "Transform an integer token to an int value."),
    ("def FLOAT(self, token):", "Transform a float token to a float value."),
    ("def STRING(self, token):", "Transform a string token, stripping surrounding quotes."),
    ("def dim_list(self, *dims):", "Collect dimension values into a list."),
    ("def SCALAR_TYPE(self, token):", "Transform a scalar type token to a string."),
    ("def scalar_type(self, val):", "Return the scalar type value."),
    ("def LAYOUT(self, token):", "Transform a layout token to a string."),
    ("def layout(self, val):", "Return the layout value."),
    ("def tensor_type(self, *args):", "Construct a TensorType from shape, dtype, and optional layout."),
    ("def param(self, name, typ):", "Construct a Param AST node from name and type."),
    ("def param_list(self, *params):", "Collect parameters into a list."),
    ("def named_arg(self, key, value):", "Construct a named argument tuple."),
    ("def positional_arg(self, name):", "Construct a positional argument tuple."),
    ("def arg_list(self, *args):", "Collect arguments into a list."),
    ("def op_call(self, op_name, *rest):", "Construct an OpCall node from operator name and arguments."),
    ("def let_stmt(self, name, value):", "Construct a LetStmt from variable name and value."),
    ("def return_stmt(self, name):", "Construct a ReturnStmt from variable name."),
    ("def kernel_body(self, *stmts):", "Collect kernel body statements into a list."),
    ("def kernel_def(self, name, *rest):", "Construct a KernelDef from name, params, return type, and body."),
    ("def annotation(self, key, value):", "Construct an Annotation from key and value."),
    ("def bool_true(self):", "Return True for a boolean true literal."),
    ("def bool_false(self):", "Return False for a boolean false literal."),
    ("def ident_value(self, name):", "Return an identifier value."),
    ("def array(self, *items):", "Collect array items into a list."),
    ("def map_entry(self, key, value):", "Construct a map entry tuple from key and value."),
    ("def map(self, *entries):", "Construct a dict from map entries."),
    ("def strategy_value(self, val):", "Return a strategy value."),
    ("def strategy_kwarg(self, key, value):", "Construct a strategy keyword argument tuple."),
    ("def strategy_args(self, *kwargs):", "Collect strategy keyword arguments into a dict."),
    ("def strategy_action(self, action_name, args):", "Construct a strategy action tuple from name and arguments."),
    ("def strategy_stmt(self, *parts):", "Construct a StrategyAction with optional annotation."),
    ("def strategy_body(self, *stmts):", "Collect strategy statements into a list."),
    ("def strategy_def(self, name, target, body):", "Construct a StrategyDef from name, target, and body."),
    ("def import_stmt(self, path, alias):", "Construct an ImportStmt from path and alias."),
    ("def start(self, *items):", "Construct the top-level Program from parsed items."),
])

# ─── arke/ir/strategy.py ───
total += process_file("arke/ir/strategy.py", [
    ("def decision_count(self) -> int:", "Return the number of optimization decisions."),
    ("def tile(self, loop: str, factors: list[int],", "Add a tile decision for the given loop with specified factors."),
    ("def reorder(self, order: list[str],", "Add a reorder decision with the specified loop order."),
    ("def fuse(self, ops: list[str], fusion_type: str", "Add a fusion decision for the specified operators."),
    ("def parallel(self, loops: list[str], mapping: dict[str, str],", "Add a parallelism decision mapping loops to hardware dimensions."),
    ("def place(self, tensor: str, memory: str,", "Add a memory placement decision for a tensor."),
    ("def to_dict(self) -> dict:", "Serialize the strategy IR to a plain dict."),
    ("def to_json(self, indent: int = 2) -> str:", "Serialize the strategy IR to a JSON string."),
    ("def to_file(self, path: str) -> None:", "Save the strategy IR to a JSON file."),
    ("def from_dict(cls, data: dict) -> StrategyIR:", "Deserialize a StrategyIR from a plain dict."),
    ("def from_json(cls, json_str: str) -> StrategyIR:", "Deserialize a StrategyIR from a JSON string."),
])

# ─── arke/ir/semantic.py ───
total += process_file("arke/ir/semantic.py", [
    ("def to_tensor_desc(self) -> TensorDesc:", "Convert this parameter to a TensorDesc."),
    # ParamRef
    ("class ParamRef:\n    \"\"\"Reference to a kernel parameter.\"\"\"\n    name: str\n\n    def to_dict(self) -> dict:", "Serialize this parameter reference to a dict."),
    ("def from_dict(cls, d: dict) -> ParamRef:", "Deserialize a ParamRef from a dict."),
    # NodeRef
    ("class NodeRef:\n    \"\"\"Reference to a previous node's output.\"\"\"\n    id: str\n\n    def to_dict(self) -> dict:", "Serialize this node reference to a dict."),
    ("def from_dict(cls, d: dict) -> NodeRef:", "Deserialize a NodeRef from a dict."),
])

# ─── arke/parser/ast_nodes.py ───
total += process_file("arke/parser/ast_nodes.py", [
    ("def get_kernel(self, name: str) -> KernelDef | None:", "Get a kernel definition by name, or None if not found."),
    ("def get_strategy(self, name: str) -> StrategyDef | None:", "Get a strategy definition by name, or None if not found."),
])

# ─── arke/learn/trajectory.py ───
total += process_file("arke/learn/trajectory.py", [
    ("def to_dict(self) -> dict[str, Any]:", "Serialize this trajectory record to a dict."),
    ("def __init__(self, path: str | Path):", "Initialize the trajectory writer for the given file path."),
    ("def flush(self) -> None:", "Flush buffered trajectory data to disk."),
    ("def close(self) -> None:", "Close the trajectory file."),
])

# ─── arke/engine/accuracy.py ───
# AccuracyMetrics.to_dict
total += process_file("arke/engine/accuracy.py", [
    # The to_dict methods: search for unique context
    ("    total_elements: int = 0\n    nontrivial_elements: int = 0  # |ref| > epsilon\n\n    def to_dict(self) -> dict[str, Any]:", "Serialize accuracy metrics to a dict."),
    # CompareConfig.to_dict
    ("    num_trials: int = 5\n\n    def to_dict(self) -> dict[str, Any]:", "Serialize comparison config to a dict."),
    # CompareResult.to_dict
    ("    timestamp: float = field(default_factory=time.time)\n\n    def to_dict(self) -> dict[str, Any]:", "Serialize comparison result to a dict."),
    # BenchmarkResult.to_dict  
    ("    environment: dict[str, Any] = field(default_factory=dict)\n\n    def to_dict(self) -> dict[str, Any]:", "Serialize benchmark result to a dict."),
    # med, max_, sum_
    ("def med(attr: str) -> float:", "Compute median of an attribute across metrics."),
    ("def max_(attr: str) -> float:", "Compute max of an attribute across metrics."),
    ("def sum_(attr: str) -> int:", "Compute sum of an attribute across metrics."),
])

# ─── arke/engine/env.py ───
total += process_file("arke/engine/env.py", [
    ("def __init__(self, semantic: SemanticIR, target_hw: str):", "Initialize the environment with a semantic IR and hardware target."),
])

# ─── arke/engine/reference_sources.py ───
total += process_file("arke/engine/reference_sources.py", [
    # NumPyCPUSource methods
    ("def generate_reference(\n        self,\n        semantic_ir: SemanticIR,\n        inputs: dict[str, np.ndarray],\n    ) -> np.ndarray:\n        from arke.engine.numerical_check", "Compute reference output using NumPy at the configured precision."),
    ("def generate_inputs(\n        self,\n        semantic_ir: SemanticIR,\n        seed: int = 42,\n        input_type: str = \"normal\",\n    ) -> dict[str, np.ndarray]:\n        rng = np.random.RandomState", "Generate random input tensors for accuracy comparison."),
    # TorchGPUSource.__init__
    ("def __init__(self, device: str = \"cuda\"):\n        self.device = device", "Initialize with the specified GPU device."),
    # TorchGPUSource.generate_reference  
    ("def generate_reference(\n        self,\n        semantic_ir: SemanticIR,\n        inputs: dict[str, np.ndarray],\n    ) -> np.ndarray:\n        import torch", "Compute reference output using PyTorch on GPU."),
    # TorchGPUSource.generate_inputs
    ("def generate_inputs(\n        self,\n        semantic_ir: SemanticIR,\n        seed: int = 42,\n        input_type: str = \"normal\",\n    ) -> dict[str, np.ndarray]:\n        # Delegate", "Generate random input tensors, delegating to NumPy source."),
    # CustomSource.__init__
    ("def __init__(self, ref_data: dict[str, np.ndarray] | None = None):", "Initialize with optional pre-computed reference data."),
    # CustomSource.set_reference
    ("def set_reference(self, inputs: dict[str, np.ndarray], output: np.ndarray) -> None:", "Set the reference inputs and expected output."),
    # CustomSource.generate_reference
    ("def generate_reference(\n        self,\n        semantic_ir: SemanticIR,\n        inputs: dict[str, np.ndarray],\n    ) -> np.ndarray:\n        if \"output\" not in", "Return the stored reference output."),
    # CustomSource.generate_inputs
    ("def generate_inputs(\n        self,\n        semantic_ir: SemanticIR,\n        seed: int = 42,\n        input_type: str = \"normal\",\n    ) -> dict[str, np.ndarray]:\n        # Return stored", "Return stored inputs or generate random ones as fallback."),
])

# ─── arke/engine/validator.py ───
total += process_file("arke/engine/validator.py", [
    ("violations(self) -> list[str]:", "Return all violation messages from all checks."),
    ("def validate(\n        self,\n        semantic: SemanticIR,", "Validate strategy IR against hardware constraints and IR invariants."),
])

# ─── arke/agent/session.py ───
total += process_file("arke/agent/session.py", [
    ("def decisions_remaining(self) -> int:", "Return the number of decisions remaining in the budget."),
    ("def compiles_remaining(self) -> int:", "Return the number of compiles remaining in the budget."),
    ("def exhausted(self) -> bool:", "Return True if the decision budget is exhausted."),
    ("def should_warn(self) -> bool:", "Return True if the budget warning threshold has been reached."),
    ("def use_decision(self) -> None:", "Record usage of one decision from the budget."),
    ("def use_compile(self) -> None:", "Record usage of one compile from the budget."),
    ("def to_dict(self) -> dict[str, Any]:\n        return {\n            \"decisions_used\":", "Serialize budget state to a dict."),
    ("def tool_schemas(self) -> list[dict[str, Any]]:", "Return the tool schemas for LLM tool-use."),
    ("def system_prompt(self) -> str:", "Return the system prompt from the conversation."),
    ("def duration_seconds(self) -> float:", "Return elapsed seconds since session creation."),
])

# ─── arke/agent/llm_config.py ───
total += process_file("arke/agent/llm_config.py", [
    ("def get_primary(self) -> tuple[ProviderConfig, ModelConfig]:", "Resolve the primary provider and model config."),
])

# ─── arke/agent/runner.py ───
total += process_file("arke/agent/runner.py", [
    ("def __init__(self, config: LLMConfig, timeout: float = 300.0):", "Initialize the LLM runner with config and timeout."),
])

# ─── arke/backend/triton_backend.py ───
total += process_file("arke/backend/triton_backend.py", [
    ("def __init__(self) -> None:\n        self._engine = TritonTemplateEngine()", "Initialize with Triton template engine and compiler."),
])

# ─── arke/backend/triton_template_engine.py ───
total += process_file("arke/backend/triton_template_engine.py", [
    ("def __init__(self) -> None:\n        self._env = Environment(", "Initialize the Jinja2 template environment."),
])

# ─── arke/ir/builder.py ───
total += process_file("arke/ir/builder.py", [
    ("def __init__(self, name: str):\n        self.name = name\n        self._params", "Initialize a kernel builder with the given kernel name."),
])

# ─── arke/pipeline.py ───
total += process_file("arke/pipeline.py", [
    ("def __init__(self) -> None:\n        self.numerical_validator = NumericalValidator()", "Initialize the pipeline with a numerical validator."),
])

# ─── arke/integration/custom_ops.py ───
total += process_file("arke/integration/custom_ops.py", [
    ("def make_conv1d_fwd(mod):\n                def forward(x):\n                    out = torch.ops.arke.matmul(x, mod.weight)", "Create a forward function that uses Arke matmul for Conv1D."),
    ("            def make_linear_fwd(mod):\n                def forward(x):\n                    out = torch.ops.arke.matmul(x, mod.weight.t()", "Create a forward function that uses Arke matmul for Linear."),
    ("def arke_eager_attention_forward(", "Arke-patched eager attention using custom softmax op."),
])

# ─── arke/integration/gpt2_e2e.py ───
total += process_file("arke/integration/gpt2_e2e.py", [
    ("def make_conv1d_fwd(mod, c):\n                def forward(x):\n                    out = c.matmul(x, mod.weight)", "Create a forward function using cached Arke matmul for Conv1D."),
    ("def make_linear_fwd(mod, c):\n                def forward(x):\n                    out = c.matmul(x, mod.weight.t()", "Create a forward function using cached Arke matmul for Linear."),
    ("def main():\n    parser = argparse.ArgumentParser(", "CLI entry point for GPT-2 E2E benchmarking."),
])

# ─── arke/integration/kernel_cache.py ───
total += process_file("arke/integration/kernel_cache.py", [
    ("def __init__(self):\n        self._backend = TritonBackend()", "Initialize the kernel cache with empty matmul and softmax caches."),
    ("def stats(self) -> dict:", "Return cache statistics with counts of compiled shapes."),
])

# Need to handle forward() inner functions in custom_ops.py and gpt2_e2e.py
# Those are nested, let's handle them
total += process_file("arke/integration/custom_ops.py", [
    # The forward inside make_conv1d_fwd
    ("def make_conv1d_fwd(mod):\n", ""),  # already handled above; skip
])

print(f"\nTotal docstrings added: {total}")
