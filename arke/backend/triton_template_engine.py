# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — Triton Template Engine.

Translates Strategy IR → Triton source code via Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

# Directory containing .j2 templates
_TEMPLATE_DIR = Path(__file__).parent / "triton_templates"

# Map from dtype strings to Triton type expressions
_DTYPE_MAP: dict[str, str] = {
    "float32": "tl.float32",
    "float16": "tl.float16",
    "bfloat16": "tl.bfloat16",
    "int32": "tl.int32",
    "int64": "tl.int64",
}

# Default tiling parameters (fallback when strategy doesn't specify)
_DEFAULT_TILE = {"block_m": 64, "block_n": 64, "block_k": 32}


class TritonTemplateEngine:
    """Translates Strategy IR → Triton source code via Jinja2 templates."""

    def __init__(self) -> None:
        """Initialize the Jinja2 template environment."""
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ─── Public API ────────────────────────────────────────────

    def translate(self, semantic: SemanticIR, strategy: StrategyIR) -> str:
        """Generate Triton Python code from IR + strategy."""
        # 1. Determine which template to use
        template_name, primary_op = self._select_template(semantic)
        template = self._env.get_template(template_name)

        # 2. Build render context
        ctx = self._build_context(semantic, strategy, primary_op)

        # 3. Render
        return template.render(**ctx)

    # ─── Template selection ────────────────────────────────────

    def _select_template(self, semantic: SemanticIR) -> tuple[str, str]:
        """Determine which template to use based on ops in the IR.

        Returns (template_filename, primary_op_name).
        """
        ops = [node.op for node in semantic.nodes]

        if "matmul" in ops or "batch_matmul" in ops:
            return "matmul.py.j2", "matmul"
        if "softmax" in ops:
            return "softmax.py.j2", "softmax"

        norm_ops = {"layernorm", "rmsnorm"}
        if any(op in norm_ops for op in ops):
            first_norm = next(op for op in ops if op in norm_ops)
            return "layernorm.py.j2", first_norm

        elementwise_ops = {"relu", "gelu", "silu", "add", "mul"}
        if any(op in elementwise_ops for op in ops):
            first_ew = next(op for op in ops if op in elementwise_ops)
            return "elementwise.py.j2", first_ew

        # Fallback: pick first op and hope we have a template
        first_op = ops[0] if ops else "matmul"
        return f"{first_op}.py.j2", first_op

    # ─── Context building ──────────────────────────────────────

    def _build_context(
        self, semantic: SemanticIR, strategy: StrategyIR, primary_op: str
    ) -> dict:
        """Build the Jinja2 template context dict."""
        kernel_name = semantic.kernel_id or f"arke_{primary_op}"
        # Sanitize to valid Python identifier
        kernel_name = kernel_name.replace("-", "_").replace(".", "_")

        ctx: dict = {"kernel_name": kernel_name}

        if primary_op == "matmul":
            tile = self._extract_tile_params(strategy)
            ctx.update(tile)
            ctx["fused_activation"] = self._detect_fused_activation(
                semantic, strategy
            )
            ctx["output_dtype"] = self._resolve_output_dtype(semantic)
        elif primary_op == "softmax":
            # softmax template needs kernel_name only; BLOCK_N is computed at runtime
            pass
        elif primary_op in ("layernorm", "rmsnorm"):
            ctx["norm_type"] = primary_op
        elif primary_op in ("relu", "gelu", "silu"):
            ctx["activation"] = primary_op

        return ctx

    # ─── Tile parameter extraction ─────────────────────────────

    def _extract_tile_params(self, strategy: StrategyIR) -> dict:
        """Extract BLOCK_M, BLOCK_N, BLOCK_K from tile decisions.

        Scans strategy decisions for kind=='tile' and maps loop names
        (i/m → block_m, j/n → block_n, k → block_k) to tile sizes.
        Returns a dict with keys block_m, block_n, block_k.
        """
        result = dict(_DEFAULT_TILE)  # start from defaults

        loop_to_block = {
            "i": "block_m",
            "m": "block_m",
            "j": "block_n",
            "n": "block_n",
            "k": "block_k",
        }

        for decision in strategy.decisions:
            if decision.kind != "tile":
                continue
            loop = decision.params.get("loop", "")
            factors = decision.params.get("factors", [])
            block_key = loop_to_block.get(loop)
            if block_key and factors:
                # Use the first factor as the block size
                result[block_key] = factors[0]

        return result

    # ─── Fused activation detection ────────────────────────────

    def _detect_fused_activation(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> str | None:
        """Check if an epilogue activation is fused.

        Looks for:
        1. A 'fuse' decision in the strategy whose type is 'epilogue'
        2. The fused ops include an activation (relu, gelu)
        Also checks if the SemanticIR has a fusion group of type 'epilogue'.
        """
        # Check strategy decisions for fuse-type decisions
        fused_ops: set[str] = set()
        for decision in strategy.decisions:
            if decision.kind == "fuse":
                # LLM may use 'nodes' or 'ops' key
                ops = decision.params.get("ops", []) or decision.params.get("nodes", [])
                ftype = decision.params.get("type", "")
                if ftype == "epilogue":
                    fused_ops.update(ops)

        # Check semantic IR fusion groups
        for fg in semantic.fusion_groups:
            if fg.fusion_type == "epilogue":
                fused_ops.update(fg.nodes)

        # Resolve node IDs to op names
        activation_ops = {"relu", "gelu"}
        for node in semantic.nodes:
            if node.id in fused_ops and node.op in activation_ops:
                return node.op
            if node.op in fused_ops & activation_ops:
                return node.op

        return None

    # ─── Helpers ───────────────────────────────────────────────

    def _resolve_output_dtype(self, semantic: SemanticIR) -> str:
        """Resolve the output dtype for the kernel.

        Uses return_type if set, otherwise infers from first param.
        """
        if semantic.return_type:
            return _DTYPE_MAP.get(semantic.return_type.dtype, "tl.float32")

        # Infer from first param
        if semantic.params:
            return _DTYPE_MAP.get(semantic.params[0].dtype, "tl.float32")

        return "tl.float32"
