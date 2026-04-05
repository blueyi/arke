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

        # --- OT4: Attention (check first, most specific) ---
        if "paged_attention" in ops:
            return "paged_attention.py.j2", "paged_attention"
        if "multi_latent_attention" in ops:
            return "mla.py.j2", "multi_latent_attention"
        attention_ops = {"flash_attention", "grouped_query_attention", "cross_attention"}
        if any(op in attention_ops for op in ops):
            first_attn = next(op for op in ops if op in attention_ops)
            return "flash_attention.py.j2", first_attn

        # --- OT3: Fused Compound ---
        gated_ops = {"swiglu", "geglu"}
        if any(op in gated_ops for op in ops):
            first_gate = next(op for op in ops if op in gated_ops)
            return "gated_activation.py.j2", first_gate
        if "rope" in ops:
            return "rope.py.j2", "rope"
        if "fused_linear_cross_entropy" in ops:
            return "cross_entropy.py.j2", "fused_linear_cross_entropy"
        if "cross_entropy" in ops:
            return "cross_entropy.py.j2", "cross_entropy"
        quant_ops = {"quantize_per_token", "dequantize_per_channel"}
        if any(op in quant_ops for op in ops):
            first_quant = next(op for op in ops if op in quant_ops)
            return "quantize.py.j2", first_quant

        # --- OT2: Data Movement & Dense ---
        if "grouped_matmul" in ops:
            return "grouped_matmul.py.j2", "grouped_matmul"
        if "batch_matmul" in ops:
            return "batch_matmul.py.j2", "batch_matmul"
        if "matmul" in ops:
            return "matmul.py.j2", "matmul"

        transpose_ops = {"transpose", "permute", "copy_"}
        if any(op in transpose_ops for op in ops):
            first_trans = next(op for op in ops if op in transpose_ops)
            return "transpose.py.j2", first_trans

        data_move_ops = {"concat", "split"}
        if any(op in data_move_ops for op in ops):
            first_dm = next(op for op in ops if op in data_move_ops)
            return "data_movement.py.j2", first_dm

        index_ops = {"gather", "scatter", "embedding"}
        if any(op in index_ops for op in ops):
            first_idx = next(op for op in ops if op in index_ops)
            return "index_ops.py.j2", first_idx

        # --- OT0/OT1: Existing ops ---
        if "softmax" in ops:
            return "softmax.py.j2", "softmax"

        # Fused RMSNorm + Residual (check before plain norm ops)
        if "rmsnorm_residual" in ops:
            return "rmsnorm_residual.py.j2", "rmsnorm_residual"

        norm_ops = {"layernorm", "rmsnorm"}
        if any(op in norm_ops for op in ops):
            first_norm = next(op for op in ops if op in norm_ops)
            return "layernorm.py.j2", first_norm

        # Reduction ops (OT1)
        reduction_ops = {"reduce_sum", "reduce_max", "reduce_mean", "argmax"}
        if any(op in reduction_ops for op in ops):
            first_red = next(op for op in ops if op in reduction_ops)
            return "reduction.py.j2", first_red

        # Top-K
        if "topk" in ops:
            return "topk.py.j2", "topk"

        # Cumulative sum
        if "cumsum" in ops:
            return "cumsum.py.j2", "cumsum"

        # Cast
        if "cast" in ops:
            return "cast.py.j2", "cast"

        # Binary elementwise ops
        binary_ops = {"add", "mul", "where_"}
        if any(op in binary_ops for op in ops):
            first_bin = next(op for op in ops if op in binary_ops)
            return "elementwise_binary.py.j2", first_bin

        # Unary elementwise ops (activations)
        unary_ops = {"relu", "gelu", "silu", "tanh", "sigmoid", "exp", "neg", "rsqrt"}
        if any(op in unary_ops for op in ops):
            first_ew = next(op for op in ops if op in unary_ops)
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

        # Extract launch_config from strategy decisions
        launch_config = self._extract_launch_config(strategy)
        if launch_config:
            ctx["launch_config"] = launch_config

        # Extract autotune from strategy decisions
        autotune_config = self._extract_autotune(strategy)
        if autotune_config:
            ctx["autotune"] = autotune_config

        if primary_op == "matmul":
            tile = self._extract_tile_params(strategy)
            ctx.update(tile)
            ctx["fused_activation"] = self._detect_fused_activation(
                semantic, strategy
            )
            ctx["output_dtype"] = self._resolve_output_dtype(semantic)
        elif primary_op == "batch_matmul":
            tile = self._extract_tile_params(strategy)
            ctx.update(tile)
        elif primary_op == "grouped_matmul":
            tile = self._extract_tile_params(strategy)
            ctx.update(tile)
        elif primary_op == "softmax":
            # softmax template needs kernel_name only; BLOCK_N is computed at runtime
            pass
        elif primary_op in ("layernorm", "rmsnorm"):
            ctx["norm_type"] = primary_op
        elif primary_op in ("relu", "gelu", "silu", "tanh", "sigmoid", "exp", "neg", "rsqrt"):
            ctx["activation"] = primary_op
        elif primary_op in ("add", "mul", "where_"):
            ctx["binary_op"] = primary_op
        elif primary_op == "cast":
            ctx["target_dtype"] = self._resolve_output_dtype(semantic)
        elif primary_op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax"):
            ctx["reduction_op"] = primary_op
        elif primary_op == "rmsnorm_residual":
            pass  # rmsnorm_residual template needs only kernel_name
        elif primary_op == "cumsum":
            pass  # cumsum template needs only kernel_name
        elif primary_op == "topk":
            pass  # topk template needs only kernel_name
        # --- OT2: Data Movement ---
        elif primary_op in ("transpose", "permute"):
            ctx["transpose_op"] = primary_op
        elif primary_op == "copy_":
            ctx["transpose_op"] = "copy_"
        elif primary_op == "concat":
            ctx["data_op"] = "concat"
        elif primary_op == "split":
            ctx["data_op"] = "split"
        elif primary_op == "gather":
            ctx["index_op"] = "gather"
        elif primary_op == "scatter":
            ctx["index_op"] = "scatter"
        elif primary_op == "embedding":
            ctx["index_op"] = "embedding"
        # --- OT3: Fused Compound ---
        elif primary_op == "swiglu":
            ctx["gate_activation"] = "silu"
        elif primary_op == "geglu":
            ctx["gate_activation"] = "gelu"
        elif primary_op == "rope":
            pass  # rope template needs only kernel_name
        elif primary_op == "cross_entropy":
            ctx["fused_linear"] = False
        elif primary_op == "fused_linear_cross_entropy":
            ctx["fused_linear"] = True
        elif primary_op == "quantize_per_token":
            ctx["quant_op"] = "quantize"
        elif primary_op == "dequantize_per_channel":
            ctx["quant_op"] = "dequantize"
        # --- OT4: Attention ---
        elif primary_op in ("flash_attention", "grouped_query_attention", "cross_attention"):
            ctx["causal"] = primary_op != "cross_attention"
            if primary_op == "grouped_query_attention":
                ctx["gqa_groups"] = self._extract_gqa_groups(strategy)
            else:
                ctx["gqa_groups"] = 1
        elif primary_op == "multi_latent_attention":
            pass  # mla template needs only kernel_name
        elif primary_op == "paged_attention":
            pass  # paged_attention template needs only kernel_name

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

    # ─── Strategy decision extraction ─────────────────────────

    def _extract_launch_config(self, strategy: StrategyIR) -> dict | None:
        """Extract launch_config from strategy decisions."""
        for decision in strategy.decisions:
            if decision.kind == "launch_config":
                return decision.params
        return None

    def _extract_autotune(self, strategy: StrategyIR) -> dict | None:
        """Extract autotune config from strategy decisions."""
        for decision in strategy.decisions:
            if decision.kind == "autotune":
                return decision.params
        return None

    # ─── Helpers ───────────────────────────────────────────────

    def _extract_gqa_groups(self, strategy: StrategyIR) -> int:
        """Extract GQA group count from strategy decisions.

        Looks for a decision with kind='gqa' and params containing 'groups'.
        Defaults to 1 (standard MHA).
        """
        for decision in strategy.decisions:
            if decision.kind == "gqa":
                return decision.params.get("groups", 1)
            # Also check generic params
            if "gqa_groups" in decision.params:
                return decision.params["gqa_groups"]
        return 1

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
