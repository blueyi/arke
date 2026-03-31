# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Triton code generation — template rendering, tile extraction, fusion detection."""

from __future__ import annotations

import ast

import pytest

from arke.backend.triton_template_engine import TritonTemplateEngine
from arke.ir.semantic import (
    Edge,
    FusionGroup,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.strategy import StrategyIR


# ============================================================
# Helpers — build IR fixtures
# ============================================================


def _make_matmul_semantic(
    kernel_id: str = "test_matmul",
    dtype: str = "float32",
    M: int = 512,
    N: int = 512,
    K: int = 512,
) -> SemanticIR:
    """Build a minimal SemanticIR for matmul."""
    ir = SemanticIR(kernel_id=kernel_id)
    ir.add_param(Param(name="A", shape=[M, K], dtype=dtype))
    ir.add_param(Param(name="B", shape=[K, N], dtype=dtype))
    ir.return_type = TensorDesc(shape=[M, N], dtype=dtype)
    ir.add_node(Node(
        id="matmul_0",
        op="matmul",
        inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
        output=TensorDesc(shape=[M, N], dtype=dtype),
        semantics=Semantics(
            computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
            index_vars=["i", "j", "k"],
            reduction_axes=["k"],
            properties=["associative", "distributive"],
        ),
    ))
    ir.return_node = "matmul_0"
    return ir


def _make_matmul_relu_semantic(
    kernel_id: str = "test_matmul_relu",
) -> SemanticIR:
    """Build a SemanticIR for matmul + relu (epilogue fused)."""
    ir = _make_matmul_semantic(kernel_id=kernel_id)
    ir.add_node(Node(
        id="relu_0",
        op="relu",
        inputs={"X": NodeRef(id="matmul_0")},
        output=TensorDesc(shape=[512, 512], dtype="float32"),
        semantics=Semantics(
            computation="Y = max(X, 0)",
            properties=["elementwise", "monotonic"],
        ),
    ))
    ir.add_edge(Edge(from_node="matmul_0", to_node="relu_0", tensor_name="C"))
    ir.return_node = "relu_0"
    ir.add_fusion_group(FusionGroup(
        id="fg_0",
        nodes=["matmul_0", "relu_0"],
        fusion_type="epilogue",
        reason="relu is elementwise, can fuse as matmul epilogue",
    ))
    return ir


def _make_softmax_semantic(
    kernel_id: str = "test_softmax",
    M: int = 128,
    N: int = 1024,
) -> SemanticIR:
    """Build a minimal SemanticIR for softmax."""
    ir = SemanticIR(kernel_id=kernel_id)
    ir.add_param(Param(name="X", shape=[M, N], dtype="float32"))
    ir.return_type = TensorDesc(shape=[M, N], dtype="float32")
    ir.add_node(Node(
        id="softmax_0",
        op="softmax",
        inputs={"X": ParamRef(name="X")},
        output=TensorDesc(shape=[M, N], dtype="float32"),
        semantics=Semantics(
            computation="Y[i,j] = exp(X[i,j]) / sum(exp(X[i,:]), axis=j)",
            index_vars=["i", "j"],
            reduction_axes=["j"],
            properties=["row-wise"],
        ),
    ))
    ir.return_node = "softmax_0"
    return ir


def _make_strategy_with_tiles(
    kernel_id: str = "test_matmul",
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
) -> StrategyIR:
    """Build a StrategyIR with tile decisions."""
    s = StrategyIR(kernel_id=kernel_id, target_hw="nvidia_ampere")
    s.tile("i", [block_m], rationale="Tile M dimension")
    s.tile("j", [block_n], rationale="Tile N dimension")
    s.tile("k", [block_k], rationale="Tile K dimension")
    return s


def _make_strategy_with_fused_relu(kernel_id: str = "test_matmul_relu") -> StrategyIR:
    """Build a StrategyIR that fuses relu as epilogue."""
    s = _make_strategy_with_tiles(kernel_id=kernel_id)
    s.fuse(["matmul_0", "relu_0"], fusion_type="epilogue",
           rationale="Fuse relu into matmul epilogue")
    return s


# ============================================================
# Tests — Template Rendering
# ============================================================


class TestMatmulTemplateRendering:
    """Test matmul Jinja2 template rendering."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()

    def test_basic_matmul_renders(self) -> None:
        """Basic matmul without fusion should produce valid code."""
        semantic = _make_matmul_semantic()
        strategy = _make_strategy_with_tiles()
        code = self.engine.translate(semantic, strategy)

        assert "import triton" in code
        assert "import triton.language as tl" in code
        assert "test_matmul_kernel" in code
        assert "test_matmul" in code
        assert "BLOCK_M=128" in code
        assert "BLOCK_N=128" in code
        assert "BLOCK_K=32" in code
        # No activation
        assert "ACTIVATION" not in code
        assert "tl.maximum" not in code

    def test_matmul_valid_python(self) -> None:
        """Generated matmul code should be parseable Python."""
        semantic = _make_matmul_semantic()
        strategy = _make_strategy_with_tiles()
        code = self.engine.translate(semantic, strategy)
        # ast.parse will raise SyntaxError if invalid
        tree = ast.parse(code)
        assert tree is not None

    def test_matmul_with_relu_fusion(self) -> None:
        """Matmul + relu fusion should include activation code."""
        semantic = _make_matmul_relu_semantic()
        strategy = _make_strategy_with_fused_relu()
        code = self.engine.translate(semantic, strategy)

        assert "tl.maximum(acc, 0.0)" in code
        assert "ACTIVATION" in code

    def test_matmul_relu_valid_python(self) -> None:
        """Generated fused matmul+relu code should be parseable."""
        semantic = _make_matmul_relu_semantic()
        strategy = _make_strategy_with_fused_relu()
        code = self.engine.translate(semantic, strategy)
        tree = ast.parse(code)
        assert tree is not None

    def test_different_tile_sizes(self) -> None:
        """Different tile sizes should appear in the generated code."""
        semantic = _make_matmul_semantic()
        strategy = _make_strategy_with_tiles(block_m=64, block_n=256, block_k=16)
        code = self.engine.translate(semantic, strategy)

        assert "BLOCK_M=64" in code
        assert "BLOCK_N=256" in code
        assert "BLOCK_K=16" in code

    def test_default_tile_sizes(self) -> None:
        """Without tile decisions, should use defaults (64, 64, 32)."""
        semantic = _make_matmul_semantic()
        strategy = StrategyIR(kernel_id="test_matmul")  # no decisions
        code = self.engine.translate(semantic, strategy)

        assert "BLOCK_M=64" in code
        assert "BLOCK_N=64" in code
        assert "BLOCK_K=32" in code

    def test_output_dtype_float16(self) -> None:
        """Float16 inputs should produce float16 output dtype."""
        semantic = _make_matmul_semantic(dtype="float16")
        strategy = _make_strategy_with_tiles()
        code = self.engine.translate(semantic, strategy)

        assert "tl.float16" in code


class TestSoftmaxTemplateRendering:
    """Test softmax Jinja2 template rendering."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()

    def test_softmax_renders(self) -> None:
        """Softmax template should produce valid code."""
        semantic = _make_softmax_semantic()
        strategy = StrategyIR(kernel_id="test_softmax")
        code = self.engine.translate(semantic, strategy)

        assert "import triton" in code
        assert "test_softmax_kernel" in code
        assert "tl.exp" in code
        assert "tl.max" in code
        assert "tl.sum" in code

    def test_softmax_valid_python(self) -> None:
        """Generated softmax code should be parseable Python."""
        semantic = _make_softmax_semantic()
        strategy = StrategyIR(kernel_id="test_softmax")
        code = self.engine.translate(semantic, strategy)
        tree = ast.parse(code)
        assert tree is not None


# ============================================================
# Tests — Tile Parameter Extraction
# ============================================================


class TestTileParamExtraction:
    """Test _extract_tile_params from Strategy IR."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()

    def test_extracts_all_tiles(self) -> None:
        strategy = _make_strategy_with_tiles(block_m=128, block_n=64, block_k=16)
        params = self.engine._extract_tile_params(strategy)
        assert params["block_m"] == 128
        assert params["block_n"] == 64
        assert params["block_k"] == 16

    def test_defaults_when_no_tile_decisions(self) -> None:
        strategy = StrategyIR(kernel_id="test")
        params = self.engine._extract_tile_params(strategy)
        assert params["block_m"] == 64
        assert params["block_n"] == 64
        assert params["block_k"] == 32

    def test_partial_tiles(self) -> None:
        """Only some tile dimensions specified; others use defaults."""
        strategy = StrategyIR(kernel_id="test")
        strategy.tile("i", [256])
        params = self.engine._extract_tile_params(strategy)
        assert params["block_m"] == 256
        assert params["block_n"] == 64   # default
        assert params["block_k"] == 32   # default

    def test_m_n_loop_aliases(self) -> None:
        """Loop names 'm' and 'n' should also map correctly."""
        strategy = StrategyIR(kernel_id="test")
        strategy.tile("m", [32])
        strategy.tile("n", [48])
        params = self.engine._extract_tile_params(strategy)
        assert params["block_m"] == 32
        assert params["block_n"] == 48


# ============================================================
# Tests — Fused Activation Detection
# ============================================================


class TestFusedActivationDetection:
    """Test _detect_fused_activation logic."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()

    def test_detects_relu_from_strategy(self) -> None:
        semantic = _make_matmul_relu_semantic()
        strategy = _make_strategy_with_fused_relu()
        result = self.engine._detect_fused_activation(semantic, strategy)
        assert result == "relu"

    def test_detects_relu_from_fusion_group(self) -> None:
        """Should detect activation from SemanticIR fusion groups even
        without an explicit fuse decision in strategy."""
        semantic = _make_matmul_relu_semantic()
        strategy = _make_strategy_with_tiles()  # no fuse decision
        result = self.engine._detect_fused_activation(semantic, strategy)
        assert result == "relu"

    def test_no_activation_when_none_fused(self) -> None:
        semantic = _make_matmul_semantic()  # no relu node
        strategy = _make_strategy_with_tiles()
        result = self.engine._detect_fused_activation(semantic, strategy)
        assert result is None

    def test_detects_gelu_fusion(self) -> None:
        """Should detect gelu if fused."""
        ir = _make_matmul_semantic(kernel_id="matmul_gelu")
        ir.add_node(Node(
            id="gelu_0",
            op="gelu",
            inputs={"X": NodeRef(id="matmul_0")},
            output=TensorDesc(shape=[512, 512], dtype="float32"),
            semantics=Semantics(computation="Y = X * Phi(X)", properties=["elementwise"]),
        ))
        ir.add_fusion_group(FusionGroup(
            id="fg_gelu",
            nodes=["matmul_0", "gelu_0"],
            fusion_type="epilogue",
            reason="gelu epilogue fusion",
        ))
        strategy = _make_strategy_with_tiles(kernel_id="matmul_gelu")
        result = self.engine._detect_fused_activation(ir, strategy)
        assert result == "gelu"


# ============================================================
# Tests — Backend Registration
# ============================================================


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestBackendRegistration:
    """Test that TritonBackend is properly registered."""

    def test_triton_backend_registered(self) -> None:
        # Import triggers registration
        import arke.backend.triton_backend  # noqa: F401
        from arke.backend.base import get_backend, list_backends

        assert "triton" in list_backends()
        cls = get_backend("triton")
        assert cls.name == "triton"

    def test_triton_backend_translate(self) -> None:
        from arke.backend.triton_backend import TritonBackend

        backend = TritonBackend()
        semantic = _make_matmul_semantic()
        strategy = _make_strategy_with_tiles()
        code = backend.translate(semantic, strategy)
        assert "test_matmul_kernel" in code
