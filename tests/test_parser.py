# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Arke language parser."""

import pytest

from arke.ir.builder import KernelBuilder
from arke.parser.ast_nodes import (
    LetStmt,
)
from arke.parser.converter import ConversionError, ast_to_ir
from arke.parser.parser import parse_file, parse_string

# ============================================================
# Parser Tests
# ============================================================

class TestParseKernel:
    """Test kernel parsing."""

    def test_simple_matmul(self):
        src = '''
        kernel matmul(
            A: Tensor<[1024, 1024], f16>,
            B: Tensor<[1024, 1024], f16>
        ) -> Tensor<[1024, 1024], f16> {
            let C = matmul(A, B);
            return C;
        }
        '''
        prog = parse_string(src)
        assert len(prog.kernels) == 1
        k = prog.kernels[0]
        assert k.name == "matmul"
        assert len(k.params) == 2
        assert k.params[0].name == "A"
        assert k.params[0].type.shape == [1024, 1024]
        assert k.params[0].type.dtype == "f16"
        assert k.return_type.shape == [1024, 1024]
        assert len(k.body) == 2  # let + return

    def test_fused_ops(self):
        src = '''
        kernel fused(
            A: Tensor<[1024, 512], f16>,
            B: Tensor<[512, 2048], f16>
        ) -> Tensor<[1024, 2048], f16> {
            let C = matmul(A, B);
            let Y = relu(X=C);
            return Y;
        }
        '''
        prog = parse_string(src)
        k = prog.kernels[0]
        assert len(k.body) == 3  # 2 lets + return
        assert isinstance(k.body[0], LetStmt)
        assert k.body[0].name == "C"
        assert k.body[0].value.op == "matmul"

    def test_named_args(self):
        src = '''
        kernel test(X: Tensor<[100, 100], f32>) -> Tensor<[100, 100], f32> {
            let Y = relu(X=X);
            return Y;
        }
        '''
        prog = parse_string(src)
        k = prog.kernels[0]
        let_stmt = k.body[0]
        assert let_stmt.value.args == {"X": "X"}

    def test_positional_args(self):
        src = '''
        kernel test(
            A: Tensor<[64, 64], f16>,
            B: Tensor<[64, 64], f16>
        ) -> Tensor<[64, 64], f16> {
            let C = matmul(A, B);
            return C;
        }
        '''
        prog = parse_string(src)
        k = prog.kernels[0]
        let_stmt = k.body[0]
        # Positional args are stored as _pos_0, _pos_1
        assert "_pos_0" in let_stmt.value.args or "A" in let_stmt.value.args

    def test_f32_dtype(self):
        src = '''
        kernel test(X: Tensor<[256, 256], f32>) -> Tensor<[256, 256], f32> {
            let Y = relu(X=X);
            return Y;
        }
        '''
        prog = parse_string(src)
        assert prog.kernels[0].params[0].type.dtype == "f32"

    def test_comments(self):
        src = '''
        // This is a comment
        kernel test(
            X: Tensor<[100, 100], f16>  // inline comment
        ) -> Tensor<[100, 100], f16> {
            /* multi-line
               comment */
            let Y = softmax(X=X);
            return Y;
        }
        '''
        prog = parse_string(src)
        assert len(prog.kernels) == 1

    def test_multiple_dims(self):
        src = '''
        kernel test(
            X: Tensor<[4096, 64], f16>
        ) -> Tensor<[4096, 64], f16> {
            let Y = softmax(X=X);
            return Y;
        }
        '''
        prog = parse_string(src)
        assert prog.kernels[0].params[0].type.shape == [4096, 64]


class TestParseStrategy:
    """Test strategy parsing."""

    def test_basic_strategy(self):
        src = '''
        kernel matmul(
            A: Tensor<[1024, 1024], f16>,
            B: Tensor<[1024, 1024], f16>
        ) -> Tensor<[1024, 1024], f16> {
            let C = matmul(A, B);
            return C;
        }

        strategy matmul for target("nvidia_ampere") {
            tile(loop="i", factors=[64, 16])
                @rationale("Cache line optimization");
            place(tensor="A_tile", memory=shared)
                @rationale("A reused 16x");
        }
        '''
        prog = parse_string(src)
        assert len(prog.strategies) == 1
        s = prog.strategies[0]
        assert s.name == "matmul"
        assert s.target == "nvidia_ampere"
        assert len(s.actions) == 2
        assert s.actions[0].action == "tile"
        assert s.actions[0].annotation.key == "rationale"
        assert "Cache line" in s.actions[0].annotation.value


class TestParseImport:
    """Test import parsing."""

    def test_import(self):
        src = '''
        import "nvidia_ampere" as hw;
        kernel test(X: Tensor<[100, 100], f16>) -> Tensor<[100, 100], f16> {
            let Y = relu(X=X);
            return Y;
        }
        '''
        prog = parse_string(src)
        assert len(prog.imports) == 1
        assert prog.imports[0].path == "nvidia_ampere"
        assert prog.imports[0].alias == "hw"


# ============================================================
# Converter Tests
# ============================================================

class TestAstToIr:
    """Test AST → SemanticIR conversion."""

    def test_matmul_matches_builder(self):
        """Parser output should match KernelBuilder output."""
        src = '''
        kernel matmul(
            A: Tensor<[1024, 1024], f16>,
            B: Tensor<[1024, 1024], f16>
        ) -> Tensor<[1024, 1024], f16> {
            let C = matmul(A, B);
            return C;
        }
        '''
        ir_parsed = ast_to_ir(parse_string(src).kernels[0])

        b = KernelBuilder("matmul")
        b.param("A", [1024, 1024], "f16")
        b.param("B", [1024, 1024], "f16")
        m = b.op("matmul", A="A", B="B")
        b.returns(m, [1024, 1024], "f16")
        ir_built = b.build()

        assert ir_parsed.kernel_id == ir_built.kernel_id
        assert len(ir_parsed.params) == len(ir_built.params)
        assert len(ir_parsed.nodes) == len(ir_built.nodes)
        assert ir_parsed.return_node == ir_built.return_node

    def test_fused_matmul_relu(self):
        src = '''
        kernel fused_matmul_relu(
            A: Tensor<[1024, 512], f16>,
            B: Tensor<[512, 2048], f16>
        ) -> Tensor<[1024, 2048], f16> {
            let C = matmul(A, B);
            let Y = relu(X=C);
            return Y;
        }
        '''
        ir = ast_to_ir(parse_string(src).kernels[0])
        assert ir.kernel_id == "fused_matmul_relu"
        assert len(ir.nodes) == 2
        assert ir.nodes[0].op == "matmul"
        assert ir.nodes[1].op == "relu"

    def test_softmax(self):
        src = '''
        kernel my_softmax(
            X: Tensor<[4096, 4096], f16>
        ) -> Tensor<[4096, 4096], f16> {
            let Y = softmax(X=X);
            return Y;
        }
        '''
        ir = ast_to_ir(parse_string(src).kernels[0])
        assert ir.nodes[0].op == "softmax"
        assert ir.params[0].shape == [4096, 4096]

    def test_three_ops(self):
        """matmul → add → relu chain."""
        src = '''
        kernel chain(
            A: Tensor<[1024, 1024], f16>,
            B: Tensor<[1024, 1024], f16>,
            bias: Tensor<[1024, 1024], f16>
        ) -> Tensor<[1024, 1024], f16> {
            let C = matmul(A, B);
            let D = add(A=C, B=bias);
            let Y = relu(X=D);
            return Y;
        }
        '''
        ir = ast_to_ir(parse_string(src).kernels[0])
        assert len(ir.nodes) == 3
        assert [n.op for n in ir.nodes] == ["matmul", "add", "relu"]

    def test_undefined_var_error(self):
        src = '''
        kernel bad(
            A: Tensor<[100, 100], f16>
        ) -> Tensor<[100, 100], f16> {
            let Y = relu(X=Z);
            return Y;
        }
        '''
        with pytest.raises(ConversionError, match="Undefined variable 'Z'"):
            ast_to_ir(parse_string(src).kernels[0])

    def test_no_return_error(self):
        src = '''
        kernel bad(
            A: Tensor<[100, 100], f16>,
            B: Tensor<[100, 100], f16>
        ) -> Tensor<[100, 100], f16> {
            let C = matmul(A, B);
        }
        '''
        with pytest.raises(ConversionError, match="no return"):
            ast_to_ir(parse_string(src).kernels[0])


# ============================================================
# File Parsing Tests
# ============================================================

class TestParseFiles:
    """Test parsing .ak example files."""

    def test_01_matmul(self):
        prog = parse_file("examples/operators/01_matmul.ak")
        ir = ast_to_ir(prog.kernels[0])
        assert ir.kernel_id == "matmul"
        assert len(ir.params) == 2
        assert ir.nodes[0].op == "matmul"

    def test_02_softmax(self):
        prog = parse_file("examples/operators/02_softmax.ak")
        ir = ast_to_ir(prog.kernels[0])
        assert ir.kernel_id == "softmax"
        assert ir.nodes[0].op == "softmax"

    def test_05_matmul_gelu(self):
        prog = parse_file("examples/operators/05_matmul_gelu.ak")
        ir = ast_to_ir(prog.kernels[0])
        assert [n.op for n in ir.nodes] == ["matmul", "gelu"]

    def test_15_flash_attention_keywords(self):
        """Verify flash_attention.ak uses 'strategy' keyword (not 'schedule')."""
        import pathlib
        src = pathlib.Path("examples/operators/15_flash_attention.ak").read_text()
        assert "\nstrategy " in src, "flash_attention.ak must use 'strategy' keyword, not 'schedule'"
        assert "\nschedule " not in src, "flash_attention.ak still uses deprecated 'schedule' keyword"
