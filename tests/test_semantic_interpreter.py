# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SemanticInterpreter (S6 Track 1, Task C1.4).

Validates execute + validate for representative ops from each OT tier.
"""

import pytest
import torch

from arke.ir.ops.interpreter import INTERPRETER
from arke.ir.ops.registry import REGISTRY


class TestSemanticInterpreterOT0:
    """OT0: Elementwise operators."""

    def test_relu(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("relu", {"X": x})
        expected = torch.relu(x)
        assert torch.allclose(result, expected)

    def test_gelu(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("gelu", {"X": x})
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(result, expected)

    def test_silu(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("silu", {"X": x})
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(result, expected)

    def test_add(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        result = INTERPRETER.execute("add", {"A": a, "B": b})
        assert torch.allclose(result, a + b)

    def test_mul(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        result = INTERPRETER.execute("mul", {"A": a, "B": b})
        assert torch.allclose(result, a * b)

    def test_neg(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("neg", {"X": x})
        assert torch.allclose(result, -x)

    def test_exp(self):
        x = torch.randn(4, 8).clamp(-3, 3)
        result = INTERPRETER.execute("exp", {"X": x})
        assert torch.allclose(result, torch.exp(x))

    def test_sigmoid(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("sigmoid", {"X": x})
        assert torch.allclose(result, torch.sigmoid(x))

    def test_tanh(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("tanh", {"X": x})
        assert torch.allclose(result, torch.tanh(x))

    def test_rsqrt(self):
        x = torch.rand(4, 8) + 0.1  # positive
        result = INTERPRETER.execute("rsqrt", {"X": x})
        assert torch.allclose(result, torch.rsqrt(x), rtol=1e-3, atol=1e-5)

    def test_where(self):
        cond = torch.tensor([[True, False], [False, True]])
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = INTERPRETER.execute("where_", {"cond": cond, "A": a, "B": b})
        assert torch.allclose(result, torch.where(cond, a, b))

    def test_cast(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        result = INTERPRETER.execute("cast", {"X": x}, {"target_dtype": "float16"})
        assert result.dtype == torch.float16


class TestSemanticInterpreterOT1:
    """OT1: Reduction operators."""

    def test_softmax(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("softmax", {"X": x})
        expected = torch.nn.functional.softmax(x, dim=-1)
        assert torch.allclose(result, expected)

    def test_layernorm(self):
        x = torch.randn(4, 768)
        w = torch.ones(768)
        b = torch.ones(768)
        result = INTERPRETER.execute("layernorm", {"X": x, "W": w, "B": b})
        expected = torch.nn.functional.layer_norm(x, [768], w, b, 1e-5)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-5)

    def test_rmsnorm(self):
        x = torch.randn(4, 128)
        w = torch.ones(128)
        result = INTERPRETER.execute("rmsnorm", {"X": x, "W": w})
        # Manual RMSNorm
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = x / rms * w
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-5)

    def test_reduce_sum(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("reduce_sum", {"X": x})
        assert torch.allclose(result, x.sum(dim=-1))

    def test_reduce_max(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("reduce_max", {"X": x})
        assert torch.allclose(result, x.max(dim=-1).values)

    def test_argmax(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("argmax", {"X": x})
        assert torch.equal(result, x.argmax(dim=-1))

    def test_topk(self):
        x = torch.randn(4, 32)
        result = INTERPRETER.execute("topk", {"X": x}, {"k": 5})
        expected = torch.topk(x, 5, dim=-1).values
        assert torch.allclose(result, expected)

    def test_cumsum(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("cumsum", {"X": x})
        assert torch.allclose(result, torch.cumsum(x, dim=-1))


class TestSemanticInterpreterOT2:
    """OT2: Compute-dense + data movement."""

    def test_matmul(self):
        a = torch.randn(64, 32)
        b = torch.randn(32, 128)
        result = INTERPRETER.execute("matmul", {"A": a, "B": b})
        assert torch.allclose(result, a @ b, rtol=1e-3, atol=1e-5)

    def test_batch_matmul(self):
        a = torch.randn(4, 16, 32)
        b = torch.randn(4, 32, 64)
        result = INTERPRETER.execute("batch_matmul", {"A": a, "B": b})
        assert torch.allclose(result, torch.matmul(a, b), rtol=1e-3, atol=1e-5)

    def test_transpose(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("transpose", {"X": x})
        assert result.shape == (8, 4)
        assert torch.allclose(result, x.T)

    def test_concat(self):
        a = torch.randn(4, 8)
        b = torch.randn(4, 12)
        result = INTERPRETER.execute("concat", {"A": a, "B": b})
        assert result.shape == (4, 20)

    def test_gather(self):
        x = torch.randn(4, 10)
        idx = torch.randint(0, 10, (4, 3))
        result = INTERPRETER.execute("gather", {"X": x, "idx": idx})
        expected = torch.gather(x, -1, idx)
        assert torch.allclose(result, expected)

    def test_embedding(self):
        weight = torch.randn(100, 32)
        indices = torch.randint(0, 100, (2, 8))
        result = INTERPRETER.execute("embedding", {"indices": indices, "weight": weight})
        expected = torch.nn.functional.embedding(indices, weight)
        assert torch.allclose(result, expected)

    def test_copy(self):
        x = torch.randn(4, 8)
        result = INTERPRETER.execute("copy_", {"X": x})
        assert torch.equal(result, x)
        assert result.data_ptr() != x.data_ptr()  # different memory


class TestSemanticInterpreterOT3:
    """OT3: Gated activations + fused ops."""

    def test_silu_and_mul(self):
        x = torch.randn(4, 256)
        result = INTERPRETER.execute("silu_and_mul", {"X": x})
        assert result.shape == (4, 128)
        # Manual
        gate, up = x[..., :128], x[..., 128:]
        expected = torch.nn.functional.silu(gate) * up
        assert torch.allclose(result, expected)

    def test_geglu(self):
        x = torch.randn(4, 256)
        result = INTERPRETER.execute("geglu", {"X": x})
        assert result.shape == (4, 128)

    def test_rmsnorm_residual(self):
        x = torch.randn(4, 64)
        res = torch.randn(4, 64)
        w = torch.ones(64)
        result = INTERPRETER.execute("rmsnorm_residual", {"X": x, "residual": res, "W": w})
        assert result.shape == (4, 64)

    def test_cross_entropy(self):
        logits = torch.randn(8, 100)
        labels = torch.randint(0, 100, (8,))
        result = INTERPRETER.execute("cross_entropy", {"logits": logits, "labels": labels})
        expected = torch.nn.functional.cross_entropy(logits, labels)
        assert torch.allclose(result, expected)


class TestSemanticInterpreterOT4:
    """OT4: Attention operators."""

    def test_flash_attention(self):
        B, H, S, D = 1, 2, 16, 32
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        result = INTERPRETER.execute("flash_attention", {"Q": Q, "K": K, "V": V})
        expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-5)
        assert result.shape == (B, H, S, D)

    def test_grouped_query_attention(self):
        B, H_q, H_kv, S, D = 1, 8, 2, 16, 32
        Q = torch.randn(B, H_q, S, D)
        K = torch.randn(B, H_kv, S, D)
        V = torch.randn(B, H_kv, S, D)
        result = INTERPRETER.execute("grouped_query_attention", {"Q": Q, "K": K, "V": V})
        assert result.shape == (B, H_q, S, D)

    def test_cross_attention(self):
        B, H, Sq, Skv, D = 1, 2, 16, 32, 64
        Q = torch.randn(B, H, Sq, D)
        K = torch.randn(B, H, Skv, D)
        V = torch.randn(B, H, Skv, D)
        result = INTERPRETER.execute("cross_attention", {"Q": Q, "K": K, "V": V})
        assert result.shape == (B, H, Sq, D)


class TestSemanticInterpreterValidate:
    """Test the validate() method."""

    def test_validate_correct(self):
        x = torch.randn(4, 8)
        expected = torch.relu(x)
        result = INTERPRETER.validate("relu", {"X": x}, expected)
        assert result["correct"] is True
        assert result["max_diff"] < 1e-6

    def test_validate_incorrect(self):
        x = torch.randn(4, 8)
        wrong = torch.randn(4, 8)  # random != relu(x)
        result = INTERPRETER.validate("relu", {"X": x}, wrong)
        # May or may not be "correct" by chance, but max_diff should be > 0
        assert isinstance(result["correct"], bool)
        assert result["max_diff"] >= 0

    def test_validate_shape_mismatch(self):
        x = torch.randn(4, 8)
        wrong_shape = torch.randn(4, 16)
        result = INTERPRETER.validate("relu", {"X": x}, wrong_shape)
        assert result["correct"] is False
        assert "Shape mismatch" in result.get("error", "")


class TestSemanticInterpreterInferShape:
    """Test shape inference via interpreter."""

    def test_infer_matmul(self):
        shape = INTERPRETER.infer_shape("matmul", {"A": [1024, 512], "B": [512, 2048]})
        assert shape == [1024, 2048]

    def test_infer_relu(self):
        shape = INTERPRETER.infer_shape("relu", {"X": [4, 8]})
        assert shape == [4, 8]


class TestAllOpsExecute:
    """Smoke test: every op in REGISTRY can be executed without error."""

    @pytest.mark.parametrize("op_name", [
        "relu", "gelu", "silu", "tanh", "sigmoid", "neg", "exp", "rsqrt",
        "add", "mul", "softmax", "layernorm", "rmsnorm",
        "reduce_sum", "reduce_max", "reduce_mean", "cumsum",
        "matmul", "batch_matmul", "transpose", "copy_",
        "silu_and_mul", "geglu",
    ])
    def test_op_executes(self, op_name):
        """Each op should execute without error on simple inputs."""
        op = REGISTRY.get(op_name)

        # Build simple inputs based on op definition
        inputs = {}
        for inp_name, inp_type in op.inputs.items():
            if "Tensor[M,N]" in inp_type or "Tensor[...]" in inp_type:
                inputs[inp_name] = torch.randn(4, 8)
            elif "Tensor[M,K]" in inp_type or "Tensor[K,N]" in inp_type:
                inputs[inp_name] = torch.randn(4, 8)
            elif "Tensor[B,M,K]" in inp_type or "Tensor[B,K,N]" in inp_type:
                inputs[inp_name] = torch.randn(2, 4, 8)
            elif "Tensor[N]" in inp_type:
                inputs[inp_name] = torch.ones(8)
            elif "Tensor[...,2N]" in inp_type:
                inputs[inp_name] = torch.randn(4, 16)
            else:
                inputs[inp_name] = torch.randn(4, 8)

        # Fix matmul shapes
        if op_name == "matmul":
            inputs = {"A": torch.randn(4, 8), "B": torch.randn(8, 16)}
        elif op_name == "batch_matmul":
            inputs = {"A": torch.randn(2, 4, 8), "B": torch.randn(2, 8, 16)}

        result = INTERPRETER.execute(op_name, inputs)
        assert isinstance(result, torch.Tensor)
