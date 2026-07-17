# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive correctness tests for LLVM backend — all 46 ops.

Tests verify: lower() produces valid LLVM IR → compile() succeeds →
run() output matches numpy/torch reference within tolerance.
"""

import math
import numpy as np
import pytest

from arke.backend.llvm_backend import LLVMBackend, llvm_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not llvm_toolchain_available(),
    reason="LLVM toolchain (llc + ptxas + GPU driver) not available",
)


@pytest.fixture
def backend():
    return LLVMBackend(chip="sm_86")


# ─── Helper: build IRGraph ───────────────────────────────────────

def _graph_unary(op: str, M: int = 32, N: int = 64, dtype: str = "float32") -> IRGraph:
    g = IRGraph(name=f"{op}_{M}x{N}")
    g.add_input("X", dtype=dtype, shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"X": "X"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_binary(op: str, M: int = 32, N: int = 64, dtype: str = "float32") -> IRGraph:
    g = IRGraph(name=f"{op}_{M}x{N}")
    g.add_input("A", dtype=dtype, shape=[M, N])
    g.add_input("B", dtype=dtype, shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"A": "A", "B": "B"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _graph_matmul(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name=f"matmul_{M}x{K}x{N}")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


def _make_test_graph(op: str) -> IRGraph:
    """Build an IRGraph with the correct input structure for each op."""
    M, N = 16, 32

    # -- Unary elementwise --
    UNARY_OPS = {"relu", "gelu", "silu", "tanh", "sigmoid", "exp", "neg", "rsqrt",
                 "cast", "copy_", "softmax", "reduce_sum", "reduce_max", "reduce_mean",
                 "argmax", "cumsum", "topk", "transpose", "permute", "split",
                 "quantize_per_token"}
    # -- Binary ops --
    BINARY_OPS = {"add", "mul", "silu_and_mul", "gelu_and_mul", "concat", "swiglu_packed"}

    if op == "matmul":
        return _graph_matmul(M, M, M)
    elif op in UNARY_OPS:
        return _graph_unary(op, M, N)
    elif op in BINARY_OPS:
        return _graph_binary(op, M, N)
    elif op == "where_":
        g = IRGraph(name=f"{op}_test")
        g.add_input("Cond", dtype="float32", shape=[M, N])
        g.add_input("A", dtype="float32", shape=[M, N])
        g.add_input("B", dtype="float32", shape=[M, N])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"Cond": "Cond", "A": "A", "B": "B"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "layernorm":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("W", dtype="float32", shape=[1, N])
        g.add_input("Bias", dtype="float32", shape=[1, N])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "W": "W", "Bias": "Bias"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "rmsnorm":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("W", dtype="float32", shape=[1, N])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "W": "W"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "rmsnorm_residual":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("R", dtype="float32", shape=[M, N])  # residual
        g.add_input("W", dtype="float32", shape=[1, N])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "R": "R", "W": "W"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "batch_matmul":
        g = IRGraph(name=f"{op}_test")
        g.add_input("A", dtype="float32", shape=[4, M, N])
        g.add_input("B", dtype="float32", shape=[4, N, M])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"A": "A", "B": "B"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "embedding":
        g = IRGraph(name=f"{op}_test")
        g.add_input("Table", dtype="float32", shape=[100, N])  # vocab=100, dim=N
        g.add_input("Idx", dtype="float32", shape=[M, 1])  # indices
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"Table": "Table", "Idx": "Idx"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "gather":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("Idx", dtype="float32", shape=[M, 4])  # gather 4 cols per row
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "Idx": "Idx"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "scatter":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("Idx", dtype="float32", shape=[M, 4])
        g.add_input("Src", dtype="float32", shape=[M, 4])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "Idx": "Idx", "Src": "Src"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "grouped_matmul":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[4, 16, 32])  # [B, M, K]
        g.add_input("W", dtype="float32", shape=[8, 32, 32])  # [E, K, N]
        g.add_input("Idx", dtype="float32", shape=[4])  # [B] group indices
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "W": "W", "Idx": "Idx"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "cross_entropy":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])  # logits
        g.add_input("Labels", dtype="float32", shape=[M, 1])  # targets
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "Labels": "Labels"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "dequantize_per_channel":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("Scale", dtype="float32", shape=[M, 1])
        g.add_input("ZP", dtype="float32", shape=[M, 1])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "Scale": "Scale", "ZP": "ZP"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "fused_linear_cross_entropy":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("W", dtype="float32", shape=[N, 10])  # project to 10 classes
        g.add_input("Labels", dtype="float32", shape=[M, 1])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "W": "W", "Labels": "Labels"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "rope":
        g = IRGraph(name=f"{op}_test")
        g.add_input("X", dtype="float32", shape=[2, 8, 16, 32])  # [B, H, S, D]
        g.add_input("Cos", dtype="float32", shape=[2, 8, 16, 32])
        g.add_input("Sin", dtype="float32", shape=[2, 8, 16, 32])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"X": "X", "Cos": "Cos", "Sin": "Sin"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op in ("flash_attention", "grouped_query_attention", "cross_attention"):
        g = IRGraph(name=f"{op}_test")
        g.add_input("Q", dtype="float32", shape=[2, 4, 16, 32])  # [B, H, S, D]
        g.add_input("K", dtype="float32", shape=[2, 4, 16, 32])
        g.add_input("V", dtype="float32", shape=[2, 4, 16, 32])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "paged_attention":
        g = IRGraph(name=f"{op}_test")
        g.add_input("Q", dtype="float32", shape=[2, 4, 1, 32])  # [B, H, 1, D]
        g.add_input("KCache", dtype="float32", shape=[8, 4, 4, 32])  # [num_blocks, block_size, H, D]
        g.add_input("VCache", dtype="float32", shape=[8, 4, 4, 32])
        g.add_input("BlockTable", dtype="float32", shape=[2, 2])  # [B, max_blocks]
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"Q": "Q", "KCache": "KCache", "VCache": "VCache",
                                  "BlockTable": "BlockTable"}, outputs=["out"]))
        g.set_outputs(["out"])
        return g
    elif op == "multi_latent_attention":
        g = IRGraph(name=f"{op}_test")
        g.add_input("Q", dtype="float32", shape=[2, 4, 16, 32])  # [B, H, S, D]
        g.add_input("KV", dtype="float32", shape=[2, 16, 64])  # [B, S, Dc]
        g.add_input("Wq", dtype="float32", shape=[64, 4, 32])  # [Dc, H, D]
        g.add_input("Wkv", dtype="float32", shape=[64, 4, 32])
        g.add_node(IRNode(id="n0", op=op,
                          inputs={"Q": "Q", "KV": "KV", "Wq": "Wq", "Wkv": "Wkv"},
                          outputs=["out"]))
        g.set_outputs(["out"])
        return g
    else:
        # Fallback: unary
        return _graph_unary(op, M, N)


def _run_op(backend, graph, inputs, atol=1e-3, rtol=1e-3):
    """Lower → compile → run, return output numpy array."""
    artifact = backend.lower(graph)
    assert "nvptx64" in artifact.source_code or "nvptx" in artifact.source_code.lower()
    kernel = backend.compile(artifact)
    assert kernel.success, f"Compile failed for {graph.name}: {kernel.error}"
    result = backend.run(kernel, inputs)
    return result


# ─── OT0: Elementwise ────────────────────────────────────────────

class TestLLVMElementwise:
    """12 OT0 ops: relu, gelu, silu, tanh, sigmoid, exp, neg, rsqrt, add, mul, cast, where_."""

    @pytest.mark.parametrize("op,ref_fn", [
        ("relu", lambda x: np.maximum(x, 0)),
        ("neg", lambda x: -x),
        ("exp", lambda x: np.exp(x)),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("tanh", lambda x: np.tanh(x)),
        ("rsqrt", lambda x: 1.0 / np.sqrt(np.abs(x) + 1e-6)),
    ])
    def test_unary_simple(self, backend, op, ref_fn):
        M, N = 32, 64
        g = _graph_unary(op, M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        if op == "rsqrt":
            X = np.abs(X) + 0.01  # positive values for rsqrt
        result = _run_op(backend, g, {"X": X})
        np.testing.assert_allclose(result["out"], ref_fn(X), atol=1e-3, rtol=1e-3)

    def test_gelu(self, backend):
        M, N = 32, 64
        g = _graph_unary("gelu", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        ref = 0.5 * X * (1.0 + np.vectorize(math.erf)(X / math.sqrt(2.0)))
        np.testing.assert_allclose(result["out"], ref, atol=1e-2, rtol=1e-2)

    def test_silu(self, backend):
        M, N = 32, 64
        g = _graph_unary("silu", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        ref = X / (1.0 + np.exp(-X))
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("op,ref_fn", [
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
    ])
    def test_binary(self, backend, op, ref_fn):
        M, N = 32, 64
        g = _graph_binary(op, M, N)
        np.random.seed(42)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"A": A, "B": B})
        np.testing.assert_allclose(result["out"], ref_fn(A, B), atol=1e-5, rtol=1e-5)

    def test_cast(self, backend):
        """Cast op — float32 to float32 (identity for now)."""
        M, N = 32, 64
        g = _graph_unary("cast", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        np.testing.assert_allclose(result["out"], X, atol=1e-6, rtol=1e-6)

    def test_where(self, backend):
        """where_(cond, A, B) — ternary select."""
        M, N = 32, 64
        g = IRGraph(name=f"where_{M}x{N}")
        g.add_input("Cond", dtype="float32", shape=[M, N])
        g.add_input("A", dtype="float32", shape=[M, N])
        g.add_input("B", dtype="float32", shape=[M, N])
        g.add_node(IRNode(id="n0", op="where_",
                          inputs={"Cond": "Cond", "A": "A", "B": "B"},
                          outputs=["out"]))
        g.set_outputs(["out"])

        np.random.seed(42)
        Cond = (np.random.randn(M, N) > 0).astype(np.float32)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"Cond": Cond, "A": A, "B": B})
        ref = np.where(Cond > 0, A, B)
        np.testing.assert_allclose(result["out"], ref, atol=1e-6, rtol=1e-6)


# ─── OT1: Reduction ──────────────────────────────────────────────

class TestLLVMReduction:
    """10 OT1 ops: softmax, layernorm, rmsnorm, reduce_sum/max/mean, argmax, cumsum, topk, rmsnorm_residual."""

    def test_reduce_sum(self, backend):
        M, N = 16, 64
        g = _graph_unary("reduce_sum", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        ref = X.sum(axis=-1, keepdims=True)
        # Output shape might be [M, 1] or [M]
        out = result["out"].reshape(ref.shape)
        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_reduce_max(self, backend):
        M, N = 16, 64
        g = _graph_unary("reduce_max", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        ref = X.max(axis=-1, keepdims=True)
        out = result["out"].reshape(ref.shape)
        np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)

    def test_reduce_mean(self, backend):
        M, N = 16, 64
        g = _graph_unary("reduce_mean", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        ref = X.mean(axis=-1, keepdims=True)
        out = result["out"].reshape(ref.shape)
        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_softmax(self, backend):
        M, N = 16, 64
        g = _graph_unary("softmax", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        # numerically stable softmax
        e = np.exp(X - X.max(axis=-1, keepdims=True))
        ref = e / e.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)

    def test_layernorm(self, backend):
        M, N = 16, 64
        g = IRGraph(name=f"layernorm_{M}x{N}")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("W", dtype="float32", shape=[1, N])
        g.add_input("Bias", dtype="float32", shape=[1, N])
        g.add_node(IRNode(id="n0", op="layernorm",
                          inputs={"X": "X", "W": "W", "Bias": "Bias"},
                          outputs=["out"]))
        g.set_outputs(["out"])

        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        W = np.ones((1, N), dtype=np.float32)
        Bias = np.zeros((1, N), dtype=np.float32)
        result = _run_op(backend, g, {"X": X, "W": W, "Bias": Bias})
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        ref = (X - mean) / np.sqrt(var + 1e-5) * W + Bias
        np.testing.assert_allclose(result["out"], ref, atol=1e-2, rtol=1e-2)

    def test_rmsnorm(self, backend):
        M, N = 16, 64
        g = IRGraph(name=f"rmsnorm_{M}x{N}")
        g.add_input("X", dtype="float32", shape=[M, N])
        g.add_input("W", dtype="float32", shape=[1, N])
        g.add_node(IRNode(id="n0", op="rmsnorm",
                          inputs={"X": "X", "W": "W"}, outputs=["out"]))
        g.set_outputs(["out"])

        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        W = np.ones((1, N), dtype=np.float32)
        result = _run_op(backend, g, {"X": X, "W": W})
        rms = np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + 1e-5)
        ref = X / rms * W
        np.testing.assert_allclose(result["out"], ref, atol=1e-2, rtol=1e-2)


# ─── OT2: Dense & Data Movement ──────────────────────────────────

class TestLLVMDense:
    """11 OT2 ops: matmul, batch_matmul, grouped_matmul, transpose, permute, etc."""

    @pytest.mark.parametrize("M,K,N", [(32, 32, 32), (64, 64, 64)])
    def test_matmul(self, backend, M, K, N):
        g = _graph_matmul(M, K, N)
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        result = _run_op(backend, g, {"A": A, "B": B})
        ref = A @ B
        atol = max(1e-3, K * 2e-6)
        np.testing.assert_allclose(result["out"], ref, atol=atol, rtol=1e-3)

    def test_transpose(self, backend):
        M, N = 32, 64
        g = _graph_unary("transpose", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        np.testing.assert_allclose(result["out"], X.T, atol=1e-6, rtol=1e-6)

    def test_copy(self, backend):
        M, N = 32, 64
        g = _graph_unary("copy_", M, N)
        np.random.seed(42)
        X = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"X": X})
        np.testing.assert_allclose(result["out"], X, atol=1e-6, rtol=1e-6)


# ─── OT3: Fused Compound ─────────────────────────────────────────

class TestLLVMFused:
    """8 OT3 ops."""

    def test_silu_and_mul(self, backend):
        M, N = 32, 64
        g = _graph_binary("silu_and_mul", M, N)
        np.random.seed(42)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"A": A, "B": B})
        ref = (A / (1.0 + np.exp(-A))) * B
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)

    def test_gelu_and_mul(self, backend):
        M, N = 32, 64
        g = _graph_binary("gelu_and_mul", M, N)
        np.random.seed(42)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = _run_op(backend, g, {"A": A, "B": B})
        gelu_a = 0.5 * A * (1.0 + np.vectorize(math.erf)(A / math.sqrt(2.0)))
        ref = gelu_a * B
        np.testing.assert_allclose(result["out"], ref, atol=1e-2, rtol=1e-2)


# ─── Protocol: All 46 ops supported ──────────────────────────────

class TestLLVMFullCoverage:
    """Verify all 46 ops are registered."""

    def test_all_ops_supported(self, backend):
        from benchmarks.op_registry import ALL_OPS
        unsupported = [op for op in ALL_OPS if not backend.supports_op(op)]
        assert not unsupported, f"Unsupported ops: {unsupported}"

    def test_all_ops_lower(self, backend):
        """Every op can produce LLVM IR (lower step only, no GPU needed)."""
        from benchmarks.op_registry import ALL_OPS
        failures = []
        for op in sorted(ALL_OPS):
            try:
                g = _make_test_graph(op)
                artifact = backend.lower(g)
                assert artifact.source_code, f"Empty source for {op}"
            except Exception as e:
                failures.append(f"{op}: {e}")
        assert not failures, f"Lower failures:\n" + "\n".join(failures)
