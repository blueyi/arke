# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C4 — Triton backend dispatcher end-to-end (lower → compile → run).

Validates that TritonBackend after the S7 codegen rewrite:
  - dispatches real Triton kernels via generate_kernel() (not interpreter)
  - threads multi-node graphs correctly
  - falls back to the SemanticInterpreter when a node has no template
  - exposes num_real_kernels / num_fallback metadata for diagnostics
"""

from __future__ import annotations

import pytest
import torch

from arke.backend.triton_backend import TritonBackend
from arke.ir.graph import IRGraph, IRNode


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TritonBackend dispatch path requires CUDA",
)

DEV = "cuda"


# ── helpers ────────────────────────────────────────────────────

def _matmul_graph(M=128, K=64, N=128, dtype="float16"):
    g = IRGraph(name="matmul")
    g.add_input("A", dtype=dtype, shape=[M, K])
    g.add_input("B", dtype=dtype, shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul",
                      inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


def _softmax_graph(M=32, N=1024, dtype="float16"):
    g = IRGraph(name="softmax")
    g.add_input("X", dtype=dtype, shape=[M, N])
    g.add_node(IRNode(id="n0", op="softmax",
                      inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _layernorm_graph(M=4, N=768, dtype="float16"):
    g = IRGraph(name="layernorm")
    g.add_input("X", dtype=dtype, shape=[M, N])
    g.add_input("W", dtype=dtype, shape=[N])
    g.add_input("B", dtype=dtype, shape=[N])
    g.add_node(IRNode(id="n0", op="layernorm",
                      inputs={"X": "X", "W": "W", "B": "B"},
                      outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _relu_matmul_graph(M=64, K=32, N=128, dtype="float16"):
    g = IRGraph(name="relu_matmul")
    g.add_input("A", dtype=dtype, shape=[M, K])
    g.add_input("B", dtype=dtype, shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul",
                      inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.add_node(IRNode(id="n1", op="relu",
                      inputs={"X": "C"}, outputs=["D"]))
    g.set_outputs(["D"])
    return g


# ── 1. lower() drives real codegen ────────────────────────────

class TestLowerCodegen:

    def test_matmul_codegen_real_kernel(self):
        tb = TritonBackend()
        art = tb.lower(_matmul_graph())
        assert art.metadata["num_real_kernels"] == 1
        assert art.metadata["num_fallback"] == 0
        # source should contain rendered Triton kernel body
        assert "@triton.jit" in art.source_code
        assert "tl.load" in art.source_code

    def test_softmax_codegen_real_kernel(self):
        tb = TritonBackend()
        art = tb.lower(_softmax_graph())
        assert art.metadata["num_real_kernels"] == 1
        assert "@triton.jit" in art.source_code

    def test_multi_node_codegen(self):
        tb = TritonBackend()
        art = tb.lower(_relu_matmul_graph())
        assert art.metadata["num_real_kernels"] == 2
        assert art.metadata["num_fallback"] == 0


# ── 2. dispatcher numerical correctness ───────────────────────

class TestDispatchCorrectness:

    def test_matmul_dispatch(self):
        M, K, N = 128, 64, 128
        A = torch.randn(M, K, device=DEV, dtype=torch.float16)
        B = torch.randn(K, N, device=DEV, dtype=torch.float16)
        tb = TritonBackend()
        ker = tb.compile(tb.lower(_matmul_graph(M, K, N)))
        out = tb.run(ker, {"A": A, "B": B})
        assert "C" in out
        expected = A @ B
        assert torch.allclose(out["C"], expected, rtol=1e-2, atol=1e-2)

    def test_softmax_dispatch(self):
        M, N = 32, 1024
        X = torch.randn(M, N, device=DEV, dtype=torch.float16)
        tb = TritonBackend()
        ker = tb.compile(tb.lower(_softmax_graph(M, N)))
        out = tb.run(ker, {"X": X})
        expected = torch.nn.functional.softmax(X.float(), dim=-1).to(torch.float16)
        assert torch.allclose(out["Y"], expected, rtol=1e-2, atol=1e-3)

    def test_layernorm_dispatch(self):
        M, N = 4, 768
        X = torch.randn(M, N, device=DEV, dtype=torch.float16)
        W = torch.ones(N, device=DEV, dtype=torch.float16)
        B = torch.zeros(N, device=DEV, dtype=torch.float16)
        tb = TritonBackend()
        ker = tb.compile(tb.lower(_layernorm_graph(M, N)))
        out = tb.run(ker, {"X": X, "W": W, "B": B})
        expected = torch.nn.functional.layer_norm(
            X.float(), [N], W.float(), B.float()
        ).to(torch.float16)
        assert torch.allclose(out["Y"], expected, rtol=1e-2, atol=1e-2)

    def test_multi_node_dispatch_chains_kernels(self):
        """relu(matmul(A,B)) — must thread C from node0 into node1."""
        M, K, N = 64, 32, 128
        A = torch.randn(M, K, device=DEV, dtype=torch.float16)
        B = torch.randn(K, N, device=DEV, dtype=torch.float16)
        tb = TritonBackend()
        art = tb.lower(_relu_matmul_graph(M, K, N))
        ker = tb.compile(art)
        out = tb.run(ker, {"A": A, "B": B})
        expected = torch.relu(A @ B)
        assert torch.allclose(out["D"], expected, rtol=1e-2, atol=1e-2)
        # Both nodes used real kernels
        assert ker.metadata["num_real_kernels"] == 2


# ── 3. fallback safety net ────────────────────────────────────

class TestFallbackSafetyNet:

    def test_codegen_failure_falls_back_to_interpreter(self, monkeypatch):
        """If generate_kernel raises, the node must be marked as fallback
        and the dispatcher must still produce correct output via the
        SemanticInterpreter."""
        from arke.backend.kernel_cache import KERNEL_CACHE
        import arke.backend.kernel_cache as kc_mod

        # Clear cache so the monkeypatch actually intercepts (a previous
        # test may have cached the real matmul wrapper).
        KERNEL_CACHE.clear()

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic codegen failure")

        # KernelCache.get_or_build → generate_kernel inside kernel_cache module
        monkeypatch.setattr(kc_mod, "generate_kernel", _boom)

        tb = TritonBackend()
        art = tb.lower(_matmul_graph(64, 32, 64))
        # codegen failed → must be in fallback bucket
        assert art.metadata["num_real_kernels"] == 0
        assert art.metadata["num_fallback"] == 1
        # And the dispatcher still gives correct output via interpreter.
        A = torch.randn(64, 32, device=DEV, dtype=torch.float16)
        B = torch.randn(32, 64, device=DEV, dtype=torch.float16)
        ker = tb.compile(art)
        out = tb.run(ker, {"A": A, "B": B})
        assert torch.allclose(out["C"], A @ B, rtol=1e-2, atol=1e-2)

    def test_runtime_wrapper_exception_falls_back(self, monkeypatch):
        """If the compiled wrapper raises at run time, the dispatcher
        retries via the SemanticInterpreter — graph still produces output."""
        from arke.backend.kernel_cache import KERNEL_CACHE
        import arke.backend.kernel_cache as kc_mod
        KERNEL_CACHE.clear()

        orig_gen = kc_mod.generate_kernel

        def _wrap(*args, **kwargs):
            real = orig_gen(*args, **kwargs)

            def _bad(*a, **kw):
                raise RuntimeError("synthetic runtime failure")
            return _bad

        monkeypatch.setattr(kc_mod, "generate_kernel", _wrap)

        tb = TritonBackend()
        art = tb.lower(_matmul_graph(64, 32, 64))
        # lower() succeeds (wrapper was constructed)
        assert art.metadata["num_real_kernels"] == 1
        A = torch.randn(64, 32, device=DEV, dtype=torch.float16)
        B = torch.randn(32, 64, device=DEV, dtype=torch.float16)
        ker = tb.compile(art)
        # Wrapper raises at run-time → interpreter takes over.
        out = tb.run(ker, {"A": A, "B": B})
        assert torch.allclose(out["C"], A @ B, rtol=1e-2, atol=1e-2)
        # Clean up cache so the bogus wrapper doesn't pollute subsequent tests.
        KERNEL_CACHE.clear()


# ── 4. compile() packaging ────────────────────────────────────

class TestCompilePackaging:

    def test_compile_success_with_real_kernel(self):
        tb = TritonBackend()
        art = tb.lower(_matmul_graph())
        ker = tb.compile(art)
        assert ker.success
        assert ker.compiled_fn is not None
        assert ker.metadata["num_real_kernels"] == 1
        assert ker.metadata["graph_name"] == "matmul"

    def test_run_failed_kernel_raises(self):
        from arke.backend.protocol import CompiledKernel
        tb = TritonBackend()
        bad = CompiledKernel.fail("synthetic fail")
        with pytest.raises(RuntimeError, match="Cannot run failed"):
            tb.run(bad, {})
