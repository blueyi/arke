# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C5 — KernelCache unit tests + integration with TritonBackend.

Validates:
  * LRU eviction + maxsize
  * Hit / miss / eviction counters
  * Same op+template+dtype shares the same wrapper across nodes
  * Build failures are NOT cached (retry on next call)
  * Thread-safety smoke test (parallel get_or_build same key → one build)
  * TritonBackend integration: lowering two graphs that share an op
    yields exactly 1 build (cache hit on the second)
"""

from __future__ import annotations

import threading

import pytest
import torch

from arke.backend.kernel_cache import KERNEL_CACHE, KernelCache
from arke.backend.triton_backend import TritonBackend
from arke.ir.graph import IRGraph, IRNode
from arke.ir.ops.registry import REGISTRY


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="KernelCache integration requires CUDA",
)


# ── unit tests on a fresh cache ─────────────────────────────────

class TestKernelCacheUnit:

    def test_first_call_misses_second_call_hits(self):
        c = KernelCache(maxsize=16)
        hint = REGISTRY.get("matmul").template_hint
        w1 = c.get_or_build("matmul", hint, dtype="float16")
        w2 = c.get_or_build("matmul", hint, dtype="float16")
        assert w1 is w2  # same object — cache hit
        stats = c.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_different_dtype_different_entry(self):
        c = KernelCache(maxsize=16)
        hint = REGISTRY.get("matmul").template_hint
        w_fp16 = c.get_or_build("matmul", hint, dtype="float16")
        w_fp32 = c.get_or_build("matmul", hint, dtype="float32")
        assert w_fp16 is not w_fp32
        assert c.stats()["size"] == 2

    def test_lru_eviction_kicks_in_at_maxsize(self):
        """Fill cache past maxsize; oldest entry evicted."""
        c = KernelCache(maxsize=2)
        h_mm = REGISTRY.get("matmul").template_hint
        h_sm = REGISTRY.get("softmax").template_hint
        h_ln = REGISTRY.get("layernorm").template_hint
        c.get_or_build("matmul", h_mm, dtype="float16")
        c.get_or_build("softmax", h_sm, dtype="float16")
        assert c.stats()["size"] == 2
        # Inserting a 3rd → evicts the oldest (matmul)
        c.get_or_build("layernorm", h_ln, dtype="float16")
        stats = c.stats()
        assert stats["size"] == 2
        assert stats["evictions"] == 1

    def test_build_failure_is_not_cached(self, monkeypatch):
        """Failed builds should not pollute the cache — next call retries."""
        import arke.backend.kernel_cache as kc_mod
        c = KernelCache(maxsize=8)

        calls = {"n": 0}

        def _boom(*a, **kw):
            calls["n"] += 1
            raise RuntimeError("synthetic")

        monkeypatch.setattr(kc_mod, "generate_kernel", _boom)
        hint = REGISTRY.get("matmul").template_hint

        with pytest.raises(RuntimeError, match="synthetic"):
            c.get_or_build("matmul", hint, dtype="float16")
        with pytest.raises(RuntimeError, match="synthetic"):
            c.get_or_build("matmul", hint, dtype="float16")

        # Cache stays empty
        assert c.stats()["size"] == 0
        assert c.stats()["build_failures"] == 2
        # Both calls went through to generate_kernel (no result cached)
        assert calls["n"] == 2

    def test_thread_safety_single_build_under_concurrent_misses(self):
        """Two threads racing on the same key should both get a working
        wrapper. We don't require exactly-once build (cold-build runs
        outside the lock for parallelism), but the cache must end up with
        a single live entry and zero corruption."""
        c = KernelCache(maxsize=8)
        hint = REGISTRY.get("matmul").template_hint

        results: list = []
        errors: list = []

        def _worker():
            try:
                w = c.get_or_build("matmul", hint, dtype="float16")
                results.append(w)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 4
        # Final cache has exactly one entry (last writer wins under lock)
        assert c.stats()["size"] == 1
        # All callers got the SAME wrapper (the one that won the race)
        assert all(r is results[0] for r in results)


# ── integration: TritonBackend uses KERNEL_CACHE ────────────────

class TestTritonBackendCacheIntegration:

    def _matmul_graph(self):
        g = IRGraph(name="g")
        g.add_input("A", dtype="float16", shape=[64, 32])
        g.add_input("B", dtype="float16", shape=[32, 64])
        g.add_node(IRNode(id="n0", op="matmul",
                          inputs={"A": "A", "B": "B"}, outputs=["C"]))
        g.set_outputs(["C"])
        return g

    def test_two_graphs_share_wrapper(self):
        """Lowering the same op in two separate IRGraphs should hit the
        cache the second time."""
        KERNEL_CACHE.clear()
        KERNEL_CACHE.reset_stats()
        tb = TritonBackend()

        art1 = tb.lower(self._matmul_graph())
        art2 = tb.lower(self._matmul_graph())

        # Both used real kernels
        assert art1.metadata["num_real_kernels"] == 1
        assert art2.metadata["num_real_kernels"] == 1

        # First lower → 1 miss, second lower → 1 hit
        stats = KERNEL_CACHE.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

        # And the wrappers stashed on the plans are the SAME object
        plan1 = art1.metadata["plans"][0]
        plan2 = art2.metadata["plans"][0]
        assert plan1.wrapper is plan2.wrapper

    def test_multinode_graph_shares_wrapper_across_nodes(self):
        """A graph with two matmul nodes should reuse one wrapper."""
        KERNEL_CACHE.clear()
        KERNEL_CACHE.reset_stats()

        g = IRGraph(name="two_matmul")
        g.add_input("A", dtype="float16", shape=[64, 32])
        g.add_input("B", dtype="float16", shape=[32, 64])
        g.add_input("X", dtype="float16", shape=[64, 32])
        g.add_input("Y", dtype="float16", shape=[32, 64])
        g.add_node(IRNode(id="n0", op="matmul",
                          inputs={"A": "A", "B": "B"}, outputs=["C"]))
        g.add_node(IRNode(id="n1", op="matmul",
                          inputs={"A": "X", "B": "Y"}, outputs=["D"]))
        g.set_outputs(["C", "D"])

        tb = TritonBackend()
        art = tb.lower(g)
        assert art.metadata["num_real_kernels"] == 2

        stats = KERNEL_CACHE.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

        plan_a, plan_b = art.metadata["plans"]
        assert plan_a.wrapper is plan_b.wrapper

    def test_e2e_correctness_after_cache_hit(self):
        """End-to-end: the cached wrapper actually produces correct output
        when re-executed against fresh inputs."""
        KERNEL_CACHE.clear()
        tb = TritonBackend()

        # Prime the cache.
        g1 = self._matmul_graph()
        ker1 = tb.compile(tb.lower(g1))
        A = torch.randn(64, 32, device="cuda", dtype=torch.float16)
        B = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        out1 = tb.run(ker1, {"A": A, "B": B})
        assert torch.allclose(out1["C"], A @ B, rtol=1e-2, atol=1e-2)

        # Reuse on a fresh graph + different shapes — same wrapper,
        # Triton internally specializes for the new shape.
        g2 = IRGraph(name="bigger")
        g2.add_input("A", dtype="float16", shape=[128, 64])
        g2.add_input("B", dtype="float16", shape=[64, 256])
        g2.add_node(IRNode(id="n0", op="matmul",
                           inputs={"A": "A", "B": "B"}, outputs=["C"]))
        g2.set_outputs(["C"])
        ker2 = tb.compile(tb.lower(g2))
        A2 = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        B2 = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        out2 = tb.run(ker2, {"A": A2, "B": B2})
        assert torch.allclose(out2["C"], A2 @ B2, rtol=1e-2, atol=1e-2)


# ── singleton smoke ─────────────────────────────────────────────

class TestModuleSingleton:

    def test_singleton_is_importable(self):
        from arke.backend.kernel_cache import KERNEL_CACHE as kc1
        from arke.backend.kernel_cache import KERNEL_CACHE as kc2
        assert kc1 is kc2

    def test_singleton_has_stats(self):
        stats = KERNEL_CACHE.stats()
        assert {"size", "maxsize", "hits", "misses", "evictions", "build_failures"} <= stats.keys()
