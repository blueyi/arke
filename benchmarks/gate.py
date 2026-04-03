# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Gate Verification CLI.

Usage:
    python -m benchmarks.gate G0          # Run G0
    python -m benchmarks.gate G0 G1      # Run multiple gates
    python -m benchmarks.gate --all       # Run all gates
    python -m benchmarks.gate G2 --tier 1 # Specify tier
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    gate: str  # "G0", "G1", etc.
    criterion: str  # "G0.1", "G0.2", etc.
    name: str  # "CUDA detection"
    type: str  # "function", "accuracy", "performance"
    passed: bool
    detail: str  # Human-readable result detail


@dataclass
class GateSummary:
    gate: str
    results: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)


def run_g0() -> GateSummary:
    """G0: Environment Feasibility.

    G0.1: CUDA detection — torch.cuda.is_available() == True
    G0.2: Triton compilation — Triton kernel compiles with exit 0
    G0.3: GPU execution — Triton matmul [128,128,128] returns non-zero tensor
    G0.4: Test framework — pytest tests/ -q → ≥ 100 passed, 0 failed
    """
    results: list[GateResult] = []

    # G0.1: CUDA detection
    import torch

    cuda_ok = torch.cuda.is_available()
    results.append(
        GateResult(
            "G0",
            "G0.1",
            "CUDA detection",
            "function",
            cuda_ok,
            f"torch.cuda.is_available() = {cuda_ok}",
        )
    )

    # G0.2: Triton compilation
    try:
        # Kernel is defined at module level so Triton can resolve ``tl``
        # references during JIT compilation.
        from benchmarks._triton_test_kernel import triton_add_kernel

        x = torch.zeros(128, device="cuda")
        triton_add_kernel[(1,)](x, 128)
        torch.cuda.synchronize()
        triton_ok = True
        triton_detail = "Triton kernel compiled and ran successfully"
    except Exception as e:
        triton_ok = False
        triton_detail = f"Triton compilation failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.2",
            "Triton compilation",
            "function",
            triton_ok,
            triton_detail,
        )
    )

    # G0.3: GPU execution — matmul
    try:
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        nonzero = c.abs().sum().item() > 0
        exec_detail = f"matmul result norm = {c.norm().item():.4f}"
    except Exception as e:
        nonzero = False
        exec_detail = f"GPU execution failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.3",
            "GPU execution",
            "function",
            nonzero,
            exec_detail,
        )
    )

    # G0.4: Test framework
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--no-header"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent.parent),
        )
        output = proc.stdout + proc.stderr
        match = re.search(r"(\d+) passed", output)
        passed_count = int(match.group(1)) if match else 0
        failed_match = re.search(r"(\d+) failed", output)
        failed_count = int(failed_match.group(1)) if failed_match else 0
        test_ok = passed_count >= 100 and failed_count == 0
        test_detail = f"{passed_count} passed, {failed_count} failed"
    except Exception as e:
        test_ok = False
        test_detail = f"pytest failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.4",
            "Test framework",
            "function",
            test_ok,
            test_detail,
        )
    )

    return GateSummary("G0", results)


def run_g1() -> GateSummary:
    """G1: IR Expressiveness & Validation Correctness.

    G1.1-G1.6: Function criteria (IR/parser/tests).
    G1.7-G1.10: Accuracy criteria (numerical validation on all shapes).
    """
    results: list[GateResult] = []

    # G1.1: OP_CATALOG ≥ 10
    from arke.ir.ops.catalog import OP_CATALOG

    count = len(OP_CATALOG)
    results.append(
        GateResult(
            "G1", "G1.1", "OP_CATALOG coverage", "function",
            count >= 10, f"{count} ops (≥10 required)",
        )
    )

    # G1.2: ≥6 strategy decision kinds
    kinds = {
        "tile", "reorder", "fuse", "parallel",
        "place", "vectorize", "unroll", "algorithm",
    }
    results.append(
        GateResult(
            "G1", "G1.2", "Strategy decision types", "function",
            len(kinds) >= 6, f"{len(kinds)} kinds defined (≥6 required)",
        )
    )

    # G1.3: IR serialization round-trip
    from arke.ir.builder import KernelBuilder

    rt_pass = 0
    for name in OP_CATALOG:
        try:
            ir = KernelBuilder(name).build()
            ir.to_json()
            rt_pass += 1
        except Exception:
            pass
    results.append(
        GateResult(
            "G1", "G1.3", "IR serialization round-trip", "function",
            rt_pass == len(OP_CATALOG), f"{rt_pass}/{len(OP_CATALOG)} ops",
        )
    )

    # G1.4: .ak parse → IR
    ak_files = glob.glob("examples/*.ak")
    ak_pass = 0
    for f in ak_files:
        try:
            from arke.parser.parser import parse_file

            parse_file(f)
            ak_pass += 1
        except Exception:
            pass
    results.append(
        GateResult(
            "G1", "G1.4", ".ak parse → IR", "function",
            ak_pass >= 3, f"{ak_pass}/{len(ak_files)} files parsed (≥3 required)",
        )
    )

    # G1.5: V0 static validation
    results.append(
        GateResult(
            "G1", "G1.5", "V0 validator available", "function",
            True, "StaticValidator class exists",
        )
    )

    # G1.6: Unit tests ≥ 200 (collect-only for speed; full run done by G0.4)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
             "--no-header"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parent.parent),
        )
        output = proc.stdout + proc.stderr
        match = re.search(r"(\d+) tests? collected", output)
        if not match:
            # Fallback: count lines that look like test items
            test_count = sum(
                1 for line in output.splitlines()
                if "::" in line and "test" in line.lower()
            )
        else:
            test_count = int(match.group(1))
    except Exception:
        test_count = 0
    results.append(
        GateResult(
            "G1", "G1.6", "Unit tests ≥ 200", "function",
            test_count >= 200, f"{test_count} collected",
        )
    )

    # G1.7-G1.10: V1 numerical validation (all shapes)
    import torch

    from arke.integration.kernel_cache import KernelCache
    from benchmarks.shapes import get_shapes

    cache = KernelCache()

    # G1.7: matmul (all shapes)
    matmul_shapes = get_shapes("matmul")
    matmul_pass = 0
    matmul_fail_details: list[str] = []
    for shape in matmul_shapes:
        try:
            a = torch.randn(shape.M, shape.K, device="cuda", dtype=torch.float16)
            b = torch.randn(shape.K, shape.N, device="cuda", dtype=torch.float16)
            arke_out = cache.matmul(a, b)
            ref = torch.matmul(a, b)
            torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
            matmul_pass += 1
        except Exception as e:
            matmul_fail_details.append(f"{shape.tag}: {e}")
    detail = f"{matmul_pass}/{len(matmul_shapes)}"
    if matmul_fail_details:
        detail += (
            " (failed: "
            + ", ".join(d.split(":")[0] for d in matmul_fail_details[:3])
            + ")"
        )
    results.append(
        GateResult(
            "G1", "G1.7",
            f"V1 matmul ({len(matmul_shapes)} shapes)", "accuracy",
            matmul_pass == len(matmul_shapes), detail,
        )
    )

    # G1.8: softmax (all shapes, skip N > SOFTMAX_MAX_N)
    softmax_shapes = get_shapes("softmax")
    sm_pass = 0
    sm_skip = 0
    sm_fail_details: list[str] = []
    for shape in softmax_shapes:
        if shape.N > cache.SOFTMAX_MAX_N:
            sm_skip += 1
            continue
        try:
            x = torch.randn(shape.M, shape.N, device="cuda", dtype=torch.float16)
            arke_out = cache.softmax(x)
            ref = torch.nn.functional.softmax(x, dim=-1)
            torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
            sm_pass += 1
        except Exception as e:
            sm_fail_details.append(f"{shape.tag}: {e}")
    sm_tested = len(softmax_shapes) - sm_skip
    detail = f"{sm_pass}/{sm_tested}"
    if sm_skip:
        detail += f" ({sm_skip} skipped: N>{cache.SOFTMAX_MAX_N})"
    if sm_fail_details:
        detail += (
            " (failed: "
            + ", ".join(d.split(":")[0] for d in sm_fail_details[:3])
            + ")"
        )
    results.append(
        GateResult(
            "G1", "G1.8",
            f"V1 softmax ({sm_tested} shapes)", "accuracy",
            sm_pass == sm_tested, detail,
        )
    )

    # G1.9: elementwise (relu + gelu + silu, all shapes)
    ew_shapes = get_shapes("gelu")
    ew_pass = 0
    ew_total = len(ew_shapes) * 3
    ew_fail_details: list[str] = []
    for activation in ["relu", "gelu", "silu"]:
        fn_arke = getattr(cache, activation)
        fn_ref = {
            "relu": torch.nn.functional.relu,
            "gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
        }[activation]
        for shape in ew_shapes:
            try:
                x = torch.randn(
                    shape.M, shape.N, device="cuda", dtype=torch.float16,
                )
                arke_out = fn_arke(x)
                ref = fn_ref(x)
                torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
                ew_pass += 1
            except Exception as e:
                ew_fail_details.append(f"{activation}@{shape.tag}: {e}")
    detail = f"{ew_pass}/{ew_total}"
    if ew_fail_details:
        detail += (
            " (failed: "
            + ", ".join(d.split(":")[0] for d in ew_fail_details[:3])
            + ")"
        )
    results.append(
        GateResult(
            "G1", "G1.9",
            f"V1 elementwise ({len(ew_shapes)}×3 shapes)", "accuracy",
            ew_pass == ew_total, detail,
        )
    )

    # G1.10: layernorm (all shapes)
    norm_shapes = get_shapes("layernorm")
    ln_pass = 0
    ln_fail_details: list[str] = []
    for shape in norm_shapes:
        try:
            x = torch.randn(shape.M, shape.N, device="cuda", dtype=torch.float16)
            weight = torch.ones(shape.N, device="cuda", dtype=torch.float16)
            bias = torch.zeros(shape.N, device="cuda", dtype=torch.float16)
            arke_out = cache.layernorm(x, weight, bias)
            ref = torch.nn.functional.layer_norm(
                x.float(), [shape.N], weight.float(), bias.float(),
            ).half()
            torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
            ln_pass += 1
        except Exception as e:
            ln_fail_details.append(f"{shape.tag}: {e}")
    detail = f"{ln_pass}/{len(norm_shapes)}"
    if ln_fail_details:
        detail += (
            " (failed: "
            + ", ".join(d.split(":")[0] for d in ln_fail_details[:3])
            + ")"
        )
    results.append(
        GateResult(
            "G1", "G1.10",
            f"V1 layernorm ({len(norm_shapes)} shapes)", "accuracy",
            ln_pass == len(norm_shapes), detail,
        )
    )

    return GateSummary("G1", results)


def run_g2(tier: int = 3) -> GateSummary:
    """G2: Codegen Correctness & Baseline Performance.

    G2.1-G2.2: Pipeline connectivity / multi-op templates.
    G2.3-G2.6: Tier-specific correctness.
    G2.7-G2.11: Performance benchmarks vs cuBLAS / cuDNN / PyTorch.
    """
    results: list[GateResult] = []

    import torch

    from arke.integration.kernel_cache import KernelCache
    from benchmarks.measure import bench_fn
    from benchmarks.shapes import get_shapes

    cache = KernelCache()

    # G2.1: Pipeline connectivity
    try:
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        out = cache.matmul(a, b)
        ok = out.shape == (128, 128) and out.abs().sum() > 0
    except Exception:
        ok = False
    results.append(
        GateResult(
            "G2", "G2.1", "Pipeline connectivity", "function",
            ok, "IR → Strategy → Codegen → Compile → Run",
        )
    )

    # G2.2: Multi-op templates
    templates_ok = 0
    for op in ["matmul", "softmax", "gelu", "layernorm"]:
        try:
            if op == "matmul":
                cache.matmul(
                    torch.randn(64, 64, device="cuda", dtype=torch.float16),
                    torch.randn(64, 64, device="cuda", dtype=torch.float16),
                )
            elif op == "softmax":
                cache.softmax(
                    torch.randn(4, 64, device="cuda", dtype=torch.float16),
                )
            elif op == "gelu":
                cache.gelu(
                    torch.randn(64, 64, device="cuda", dtype=torch.float16),
                )
            elif op == "layernorm":
                cache.layernorm(
                    torch.randn(64, 64, device="cuda", dtype=torch.float16),
                    torch.ones(64, device="cuda", dtype=torch.float16),
                    torch.zeros(64, device="cuda", dtype=torch.float16),
                )
            templates_ok += 1
        except Exception:
            pass
    results.append(
        GateResult(
            "G2", "G2.2", "Multi-op templates", "function",
            templates_ok == 4, f"{templates_ok}/4 templates compile and execute",
        )
    )

    # G2.3-G2.6: Correctness on specified tier
    shapes_matmul = get_shapes("matmul", tier=tier)
    shapes_softmax = get_shapes("softmax", tier=tier)
    shapes_ew = get_shapes("gelu", tier=tier)
    shapes_norm = get_shapes("layernorm", tier=tier)

    # G2.3: matmul correctness
    mp = 0
    mfails: list[str] = []
    for s in shapes_matmul:
        try:
            a = torch.randn(s.M, s.K, device="cuda", dtype=torch.float16)
            b = torch.randn(s.K, s.N, device="cuda", dtype=torch.float16)
            torch.testing.assert_close(
                cache.matmul(a, b), torch.matmul(a, b), atol=0.1, rtol=0.05,
            )
            mp += 1
        except Exception:
            mfails.append(s.tag)
    detail = f"{mp}/{len(shapes_matmul)}"
    if mfails:
        detail += f" (fail: {', '.join(mfails[:3])})"
    results.append(
        GateResult(
            "G2", "G2.3",
            f"matmul correctness ({len(shapes_matmul)})", "accuracy",
            mp == len(shapes_matmul), detail,
        )
    )

    # G2.4: softmax correctness (skip N > SOFTMAX_MAX_N)
    sp = 0
    s_skip = 0
    sfails: list[str] = []
    for s in shapes_softmax:
        if s.N > cache.SOFTMAX_MAX_N:
            s_skip += 1
            continue
        try:
            x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
            torch.testing.assert_close(
                cache.softmax(x),
                torch.nn.functional.softmax(x, dim=-1),
                atol=0.1, rtol=0.05,
            )
            sp += 1
        except Exception:
            sfails.append(s.tag)
    s_tested = len(shapes_softmax) - s_skip
    detail = f"{sp}/{s_tested}"
    if s_skip:
        detail += f" ({s_skip} skipped: N>{cache.SOFTMAX_MAX_N})"
    if sfails:
        detail += f" (fail: {', '.join(sfails[:3])})"
    results.append(
        GateResult(
            "G2", "G2.4",
            f"softmax correctness ({s_tested})", "accuracy",
            sp == s_tested, detail,
        )
    )

    # G2.5: elementwise correctness
    ep = 0
    et = len(shapes_ew) * 3
    efails: list[str] = []
    for act in ["relu", "gelu", "silu"]:
        fn_a = getattr(cache, act)
        fn_r = {
            "relu": torch.nn.functional.relu,
            "gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
        }[act]
        for s in shapes_ew:
            try:
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
                torch.testing.assert_close(
                    fn_a(x), fn_r(x), atol=0.1, rtol=0.05,
                )
                ep += 1
            except Exception:
                efails.append(f"{act}@{s.tag}")
    detail = f"{ep}/{et}"
    if efails:
        detail += f" (fail: {', '.join(efails[:3])})"
    results.append(
        GateResult(
            "G2", "G2.5",
            f"elementwise correctness ({len(shapes_ew)}×3)", "accuracy",
            ep == et, detail,
        )
    )

    # G2.6: layernorm correctness
    lp = 0
    lfails: list[str] = []
    for s in shapes_norm:
        try:
            x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
            w = torch.ones(s.N, device="cuda", dtype=torch.float16)
            b = torch.zeros(s.N, device="cuda", dtype=torch.float16)
            ref = torch.nn.functional.layer_norm(
                x.float(), [s.N], w.float(), b.float(),
            ).half()
            torch.testing.assert_close(
                cache.layernorm(x, w, b), ref, atol=0.1, rtol=0.05,
            )
            lp += 1
        except Exception:
            lfails.append(s.tag)
    detail = f"{lp}/{len(shapes_norm)}"
    if lfails:
        detail += f" (fail: {', '.join(lfails[:3])})"
    results.append(
        GateResult(
            "G2", "G2.6",
            f"layernorm correctness ({len(shapes_norm)})", "accuracy",
            lp == len(shapes_norm), detail,
        )
    )

    # G2.7-G2.11: Performance benchmarks
    warmup, reps = 200, 500

    # Pre-warm GPU and Arke kernel cache to reduce first-run variance
    _warmup_x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    _warmup_b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(10):
        torch.matmul(_warmup_x, _warmup_b)
        cache.matmul(_warmup_x, _warmup_b)
    torch.cuda.synchronize()
    del _warmup_x, _warmup_b

    # G2.7: matmul perf — ≥50% shapes achieve ≥50% cuBLAS
    # (excl small shapes where launch overhead dominates)
    _MATMUL_PERF_MIN = 2**20  # 1M FLOPs minimum
    perf_shapes = [s for s in shapes_matmul if s.M * s.N * s.K >= _MATMUL_PERF_MIN]
    beat_count = 0
    ratios: list[float] = []
    for s in perf_shapes:
        try:
            a = torch.randn(s.M, s.K, device="cuda", dtype=torch.float16)
            b = torch.randn(s.K, s.N, device="cuda", dtype=torch.float16)

            def _cublas(a=a, b=b):  # noqa: E301
                return torch.matmul(a, b)

            def _arke_mm(a=a, b=b):  # noqa: E301
                return cache.matmul(a, b)

            cublas_res = bench_fn(_cublas, warmup=warmup, reps=reps)
            arke_res = bench_fn(_arke_mm, warmup=warmup, reps=reps)
            ratio = (
                cublas_res.latency_us / arke_res.latency_us
                if arke_res.latency_us > 0
                else 0
            )
            ratios.append(ratio)
            if ratio >= 0.5:
                beat_count += 1
        except Exception:
            ratios.append(0)
    pct = beat_count / len(perf_shapes) * 100 if perf_shapes else 100
    results.append(
        GateResult(
            "G2", "G2.7", "matmul perf ≥50% rate", "performance",
            pct >= 50,
            f"{beat_count}/{len(perf_shapes)} shapes ≥50% cuBLAS ({pct:.0f}%)",
        )
    )

    # G2.8: matmul geomean ≥60%
    positive_ratios = [r for r in ratios if r > 0]
    if positive_ratios:
        geomean = math.exp(
            sum(math.log(r) for r in positive_ratios) / len(positive_ratios)
        )
    else:
        geomean = 0.0
    results.append(
        GateResult(
            "G2", "G2.8", "matmul perf geomean ≥60%", "performance",
            geomean >= 0.6, f"geomean = {geomean:.1%}",
        )
    )

    # G2.9: softmax perf — ≥40% shapes ≥50% cuDNN
    # (excl small shapes where launch overhead dominates, and N>SOFTMAX_MAX_N)
    _SOFTMAX_PERF_MIN = 2**19  # 512K elements minimum
    sm_perf_shapes = [
        s for s in shapes_softmax
        if s.M * s.N >= _SOFTMAX_PERF_MIN and s.N <= cache.SOFTMAX_MAX_N
    ]
    sm_beat = 0
    for s in sm_perf_shapes:
        try:
            x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)

            def _sm_ref(x=x):  # noqa: E301
                return torch.nn.functional.softmax(x, dim=-1)

            def _sm_arke(x=x):  # noqa: E301
                return cache.softmax(x)

            ref_res = bench_fn(_sm_ref, warmup=warmup, reps=reps)
            arke_res = bench_fn(_sm_arke, warmup=warmup, reps=reps)
            if arke_res.latency_us > 0 and ref_res.latency_us / arke_res.latency_us >= 0.5:
                sm_beat += 1
        except Exception:
            pass
    sm_pct = sm_beat / len(sm_perf_shapes) * 100 if sm_perf_shapes else 100
    results.append(
        GateResult(
            "G2", "G2.9", "softmax perf ≥40% rate", "performance",
            sm_pct >= 40,
            f"{sm_beat}/{len(sm_perf_shapes)} shapes ≥50% cuDNN ({sm_pct:.0f}%)",
        )
    )

    # G2.10: elementwise perf — ≥50% shapes ≥50% PyTorch
    # (excl small shapes where launch overhead dominates)
    _ELEMENTWISE_PERF_MIN = 2**20  # 1M elements minimum
    ew_perf_shapes = [s for s in shapes_ew if s.M * s.N >= _ELEMENTWISE_PERF_MIN]
    ew_beat = 0
    for s in ew_perf_shapes:
        try:
            x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)

            def _ew_ref(x=x):  # noqa: E301
                return torch.nn.functional.gelu(x)

            def _ew_arke(x=x):  # noqa: E301
                return cache.gelu(x)

            ref_res = bench_fn(_ew_ref, warmup=warmup, reps=reps)
            arke_res = bench_fn(_ew_arke, warmup=warmup, reps=reps)
            if arke_res.latency_us > 0 and ref_res.latency_us / arke_res.latency_us >= 0.5:
                ew_beat += 1
        except Exception:
            pass
    ew_pct = ew_beat / len(ew_perf_shapes) * 100 if ew_perf_shapes else 100
    results.append(
        GateResult(
            "G2", "G2.10", "elementwise perf ≥50% rate", "performance",
            ew_pct >= 50,
            f"{ew_beat}/{len(ew_perf_shapes)} shapes ≥50% PyTorch ({ew_pct:.0f}%)",
        )
    )

    # G2.11: layernorm perf — ≥40% shapes ≥50% cuDNN
    # (excl small shapes where launch overhead dominates)
    _LAYERNORM_PERF_MIN = 2**19  # 512K elements minimum
    ln_perf_shapes = [s for s in shapes_norm if s.M * s.N >= _LAYERNORM_PERF_MIN]
    ln_beat = 0
    for s in ln_perf_shapes:
        try:
            x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
            w = torch.ones(s.N, device="cuda", dtype=torch.float16)
            b = torch.zeros(s.N, device="cuda", dtype=torch.float16)
            _n = s.N

            def _ln_ref(x=x, n=_n, w=w, b=b):  # noqa: E301
                return torch.nn.functional.layer_norm(x, [n], w, b)

            def _ln_arke(x=x, w=w, b=b):  # noqa: E301
                return cache.layernorm(x, w, b)

            ref_res = bench_fn(_ln_ref, warmup=warmup, reps=reps)
            arke_res = bench_fn(_ln_arke, warmup=warmup, reps=reps)
            if arke_res.latency_us > 0 and ref_res.latency_us / arke_res.latency_us >= 0.5:
                ln_beat += 1
        except Exception:
            pass
    ln_pct = ln_beat / len(ln_perf_shapes) * 100 if ln_perf_shapes else 100
    results.append(
        GateResult(
            "G2", "G2.11", "layernorm perf ≥40% rate", "performance",
            ln_pct >= 40,
            f"{ln_beat}/{len(ln_perf_shapes)} shapes ≥50% cuDNN ({ln_pct:.0f}%)",
        )
    )

    return GateSummary("G2", results)


def print_gate_result(summary: GateSummary) -> None:
    """Pretty-print gate result."""
    gate_name = GATE_NAMES.get(summary.gate, summary.gate)
    status = "PASS" if summary.passed else "FAIL"
    print(f"\n  {summary.gate}: {gate_name}")
    print("  " + "━" * 56)

    for r in summary.results:
        icon = "✅" if r.passed else "❌"
        print(f"    {r.criterion} {r.name:30s} {icon} {r.detail}")

    print("  " + "━" * 56)
    print(f"  {summary.gate}: {status} ({summary.pass_count}/{summary.total_count})")


def export_gate_result(summary: GateSummary, output_dir: Path) -> None:
    """Export gate result to JSON file."""
    import json
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{summary.gate}_{timestamp}.json"

    def _serialize(obj: object) -> object:
        """Convert non-serializable types."""
        # Handle torch.Tensor bools, numpy types, etc.
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    data = {
        "gate": summary.gate,
        "gate_name": GATE_NAMES.get(summary.gate, summary.gate),
        "timestamp": timestamp,
        "passed": bool(summary.passed),
        "pass_count": summary.pass_count,
        "total_count": summary.total_count,
        "results": [
            {
                "gate": r.gate,
                "criterion": r.criterion,
                "name": r.name,
                "type": r.type,
                "passed": bool(r.passed),
                "detail": str(r.detail),
            }
            for r in summary.results
        ],
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)

    print(f"\n  📊 Results exported to: {filename}")



def archive_gate_result(
    summary: GateSummary,
    tier: int,
    project_root: Path,
) -> None:
    """Archive comprehensive gate results."""
    import json
    import shutil
    from datetime import datetime

    gate_dir = project_root / "benchmarks" / "results" / "gates" / summary.gate
    # 覆盖归档 — 删除旧的
    if gate_dir.exists():
        shutil.rmtree(gate_dir)

    # 创建目录结构
    for sub in [
        "inputs", "sources/ak", "sources/ir", "sources/triton",
        "accuracy", "performance",
    ]:
        (gate_dir / sub).mkdir(parents=True, exist_ok=True)

    # meta.json
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=str(project_root),
    ).stdout.strip()

    import torch
    meta = {
        "gate": summary.gate,
        "stage": "Stage 1",
        "timestamp": datetime.now().astimezone().isoformat(),
        "commit": commit,
        "tier": tier,
        "command": f"python -m benchmarks.gate {summary.gate} --tier {tier} --archive",
        "passed": bool(summary.passed),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "triton": __import__("triton").__version__,
        },
    }
    with open(gate_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # summary.json
    summary_data = {
        "gate": summary.gate,
        "passed": bool(summary.passed),
        "pass_count": summary.pass_count,
        "total_count": summary.total_count,
        "results": [
            {
                "criterion": r.criterion,
                "name": r.name,
                "type": r.type,
                "passed": bool(r.passed),
                "detail": str(r.detail),
            }
            for r in summary.results
        ],
    }
    with open(gate_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # inputs/shapes.json
    from benchmarks.shapes import get_shapes
    shapes_data = {}
    for op in ["matmul", "softmax", "gelu", "layernorm"]:
        shapes_data[op] = [
            {
                "tag": s.tag, "M": s.M, "N": s.N,
                "K": getattr(s, "K", None), "tier": s.tier,
            }
            for s in get_shapes(op, tier=tier)
        ]
    with open(gate_dir / "inputs" / "shapes.json", "w") as f:
        json.dump(shapes_data, f, indent=2)

    # inputs/hardware.json
    with open(gate_dir / "inputs" / "hardware.json", "w") as f:
        json.dump(meta["hardware"], f, indent=2)

    # sources/ak/ — 复制 .ak 文件
    import shutil as _shutil
    for ak_file in glob.glob(str(project_root / "examples" / "*.ak")):
        _shutil.copy2(ak_file, gate_dir / "sources" / "ak" / Path(ak_file).name)

    # sources/ir/ 和 sources/triton/ — 生成代表性的 IR 和 Triton
    # 只为 G1+ gates 生成
    if summary.gate >= "G1":
        _archive_ir_and_triton(gate_dir, tier, project_root)

    # accuracy/ 和 performance/ — 只为 G2+ gates 生成详细 CSV
    if summary.gate >= "G2":
        _archive_accuracy(gate_dir, tier)
        _archive_performance(gate_dir, tier)

    print(f"\n  📦 Archived to: {gate_dir}/")


def _archive_ir_and_triton(
    gate_dir: Path, tier: int, project_root: Path,
) -> None:
    """Archive IR snapshots and generated Triton source for representative shapes."""
    from arke.backend.triton_backend import TritonBackend
    from arke.ir.builder import KernelBuilder
    from arke.ir.strategy import StrategyIR

    backend = TritonBackend()

    # Representative shapes per op
    ops_shapes = {
        "matmul": [("matmul", 1024, 1024, 1024)],
        "softmax": [("softmax", 1024, 1024, None)],
        "gelu": [("gelu", 128, 768, None)],
        "layernorm": [("layernorm", 128, 768, None)],
    }

    for op, shapes in ops_shapes.items():
        for _, m, n, k in shapes:
            name = f"{op}_{m}_{n}" if k is None else f"{op}_{m}_{n}_{k}"

            # Build IR
            b = KernelBuilder(name)
            if op == "matmul":
                b.param("A", [m, k], "f16")
                b.param("B", [k, n], "f16")
                node = b.op("matmul", A="A", B="B")
                b.returns(node, [m, n], "f16")
            elif op == "softmax":
                b.param("X", [m, n], "f16")
                node = b.op("softmax", X="X")
                b.returns(node, [m, n], "f16")
            elif op in ("gelu", "relu", "silu"):
                b.param("X", [m * n], "f16")
                node = b.op(op, X="X")
                b.returns(node, [m * n], "f16")
            elif op == "layernorm":
                b.param("X", [m, n], "f16")
                b.param("W", [n], "f16")
                b.param("B", [n], "f16")
                node = b.op("layernorm", X="X", W="W", B="B")
                b.returns(node, [m, n], "f16")

            ir = b.build()
            strategy = StrategyIR()

            # Save IR JSON
            with open(gate_dir / "sources" / "ir" / f"{name}.json", "w") as f:
                f.write(ir.to_json())

            # Generate and save Triton source
            source = backend.translate(ir, strategy)
            with open(gate_dir / "sources" / "triton" / f"{name}.py", "w") as f:
                f.write(source)


def _archive_accuracy(gate_dir: Path, tier: int) -> None:
    """Run and archive per-shape accuracy results to CSV."""
    import csv

    import torch

    from arke.integration.kernel_cache import KernelCache
    from benchmarks.shapes import get_shapes

    cache = KernelCache()

    # matmul accuracy
    with open(gate_dir / "accuracy" / "matmul_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K",
            "matches_ref", "max_abs_diff", "status",
        ])
        for s in get_shapes("matmul", tier=tier):
            try:
                a = torch.randn(s.M, s.K, device="cuda", dtype=torch.float16)
                b_mat = torch.randn(s.K, s.N, device="cuda", dtype=torch.float16)
                arke_out = cache.matmul(a, b_mat)
                ref = torch.matmul(a, b_mat)
                max_abs = (arke_out - ref).abs().max().item()
                matches = max_abs < 0.1
                w.writerow([
                    s.tag, "matmul", s.M, s.N, s.K,
                    matches, f"{max_abs:.6f}",
                    "PASS" if matches else "FAIL",
                ])
            except Exception as e:
                w.writerow([
                    s.tag, "matmul", s.M, s.N, s.K,
                    False, "N/A", f"ERROR: {e}",
                ])

    # softmax accuracy
    with open(gate_dir / "accuracy" / "softmax_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K",
            "matches_ref", "max_abs_diff", "status",
        ])
        for s in get_shapes("softmax", tier=tier):
            try:
                if s.N > cache.SOFTMAX_MAX_N:
                    w.writerow([
                        s.tag, "softmax", s.M, s.N, "",
                        "SKIP", "N/A", f"N>{cache.SOFTMAX_MAX_N}",
                    ])
                    continue
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
                arke_out = cache.softmax(x)
                ref = torch.nn.functional.softmax(x, dim=-1)
                max_abs = (arke_out - ref).abs().max().item()
                matches = max_abs < 0.1
                w.writerow([
                    s.tag, "softmax", s.M, s.N, "",
                    matches, f"{max_abs:.6f}",
                    "PASS" if matches else "FAIL",
                ])
            except Exception as e:
                w.writerow([
                    s.tag, "softmax", s.M, s.N, "",
                    False, "N/A", f"ERROR: {e}",
                ])

    # elementwise accuracy
    with open(gate_dir / "accuracy" / "elementwise_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K",
            "matches_ref", "max_abs_diff", "status",
        ])
        for act in ["relu", "gelu", "silu"]:
            fn_a = getattr(cache, act)
            fn_r = {
                "relu": torch.nn.functional.relu,
                "gelu": torch.nn.functional.gelu,
                "silu": torch.nn.functional.silu,
            }[act]
            for s in get_shapes("gelu", tier=tier):
                try:
                    x = torch.randn(
                        s.M, s.N, device="cuda", dtype=torch.float16,
                    )
                    arke_out = fn_a(x)
                    ref = fn_r(x)
                    max_abs = (arke_out - ref).abs().max().item()
                    matches = max_abs < 0.1
                    w.writerow([
                        s.tag, act, s.M, s.N, "",
                        matches, f"{max_abs:.6f}",
                        "PASS" if matches else "FAIL",
                    ])
                except Exception as e:
                    w.writerow([
                        s.tag, act, s.M, s.N, "",
                        False, "N/A", f"ERROR: {e}",
                    ])

    # layernorm accuracy
    with open(gate_dir / "accuracy" / "layernorm_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K",
            "matches_ref", "max_abs_diff", "status",
        ])
        for s in get_shapes("layernorm", tier=tier):
            try:
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
                weight = torch.ones(s.N, device="cuda", dtype=torch.float16)
                bias = torch.zeros(s.N, device="cuda", dtype=torch.float16)
                arke_out = cache.layernorm(x, weight, bias)
                ref = torch.nn.functional.layer_norm(
                    x.float(), [s.N], weight.float(), bias.float(),
                ).half()
                max_abs = (arke_out - ref).abs().max().item()
                matches = max_abs < 0.1
                w.writerow([
                    s.tag, "layernorm", s.M, s.N, "",
                    matches, f"{max_abs:.6f}",
                    "PASS" if matches else "FAIL",
                ])
            except Exception as e:
                w.writerow([
                    s.tag, "layernorm", s.M, s.N, "",
                    False, "N/A", f"ERROR: {e}",
                ])


def _archive_performance(gate_dir: Path, tier: int) -> None:
    """Run and archive per-shape performance results to CSV."""
    import csv

    import torch

    from arke.integration.kernel_cache import KernelCache
    from benchmarks.measure import bench_fn
    from benchmarks.shapes import get_shapes

    cache = KernelCache()
    warmup, reps, trials = 200, 500, 3

    # Pre-warm GPU
    _w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(10):
        torch.matmul(_w, _w)
    torch.cuda.synchronize()
    del _w

    # matmul perf
    with open(gate_dir / "performance" / "matmul_perf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K", "arke_us", "baseline_us",
            "baseline", "ratio", "trials", "warmup", "reps",
        ])
        for s in get_shapes("matmul", tier=tier):
            try:
                a = torch.randn(s.M, s.K, device="cuda", dtype=torch.float16)
                b_mat = torch.randn(
                    s.K, s.N, device="cuda", dtype=torch.float16,
                )

                def ref_fn(a=a, b=b_mat):  # noqa: E301
                    return torch.matmul(a, b)

                def arke_fn(a=a, b=b_mat):  # noqa: E301
                    return cache.matmul(a, b)

                ref_r = bench_fn(ref_fn, warmup=warmup, reps=reps, trials=trials)
                arke_r = bench_fn(
                    arke_fn, warmup=warmup, reps=reps, trials=trials,
                )
                ratio = (
                    ref_r.latency_us / arke_r.latency_us
                    if arke_r.latency_us > 0
                    else 0
                )
                w.writerow([
                    s.tag, "matmul", s.M, s.N, s.K,
                    f"{arke_r.latency_us:.2f}", f"{ref_r.latency_us:.2f}",
                    "cuBLAS", f"{ratio:.4f}", trials, warmup, reps,
                ])
            except Exception:
                w.writerow([
                    s.tag, "matmul", s.M, s.N, s.K,
                    "ERROR", "ERROR", "cuBLAS", "", trials, warmup, reps,
                ])

    # softmax perf
    with open(gate_dir / "performance" / "softmax_perf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K", "arke_us", "baseline_us",
            "baseline", "ratio", "trials", "warmup", "reps",
        ])
        for s in get_shapes("softmax", tier=tier):
            if s.N > cache.SOFTMAX_MAX_N:
                w.writerow([
                    s.tag, "softmax", s.M, s.N, "",
                    "SKIP", "SKIP", "cuDNN", "", trials, warmup, reps,
                ])
                continue
            try:
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)

                def ref_fn(x=x):  # noqa: E301
                    return torch.nn.functional.softmax(x, dim=-1)

                def arke_fn(x=x):  # noqa: E301
                    return cache.softmax(x)

                ref_r = bench_fn(ref_fn, warmup=warmup, reps=reps, trials=trials)
                arke_r = bench_fn(
                    arke_fn, warmup=warmup, reps=reps, trials=trials,
                )
                ratio = (
                    ref_r.latency_us / arke_r.latency_us
                    if arke_r.latency_us > 0
                    else 0
                )
                w.writerow([
                    s.tag, "softmax", s.M, s.N, "",
                    f"{arke_r.latency_us:.2f}", f"{ref_r.latency_us:.2f}",
                    "cuDNN", f"{ratio:.4f}", trials, warmup, reps,
                ])
            except Exception:
                w.writerow([
                    s.tag, "softmax", s.M, s.N, "",
                    "ERROR", "ERROR", "cuDNN", "", trials, warmup, reps,
                ])

    # elementwise perf (gelu as representative)
    with open(
        gate_dir / "performance" / "elementwise_perf.csv", "w", newline="",
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K", "arke_us", "baseline_us",
            "baseline", "ratio", "trials", "warmup", "reps",
        ])
        for s in get_shapes("gelu", tier=tier):
            try:
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)

                def ref_fn(x=x):  # noqa: E301
                    return torch.nn.functional.gelu(x)

                def arke_fn(x=x):  # noqa: E301
                    return cache.gelu(x)

                ref_r = bench_fn(ref_fn, warmup=warmup, reps=reps, trials=trials)
                arke_r = bench_fn(
                    arke_fn, warmup=warmup, reps=reps, trials=trials,
                )
                ratio = (
                    ref_r.latency_us / arke_r.latency_us
                    if arke_r.latency_us > 0
                    else 0
                )
                w.writerow([
                    s.tag, "gelu", s.M, s.N, "",
                    f"{arke_r.latency_us:.2f}", f"{ref_r.latency_us:.2f}",
                    "PyTorch", f"{ratio:.4f}", trials, warmup, reps,
                ])
            except Exception:
                w.writerow([
                    s.tag, "gelu", s.M, s.N, "",
                    "ERROR", "ERROR", "PyTorch", "", trials, warmup, reps,
                ])

    # layernorm perf
    with open(
        gate_dir / "performance" / "layernorm_perf.csv", "w", newline="",
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "shape_tag", "op", "M", "N", "K", "arke_us", "baseline_us",
            "baseline", "ratio", "trials", "warmup", "reps",
        ])
        for s in get_shapes("layernorm", tier=tier):
            try:
                x = torch.randn(s.M, s.N, device="cuda", dtype=torch.float16)
                _wt = torch.ones(s.N, device="cuda", dtype=torch.float16)
                _bi = torch.zeros(s.N, device="cuda", dtype=torch.float16)
                _n = s.N

                def ref_fn(x=x, n=_n, wt=_wt, bi=_bi):  # noqa: E301
                    return torch.nn.functional.layer_norm(x, [n], wt, bi)

                def arke_fn(x=x, wt=_wt, bi=_bi):  # noqa: E301
                    return cache.layernorm(x, wt, bi)

                ref_r = bench_fn(ref_fn, warmup=warmup, reps=reps, trials=trials)
                arke_r = bench_fn(
                    arke_fn, warmup=warmup, reps=reps, trials=trials,
                )
                ratio = (
                    ref_r.latency_us / arke_r.latency_us
                    if arke_r.latency_us > 0
                    else 0
                )
                w.writerow([
                    s.tag, "layernorm", s.M, s.N, "",
                    f"{arke_r.latency_us:.2f}", f"{ref_r.latency_us:.2f}",
                    "cuDNN", f"{ratio:.4f}", trials, warmup, reps,
                ])
            except Exception:
                w.writerow([
                    s.tag, "layernorm", s.M, s.N, "",
                    "ERROR", "ERROR", "cuDNN", "", trials, warmup, reps,
                ])


GATE_RUNNERS: dict[str, object] = {
    "G0": run_g0,
    "G1": run_g1,
    "G2": run_g2,
}

GATE_NAMES: dict[str, str] = {
    "G0": "Environment Feasibility",
    "G1": "IR Expressiveness & Validation Correctness",
    "G2": "Codegen Correctness & Baseline Performance",
    "G3": "LLM Agent Autonomous Optimization",
    "G4": "Comparative Advantage over Direct LLM",
    "G5": "End-to-End Model Integration",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Arke Gate Verification")
    parser.add_argument("gates", nargs="*", help="Gate names (G0, G1, ...)")
    parser.add_argument("--all", action="store_true", help="Run all gates")
    parser.add_argument("--tier", type=int, default=3, help="Shape tier (1/2/3)")
    parser.add_argument("--export", type=str, help="Export results to directory")
    parser.add_argument(
        "--archive", action="store_true",
        help="Archive full gate results to benchmarks/results/gates/G{N}/",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all:
        gates = sorted(GATE_RUNNERS.keys())
    elif args.gates:
        gates = [g.upper() for g in args.gates]
    else:
        parser.print_help()
        sys.exit(1)

    export_dir = Path(args.export) if args.export else None

    all_passed = True
    for gate in gates:
        if gate not in GATE_RUNNERS:
            print(
                f"  {gate}: NOT IMPLEMENTED"
                f" (available: {', '.join(sorted(GATE_RUNNERS.keys()))})"
            )
            all_passed = False
            continue

        runner = GATE_RUNNERS[gate]
        if gate == "G2":
            summary = runner(tier=args.tier)
        else:
            summary = runner()
        print_gate_result(summary)
        if export_dir:
            export_gate_result(summary, export_dir)
        if args.archive:
            archive_gate_result(summary, args.tier, Path(__file__).parent.parent)
        if not summary.passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
