# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""L3 End-to-End Model Benchmark Runner.

GPT-2 Small inference benchmark across three modes:
  - eager (PyTorch default)
  - torch.compile
  - Arke-patched (KernelCache monkey-patch)

Reports latency, correctness, and memory for seq_len = 128, 256, 512.

Usage:
    python -m benchmarks.bench_l3 --all
    python -m benchmarks.bench_l3 --seq-len 128
    python -m benchmarks.bench_l3 --seq-len 128,256,512
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.hardware import collect_hardware_info

logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENS = [128, 256, 512]
WARMUP_RUNS = 10
MEASURE_RUNS = 50


@dataclass
class E2EResult:
    """Result of one mode × one seq_len."""

    model: str
    seq_len: int
    mode: str
    source: str
    mean_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    peak_memory_mb: float
    correct: bool
    max_logit_diff: float
    mean_logit_diff: float
    top1_match: bool


# ── Model loading ───────────────────────────────────────────


def _load_gpt2(device: str = "cuda") -> tuple:
    """Load GPT-2 Small model and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).half()
    model.eval()
    return model, tokenizer


def _make_input(tokenizer, seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Create padded input_ids of exact seq_len."""
    text = "The future of artificial intelligence is"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < seq_len:
        pad = torch.zeros(
            1, seq_len - input_ids.shape[1],
            dtype=torch.long, device=device,
        )
        input_ids = torch.cat([input_ids, pad], dim=1)
    elif input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, :seq_len]
    return input_ids


# ── Profiling ───────────────────────────────────────────────


def _profile(
    model,
    input_ids: torch.Tensor,
    warmup: int = WARMUP_RUNS,
    runs: int = MEASURE_RUNS,
) -> dict[str, float]:
    """Profile forward pass latency (ms)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    with torch.no_grad():
        for i in range(runs):
            start_events[i].record()
            model(input_ids)
            end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": times_ms[len(times_ms) // 2],
    }


def _get_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Get model output logits."""
    with torch.no_grad():
        return model(input_ids).logits.clone()


# ── Mode runners ────────────────────────────────────────────


def _run_eager(
    model, input_ids: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, float]:
    """Run eager mode benchmark."""
    torch.cuda.reset_peak_memory_stats()
    logits = _get_logits(model, input_ids)
    perf = _profile(model, input_ids)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return perf, logits, peak_mb


def _run_compile(
    model, input_ids: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, float] | None:
    """Run torch.compile mode benchmark."""
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        torch.cuda.reset_peak_memory_stats()
        logits = _get_logits(compiled, input_ids)
        perf = _profile(compiled, input_ids)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        return perf, logits, peak_mb
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
        return None


def _run_arke(
    model, input_ids: torch.Tensor, seq_len: int,
) -> tuple[dict[str, float], torch.Tensor, float] | None:
    """Run Arke-patched mode benchmark."""
    try:
        from arke.integration.gpt2_e2e import get_gpt2_shapes, patch_gpt2_fast
        from arke.integration.kernel_cache import KernelCache

        cache = KernelCache()
        shapes = get_gpt2_shapes(seq_len)
        cache.precompile_matmul(shapes["matmul"])

        n_linear, n_softmax = patch_gpt2_fast(model, cache)
        logger.info(f"  Arke patched: {n_linear} linear, {n_softmax} softmax")

        torch.cuda.reset_peak_memory_stats()
        logits = _get_logits(model, input_ids)
        perf = _profile(model, input_ids)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        return perf, logits, peak_mb
    except Exception as e:
        logger.warning(f"Arke patching failed: {e}")
        return None


# ── Correctness ─────────────────────────────────────────────


def _check_correctness(
    ref_logits: torch.Tensor,
    test_logits: torch.Tensor,
) -> tuple[bool, float, float, bool]:
    """Compare logits; return (correct, max_diff, mean_diff, top1_match)."""
    diff = (test_logits.float() - ref_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_top = ref_logits[:, -1, :].argmax(dim=-1)
    test_top = test_logits[:, -1, :].argmax(dim=-1)
    top1_match = (ref_top == test_top).all().item()
    correct = max_diff < 5.0  # fp16 tolerance
    return correct, max_diff, mean_diff, top1_match


# ── Main runner ─────────────────────────────────────────────


def run_l3_single(
    seq_len: int,
) -> list[E2EResult]:
    """Run L3 benchmark for a single seq_len."""
    logger.info(f"\n{'='*60}")
    logger.info(f"L3 E2E Benchmark: GPT-2 Small, seq_len={seq_len}")
    logger.info(f"{'='*60}")

    results: list[E2EResult] = []

    # Load fresh model for each seq_len to avoid cross-contamination
    model, tokenizer = _load_gpt2()
    input_ids = _make_input(tokenizer, seq_len)

    # 1. Eager mode (reference)
    logger.info("\n--- Eager mode ---")
    eager_perf, eager_logits, eager_mem = _run_eager(model, input_ids)
    logger.info(
        f"  Mean: {eager_perf['mean_ms']:.2f} ms, "
        f"Peak mem: {eager_mem:.1f} MB"
    )
    results.append(E2EResult(
        model="GPT-2 Small",
        seq_len=seq_len,
        mode="eager",
        source=(
            f"PyTorch {torch.__version__} eager mode | "
            "https://pytorch.org | License: BSD-3-Clause"
        ),
        mean_ms=eager_perf["mean_ms"],
        min_ms=eager_perf["min_ms"],
        max_ms=eager_perf["max_ms"],
        median_ms=eager_perf["median_ms"],
        peak_memory_mb=eager_mem,
        correct=True,
        max_logit_diff=0.0,
        mean_logit_diff=0.0,
        top1_match=True,
    ))

    # 2. torch.compile
    logger.info("\n--- torch.compile mode ---")
    # Reload model to clear any state
    del model
    torch.cuda.empty_cache()
    model, _ = _load_gpt2()
    compile_result = _run_compile(model, input_ids)
    if compile_result is not None:
        comp_perf, comp_logits, comp_mem = compile_result
        correct, max_d, mean_d, top1 = _check_correctness(
            eager_logits, comp_logits
        )
        logger.info(
            f"  Mean: {comp_perf['mean_ms']:.2f} ms, "
            f"Peak mem: {comp_mem:.1f} MB, "
            f"Correct: {correct}"
        )
        results.append(E2EResult(
            model="GPT-2 Small",
            seq_len=seq_len,
            mode="torch.compile",
            source=(
                f"torch.compile (Inductor) via PyTorch {torch.__version__} | "
                "https://pytorch.org | License: BSD-3-Clause"
            ),
            mean_ms=comp_perf["mean_ms"],
            min_ms=comp_perf["min_ms"],
            max_ms=comp_perf["max_ms"],
            median_ms=comp_perf["median_ms"],
            peak_memory_mb=comp_mem,
            correct=correct,
            max_logit_diff=max_d,
            mean_logit_diff=mean_d,
            top1_match=top1,
        ))

    # 3. Arke-patched
    logger.info("\n--- Arke-patched mode ---")
    del model
    torch.cuda.empty_cache()
    model, _ = _load_gpt2()
    arke_result = _run_arke(model, input_ids, seq_len)
    if arke_result is not None:
        arke_perf, arke_logits, arke_mem = arke_result
        correct, max_d, mean_d, top1 = _check_correctness(
            eager_logits, arke_logits
        )
        arke_src = "unknown"
        try:
            import arke as _arke
            arke_src = getattr(_arke, "__version__", "unknown")
        except Exception:
            pass
        logger.info(
            f"  Mean: {arke_perf['mean_ms']:.2f} ms, "
            f"Peak mem: {arke_mem:.1f} MB, "
            f"Correct: {correct}"
        )
        results.append(E2EResult(
            model="GPT-2 Small",
            seq_len=seq_len,
            mode="arke",
            source=(
                f"Arke {arke_src} (KernelCache monkey-patch) | "
                "https://github.com/arke-ai/arke | License: Apache-2.0"
            ),
            mean_ms=arke_perf["mean_ms"],
            min_ms=arke_perf["min_ms"],
            max_ms=arke_perf["max_ms"],
            median_ms=arke_perf["median_ms"],
            peak_memory_mb=arke_mem,
            correct=correct,
            max_logit_diff=max_d,
            mean_logit_diff=mean_d,
            top1_match=top1,
        ))

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def save_results(
    results: list[E2EResult],
    output_dir: Path,
) -> Path:
    """Save L3 results as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gpt2_results.csv"

    fieldnames = [
        "model", "seq_len", "mode", "source",
        "mean_ms", "min_ms", "max_ms", "median_ms",
        "peak_memory_mb", "correct", "max_logit_diff",
        "mean_logit_diff", "top1_match",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r.model,
                "seq_len": r.seq_len,
                "mode": r.mode,
                "source": r.source,
                "mean_ms": f"{r.mean_ms:.2f}",
                "min_ms": f"{r.min_ms:.2f}",
                "max_ms": f"{r.max_ms:.2f}",
                "median_ms": f"{r.median_ms:.2f}",
                "peak_memory_mb": f"{r.peak_memory_mb:.1f}",
                "correct": r.correct,
                "max_logit_diff": f"{r.max_logit_diff:.4f}",
                "mean_logit_diff": f"{r.mean_logit_diff:.6f}",
                "top1_match": r.top1_match,
            })

    return csv_path


def print_comparison_table(results: list[E2EResult]) -> None:
    """Print a comparison table of L3 results."""
    print(f"\n{'='*80}")
    print("L3 E2E GPT-2 Small — Comparison")
    print(f"{'='*80}")
    print(
        f"{'SeqLen':>7s} {'Mode':>15s} {'Mean(ms)':>10s} {'Min(ms)':>10s} "
        f"{'Memory(MB)':>11s} {'Correct':>8s} {'Top1':>5s}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.seq_len:>7d} {r.mode:>15s} {r.mean_ms:>10.2f} {r.min_ms:>10.2f} "
            f"{r.peak_memory_mb:>11.1f} {'✅' if r.correct else '❌':>8s} "
            f"{'✅' if r.top1_match else '❌':>5s}"
        )

    print(f"{'='*80}")


def run_l3(
    seq_lens: list[int],
    output_dir: str = "benchmarks/results",
) -> list[E2EResult]:
    """Run L3 benchmark suite across seq_lens."""
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    base_dir = Path(output_dir) / "L3" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save hardware info
    hw = collect_hardware_info()
    hw.save(str(base_dir / "hardware.json"))

    # Save sources manifest
    sources_manifest: dict[str, dict[str, str]] = {
        "eager": {
            "description": "PyTorch eager mode (SDPA attention)",
            "source": f"PyTorch {torch.__version__}",
        },
        "torch.compile": {
            "description": "torch.compile (Inductor, reduce-overhead)",
            "source": f"PyTorch {torch.__version__}",
        },
    }
    try:
        import arke as _arke
        v = getattr(_arke, "__version__", "unknown")
        sources_manifest["arke"] = {
            "description": "Arke KernelCache monkey-patch",
            "source": f"Arke {v}",
        }
    except ImportError:
        pass

    with open(base_dir / "sources.json", "w") as f:
        json.dump(sources_manifest, f, indent=2)

    # Save config
    config = {
        "timestamp": timestamp,
        "model": "GPT-2 Small",
        "seq_lens": seq_lens,
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "layer": "L3",
    }
    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_results: list[E2EResult] = []
    for seq_len in seq_lens:
        results = run_l3_single(seq_len)
        all_results.extend(results)

    csv_path = save_results(all_results, base_dir)
    logger.info(f"Saved: {csv_path}")

    print_comparison_table(all_results)
    print(f"\nResults saved to: {base_dir}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3 E2E Model Benchmark (GPT-2 Small)"
    )
    parser.add_argument(
        "--seq-len",
        type=str,
        default=None,
        help="Comma-separated seq_lens (default: 128,256,512)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all default seq_lens (128, 256, 512)",
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all or args.seq_len is None:
        seq_lens = DEFAULT_SEQ_LENS
    else:
        seq_lens = [int(s.strip()) for s in args.seq_len.split(",")]

    run_l3(seq_lens=seq_lens, output_dir=args.output)


if __name__ == "__main__":
    main()
