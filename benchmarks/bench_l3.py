# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""L3 end-to-end model benchmark runner for Stage 8.

The runner establishes the repeatable G8[4] artifact contract for GPT-2:
PyTorch eager is the correctness/performance reference and torch.compile
(Inductor) is measured against it.  A CPU-safe mock model is available for CI
and contract tests when CUDA, transformers, or model downloads are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.hardware import collect_hardware_info

logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENS = [128, 512, 1024]
DEFAULT_OUTPUT_DIR = "benchmarks/results/phase1/stage8/track4/l3"
# D7-E1.6 (commit 145d772 follow-up): bumped from 5 → 10 because the diagnostic
# script proved warmup=5 still lets the first-call compile overhead bleed into
# the measurement window, producing apparent ~0.81x regressions that vanish at
# warmup=10. See diagnosis.md in benchmarks/results/phase1/stage8/track4/diagnose_2026-05-16/.
DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASURE_RUNS = 20
G8_GPT2_TARGET_RATIO = 0.95

# D7-E1.6: GPT-2 has 12 transformer layers + lm_head + embeddings, so a single
# forward pass under multiple seq_lens can register >8 distinct dynamo cache
# entries (the default ``torch._dynamo.config.cache_size_limit``). Once the
# limit is hit, dynamo evicts entries → re-compilation thrash → false-negative
# regressions vs eager. Diagnostic dynamo_explain.txt confirmed the eviction
# path. 64 is comfortably above 12 layers × 3 seq_lens × a few side-paths.
_DYNAMO_CACHE_SIZE_LIMIT = 64
try:
    import torch._dynamo  # noqa: F401  (registers config)
    torch._dynamo.config.cache_size_limit = _DYNAMO_CACHE_SIZE_LIMIT
except Exception as _e:  # pragma: no cover  (older torch w/o _dynamo)
    logger.warning("Could not bump torch._dynamo.config.cache_size_limit: %s", _e)


# Canonical mode names used throughout bench_l3 (CSV `mode` column, summary.json
# filter keys, tests). The CLI accepts a few common aliases (underscore,
# hyphen, dotted) and normalizes them via `_normalize_mode` so summary.json's
# filter expression (`r.mode == MODE_TORCH_COMPILE`) doesn't silently miss rows.
# Bug fix 2026-05-17: previously `--modes eager,torch_compile` (an ergonomic
# spelling that avoids shell-dot quirks) produced compile_rows=0 because the
# build_summary filter expected `torch.compile`.
MODE_EAGER = "eager"
MODE_TORCH_COMPILE = "torch.compile"

_MODE_ALIASES: dict[str, str] = {
    "eager": MODE_EAGER,
    "torch.compile": MODE_TORCH_COMPILE,
    "torch_compile": MODE_TORCH_COMPILE,
    "torch-compile": MODE_TORCH_COMPILE,
    "torchcompile": MODE_TORCH_COMPILE,
}


def _normalize_mode(mode: str) -> str:
    """Map a user-supplied --modes token to the canonical bench_l3 mode name.

    Raises ValueError if the token doesn't match a known mode — better to fail
    loud at parse time than to silently produce empty compile_rows in summary.
    """
    canonical = _MODE_ALIASES.get(mode.strip().lower())
    if canonical is None:
        known = sorted(set(_MODE_ALIASES.values()))
        aliases = sorted(_MODE_ALIASES.keys())
        raise ValueError(
            f"Unknown bench_l3 mode {mode!r}; "
            f"canonical modes: {known}; accepted aliases: {aliases}"
        )
    return canonical


def _normalize_modes(modes: list[str]) -> list[str]:
    return [_normalize_mode(m) for m in modes]


@dataclass
class E2EResult:
    """Result of one model x sequence length x execution mode."""

    model: str
    seq_len: int
    mode: str
    source: str
    status: str
    reason: str = ""
    mean_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    median_ms: float | None = None
    peak_memory_mb: float | None = None
    correct: bool | None = None
    max_logit_diff: float | None = None
    mean_logit_diff: float | None = None
    top1_match: bool | None = None
    ratio_vs_eager: float | None = None

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.6f}"
            elif value is None:
                row[key] = ""
        return row


class _MockTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del text, return_tensors
        return {"input_ids": torch.arange(8, dtype=torch.long).unsqueeze(0)}


class _MockGPT2(torch.nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = torch.nn.Parameter(torch.linspace(0.01, 0.64, vocab_size))

    def forward(self, input_ids: torch.Tensor):
        base = input_ids.float().unsqueeze(-1)
        logits = base + self.weight.view(1, 1, -1)
        return type("MockCausalLMOutput", (), {"logits": logits})()


# -- Model loading ---------------------------------------------------------


def _load_gpt2(
    *,
    device: str,
    dtype: torch.dtype,
    model_name: str = "gpt2",
) -> tuple[torch.nn.Module, Any]:
    """Load GPT-2 model/tokenizer lazily so tests can run without transformers."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def _load_causal_lm(
    *,
    device: str,
    dtype: torch.dtype,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
) -> tuple[torch.nn.Module, Any]:
    """Generic HF causal-LM loader (AutoModelForCausalLM + AutoTokenizer).

    Used for the LLaMA-architecture E2E endpoint (D4=L2, Leon-approved
    2026-06-25): an ungated LLaMA-arch model (TinyLlama-1.1B) validates the
    LLaMA-family path on the 6GB dev box; full LLaMA-2 7B is deferred to a
    larger GPU. Works for any HF causal LM whose weights fit VRAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def _load_mock_model(
    *,
    device: str,
    dtype: torch.dtype,
    model_name: str = "mock-gpt2",
) -> tuple[torch.nn.Module, _MockTokenizer]:
    del model_name
    model = _MockGPT2().to(device=device, dtype=dtype)
    model.eval()
    return model, _MockTokenizer()


def _make_input(tokenizer: Any, seq_len: int, *, device: str) -> torch.Tensor:
    text = "The future of artificial intelligence is"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    if input_ids.shape[1] < seq_len:
        pad = torch.zeros(
            input_ids.shape[0],
            seq_len - input_ids.shape[1],
            dtype=torch.long,
            device=device,
        )
        input_ids = torch.cat([input_ids, pad], dim=1)
    elif input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, :seq_len]
    return input_ids


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "f16": torch.float16,
        "float32": torch.float32,
        "f32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


# -- Profiling -------------------------------------------------------------


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def _get_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits.detach().clone()


def _profile(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
    _sync(device)

    if device == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        with torch.no_grad():
            for idx in range(runs):
                starts[idx].record()
                model(input_ids)
                ends[idx].record()
        _sync(device)
        times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    else:
        times_ms = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                model(input_ids)
                times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms.sort()
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": times_ms[len(times_ms) // 2],
    }


def _check_correctness(
    ref_logits: torch.Tensor,
    test_logits: torch.Tensor,
) -> tuple[bool, float, float, bool]:
    diff = (test_logits.float() - ref_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_top = ref_logits[:, -1, :].argmax(dim=-1)
    test_top = test_logits[:, -1, :].argmax(dim=-1)
    top1_match = bool((ref_top == test_top).all().item())
    correct = top1_match and max_diff < 5.0
    return correct, max_diff, mean_diff, top1_match


def _run_mode(
    mode: str,
    *,
    model_loader: Callable[..., tuple[torch.nn.Module, Any]],
    tokenizer: Any,
    input_ids: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    model_name: str,
    warmup: int,
    runs: int,
    reference_logits: torch.Tensor | None = None,
    eager_mean_ms: float | None = None,
) -> tuple[E2EResult, torch.Tensor | None]:
    try:
        model, _ = model_loader(device=device, dtype=dtype, model_name=model_name)
        if mode == "torch.compile":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            # D7-E1.6: dynamic=True compiles a single shape-generic graph instead
            # of one Inductor kernel per seq_len. With static specialization the
            # per-seq forward generates fresh dynamo entries (12 layers × N seqs)
            # which thrashes the cache even after bumping cache_size_limit on
            # tight VRAM. dynamic=True keeps compile reused across {128,256,512}
            # and is supported alongside mode="reduce-overhead" in torch >= 2.5.
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        _reset_peak_memory(device)
        logits = _get_logits(model, input_ids)
        perf = _profile(model, input_ids, device=device, warmup=warmup, runs=runs)
        peak_mb = _peak_memory_mb(device)

        if reference_logits is None:
            correct, max_diff, mean_diff, top1 = True, 0.0, 0.0, True
        else:
            correct, max_diff, mean_diff, top1 = _check_correctness(reference_logits, logits)

        ratio = None
        if eager_mean_ms and perf["mean_ms"] > 0:
            ratio = eager_mean_ms / perf["mean_ms"]

        source = _source_for_mode(mode)
        return E2EResult(
            model=model_name,
            seq_len=input_ids.shape[1],
            mode=mode,
            source=source,
            status="ok",
            mean_ms=perf["mean_ms"],
            min_ms=perf["min_ms"],
            max_ms=perf["max_ms"],
            median_ms=perf["median_ms"],
            peak_memory_mb=peak_mb,
            correct=correct,
            max_logit_diff=max_diff,
            mean_logit_diff=mean_diff,
            top1_match=top1,
            ratio_vs_eager=ratio or 1.0 if reference_logits is None else ratio,
        ), logits
    except Exception as exc:
        return E2EResult(
            model=model_name,
            seq_len=input_ids.shape[1],
            mode=mode,
            source=_source_for_mode(mode),
            status="error",
            reason=str(exc),
            correct=False,
            top1_match=False,
        ), None
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()


def _source_for_mode(mode: str) -> str:
    if mode == "eager":
        return f"PyTorch {torch.__version__} eager | https://pytorch.org | BSD-3-Clause"
    if mode == "torch.compile":
        return (
            f"torch.compile Inductor via PyTorch {torch.__version__} | "
            "https://pytorch.org | BSD-3-Clause"
        )
    return mode


# -- Runner / artifacts ----------------------------------------------------


def run_l3_single(
    seq_len: int,
    *,
    model_name: str = "gpt2",
    model_loader: Callable[..., tuple[torch.nn.Module, Any]] = _load_gpt2,
    device: str = "auto",
    dtype: str = "auto",
    modes: list[str] | None = None,
    warmup: int = DEFAULT_WARMUP_RUNS,
    runs: int = DEFAULT_MEASURE_RUNS,
) -> list[E2EResult]:
    device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype, device)
    modes = modes or [MODE_EAGER, MODE_TORCH_COMPILE]

    _, tokenizer = model_loader(device=device, dtype=torch_dtype, model_name=model_name)
    input_ids = _make_input(tokenizer, seq_len, device=device)

    results: list[E2EResult] = []
    eager_result, eager_logits = _run_mode(
        "eager",
        model_loader=model_loader,
        tokenizer=tokenizer,
        input_ids=input_ids,
        device=device,
        dtype=torch_dtype,
        model_name=model_name,
        warmup=warmup,
        runs=runs,
    )
    del tokenizer
    results.append(eager_result)

    if eager_result.status != "ok" or eager_logits is None or eager_result.mean_ms is None:
        return results

    for mode in modes:
        if mode == "eager":
            continue
        result, _ = _run_mode(
            mode,
            model_loader=model_loader,
            tokenizer=None,
            input_ids=input_ids,
            device=device,
            dtype=torch_dtype,
            model_name=model_name,
            warmup=warmup,
            runs=runs,
            reference_logits=eager_logits,
            eager_mean_ms=eager_result.mean_ms,
        )
        results.append(result)
    return results


def run_l3(
    seq_lens: list[int],
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model: str = "gpt2",
    device: str = "auto",
    dtype: str = "auto",
    modes: list[str] | None = None,
    warmup: int = DEFAULT_WARMUP_RUNS,
    runs: int = DEFAULT_MEASURE_RUNS,
    mock: bool = False,
    arch: str = "gpt2",
) -> list[E2EResult]:
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    base_dir = Path(output_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    # Loader selection: mock (CI) → mock; arch=gpt2 → GPT-2-specific loader;
    # any other arch (e.g. 'llama' / 'causal_lm') → generic AutoModelForCausalLM
    # (D4=L2 LLaMA-family endpoint, Leon-approved 2026-06-25).
    if mock:
        model_loader = _load_mock_model
    elif arch == "gpt2":
        model_loader = _load_gpt2
    else:
        model_loader = _load_causal_lm
    model_name = "mock-gpt2" if mock else model
    modes = modes or [MODE_EAGER, MODE_TORCH_COMPILE]

    _write_provenance(
        base_dir,
        model=model_name,
        seq_lens=seq_lens,
        device=resolved_device,
        dtype=dtype,
        modes=modes,
        warmup=warmup,
        runs=runs,
        mock=mock,
    )

    all_results: list[E2EResult] = []
    for seq_len in seq_lens:
        logger.info("Running L3 %s seq_len=%s modes=%s", model_name, seq_len, modes)
        all_results.extend(run_l3_single(
            seq_len,
            model_name=model_name,
            model_loader=model_loader,
            device=resolved_device,
            dtype=dtype,
            modes=modes,
            warmup=warmup,
            runs=runs,
        ))

    save_results(all_results, base_dir)
    summary = build_summary(all_results, target_ratio=G8_GPT2_TARGET_RATIO)
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_comparison_table(all_results)
    print(f"\nResults saved to: {base_dir}")
    return all_results


def save_results(results: list[E2EResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gpt2_results.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(E2EResult(
        model="", seq_len=0, mode="", source="", status=""
    )).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())
    (output_dir / "results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n"
    )


def build_summary(results: list[E2EResult], *, target_ratio: float) -> dict[str, Any]:
    compile_rows = [r for r in results if r.mode == MODE_TORCH_COMPILE]
    eager_rows = [r for r in results if r.mode == MODE_EAGER]
    successful_compile = [
        r
        for r in compile_rows
        if r.status == "ok"
        and r.correct
        and r.top1_match
        and (r.ratio_vs_eager or 0.0) >= target_ratio
    ]
    ratios = [r.ratio_vs_eager for r in compile_rows if r.ratio_vs_eager is not None]
    return {
        "schema": "stage8-l3-gpt2-v1",
        "target_ratio_vs_eager": target_ratio,
        "rows": len(results),
        "eager_rows": len(eager_rows),
        "compile_rows": len(compile_rows),
        "compile_success_rows": len(successful_compile),
        "g8_gpt2_pass": bool(compile_rows) and len(successful_compile) == len(compile_rows),
        "min_compile_ratio_vs_eager": min(ratios) if ratios else None,
        "geomean_compile_ratio_vs_eager": _geomean(ratios),
        "all_correct": all((r.correct is not False) for r in results if r.status == "ok"),
        "errors": [asdict(r) for r in results if r.status != "ok"],
    }


def _write_provenance(base_dir: Path, **config: Any) -> None:
    try:
        hw = collect_hardware_info()
        if hasattr(hw, "save"):
            hw.save(str(base_dir / "hardware.json"))
        else:
            (base_dir / "hardware.json").write_text(json.dumps(hw, indent=2, default=str))
    except Exception as exc:
        (base_dir / "hardware.json").write_text(json.dumps({"error": str(exc)}, indent=2))

    (base_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (base_dir / "sources.json").write_text(json.dumps({
        "eager": _source_for_mode("eager"),
        "torch.compile": _source_for_mode("torch.compile"),
    }, indent=2) + "\n")


def _geomean(values: list[float]) -> float | None:
    positives = [v for v in values if v and v > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def print_comparison_table(results: list[E2EResult]) -> None:
    print("\n" + "=" * 92)
    print("L3 E2E GPT-2 — eager vs torch.compile")
    print("=" * 92)
    print(
        f"{'SeqLen':>7s} {'Mode':>15s} {'Status':>8s} {'Mean(ms)':>10s} "
        f"{'Ratio':>8s} {'Memory(MB)':>11s} {'Correct':>8s} {'Top1':>5s}"
    )
    print("-" * 92)
    for result in results:
        mean = "" if result.mean_ms is None else f"{result.mean_ms:.2f}"
        ratio = "" if result.ratio_vs_eager is None else f"{result.ratio_vs_eager:.3f}"
        mem = "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.1f}"
        correct = "" if result.correct is None else ("yes" if result.correct else "no")
        top1 = "" if result.top1_match is None else ("yes" if result.top1_match else "no")
        print(
            f"{result.seq_len:>7d} {result.mode:>15s} {result.status:>8s} "
            f"{mean:>10s} {ratio:>8s} {mem:>11s} {correct:>8s} {top1:>5s}"
        )
    print("=" * 92)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 L3 GPT-2 benchmark")
    parser.add_argument("--seq-len", type=str, default=None, help="Comma-separated seq lens")
    parser.add_argument("--all", action="store_true", help="Run default seq lens")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--arch",
        default="gpt2",
        help="Model architecture loader: 'gpt2' (GPT2LMHeadModel) or any other "
             "value (e.g. 'llama'/'causal_lm') → generic AutoModelForCausalLM.",
    )
    parser.add_argument("--modes", default="eager,torch.compile")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURE_RUNS)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use CPU/GPU-safe mock GPT-2 contract model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned config without loading models",
    )
    parser.add_argument("--list-seq-lens", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_seq_lens:
        print(",".join(str(x) for x in DEFAULT_SEQ_LENS))
        return

    seq_lens = (
        DEFAULT_SEQ_LENS
        if args.all or args.seq_len is None
        else _parse_int_list(args.seq_len)
    )
    modes = _normalize_modes(_parse_str_list(args.modes))

    if args.dry_run:
        print(json.dumps({
            "model": "mock-gpt2" if args.mock else args.model,
            "seq_lens": seq_lens,
            "modes": modes,
            "device": args.device,
            "dtype": args.dtype,
            "output": args.output,
            "warmup": args.warmup,
            "runs": args.runs,
            "mock": args.mock,
        }, indent=2))
        return

    run_l3(
        seq_lens,
        output_dir=args.output,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        modes=modes,
        warmup=args.warmup,
        runs=args.runs,
        mock=args.mock,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
