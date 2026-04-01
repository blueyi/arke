# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark execution — runs Arke and Direct-Triton on all tasks.

Usage:
    python -m benchmarks.run --method arke --trials 3
    python -m benchmarks.run --method direct --trials 3
    python -m benchmarks.run --method both --trials 3
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from arke.agent.llm_config import load_from_openclaw
from arke.agent.runner import LLMRunner
from benchmarks.runner import BenchmarkReport, TaskSummary, TrialResult
from benchmarks.tasks import BENCHMARK_TASKS, BenchmarkTask

logger = logging.getLogger(__name__)


def _make_archive_dir(output_dir: str, timestamp: str) -> Path:
    """Create a timestamped archive directory.

    Structure: {output_dir}/{phase}/{timestamp}/
    Phase is inferred from the output_dir name or defaults to 'run'.
    """
    # Use parent dir name as phase identifier
    base = Path(output_dir)
    archive = base / timestamp
    archive.mkdir(parents=True, exist_ok=True)
    return archive


def _save_kernel(
    archive_dir: Path,
    task_name: str,
    method: str,
    trial: int,
    code: str,
) -> str | None:
    """Save generated Triton kernel code to archive.

    Returns the saved file path, or None if no code.
    """
    if not code:
        return None
    kernel_dir = archive_dir / "triton_kernels" / method
    kernel_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{task_name}_t{trial}.py"
    filepath = kernel_dir / filename
    filepath.write_text(code)
    return str(filepath)


def _save_arke_ir(
    archive_dir: Path,
    task: BenchmarkTask,
) -> str:
    """Save Arke IR source (SemanticIR JSON) to archive.

    Returns the saved file path.
    """
    ir_dir = archive_dir / "arke_ir"
    ir_dir.mkdir(parents=True, exist_ok=True)
    filepath = ir_dir / f"{task.name}.json"
    filepath.write_text(task.semantic_ir.to_json())
    return str(filepath)


def run_arke_trial(
    task: BenchmarkTask,
    runner: LLMRunner,
    trial: int,
    max_turns: int = 25,
    archive_dir: Path | None = None,
) -> TrialResult:
    """Run one Arke optimization trial on a task."""
    logger.info(f"[Arke] {task.name} trial {trial}")
    start = time.time()

    try:
        result = runner.optimize(
            semantic_ir=task.semantic_ir,
            target_hw=task.target_hw,
            max_turns=max_turns,
        )

        # Extract performance from session summary
        summary = result.session_summary
        best_perf = summary.get("best_performance", {})
        vs_baseline = best_perf.get("vs_baseline") if best_perf else None
        latency = best_perf.get("latency_us") if best_perf else None
        tflops_val = best_perf.get("tflops") if best_perf else None

        # Check correctness from trajectory
        # (look for last verify_correctness result)
        correct = False
        for entry in reversed(result.trajectory):
            if entry.get("tool") == "verify_correctness":
                verify_result = entry.get("result", {})
                correct = verify_result.get("passed", False)
                break

        # Save generated kernel file
        kernel_path = _save_kernel(
            archive_dir, task.name, "arke", trial, result.generated_code
        ) if archive_dir else None
        if kernel_path:
            logger.info(f"  Saved kernel: {kernel_path}")

        return TrialResult(
            task_name=task.name,
            method="arke",
            trial=trial,
            correct=correct,
            vs_baseline=vs_baseline,
            latency_us=latency,
            tflops=tflops_val,
            decisions=result.decisions,
            tool_calls=result.tool_calls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            duration_s=time.time() - start,
        )
    except Exception as e:
        logger.error(f"[Arke] {task.name} trial {trial} failed: {e}")
        return TrialResult(
            task_name=task.name,
            method="arke",
            trial=trial,
            correct=False,
            error=str(e),
            duration_s=time.time() - start,
        )


def _build_direct_prompt(task: BenchmarkTask) -> str:
    """Build prompt for LLM to write Triton kernel directly."""
    ir = task.semantic_ir
    params_desc = []
    for p in ir.params:
        params_desc.append(f"  {p.name}: shape={p.shape}, dtype={p.dtype}")

    ops_desc = []
    for node in ir.nodes:
        inputs_str = ", ".join(
            f"{k}={v}" for k, v in node.inputs.items()
        )
        ops_desc.append(f"  {node.id} = {node.op}({inputs_str})")

    out_shape = ir.return_type.shape if ir.return_type else "inferred"
    out_dtype = ir.return_type.dtype if ir.return_type else "f16"

    return f"""Write a complete, optimized Triton kernel for the following computation.

Kernel: {ir.kernel_id}
Parameters:
{chr(10).join(params_desc)}

Computation:
{chr(10).join(ops_desc)}

Output: {ir.return_node} with shape {out_shape}, dtype {out_dtype}

Target: NVIDIA Ampere (RTX 3060, 28 SMs, 128KB shared mem, warp=32)

Requirements:
1. Write a complete Python file with `import triton` and `import triton.language as tl`
2. Define the Triton kernel with `@triton.jit`
3. Define a wrapper function named `{ir.kernel_id}` that:
   - Takes PyTorch tensors as arguments (in order: {', '.join(p.name for p in ir.params)})
   - Allocates the output tensor
   - Launches the kernel with appropriate grid
   - Returns the output tensor
4. Optimize for maximum GPU performance (good tile sizes, memory coalescing, etc.)

Return ONLY the Python code, no explanation. The code must be directly executable."""


def _extract_code(response_text: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code between ```python and ```
    if "```python" in response_text:
        parts = response_text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()

    # Try ``` blocks
    if "```" in response_text:
        parts = response_text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            return code.strip()

    # Assume entire response is code
    return response_text.strip()


def run_direct_trial(
    task: BenchmarkTask,
    runner: LLMRunner,
    trial: int,
    archive_dir: Path | None = None,
) -> TrialResult:
    """Run one LLM-direct-Triton trial on a task.

    The LLM writes Triton kernel code directly (no IR, no validation).
    We then compile and verify the result.
    """

    logger.info(f"[Direct] {task.name} trial {trial}")
    start = time.time()

    try:
        # Use the same LLM to write Triton code directly
        spec = runner.config.primary
        provider, model = runner.config.get_provider_and_model(spec)

        prompt = _build_direct_prompt(task)
        messages = [{"role": "user", "content": prompt}]

        # Call LLM (no tools, just code generation)
        response = runner._call_llm(
            provider, model, messages, tools=False
        )

        if response is None:
            return TrialResult(
                task_name=task.name,
                method="direct",
                trial=trial,
                correct=False,
                error="LLM returned no response",
                duration_s=time.time() - start,
            )

        # Extract usage
        usage = response.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        # Extract code from response
        content = response.get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
            elif isinstance(block, str):
                text += block

        code = _extract_code(text)
        if not code:
            return TrialResult(
                task_name=task.name,
                method="direct",
                trial=trial,
                correct=False,
                error="Could not extract code from LLM response",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_s=time.time() - start,
            )

        # Save generated kernel file
        kernel_path = _save_kernel(
            archive_dir, task.name, "direct", trial, code
        ) if archive_dir else None
        if kernel_path:
            logger.info(f"  Saved kernel: {kernel_path}")

        # Try to compile and verify
        from arke.backend.compiler import TritonCompiler
        compiler = TritonCompiler()

        compiled = compiler.compile(code)
        if not compiled.success:
            return TrialResult(
                task_name=task.name,
                method="direct",
                trial=trial,
                correct=False,
                error=f"Compilation failed: {compiled.error}",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_s=time.time() - start,
            )

        # Verify correctness
        import torch

        from arke.engine.numerical_check import NumericalValidator

        validator = NumericalValidator()
        np_inputs = validator.generate_random_inputs(
            task.semantic_ir, seed=42 + trial
        )
        np_ref = validator.generate_reference(
            task.semantic_ir, np_inputs
        )

        # Run on GPU
        gpu_inputs = {}
        dtype_map = {
            "f16": torch.float16,
            "f32": torch.float32,
        }
        for p in task.semantic_ir.params:
            t_dtype = dtype_map.get(p.dtype, torch.float16)
            gpu_inputs[p.name] = torch.from_numpy(
                np_inputs[p.name].astype(np.float32)
            ).to(dtype=t_dtype, device="cuda")

        try:
            gpu_output = compiler.run(compiled, gpu_inputs)
            if isinstance(gpu_output, dict):
                gpu_output = gpu_output.get("output", gpu_output)
            gpu_np = gpu_output.cpu().float().numpy()
        except Exception as e:
            return TrialResult(
                task_name=task.name,
                method="direct",
                trial=trial,
                correct=False,
                error=f"GPU execution failed: {e}",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_s=time.time() - start,
            )

        # Check correctness (same-dtype)
        ref_f32 = np_ref.astype(np.float32)
        correct = bool(np.allclose(
            gpu_np, ref_f32, atol=0.1, rtol=0.05
        ))

        # Profile if correct
        vs_baseline = None
        latency_us = None
        tflops_val = None
        if correct:
            try:
                profile_result = compiler.profile(
                    code, gpu_inputs, warmup=5, runs=20
                )
                vs_baseline = profile_result.vs_baseline
                latency_us = profile_result.latency_us
                tflops_val = profile_result.tflops
            except Exception:
                pass  # Profile failure is not a correctness issue

        return TrialResult(
            task_name=task.name,
            method="direct",
            trial=trial,
            correct=correct,
            vs_baseline=vs_baseline,
            latency_us=latency_us,
            tflops=tflops_val,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_s=time.time() - start,
        )

    except Exception as e:
        logger.error(f"[Direct] {task.name} trial {trial} failed: {e}")
        return TrialResult(
            task_name=task.name,
            method="direct",
            trial=trial,
            correct=False,
            error=str(e),
            duration_s=time.time() - start,
        )


def run_benchmark(
    methods: list[str],
    trials: int = 3,
    tasks: list[BenchmarkTask] | None = None,
    output_dir: str = "benchmarks/results",
    phase: str = "run",
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    import os

    if tasks is None:
        tasks = BENCHMARK_TASKS

    # Load LLM config
    openclaw_dir = os.environ.get(
        "OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")
    )
    config = load_from_openclaw(openclaw_dir)
    runner = LLMRunner(config, timeout=300.0)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # Create archive directory: {output_dir}/{phase}/{timestamp}/
    archive_dir = Path(output_dir) / phase / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Save Arke IR source files
    for task in tasks:
        _save_arke_ir(archive_dir, task)

    try:
        for task in tasks:
            if "arke" in methods:
                summary = TaskSummary(
                    task_name=task.name, method="arke"
                )
                for t in range(trials):
                    result = run_arke_trial(
                        task, runner, t, archive_dir=archive_dir
                    )
                    summary.trials.append(result)
                    logger.info(
                        f"  [Arke] {task.name} t{t}: "
                        f"correct={result.correct}, "
                        f"perf={result.vs_baseline}"
                    )
                report.arke_results[task.name] = summary

            if "direct" in methods:
                summary = TaskSummary(
                    task_name=task.name, method="direct"
                )
                for t in range(trials):
                    result = run_direct_trial(
                        task, runner, t, archive_dir=archive_dir
                    )
                    summary.trials.append(result)
                    logger.info(
                        f"  [Direct] {task.name} t{t}: "
                        f"correct={result.correct}, "
                        f"perf={result.vs_baseline}"
                    )
                report.direct_results[task.name] = summary

    finally:
        runner.close()

    # Save results to archive
    report_json_path = str(archive_dir / "benchmark_report.json")
    report.save(report_json_path)

    # Export CSV and task catalog
    from benchmarks.export import export_csv, export_task_catalog
    csv_path = export_csv(report_json_path, str(archive_dir / "benchmark_results.csv"))
    catalog_path = export_task_catalog(str(archive_dir / "task_catalog.csv"))
    print(f"\nArchive: {archive_dir}")
    print(f"CSV results: {csv_path}")
    print(f"Task catalog: {catalog_path}")
    print(f"Arke IR: {archive_dir / 'arke_ir/'}")
    print(f"Triton kernels: {archive_dir / 'triton_kernels/'}")

    # Print Gate G4 summary
    passed, reasons = report.gate_g4_pass()
    print("\n" + "=" * 60)
    print("Gate G4 Evaluation")
    print("=" * 60)
    for reason in reasons:
        print(f"  {reason}")
    print(f"\n  Gate G4: {'PASS ✅' if passed else 'FAIL ❌'}")
    print("=" * 60)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Arke benchmarks"
    )
    parser.add_argument(
        "--method",
        choices=["arke", "direct", "both"],
        default="both",
        help="Which method(s) to benchmark",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of trials per task",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific task names (default: all)",
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
        help="Output directory",
    )
    parser.add_argument(
        "--phase", default="run",
        help="Phase/stage label for archival (e.g. phase1.5_baseline)",
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3], default=None,
        help="Tier level for task selection (1=core, 2=extended, 3=full). "
             "Overrides --tasks.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
    )

    methods = (
        ["arke", "direct"] if args.method == "both"
        else [args.method]
    )

    tasks = None
    if args.tier:
        # Use tier-based task selection from skill scripts
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_tier",
            Path(__file__).parent.parent
            / "skills" / "arke-test-coverage" / "scripts" / "run_tier.py",
        )
        run_tier_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_tier_mod)
        tasks = run_tier_mod.build_benchmark_tasks(args.tier)
        logger.info(f"Tier {args.tier}: {len(tasks)} tasks")
    elif args.tasks:
        from benchmarks.tasks import get_task
        tasks = [get_task(name) for name in args.tasks]

    run_benchmark(
        methods=methods,
        trials=args.trials,
        tasks=tasks,
        output_dir=args.output,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
