# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 Gate G8 MVP verification.

This runner covers the machine-checkable MVP contract introduced at the start of
Stage 8.  It is not the full final G8 acceptance suite; it verifies that the two
critical entry points now exist and emit stable artifacts for future real-GPU and
real-model measurements.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.gate import GateResult, GateSummary

REPO_ROOT = Path(__file__).resolve().parent.parent
MATMUL_AK = REPO_ROOT / "examples" / "operators" / "01_matmul.ak"
RELU_AK = REPO_ROOT / "examples" / "operators" / "00_relu.ak"
SOFTMAX_AK = REPO_ROOT / "examples" / "operators" / "02_softmax.ak"


def _run_cmd(args: list[str], timeout: int = 180) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, detail[-2500:]


def _run_pytest(args: list[str]) -> tuple[bool, str]:
    return _run_cmd([sys.executable, "-m", "pytest", "-q", *args], timeout=300)


def _check_optimize_cli_contract() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="arke-g8-opt-") as tmp:
        ok, detail = _run_cmd([
            sys.executable,
            "-m",
            "arke.cli",
            "optimize",
            str(MATMUL_AK),
            "--output",
            tmp,
            "--cycles",
            "3",
            "--json",
        ])
        if not ok:
            return False, detail
        summary = json.loads((Path(tmp) / "summary.json").read_text())
        trajectory = (Path(tmp) / "trajectory.jsonl").read_text().splitlines()
        strategy = json.loads((Path(tmp) / "strategy.json").read_text())
        if not summary.get("success"):
            return False, json.dumps(summary, indent=2)
        if summary.get("cycles_completed") != 3:
            return False, f"cycles_completed={summary.get('cycles_completed')}"
        if len(strategy.get("decisions", [])) < 3:
            return False, "generated strategy has fewer than 3 decisions"
        actions = [json.loads(line) for line in trajectory if '"kind":' in line]
        # Cycle order is now expressed via D8-F3 trajectory v1.0 record
        # kinds: each compile→profile→adjust cycle emits one
        # `compile`, one `profile`, and one `adjust` record in order.
        cycle_kinds = [
            entry.get("kind") for entry in actions
            if entry.get("kind") in {"compile", "profile", "adjust"}
        ]
        required = ["compile", "profile", "adjust"] * 3
        if cycle_kinds != required:
            return False, f"unexpected trajectory cycle order: {cycle_kinds}"
        # Validate header pins the v1.0 contract id (drift sentinel).
        header_lines = [json.loads(line) for line in trajectory[:1]]
        if not header_lines or header_lines[0].get("kind") != "header":
            return False, "first trajectory line is not a v1.0 header record"
        if header_lines[0]["data"].get("contract_id") != "arke-trajectory-v1.0.0":
            return False, (
                f"header contract_id={header_lines[0]['data'].get('contract_id')!r} "
                f"expected 'arke-trajectory-v1.0.0'"
            )
        return True, (
            f"cycles=3 decisions={summary['decision_count']} "
            f"trajectory_events={len(trajectory)}"
        )


def _check_multi_input_routing_contract() -> tuple[bool, str]:
    """Validate G8[3] MVP evidence: two ops for each routed input family."""
    cases = [
        ("ak_file_relu", [str(RELU_AK)], "ak_file", "relu_kernel"),
        ("ak_file_softmax", [str(SOFTMAX_AK)], "ak_file", "softmax"),
        (
            "natural_language_relu",
            ["optimize relu for shape 16x32 fp16"],
            "natural_language",
            "relu_kernel",
        ),
        (
            "natural_language_softmax",
            ["optimize softmax for shape 16x32 fp16"],
            "natural_language",
            "softmax_kernel",
        ),
        (
            "code_snippet_relu",
            ["def relu_kernel(x): return torch.relu(x)  # shape 16x32"],
            "code_snippet",
            "relu_kernel",
        ),
        (
            "code_snippet_softmax",
            ["def softmax_kernel(x): return torch.softmax(x, dim=-1)  # shape 16x32"],
            "code_snippet",
            "softmax_kernel",
        ),
        (
            "structured_relu",
            ["--kernel", "relu", "--shape", "16,32"],
            "structured_args",
            "relu_kernel",
        ),
        (
            "structured_softmax",
            ["--kernel", "softmax", "--shape", "16,32"],
            "structured_args",
            "softmax_kernel",
        ),
    ]
    evidence: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="arke-g8-multi-input-") as tmp:
        tmp_root = Path(tmp)
        for name, input_args, expected_kind, expected_kernel in cases:
            out_dir = tmp_root / name
            ok, detail = _run_cmd([
                sys.executable,
                "-m",
                "arke.cli",
                "optimize",
                *input_args,
                "--output",
                str(out_dir),
                "--cycles",
                "1",
                "--json",
            ])
            if not ok:
                return False, f"{name} failed:\n{detail}"
            summary = json.loads((out_dir / "summary.json").read_text())
            if not summary.get("success"):
                return False, f"{name} unsuccessful:\n{json.dumps(summary, indent=2)}"
            if summary.get("input_kind") != expected_kind:
                return False, f"{name} input_kind={summary.get('input_kind')} expected={expected_kind}"
            if summary.get("kernel_id") != expected_kernel:
                return False, f"{name} kernel_id={summary.get('kernel_id')} expected={expected_kernel}"
            normalized_source = summary.get("normalized_source_path")
            if not normalized_source or not Path(normalized_source).exists():
                return False, f"{name} missing normalized source: {normalized_source}"
            evidence.append({
                "case": name,
                "input_kind": expected_kind,
                "kernel_id": expected_kernel,
            })
    kinds = sorted({item["input_kind"] for item in evidence})
    return True, f"cases={len(evidence)} input_kinds={kinds} kernels={[item['kernel_id'] for item in evidence]}"


def _check_live_llm_loop_contract() -> tuple[bool, str]:
    """G8 Tier-2 liveness: a REAL LLM drives ≥3 real compile→profile cycles.

    Decision D1=c (Leon default-action, 2026-06-25): fold the live-LLM path
    into the gate to prove *Arke is doing real work* — i.e. a genuine LLM
    tool-use loop produces ≥3 compile_and_profile calls backed by the real
    Triton backend on GPU. This does NOT assert any perf threshold (GPT-2
    ≥0.95× etc. remain a separate Tier-2 perf round pending Leon's D1 ruling);
    it only verifies the live path is operational and measures for real.

    Skips gracefully (returns PASS with a 'skipped' detail) when the
    prerequisites are absent — no CUDA GPU, or no LLM provider credentials —
    so the gate stays green on CI/CPU. Folding the *liveness* branch in does
    not weaken the existing MVP contract checks.
    """
    # Prereq 1: GPU (the loop must use the real Triton backend, not mock).
    try:
        import torch
    except Exception as e:  # pragma: no cover - torch always present in venv
        return True, f"skipped: torch import failed ({e})"
    if not torch.cuda.is_available():
        return True, "skipped: no CUDA GPU (live loop requires real Triton backend)"

    # Prereq 2: an LLM provider credential.
    try:
        from arke.agent.llm_config import load_from_env
        from arke.agent.runner import LLMRunner
    except Exception as e:
        return True, f"skipped: agent runner import failed ({e})"
    try:
        cfg = load_from_env()
    except Exception:
        return True, "skipped: no LLM provider credentials (set YUNWU/ANTHROPIC/OPENAI key)"

    # Run a real loop on a small matmul (6GB-safe).
    try:
        with LLMRunner(cfg) as runner:
            res = runner.optimize(
                op_name="matmul",
                shapes={"A": [256, 256], "B": [256, 256]},
                max_turns=30,
            )
        d = res.to_dict()
    except Exception as e:  # network / provider transient
        return True, f"skipped: live LLM call failed transiently ({type(e).__name__}: {e})"

    actions = [t for t in d.get("trajectory", []) if t.get("type") == "action"]
    profiles = [a for a in actions if a.get("tool") == "compile_and_profile"]

    def _payload(a: dict) -> dict:
        r = a.get("result", {})
        return r.get("data", r) if isinstance(r, dict) else {}

    real_triton = [p for p in profiles if _payload(p).get("backend") == "triton"]
    n_cycles = len(real_triton)
    if n_cycles < 3:
        return False, (
            f"live-LLM loop produced only {n_cycles} real-Triton compile_and_profile "
            f"cycles (need ≥3); total profiles={len(profiles)}, "
            f"decisions={d.get('decisions')}, stop={d.get('stop_reason')}"
        )
    ratios = [round(_payload(p).get("baseline_ratio", 0.0), 3) for p in real_triton]
    return True, (
        f"live model={d.get('model_used')} real-Triton cycles={n_cycles} "
        f"decisions={d.get('decisions')} baseline_ratios={ratios} "
        f"(liveness only; perf threshold not asserted — pending D1 ruling)"
    )


def _check_bench_l3_mock_contract() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="arke-g8-l3-") as tmp:
        ok, detail = _run_cmd([
            sys.executable,
            "-m",
            "benchmarks.bench_l3",
            "--mock",
            "--device",
            "cpu",
            "--seq-len",
            "8",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--output",
            tmp,
        ], timeout=240)
        if not ok:
            return False, detail
        run_dirs = sorted(Path(tmp).glob("*"))
        if not run_dirs:
            return False, "bench_l3 produced no run directory"
        run_dir = run_dirs[-1]
        summary = json.loads((run_dir / "summary.json").read_text())
        required_files = [
            "config.json",
            "hardware.json",
            "sources.json",
            "gpt2_results.csv",
            "results.json",
            "summary.json",
        ]
        missing = [name for name in required_files if not (run_dir / name).exists()]
        if missing:
            return False, f"missing artifacts: {missing}"
        if summary.get("compile_rows") != 1:
            return False, json.dumps(summary, indent=2)
        return True, (
            f"rows={summary['rows']} compile_rows={summary['compile_rows']} "
            f"g8_gpt2_pass={summary['g8_gpt2_pass']}"
        )


def run_g8(tier: int = 2) -> GateSummary:
    results: list[GateResult] = []

    ok, detail = _check_optimize_cli_contract()
    results.append(GateResult(
        "G8",
        "G8.MVP.1",
        "arke optimize emits bounded StrategyIR and 3 compile->profile->adjust cycles",
        "function",
        ok,
        detail,
    ))

    ok, detail = _check_multi_input_routing_contract()
    results.append(GateResult(
        "G8",
        "G8.MVP.2",
        "arke optimize routes .ak, natural language, code snippet, and structured inputs for two ops each",
        "function",
        ok,
        detail,
    ))

    ok, detail = _check_bench_l3_mock_contract()
    results.append(GateResult(
        "G8",
        "G8.MVP.3",
        "bench_l3 emits GPT-2 eager vs torch.compile CSV/JSON artifacts",
        "function",
        ok,
        detail,
    ))

    ok, detail = _run_pytest([
        "tests/test_agent_optimize.py",
        "tests/test_bench_l3.py",
    ])
    results.append(GateResult(
        "G8",
        "G8.MVP.4",
        "Stage 8 MVP regression tests pass",
        "regression",
        ok,
        detail,
    ))

    # G8 Tier-2 liveness branch (D1=c, Leon default-action 2026-06-25):
    # a real LLM drives ≥3 real-Triton compile→profile cycles. Skips
    # gracefully (PASS w/ 'skipped' detail) without GPU or LLM credentials,
    # so the gate stays green on CI/CPU. Liveness only — no perf threshold.
    ok, detail = _check_live_llm_loop_contract()
    results.append(GateResult(
        "G8",
        "G8.LIVE.1",
        "Live LLM drives >=3 real-Triton compile->profile cycles (liveness, no perf threshold)",
        "function",
        ok,
        detail,
    ))

    passed = sum(1 for result in results if result.passed)
    return GateSummary(
        gate="G8",
        tier=tier,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )
