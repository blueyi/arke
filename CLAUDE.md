# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Arke is an AI-first operator description language, multi-layer IR, compiler toolchain, and agent
system for GPU/NPU kernels. The central design idea is a hard split between **SemanticIR** (what to
compute, immutable math) and **StrategyIR** (how to optimize, explicit `@rationale`-tagged
decisions). LLM agents are users of Arke — they choose optimizations from a bounded action space,
and the compiler verifies legality, correctness, and performance.

The active line is `v0.1.0` for the package, the `.ak` language schema, and the `.akir` IR schema —
they're versioned together. Phase 1 targets `Arke → Triton → NVIDIA`. Stage 7/8 of Phase 1 is the
active work; see `STAGE7_PROGRESS_REPORT.md` and `docs/phase1/stage*-plan.md`.

## Commands

Setup (creates `.venv/` and installs editable):

```bash
make setup-cpu         # CPU-only dev environment
make setup-gpu         # adds torch + triton
make setup-bench       # adds the benchmark stack (flag-gems, liger-kernel, ...)
```

Day-to-day:

```bash
make test                              # pytest tests/ -v --tb=short (CPU)
make test-gpu                          # ARKE_GPU_TESTS=1 pytest tests/
pytest tests/test_pipeline.py -v       # single test file
pytest tests/ -k "stage7 and lowering" # filter by name
pytest -m cuda                         # CUDA-marked tests only

make lint        # ruff check arke/ tests/ benchmarks/
make format      # ruff format ...
make check       # ruff + mypy (--ignore-missing-imports) + pytest
```

CLI:

```bash
arke compile examples/operators/01_matmul.ak              # → stdout JSON
arke compile examples/operators/01_matmul.ak -o out.akir  # → file
arke optimize <.ak file | inline source | NL request> --cycles 3 --json
arke bench --bl 2                                          # benchmark, BL=Benchmark Level
python -m benchmarks gate G6 --tier 2                      # Gate verification
python -m benchmarks status                                # resume / progress info
```

`arke optimize` is the Stage 8 MVP path: it emits `strategy.json`, `result.akir`,
`trajectory.jsonl`, and `summary.json` into `--output` (default
`benchmarks/results/phase1/stage8/track1/optimize`). It defaults to `--dry-run` (validate/lower with
no GPU execution).

## Architecture

The compilation pipeline is in `arke/compiler/pipeline.py` (`ArkePipeline.compile_file`). Flow:

```
.ak source
  → arke/lang/grammar.py (lark-based parser, grammar in arke/lang/arke.lark)
  → AST (arke/lang/ast.py)
  → arke/ir/converters.py: ast_to_semantic / ast_to_strategy
  → SemanticIR + StrategyIR (arke/ir/semantic.py, arke/ir/strategy.py)
  → arke/compiler/validator.py (validate_semantic_ir)
  → arke/compiler/lowering.py: lower_full_stack → ScheduleIR → InstructionIR
  → arke/compiler/mlir_emitter.py (emit_mlir_skeleton)
  → arke/ir/akir.py (akir_to_dict / save_akir → .akir JSON)
```

Key packages:

- `arke/lang/` — `.ak` parser. The grammar lives in `arke.lark`; transformer methods in
  `parser.py` must match Lark UPPERCASE terminal names (ruff `N802` is suppressed there).
- `arke/ir/` — SemanticIR/StrategyIR types, the `.akir` JSON serialization (`akir.py`), op registry
  (`ops/registry.py`, `ops/catalog.py`), shape engine (`ops/shape_engine.py`), reference impls
  (`ops/reference_impls.py`), JSON schemas in `schemas/`, hardware target descriptors in `targets/`.
- `arke/compiler/` — validator, pass infrastructure (`passes/`), semantic-only pipeline
  (`semantic_pipeline.py`), full lowering (`lowering/` and `lowering.py`), MLIR skeleton emitter.
- `arke/backend/` — Triton codegen (`triton_backend.py`, `triton_templates/`), `mock_backend.py` for
  no-GPU tests, `protocol.py` defines the backend contract.
- `arke/backends/mlir/` — MLIR backend assets (Phase 3 scaffolding).
- `arke/agent/` — Stage 8 optimization agent. `optimize.py` is the entry point invoked by
  `arke optimize`; `tools.py` exposes the bounded action space (analyze_compute, list_legal_actions,
  apply_decision, verify_correctness, compile_and_profile, checkpoint/rollback).
- `arke/learn/trajectory.py` — trajectory recording for the agent loop.
- `arkec/` — separate package surface (CLI launcher slot reserved per pyproject comments).

Benchmarks are a first-class subsystem, not just scripts:

- Taxonomy: **BL** (Benchmark Level 1–6) × **OT** (Operator Tier 0–4) × **ST** (Shape Tier 1–4) ×
  **L** (Evaluation Layer L1=single-op, L2=fused, L3=end-to-end model). Definitions live in
  `docs/benchmark/benchmark-design.md`. Don't invent new tier labels — reuse these.
- `benchmarks/cli.py` is the dispatcher reachable as `arke bench` and `python -m benchmarks`.
- `benchmarks/gate.py`, `gate_g6.py`, `gate_g7.py`, `gate_g8.py` — Gate verification harness.
  **Gate exit criteria are locked once finalized**; do not edit them without explicit approval (see
  `AGENTS.md`).
- `benchmarks/bench_l1.py` / `bench_l2.py` / `bench_l3.py` — per-layer runners. They support
  incremental persistence and resume; `python -m benchmarks status` reports progress.
- Results land in `benchmarks/results/{run_id}/` with `config.json`, `hardware.json`, per-layer
  per-OT CSVs, `summary.json`, `PERF_ALL.csv` (41-column unified schema in
  `docs/benchmark/benchmark-csv-spec.md`), and `report.md`.
- Snapshot files at the repo root (`.benchmark_ops_snapshot.json`,
  `.benchmark_shapes_snapshot.json`) are kept in sync via `scripts/sync_ops.py` /
  `scripts/sync_shapes.py` — re-run those after editing the op or shape registries.

## Conventions worth knowing

- Python ≥ 3.10, ruff line-length 100, target `py310`. Several naming rules are intentionally
  relaxed for math conventions (`A`, `B`, `X` tensors, `l`/`O`/`I` loop indices) — see
  `[tool.ruff.lint]` in `pyproject.toml` before "fixing" them.
- `torch` is imported defensively (`try: import torch`); CPU-only paths must keep working when it's
  absent. Mirror that pattern when adding torch-dependent code.
- Every StrategyIR `Decision` carries a `@rationale`. Don't strip or skip it — it's a contract,
  surfaced in trajectories, and asserted by tests like `test_rationale_e2e.py`.
- `AGENTS.md` describes Arke from the perspective of an LLM agent that *uses* Arke. The
  optimization-loop tools and bounded-action-space framing there are the system-under-test
  contract, not advice for editing this repo.
- `IDENTITY.md`, `SOUL.md`, `USER.md`, `HEARTBEAT.md`, `TOOLS.md` are persona/personalization
  scaffolding for the operator of the agent — they aren't part of Arke's runtime.

## Documentation map

Specs (authoritative contracts):

- `docs/spec/arke-lang-spec.md` — `.ak` language v0.1.0
- `docs/spec/arke-ir-spec.md` — multi-layer IR v0.1.0 (matches `arke/ir/schemas/*.json`)
- `docs/spec/symbolic-dimension-spec.md`, `pass-infrastructure-spec.md`, `op-registry-interface.md`

Roadmap, Gates, and stage plans:

- `docs/roadmap/plan.md` — phases, stages, Gate-purpose mapping
- `docs/phase1/stage{6,7,8,9}-plan.md` — current phase plans
- `STAGE7_PROGRESS_REPORT.md` — running status of the active stage

Architecture deep-dives are under `docs/architecture/` (e2e-flow, arke-harness,
compiler-infrastructure, naming-system, token-efficiency-analysis).
