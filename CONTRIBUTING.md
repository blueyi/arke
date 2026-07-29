# Contributing to Arke

Thanks for your interest in Arke — an AI-Native operator language, IR, compiler
toolchain, and optimization-agent system for GPU/NPU kernels.

> **Status: Pre-Alpha, NVIDIA-only.** All phase closures are NVIDIA-scoped gate
> passes, **not** release readiness. `v1.0.0` is intentionally **not** tagged.
> Expect churn. This document explains how the project is organized so you can
> find your way around and land changes that fit.

---

## Ground rules (read first)

1. **Gates are frozen contracts.** Benchmark/Phase/Stage/Gate targets,
   thresholds, exit criteria, and scoring semantics do **not** change without
   explicit project-lead approval. Development is *gate-driven*: work backward
   from a Gate's exit criteria. See `docs/roadmap/plan.md` (single source of
   truth for phases/stages/gates).
2. **The SSOT owns the operator catalog.** Never hardcode op-name lists outside
   `benchmarks/op_registry.py`. Adding an op = edit `docs/benchmark/benchmark-ops.md`
   (the authoritative markdown) + register an `OpSchema` + add a `ref_*` impl.
   `tests/test_ssot_op_registry.py` enforces this.
3. **Correctness before performance, always.** A fast-but-wrong kernel is a
   regression. Every kernel change must pass its correctness check
   (`max_abs_diff` within tolerance vs the reference) *before* any latency claim.
4. **Honest engineering findings over pretty numbers.** Record pitfalls,
   un-fixed problems, and capability limits as honest findings. Do not fabricate
   or mask results. A verification gate blocking a wrong result is correct
   behavior — point it out, don't route around it.
5. **Sync docs with code.** Lang → `docs/spec/arke-lang-spec.md`;
   IR → `docs/spec/arke-ir-spec.md` + `ir-mlir-mapping.md`;
   Compiler → `docs/architecture/arke-compiler-infrastructure.md`;
   Agent → `docs/architecture/agent-design.md`. A code change without its doc
   sync is incomplete.

---

## Repository layout

```
arke/lang/        .ak language parser (Arke-Lang)
arke/ir/          SemanticIR (what to compute) + StrategyIR (how to optimize)
arke/backend/     PRODUCTION codegen: Triton / MLIR-GPU / CUDA-C / LLVM
arke/compiler/    Pipeline: validation passes (shape/SSA/rationale) + lowering
arke/agent/       Agent session, tools, prompts, ArkeEnv optimization loop
arke/learn/       Trajectory recording + RL dataset (reads agent telemetry)
arke/integration/ torch_bridge
benchmarks/       Gate system, baselines, op_registry (SSOT), dynamic-shape track
tests/            ~2860 tests (pytest-xdist)
docs/             roadmap / spec / architecture / benchmark / audit
```

### Two things that look like duplicates but aren't

- **IR layers.** `SemanticIR` and `StrategyIR` are the load-bearing IR. The
  spec also documents `ScheduleIR`/`InstructionIR` as **Phase-future target
  contracts** (`[skeleton]` / `[Phase-future]` tags in `docs/spec/arke-ir-spec.md`
  §7–§8): they are populated structures that do **not yet drive codegen** —
  each backend does its own scheduling in codegen. Don't look in ScheduleIR for
  the scheduling logic; it isn't wired yet.
- **MLIR paths.** The production MLIR-GPU backend is `arke/backend/mlir_*`
  (singular *backend*). The plural `arke/backends/mlir/` is a legacy S7 PoC kept
  only for a contract test — do not extend it. Arke emits **upstream** MLIR
  dialects (`linalg`/`memref`/`gpu`/`nvgpu`); it does **not** define a custom
  MLIR dialect (no `.td`/tablegen).

---

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e ".[dev]"
```

GPU work needs an NVIDIA card (development target: sm_86 / CUDA 12.4) with
Triton. The MLIR-GPU backend additionally needs an MLIR install
(`--backend mlir_gpu`). Many tests and the interpreter fallback run CPU-only.

## Running tests

```bash
make test                    # full suite (pytest-xdist --dist loadfile)
pytest tests/backend/ -q     # a subset
```

On a 6 GB laptop card, isolated backend subsets can be flaky due to VRAM /
ordering. **The authoritative pass/fail signal is `make test`** (or an A/B
`git stash` comparison against `HEAD` on the same subset), never a single
isolated subset. If a test fails, first confirm it touches your change before
assuming a regression.

## Benchmarking a kernel change

```bash
python -m benchmarks.gate G6 --tier 2           # a gate
python -m benchmarks.dynamic_shape --all        # dynamic-shape cliff track
```

Report performance as a **same-day A/B** (laptop clocks drift across days;
don't compare to a historical snapshot). Always pair a latency number with a
correctness check and the baseline you measured against (cuBLAS / flash-attn /
FlagGems — see `benchmarks/golden_ladder.py` for the golden ladder).

## Backend contract (adding a backend)

Backends implement the `ArkeBackend` Protocol in `arke/backend/protocol.py`
(4 methods) and register with `BackendRegistry`. Hardware characteristics come
from a `HardwareModel` (`arke/backend/hardware.py`). This is the extension seam
for future non-NVIDIA (Ascend/AMD/…) backends — keep it clean; today only
`nvidia_sm86()` is instantiated.

## Coding conventions

- Match the style of the file you're editing; touch only what the task needs.
- Triton templates (`arke/backend/triton_templates/*.py.j2`) must obey the
  **Tensor Core dtype discipline**: feed `tl.dot` fp16 operands and accumulate
  in fp32 (`out_dtype=tl.float32` or an fp32 accumulator). Casting a dot operand
  to fp32 forces the FFMA path and idles the Tensor Cores.
  `tests/backend/test_template_tc_discipline.py` enforces this.
- Keep imports acyclic at module scope. `arke.learn` may import `arke.agent`
  telemetry contracts, not the reverse (defer any agent→learn import to call
  time).

## Commits & PRs

- One logical change per commit; explain **why** in the body, and cite the gate
  / benchmark data (with commit id) when a change is performance-motivated.
- Don't commit secrets, `.env`, large result blobs, or regenerated caches.
- Run `make test` before opening a PR; state the pass count and the platform
  you ran on.

---

By contributing you agree your contributions are licensed under the project's
[Apache-2.0 License](LICENSE).
