# Arke — Execution Plan v3.0

> Each Phase has SMART completion criteria. Gate must pass before advancing.
> Date: 2026-04-01

## Design Principles

1. **AI-First** — LLM agents make optimization decisions; compilers verify
2. **Minimal-Token** — Minimize total token consumption across the full pipeline: kernel definition, strategy search, compilation, and iterative optimization. Concise IR, structured decisions, and compiler verification eliminate the verbose trial-and-error of direct code generation
3. **Semantic/Strategy Separation** — *What to compute* and *how to optimize* are independent
4. **Compiler-Verified** — Every decision validated by deterministic checks

---

## Current Snapshot (2026-04-01)

### Completed
- ✅ GPU environment (PyTorch 2.6.0+cu124, Triton 3.2.0, RTX 3060)
- ✅ IR system (Semantic IR + Strategy IR, JSON Schema, 10 ops)
- ✅ Builder + Shape Inference (all 10 ops)
- ✅ Validation system (V0 static + V1 numerical + resource estimation)
- ✅ Legal action enumeration engine
- ✅ ArkeEnv full implementation
- ✅ Triton codegen (matmul + softmax, template engine)
- ✅ E2E pipeline (IR → strategy → codegen → GPU)
- ✅ LLM Runner (Anthropic + OpenAI API, fallback, retry)
- ✅ LLM closed-loop optimization (Sonnet 4.6, 23 tool calls, 106% cuBLAS)
- ✅ GPU correctness verification (same-dtype reference)
- ✅ Accuracy benchmark framework (10 metrics, 3-tier verdict)
- ✅ Trajectory JSONL export
- ✅ 237 tests passing

### Gate Status
- G0 ✅ — Triton matmul runs on RTX 3060
- G1 ✅ (⚠️ G1.4 needs 100% revalidation) — IR + Validation
- G2 ✅ — Manual strategy → codegen → 105-160% cuBLAS
- G3 ✅ — LLM tool-use → 106% cuBLAS + softmax correct
- G4 ✅ — Arke vs LLM-direct comparison
- G5 ✅ (3 known-fail perf) — E2E GPT-2 integration
- G6 ⬜ — Arke Lang & IR Completeness (Key Features validation)
- G7 ⬜ — E2E Autonomous Kernel Generation (I/O contract)
- G8 ⬜ — Implementation Language Assessment

---

## Phase Overview

```
Phase 1.0 ✅  Environment setup
Phase 1.1 ✅  IR + Validation foundation (⚠️ G1.4 needs 100% fix)
Phase 1.2 ✅  Codegen + E2E pipeline
Phase 1.3 ✅  LLM agent integration
Phase 1.4 ✅  LLM closed-loop optimization
Phase 1.5 ✅  Evaluation framework + comparison
Phase 1.6 ✅  .ak Parser + CLI
Phase 1.7 ✅  Whole-model E2E (G5: 3 known-fail perf criteria)
Phase 1.8 ✅  MVP release (v0.1.0)
Phase 1.9 ⬜  Arke Lang & IR Completeness (Gate G6) ← CURRENT
Phase 1.10 ⬜ E2E Autonomous Kernel Generation (Gate G7)
Phase 1.11 ⬜ Implementation Language Assessment (Gate G8)
```

---

## Phase 1.0: Environment Setup ✅

**Objective:** One-click reproducible development environment with GPU verification.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.0.1 | `make setup` creates venv + installs all deps | Fresh clone → `make setup` → no errors | ✅ |
| 1.0.2 | PyTorch detects CUDA GPU | `torch.cuda.is_available() == True` | ✅ |
| 1.0.3 | Triton compiles and runs matmul | GPU smoke test script exits 0 | ✅ |
| 1.0.4 | `pytest tests/ -q` runs without import errors | All tests collected (skip GPU-gated if no GPU) | ✅ |

---

## Phase 1.1: IR + Validation Foundation ✅

**Objective:** Semantic IR and Strategy IR with static and numerical validation covering ≥10 operators.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.1.1 | Semantic IR supports ≥10 operators | `len(OP_CATALOG) >= 10` | ✅ |
| 1.1.2 | Strategy IR supports ≥6 decision types | `kinds ⊇ {tile,fuse,place,parallel,reorder,algorithm}` | ✅ |
| 1.1.3 | JSON Schema validates IR round-trip | `jsonschema.validate(ir.to_json(), schema)` passes | ✅ |
| 1.1.4 | V0 static validation < 1ms | Validator latency measured < 1ms | ✅ |
| 1.1.5 | V1 numerical validation (NumPy reference) | 3 random-seed trials pass for matmul, softmax | ✅ |
| 1.1.6 | Shape inference for all 10 ops | `infer_shapes()` returns correct shapes | ✅ |
| 1.1.7 | ≥100 unit tests passing | `pytest` count ≥ 100 | ✅ (237) |

**Gate G1:** IR can express known-good strategy (matmul tiling + fusion) ✅

### Tasks
- [x] Op catalog P0 (10 operators)
- [x] Semantic IR dataclasses + JSON serialization
- [x] Strategy IR dataclasses + JSON serialization
- [x] JSON Schema definitions
- [x] KernelBuilder (Python → IR)
- [x] Shape inference engine (all 10 ops)
- [x] V0 static validator (shape + constraint checks)
- [x] V1 numerical validator (NumPy reference comparison)
- [x] Resource estimation (shared memory, register usage)
- [x] HW profile: `nvidia_ampere_rtx3060.json`

---

## Phase 1.2: Codegen + E2E Pipeline ✅

**Objective:** Triton code generation from Strategy IR, end-to-end pipeline producing GPU-executable kernels with **perf ≥ 70% cuBLAS**.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.2.1 | matmul Triton codegen correct | Generated kernel passes V1 numerical validation | ✅ |
| 1.2.2 | softmax Triton codegen correct | Same as above | ✅ |
| 1.2.3 | fused matmul+relu codegen correct | Fused kernel passes numerical validation | ✅ |
| 1.2.4 | GPU execution **≥ 70% cuBLAS** | `compile_and_profile()` returns `vs_baseline >= 0.7` | ✅ (105-160%) |
| 1.2.5 | Pipeline fully connected | IR → strategy → codegen → compile → profile in one call | ✅ |
| 1.2.6 | ≥9 GPU integration tests | GPU tests pass with `ARKE_GPU_TESTS=1` | ✅ |

**Gate G2:** Manual strategy → codegen → ≥70% cuBLAS ✅

### Tasks
- [x] Triton matmul template (Jinja2)
- [x] Triton softmax template
- [x] Triton matmul+relu fusion template
- [x] Template engine (strategy params → Triton template params)
- [x] TritonBackend (translate + compile + run)
- [x] cuBLAS baseline profiler (vs_baseline calculation)
- [x] E2E pipeline assembly (`pipeline.py`)
- [x] GPU integration tests

---

## Phase 1.3: LLM Agent Integration ✅

**Objective:** LLM autonomously completes optimization loop via tool-use with zero human intervention.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.3.1 | LLM uses ≥8 distinct tools | Trajectory log contains ≥8 unique tool names | ✅ (all 10) |
| 1.3.2 | LLM applies ≥4 strategy decisions | `result.decisions >= 4` | ✅ (13) |
| 1.3.3 | LLM calls verify_correctness | Tool appears in trajectory | ✅ |
| 1.3.4 | LLM calls compile_and_profile | Tool appears in trajectory | ✅ (5 times) |
| 1.3.5 | LLM uses checkpoint + rollback | Both tools appear in trajectory | ✅ |
| 1.3.6 | Fallback mechanism works | Timeout/error triggers fallback model automatically | ✅ |
| 1.3.7 | Multi-provider support | Anthropic + OpenAI-compatible APIs both functional | ✅ |
| 1.3.8 | Zero human intervention | Start → finish with no manual steps | ✅ |

### Tasks
- [x] Tool schema definitions (10 tools)
- [x] Session lifecycle manager
- [x] System prompt builder (hardware-aware)
- [x] LLM Runner (async, multi-provider)
- [x] LLM config (model selection, fallback chain)
- [x] Error recovery + retry logic
- [x] Agent matmul example script

---

## Phase 1.4: LLM Closed-Loop Optimization ✅

**Objective:** LLM-optimized kernels achieve GPU correctness and **perf ≥ 50% cuBLAS** across multiple operators.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.4.1 | LLM-optimized matmul → GPU correct | `verify_correctness` passes + GPU output matches same-dtype NumPy ref | ✅ |
| 1.4.2 | LLM-optimized matmul **≥ 50% cuBLAS** | `compile_and_profile()` returns `vs_baseline >= 0.5` | ✅ (106.1%) |
| 1.4.3 | LLM-optimized softmax → GPU correct | Same as 1.4.1 | ✅ |
| 1.4.4 | LLM-optimized fused_matmul_relu → GPU correct | Same as 1.4.1 | ✅ |
| 1.4.5 | compile_and_profile returns real GPU data | Response includes `latency_us`, `tflops`, `vs_baseline` | ✅ |
| 1.4.6 | Error recovery: LLM rollbacks on failure | Trajectory shows failed decision → rollback → success | ✅ |
| 1.4.7 | Trajectory export to JSONL | `export_trajectory()` outputs state/action/result records | ✅ |
| 1.4.8 | ≥220 tests passing | `pytest` count ≥ 220 | ✅ (237) |

**Gate G3:** LLM tool-use → matmul ≥50% cuBLAS + softmax correct ✅

### Tasks
- [x] Strategy decisions → Triton template parameter mapping
- [x] GPU correctness verification (same-dtype NumPy reference)
- [x] vs_baseline field in compile_and_profile
- [x] Accuracy benchmark framework (10 metrics, 3-tier verdict)
- [x] Reference sources (NumPyCPU, TorchGPU, Custom)
- [x] Trajectory JSONL writer
- [x] GPU correctness tests
- [x] End-to-end agent demo (matmul, softmax, fused)

---

## Phase 1.5: Evaluation Framework + Comparison ✅

**Objective:** Quantitatively prove Arke (LLM + tool-use) produces kernels that are more correct, more consistent, and **faster** than LLM-written Triton code, across ≥5 benchmark tasks.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.5.1 | ≥5 benchmark tasks defined | `benchmarks/tasks.py` contains ≥5 task definitions | ✅ (6→50) |
| 1.5.2 | Arke completes all tasks | Each task has Arke result (correctness + perf) | ✅ 6/6 |
| 1.5.3 | LLM-direct-Triton completes all tasks | Each task has direct-write result | ✅ 6/6 |
| 1.5.4 | Arke correctness ≥ direct Triton | `arke_correct_rate >= direct_correct_rate` | ✅ 100% ≥ 83% |
| 1.5.5 | **Arke mean perf ≥ direct Triton mean perf** | `mean(arke_vs_cublas) >= mean(direct_vs_cublas)` | ⚠️ 115.7% < 118.3% |
| 1.5.6 | Arke variance ≤ direct Triton | `var(arke_results) <= var(direct_results)` | ✅ (Direct fails vary) |
| 1.5.7 | Evaluation report generated | Benchmark results archived with CSV + kernels | ✅ |
| 1.5.8 | Token efficiency comparison | Arke vs direct total token consumption documented | ✅ |

**Gate G4:** Arke correctness ✅ PASS | Performance ⚠️ marginal (fusion tasks need improvement)
**Decision:** Proceed — Arke wins on reliability (100% vs 83% correct), Direct variance is high

### Gate G4 Decision Matrix
| Result | Conclusion | Next Step |
|--------|-----------|-----------|
| Arke correct + fast | ✅ Proceed | Phase 1.6–1.8 |
| Arke correct + slow | ⚠️ Arke is a verification framework | Pivot positioning |
| Arke ≈ direct Triton | ⚠️ No clear advantage | Reassess incremental value |
| Both poor | ❌ | Kill or fundamental pivot |

### Accuracy Comparison Design

**Same-dtype reference (default):** Test implementation correctness, not precision loss.
- GPU f16 kernel → NumPy CPU f16 reference
- Differences come from: reduction order, FMA, non-determinism

**Pluggable reference sources:**
- `NumPyCPUSource` — NumPy CPU, same dtype (default, GPU-free fallback)
- `TorchGPUSource` — PyTorch GPU (GPU-vs-GPU comparison)
- `CustomSource` — User-provided (e.g. Ascend reference data)

**Metrics (10):** abs/rel error (max/mean/P90/P99), ULP error, cosine similarity, sign mismatch, NaN/Inf count, zero diff rate

**3-tier verdict:** Accept / Review / Reject with per-dtype thresholds

### Tasks
- [x] Define ≥5 benchmark tasks (matmul sizes, softmax, fused ops, reduction)
- [x] Implement Arke benchmark runner
- [x] Implement LLM-direct-Triton baseline runner
- [x] Run all tasks × both methods × 3 trials
- [x] Statistical analysis (mean, variance, significance)
- [x] Token counting integration
- [x] Generate evaluation report (`benchmarks/report.md`)
- [x] Gate G4 decision

---

## Phase 1.6: .ak Parser + CLI ✅

**Objective:** Human-readable `.ak` syntax parsed into Semantic IR, with CLI commands for parse/optimize/inspect workflows.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.6.1 | Parse matmul kernel | `parser.parse("examples/01_matmul.ak")` returns AST | ✅ |
| 1.6.2 | Parse fused kernel | `parser.parse("examples/02_matmul_relu_fused.ak")` returns AST | ✅ |
| 1.6.3 | AST → Semantic IR correct | `ast_to_ir(ast)` equals `KernelBuilder.build()` output | ✅ |
| 1.6.4 | CLI `arke parse` | `arke parse kernel.ak -o kernel.json` outputs valid JSON | ✅ |
| 1.6.5 | CLI `arke optimize` | `arke optimize kernel.json --target ampere` starts LLM session | ✅ |
| 1.6.6 | CLI `arke inspect` | `arke inspect kernel.json` outputs human-readable IR | ✅ |
| 1.6.7 | ≥3 .ak examples work E2E | matmul, softmax, fused_matmul_relu: parse → optimize → GPU | ✅ |

**Dependency:** Phase 1.4 complete

### Tasks
- [x] EBNF grammar definition (`arke.lark`)
- [x] Lark parser implementation
- [x] AST node definitions
- [x] AST → Semantic IR converter
- [x] CLI entry point (`arkec` or `arke`)
- [x] `parse` subcommand
- [x] `optimize` subcommand
- [x] `inspect` subcommand
- [x] Example .ak files (≥3)

---

## Phase 1.7: Whole-Model End-to-End ✅

**Objective:** Replace kernels in a real model with Arke-optimized versions; inference correctness verified.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.7.1 | GPT-2 Small inference correct | Arke kernel output matches PyTorch eager output (same-dtype ref) | ✅ |
| 1.7.2 | **Inference latency ≤ torch.compile** | `arke_latency <= torch_compile_latency` on same hardware | ⚠️ KNOWN-FAIL |
| 1.7.3 | ≥2 ops replaced | matmul + softmax (or matmul + layernorm) | ✅ |
| 1.7.4 | Memory ≤ 6GB | Fits in RTX 3060 Laptop 6GB VRAM | ✅ |

**Gate G5:** PASS (7/7, 3 known-fail) — correctness ✅ coverage ✅ memory ✅ | latency ⚠️ known-fail

**G5 Latency Known-Fail Analysis (2026-04-04):**
- Measured: Arke 1.7-2.3× eager (threshold ≤1.15-1.20×)
- Root cause: Stage 1 monkey-patch architecture has 3 irreducible overhead sources:
  1. Triton dispatch ~60µs/call vs cuBLAS ~14µs × 49 Conv1D = ~2.3ms cumulative
  2. Python reshape/contiguous per patched module (~10-20µs each)
  3. No graph-level fusion — each kernel dispatched individually
- Mitigation attempted: monkey-patch 1.75×, +torch.compile 1.63×, custom_ops+compile 1.49×
- Resolution: Stage 2 torch.compile Inductor backend (`arke/integration/custom_ops.py` already prototyped)
- Full report: `benchmarks/results/stage1/gates/G5/REPORT.md`

**Dependency:** Phase 1.5 Gate G4 passes

### Tasks
- [x] GPT-2 Small baseline (eager + torch.compile) profiling
- [x] PyTorch custom op registration (`torch.library`)
- [x] Arke kernel integration into GPT-2 forward pass
- [x] Correctness verification (token-level output comparison)
- [x] Latency benchmark (Arke vs eager vs torch.compile)
- [x] Memory profiling

---

## Phase 1.8: MVP Release ✅

**Objective:** Publish v0.1.0 with one-click setup, passing CI, complete docs, and reproducible evaluation results.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.8.1 | `make setup` works on fresh clone | Tested on clean Ubuntu 22.04 | ✅ |
| 1.8.2 | CI green (3 Python versions) | GitHub Actions passes on 3.10, 3.11, 3.12 | ✅ |
| 1.8.3 | README complete (install + quickstart + examples) | New user can follow README and run demo | ✅ |
| 1.8.4 | API docs complete | All public classes/functions have docstrings | ✅ (99%) |
| 1.8.5 | Evaluation report published | `benchmarks/results/stage1/EVALUATION_REPORT.md` | ✅ |
| 1.8.6 | Trajectory data downloadable | JSONL files from evaluation runs publicly available | ✅ |
| 1.8.7 | v0.1.0 tag | `git tag v0.1.0` | ✅ |

### Tasks
- [x] Makefile with `setup` / `test` / `lint` / `bench` targets
- [x] CI workflow fix (lint + type check + test)
- [x] README quickstart verification on clean machine
- [x] API documentation pass
- [x] Evaluation report finalization
- [x] Trajectory data packaging
- [x] Version tag + GitHub release

---

## Phase 1.9: Arke Lang & IR Completeness (Gate G6)

> **Prerequisite:** G0-G5 must all pass at 100% (non-performance criteria).
> G6 is the foundation for ALL subsequent development. Verification must be strict.

**Objective:** Validate that Arke Lang and Arke IR satisfy all 7 Key Features with
production-grade completeness on NVIDIA GPU. This includes full `.ak → SemanticIR →
StrategyIR → Triton → GPU` pipeline, Python interoperability, and MLIR/LLVM IR
structural compatibility.

**Gate G6: REOPENED — Stage 1 strictness upgrade**

Before G6, all existing Gates (G0-G5) must be re-verified at 100% pass rate
(excluding known-fail performance criteria). Current gaps:
- G1.4: `.ak parse → IR` threshold was ≥3/5 — must be ALL files (100%)
- `04_attention.ak`: grammar doesn't support array literals / float constants in kernel args
- Strategy block parse → StrategyIR converter does not exist (`ast_to_strategy()` missing)
- `.ak → strategy → codegen → GPU` full pipeline never executed

### Completion Criteria
| # | Criterion | Verification | Key Feature | Status |
|---|-----------|-------------|------------|:------:|
| G6.1 | `.ak → SemanticIR → StrategyIR → Triton → GPU` E2E pipeline | All `examples/*.ak` parse, convert to both IRs, codegen to Triton, execute on GPU, correctness verified | AI-First + Compiler-Verified | ⬜ |
| G6.2 | `ast_to_strategy()` converter implemented | Strategy blocks in `.ak` → StrategyIR, round-trip test passes | Semantic/Strategy Separation | ⬜ |
| G6.3 | @rationale preserved through full pipeline | `.ak` @rationale → StrategyIR → codegen comments → trajectory/log; verified on ≥3 examples | @rationale Annotations | ⬜ |
| G6.4 | Token efficiency: `.ak` ≤ Triton line count | Cat A+B+C+D operators (matmul, softmax, layernorm, gelu, rmsnorm, attention): `.ak` lines < equivalent Triton kernel lines at comparable performance | Minimal-Token Efficiency | ⬜ |
| G6.5 | Python interop: IR ↔ Python dict/JSON | SemanticIR + StrategyIR round-trip: `.from_json()` / `.to_json()` / `from_dict()` / `to_dict()` for all OP_CATALOG ops | (Python interop) | ⬜ |
| G6.6 | IR ↔ MLIR structural mapping documented | `docs/spec/ir-mlir-mapping.md`: every SemanticIR/StrategyIR field → MLIR dialect op/attribute mapping | (MLIR interop prep) | ⬜ |
| G6.7 | Grammar completeness: ALL `.ak` files parse | G1.4 threshold upgraded from ≥3 to = ALL (0 failures); grammar supports array literals, float constants, all op parameter types | AI-First | ⬜ |
| G6.8 | Arke Lang + IR expression completeness (NVIDIA) | Benchmark Cat A+B+C+D operators (matmul, softmax, layernorm, rmsnorm, gelu, silu, attention) all expressible in `.ak`, convertible to both IRs, codegen to Triton, correctness verified at Tier 2 shapes | All Key Features | ⬜ |
| G6.9 | Arke Language Spec v1.0 + IR Spec v1.0 | Both spec docs updated to match implementation, tagged v1.0, consistency verified | (Spec freeze) | ⬜ |

**Dependency:** G0-G5 re-verified at 100% (non-perf), grammar fixes complete

### Tasks
- [ ] Fix grammar: support array literals (`dims=[2,3]`) and float constants (`0.125`) in kernel args
- [ ] Implement `ast_to_strategy()` converter (parser AST → StrategyIR)
- [ ] Fix G1.4 threshold: `ak_pass >= 3` → `ak_pass == len(ak_files)` in gate.py
- [ ] Write `.ak` examples for all Cat A+B+C+D operators
- [ ] Implement full `.ak → SemanticIR + StrategyIR → Triton codegen → GPU` pipeline
- [ ] Token efficiency benchmark: `.ak` lines vs Triton baseline lines
- [ ] IR-MLIR mapping document
- [ ] Language Spec v1.0 + IR Spec v1.0 update and freeze

---

## Phase 1.10: End-to-End Autonomous Kernel Generation (Gate G7)

> **Prerequisite:** G6 must pass.
> G7 defines Arke's input/output contract and validates full autonomy.

**Objective:** Validate that Arke provides a complete autonomous kernel generation
and optimization pipeline. Define the standard input formats and output artifacts.

**Note:** G7 detailed criteria will be finalized after G6 passes. Below are the
structural requirements that are already clear.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| G7.1 | **Input support**: (a) `.ak` files, (b) natural language description, (c) existing code in any language (Python/NumPy/Triton/CUDA/etc.) | Each input type → Arke pipeline → correct GPU kernel; ≥2 operators per input type | ⬜ |
| G7.2 | **Output definition**: executable Triton `.py` + Strategy IR JSON + benchmark report JSON + trajectory JSONL | Documented in `docs/spec/arke-io-spec.md` with examples | ⬜ |
| G7.3 | `arke optimize <input>` single command | CLI E2E: input → LLM optimize → Triton → GPU → benchmark report | ⬜ |
| G7.4 | LLM auto-generates Strategy (no human strategy input) | Kernel-only `.ak` (no strategy block) → LLM generates strategy → ≥80% cuBLAS | ⬜ |
| G7.5 | LLM iterative optimization (≥3 rounds) | Trajectory shows ≥3 compile → profile → adjust cycles | ⬜ |
| G7.6 | Validated on ≥2 operator types | matmul + softmax both complete full autonomous pipeline | ⬜ |
| G7.7 | **Arke I/O Spec document** | `docs/spec/arke-io-spec.md` complete | ⬜ |

**Dependency:** G6 PASS. Detailed criteria to be refined after G6 completion.

---

## Phase 1.11: Implementation Language Assessment (Gate G8)

> **Prerequisite:** G6 must pass.
> G8 evaluates whether Python is the optimal implementation language for Arke.

**Objective:** Data-driven assessment of Python vs alternatives for Arke's
implementation, with a clear decision and migration path for Stage 2+.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| G8.1 | Python advantages quantified | Document: PyTorch/Triton/MLIR bindings ecosystem, LLM API integration, prototyping speed | ⬜ |
| G8.2 | Python bottlenecks quantified | Measured: dispatch overhead, parse speed, memory footprint vs Rust/C++ theoretical | ⬜ |
| G8.3 | Critical path analysis | Identify hot path: codegen? LLM API wait? Triton compile? Python overhead % | ⬜ |
| G8.4 | Hybrid approach evaluation | Evaluate Python + Rust(pyo3) hybrid ROI | ⬜ |
| G8.5 | **Decision document** | `docs/design/language-decision.md`: conclusion + data + migration path | ⬜ |
| G8.6 | Stage 2+ language strategy | Explicitly define which modules stay Python, which may migrate | ⬜ |

**Dependency:** G6 PASS. Detailed criteria to be refined after G6 completion.

---

## Stage 2: Ascend Backend — SIMD Architecture Validation

> **Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend.
> Arke-generated Ascend Triton kernels must outperform FlagGems on Ascend.
> Simultaneously complete Arke Lang/IR to cover Category B-E operators.
>
> **Hardware target:** Huawei Ascend 910B (SIMD, CANN)
> **Backend:** triton-ascend (Ascend Triton)
> **Benchmark baseline:** FlagGems/Ascend (P1), CANN library (P0)

### Phase 2.1: Ascend Environment + Arke Lang/IR Completeness

**Objective:** Set up Ascend Triton environment; extend Arke Lang/IR to express Category B-E operators
needed for Stage 2 Gate coverage.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.1.1 | Ascend Triton environment ready | `import torch_npu; triton-ascend` smoke test passes | ⬜ |
| 2.1.2 | Arke Semantic IR: Category B ops | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` in OP_CATALOG | ⬜ |
| 2.1.3 | Arke Semantic IR: Category D ops | `swiglu`, `geglu` in OP_CATALOG | ⬜ |
| 2.1.4 | Arke Semantic IR: Category E ops | `rope`, `yarn_rope` in OP_CATALOG | ⬜ |
| 2.1.5 | Strategy IR: SIMD decision kinds | ≥3 SIMD-specific decision kinds (vector_tile, lane_mapping, simd_width) defined | ⬜ |
| 2.1.6 | Ascend hardware profile | `targets/ascend_910b.json` with SIMD width, memory hierarchy, CANN constraints | ⬜ |

### Phase 2.2: Ascend Triton Codegen

**Objective:** Arke IR → Ascend Triton → NPU execution, correctness verified on Category A+C ops.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.2.1 | matmul correctness on Ascend | Tier 2 (15 shapes): 100% pass V1 numerical | ⬜ |
| 2.2.2 | rmsnorm correctness on Ascend | Tier 2 (7 shapes): 100% pass | ⬜ |
| 2.2.3 | silu/swiglu correctness on Ascend | Tier 2 (5 shapes): 100% pass | ⬜ |
| 2.2.4 | LLM decides SIMD-aware strategy | LLM chooses different tile/mapping for Ascend vs NVIDIA on same kernel | ⬜ |

**Gate S2-G1:** matmul + rmsnorm + swiglu on Ascend — correctness 100% (Tier 2) + LLM SIMD decision verified

### Phase 2.3: Ascend Performance vs FlagGems

**Objective:** Arke/Ascend performance exceeds FlagGems/Ascend on Category A+C+D operators.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.3.1 | matmul Arke/FlagGems ≥ 1.1× (Ascend Tier 2) | `bench_l1 --op matmul --tier 2 --backend ascend` | ⬜ |
| 2.3.2 | rmsnorm Arke/FlagGems ≥ 1.0× (Ascend Tier 2) | `bench_l1 --op rmsnorm --tier 2 --backend ascend` | ⬜ |
| 2.3.3 | swiglu Arke/FlagGems ≥ 1.0× (Ascend Tier 4) | Tier 4 SwiGLU shapes (8 shapes) | ⬜ |

**Gate S2-G2:** Cat A+C+D on Ascend — Arke geomean ≥ FlagGems (Tier 2)

### Phase 2.4: Category B — FlashAttention + GQA

**Objective:** Arke expresses and generates FlashAttention/GQA on both NVIDIA and Ascend.
This is the first complex fused operator gate.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.4.1 | FlashAttention Arke Lang expression | `.ak` kernel for flash_attention compiles and runs correctly | ⬜ |
| 2.4.2 | FlashAttention NVIDIA correctness | Tier 4 FA shapes ≥12/22 (excl. OOM): 100% pass V1 | ⬜ |
| 2.4.3 | FlashAttention NVIDIA performance | ≥0.7× flash-attn-2 baseline (Tier 4 geomean) | ⬜ |
| 2.4.4 | FlashAttention Ascend correctness | Tier 4 FA shapes ≥8/22 (excl. OOM): 100% pass | ⬜ |
| 2.4.5 | GQA correctness (NVIDIA) | Tier 4 GQA 6 shapes: 100% pass | ⬜ |
| 2.4.6 | DeepSeek shapes included | ds-v2/v3 seq=512~16384 shapes pass | ⬜ |

**Gate S2-G3:** FlashAttention (Cat B) — NVIDIA ≥12 shapes correct + ≥0.7× FA-2; Ascend ≥8 shapes correct

### Phase 2.5: Category E — RoPE/YaRN + @rationale cross-arch

**Objective:** RoPE/YaRN codegen for NVIDIA + Ascend; validate @rationale improves cross-arch optimization.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.5.1 | RoPE correctness (NVIDIA) | Tier 4 RoPE 9 shapes: 100% pass | ⬜ |
| 2.5.2 | RoPE correctness (Ascend) | Tier 4 RoPE ≥5 shapes: 100% pass | ⬜ |
| 2.5.3 | YaRN RoPE long-context | DeepSeek YaRN ds-v2-8k ~ ds-v2-32k correct | ⬜ |
| 2.5.4 | @rationale cross-arch lift | With SIMD rationale > without: ≥10% perf lift on Ascend (Tier 2 matmul) | ⬜ |

**Gate S2-G4:** RoPE/YaRN (Cat E) correctness + @rationale cross-arch effect verified

### Stage 2 Summary Gate: S2-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E all passing |
| Shape coverage | Tier 2 (15) + Tier 4 sampled (≥20 shapes across all cats) |
| Ascend performance | matmul+rmsnorm+swiglu geomean ≥ FlagGems/Ascend |
| FlashAttention | ≥12 Tier 4 shapes correct on NVIDIA, ≥0.7× FA-2 |
| @rationale | Cross-arch improvement ≥10% verified |
| H4 (cross-hardware) | Same .ak → NVIDIA + Ascend both correct |

---

## Stage 3: MLIR Backend — Complete Compiler Control

> **Goal:** Replace Triton backend with Arke MLIR Dialect for both NVIDIA and Ascend.
> MLIR removes Triton's abstraction ceiling, enabling finer-grained optimization and
> more complete operator support. Performance must match or exceed Stage 2 Triton path.
>
> **Why MLIR after Triton:** Triton's Python abstraction is not AI-First — it constrains
> LLM decisions to Triton's fixed optimization model. MLIR gives Arke full control over
> the lowering pipeline, letting LLM decisions penetrate to loop nest and memory level.
>
> **Hardware target:** NVIDIA + Ascend (via NVVM dialect + AscendNPU IR)
> **Benchmark baseline:** Stage 2 Triton path results

### Phase 3.1: Arke MLIR Dialect Design

**Objective:** Define `arke.kernel` + `arke.strategy` MLIR dialect ops that can represent
all Stage 1+2 optimizations.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.1.1 | `arke.kernel` dialect defined | All Stage 1 ops expressible in arke.kernel | ⬜ |
| 3.1.2 | `arke.strategy` dialect defined | All Stage 1+2 decision kinds representable | ⬜ |
| 3.1.3 | Strategy IR → MLIR transform dialect | Decision kinds map to MLIR transform ops | ⬜ |
| 3.1.4 | Strategy IR Level 2 fields added | loop_nest, memory_access_pattern, pipeline_stage fields in Strategy IR | ⬜ |
| 3.1.5 | LLM can produce Level 2 decisions | LLM session generates ≥1 Level 2 decision in closed-loop | ⬜ |

### Phase 3.2: MLIR Path Correctness (NVIDIA)

**Objective:** All Stage 1+2 operators pass correctness via MLIR path on NVIDIA.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.2.1 | Cat A (matmul) via MLIR | Tier 2+4 (27 shapes): 100% correct | ⬜ |
| 3.2.2 | Cat C (rmsnorm, layernorm) via MLIR | Tier 2 (7 shapes each): 100% correct | ⬜ |
| 3.2.3 | Cat D (swiglu, gelu) via MLIR | Tier 4 SwiGLU (8 shapes): 100% correct | ⬜ |
| 3.2.4 | Cat B (FlashAttention) via MLIR | Tier 4 FA ≥12 shapes: 100% correct | ⬜ |
| 3.2.5 | Cat E (RoPE) via MLIR | Tier 4 RoPE 9 shapes: 100% correct | ⬜ |

**Gate S3-G1:** Cat A+C+D via MLIR — Tier 2+4 correctness 100%

### Phase 3.3: MLIR Path Performance ≥ Stage 2 Triton

**Objective:** MLIR path matches or exceeds Stage 2 Triton path on all categories.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.3.1 | matmul MLIR ≥ Triton path | Tier 2+4 geomean ≥ Stage 2 result | ⬜ |
| 3.3.2 | FlashAttention MLIR ≥ Triton path | Tier 4 FA geomean ≥ Stage 2 result | ⬜ |
| 3.3.3 | rmsnorm/swiglu MLIR ≥ Triton | Tier 2+4 geomean ≥ Stage 2 result | ⬜ |

**Gate S3-G2:** All Cat A+B+C+D MLIR geomean ≥ Stage 2 Triton path

### Phase 3.4: LLM Level-2 Decisions via MLIR

**Objective:** LLM makes loop-nest and memory-access decisions (Level 2) via Strategy IR,
and MLIR path reflects these decisions with measurable performance benefit.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.4.1 | LLM produces loop_nest decisions | LLM session uses Level 2 loop_nest field in ≥3 shapes | ⬜ |
| 3.4.2 | LLM Level 1+2 > Level 1 + default | Tier 2 matmul: LLM L1+2 ≥ LLM L1+default L2 by ≥15% | ⬜ |
| 3.4.3 | FlashAttention Level 2 optimization | FA Tier 4: LLM Level 2 decisions improve vs default by ≥20% | ⬜ |

**Gate S3-G3:** LLM Level-2 decision value verified (matmul ≥15%, FA ≥20%)

### Phase 3.5: Ascend via MLIR

**Objective:** MLIR path generates correct Ascend code via AscendNPU IR dialect.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.5.1 | MLIR → AscendNPU IR | matmul + rmsnorm correct on Ascend via MLIR path | ⬜ |
| 3.5.2 | Ascend MLIR ≥ Ascend Triton path | Tier 2 geomean: MLIR path ≥ Stage 2 Triton result | ⬜ |

**Gate S3-G4:** Ascend via MLIR correct + performance ≥ Stage 2

### Stage 3 Summary Gate: S3-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E all via MLIR path |
| Shape coverage | Tier 2 + Tier 4 across all categories |
| Performance | MLIR path geomean ≥ Stage 2 Triton (NVIDIA + Ascend) |
| LLM Level-2 | Verified benefit ≥15% on matmul, ≥20% on FlashAttention |
| Multi-hardware | NVIDIA + Ascend both via MLIR |

---

## Stage 4: LLVM IR Backend — 100% Hardware Completeness

> **Goal:** Arke IR → LLVM IR direct emission, achieving 100% hardware expression completeness
> and maximum performance headroom beyond what MLIR lowering allows.
>
> **Why LLVM IR after MLIR:** MLIR still imposes abstraction constraints in its lowering pipeline.
> Direct LLVM IR emission gives Arke full control: register allocation hints, barrier placement,
> instruction scheduling. This is where LLM Level-3 decisions become possible.
>
> **Hardware target:** NVIDIA + Ascend + AMD (≥3 backends via LLVM)
> **Benchmark baseline:** Stage 3 MLIR path results

### Phase 4.1: LLVM IR Emission Foundation

**Objective:** Arke IR → Loop Nest IR → LLVM IR, correctness verified on Cat A.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.1.1 | Loop Nest IR design | Intermediate representation between Strategy IR and LLVM IR defined | ⬜ |
| 4.1.2 | matmul → LLVM IR correct | Tier 2 matmul (15 shapes): 100% correct via LLVM path | ⬜ |
| 4.1.3 | Strategy IR Level 3 fields | register_hints, barrier_placement, instruction_scheduling fields defined | ⬜ |
| 4.1.4 | LLVM path on ≥2 backends | matmul correct on NVIDIA (NVPTX) + AMD (AMDGCN) | ⬜ |

**Gate S4-G1:** matmul via LLVM correct on ≥2 backends (Tier 2)

### Phase 4.2: LLVM Path Performance ≥ Stage 3 MLIR

**Objective:** LLVM path achieves better performance than MLIR path on Cat A+C+D.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.2.1 | matmul LLVM ≥ MLIR + 5% | Tier 2+4 geomean: LLVM ≥ Stage 3 MLIR + 5% | ⬜ |
| 4.2.2 | rmsnorm/swiglu LLVM ≥ MLIR | Tier 2+4 geomean: LLVM ≥ Stage 3 MLIR | ⬜ |

**Gate S4-G2:** Cat A+C+D LLVM geomean ≥ MLIR + 5% (NVIDIA)

### Phase 4.3: LLM Level-3 Decisions

**Objective:** LLM makes register/barrier/scheduling decisions (Level 3) via Strategy IR Level 3 fields.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.3.1 | LLM produces Level 3 decisions | LLM session uses register_hints / barrier_placement in ≥3 shapes | ⬜ |
| 4.3.2 | LLM L1+2+3 > L1+2+default L3 | Tier 2 matmul: LLM full stack ≥ L1+2+default by ≥5% | ⬜ |

**Gate S4-G3:** LLM Level-3 decision value verified (≥5% over L1+2+default)

### Phase 4.4: FlashAttention via LLVM + Multi-Hardware Parity

**Objective:** FlashAttention via LLVM path on ≥3 backends; performance within 90% of vendor libraries.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.4.1 | FlashAttention LLVM correct | Tier 4 FA ≥12 shapes on NVIDIA via LLVM: 100% | ⬜ |
| 4.4.2 | FlashAttention LLVM performance | ≥0.85× flash-attn-2 baseline (better than Stage 2's 0.7×) | ⬜ |
| 4.4.3 | MLA (DeepSeek) LLVM correct | Tier 4 MLA 8 shapes: 100% correct | ⬜ |
| 4.4.4 | ≥3 hardware backends | NVIDIA + Ascend + AMD all pass Cat A correctness | ⬜ |
| 4.4.5 | Multi-hardware perf within 90% vendor | Each backend: matmul geomean ≥90% of respective vendor lib | ⬜ |

**Gate S4-G4:** FlashAttention + MLA via LLVM correct; multi-hardware ≥90% vendor parity

### Phase 4.5: Production Release v1.0.0

**Objective:** Stable public API, comprehensive benchmark suite across ≥3 hardware platforms.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.5.1 | pip package + stable API | `pip install arke` + API docs complete | ⬜ |
| 4.5.2 | Benchmark suite ≥3 platforms | NVIDIA + Ascend + AMD results published | ⬜ |
| 4.5.3 | @rationale knowledge base ≥200 entries | Covers Cat A-E across NVIDIA + Ascend + AMD | ⬜ |
| 4.5.4 | v1.0.0 tag | `git tag v1.0.0` | ⬜ |

**Gate S4-G_FINAL:** v1.0.0 — multi-hardware parity + @rationale KB + LLM Level 1-3 full stack

### Stage 4 Summary Gate: S4-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E + F (lm_head/vocab) |
| Shape coverage | Tier 2 + Tier 4 all categories incl. MLA/GQA/YaRN |
| Performance vs Stage 3 | LLVM geomean ≥ MLIR + 5% (Cat A+C+D) |
| FlashAttention | ≥0.85× FA-2 baseline via LLVM |
| Multi-hardware | ≥3 backends ≥90% respective vendor libs |
| LLM Level-3 | Verified benefit ≥5% over L1+2+default |
| @rationale KB | ≥200 entries, cross-hardware coverage |

---

## Stage Dependency Chain

```
Stage 1 ✅  SIMT feasibility (NVIDIA Triton)      → proves H1/H2/H3
    ↓ G5 PASS
Stage 2     SIMD feasibility (Ascend Triton)      → proves H4 (cross-arch)
    ↓ S2-G_FINAL PASS
Stage 3     MLIR backend (full compiler control)  → deeper LLM decisions (Level 2)
    ↓ S3-G_FINAL PASS
Stage 4     LLVM IR backend (100% HW completeness)→ LLM Level 1-3 full stack
    ↓ S4-G_FINAL PASS
  v1.0.0
```

**Gate benchmark coverage requirement (Stage 2+):**
- Every Gate must cover ≥3 Operator Categories (A-G, see BENCHMARK.md §2)
- Shape coverage must include Tier 4 shapes from `docs/design/BENCHMARK.md`
- DeepSeek shapes (seq=512~163840) must be included in FlashAttention + MLA Gates

---

## Phase Entry/Exit Checklist

Before advancing to the next Phase:

```
□ All completion criteria met (100%)
□ Corresponding Gate passed (if applicable)
□ All existing tests still pass (no regression)
□ Code committed + pushed
```

### Exception Handling

- **Criterion unachievable:** Analyze root cause, discuss with Leon whether to relax or skip
- **New required work discovered:** Add to current Phase criteria (don't defer)
- **Gate failure:** Follow decision matrix; may pivot or kill

---

## Risk Matrix

| Risk | Affects | Mitigation |
|------|:-------:|-----------|
| LLM decisions don't map to Triton templates | Phase 1.4 | Expand template coverage + parameter adaptation layer |
| compile_and_profile errors in LLM session | Phase 1.4 | Better error messages + graceful degradation |
| 6GB VRAM insufficient for large shapes | Phase 1.7 | Limit shapes to ≤2048 |
| Arke doesn't outperform direct Triton | Phase 1.5 | Gate G4 decision matrix |
| API timeout / rate limit | Phase 1.4-1.5 | Retry + fallback + prefer Sonnet over Opus |

---

*Plan version: v3.0 | Created: 2026-04-01 | Last updated: 2026-04-05*
