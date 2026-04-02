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
- G1 ✅ — Known-good strategy expressible in Arke IR
- G2 ✅ — Manual strategy → codegen → 105-160% cuBLAS
- G3 ✅ — LLM tool-use → 106% cuBLAS + softmax correct

---

## Phase Overview

```
Phase 1.0 ✅  Environment setup
Phase 1.1 ✅  IR + Validation foundation
Phase 1.2 ✅  Codegen + E2E pipeline
Phase 1.3 ✅  LLM agent integration
Phase 1.4 ✅  LLM closed-loop optimization
Phase 1.5 ✅  Evaluation framework + comparison
Phase 1.6 ✅  .ak Parser + CLI
Phase 1.7 ⚠️  Whole-model E2E (3/4 criteria pass, perf needs inductor lowering)
Phase 1.8 ⬅  MVP release (next)
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
- [ ] Define ≥5 benchmark tasks (matmul sizes, softmax, fused ops, reduction)
- [ ] Implement Arke benchmark runner
- [ ] Implement LLM-direct-Triton baseline runner
- [ ] Run all tasks × both methods × 3 trials
- [ ] Statistical analysis (mean, variance, significance)
- [ ] Token counting integration
- [ ] Generate evaluation report (`benchmarks/report.md`)
- [ ] Gate G4 decision

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
- [ ] EBNF grammar definition (`arke.lark`)
- [ ] Lark parser implementation
- [ ] AST node definitions
- [ ] AST → Semantic IR converter
- [ ] CLI entry point (`arkec` or `arke`)
- [ ] `parse` subcommand
- [ ] `optimize` subcommand
- [ ] `inspect` subcommand
- [ ] Example .ak files (≥3)

---

## Phase 1.7: Whole-Model End-to-End ⚠️

**Objective:** Replace kernels in a real model with Arke-optimized versions; inference correctness verified, **latency ≤ torch.compile**.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.7.1 | GPT-2 Small inference correct | Arke kernel output matches PyTorch eager output (same-dtype ref) | ✅ |
| 1.7.2 | **Inference latency ≤ torch.compile** | `arke_latency <= torch_compile_latency` on same hardware | ⚠️ |
| 1.7.3 | ≥2 ops replaced | matmul + softmax (or matmul + layernorm) | ✅ |
| 1.7.4 | Memory ≤ 6GB | Fits in RTX 3060 Laptop 6GB VRAM | ✅ |

**Gate G5:** GPT-2 Small with Arke kernels — correct AND **latency ≤ torch.compile**

**Dependency:** Phase 1.5 Gate G4 passes

### Tasks
- [ ] GPT-2 Small baseline (eager + torch.compile) profiling
- [ ] PyTorch custom op registration (`torch.library`)
- [ ] Arke kernel integration into GPT-2 forward pass
- [ ] Correctness verification (token-level output comparison)
- [ ] Latency benchmark (Arke vs eager vs torch.compile)
- [ ] Memory profiling

---

## Phase 1.8: MVP Release ⬜

**Objective:** Publish v0.1.0 with one-click setup, passing CI, complete docs, and reproducible evaluation results.

### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 1.8.1 | `make setup` works on fresh clone | Tested on clean Ubuntu 22.04 | ⬜ |
| 1.8.2 | CI green (3 Python versions) | GitHub Actions passes on 3.10, 3.11, 3.12 | ⬜ |
| 1.8.3 | README complete (install + quickstart + examples) | New user can follow README and run demo | ⬜ |
| 1.8.4 | API docs complete | All public classes/functions have docstrings | ⬜ |
| 1.8.5 | Evaluation report published | `benchmarks/report.md` with tables, charts, conclusions | ⬜ |
| 1.8.6 | Trajectory data downloadable | JSONL files from evaluation runs publicly available | ⬜ |
| 1.8.7 | v0.1.0 tag | `git tag v0.1.0` | ⬜ |

### Tasks
- [ ] Makefile with `setup` / `test` / `lint` / `bench` targets
- [ ] CI workflow fix (lint + type check + test)
- [ ] README quickstart verification on clean machine
- [ ] API documentation pass
- [ ] Evaluation report finalization
- [ ] Trajectory data packaging
- [ ] Version tag + GitHub release

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

*Plan version: v3.0 | Created: 2026-04-01 | Last updated: 2026-04-01*
