# Phase 1 — Stage 6: Compiler Infrastructure

> Gate G6 exit criteria → [plan.md](../roadmap/plan.md#stage-6-g6-compiler-infrastructure--%E2%AC%9C--current)

**Status:** ✅ 7/7 criteria PASS (100%) — closed 2026-06-06 on feat/g6-closure

**Objective:** Refactor the compiler toolchain into a clean, extensible architecture. OpRegistry as single source of truth, Pass pipeline for composable transformations, Backend abstraction for multi-target support. **Validated through full 46-op correctness and performance.**

**Depends on:** S0-S5 (all passed)
**Blocks:** S7 (Lang & IR v0.1.0 needs Pass pipeline, OpRegistry, Backend abstraction)

---

## Gate Criteria Breakdown

**BL Exit:** BL4×L1 — Full 46 ops correctness 100% + perf ≥1.00× P3 (eager). Every infra component is validated through real operator execution.

| # | Criterion | Verification | Operator Validation |
|:-:|:----------|:-------------|:-------------------|
| 1 | OpRegistry: single source of truth for all 46 ops (adding op ≤ 2 files) | `scripts/verify_op_registry.py` — adding op requires ≤2 file changes | All 46 ops registered, metadata complete |
| 2 | SemanticInterpreter: PyTorch eager executor, correctness 100% | `pytest tests/test_semantic_interpreter.py` — all 46 ops correct | OT0 relu/gelu → OT4 flash_attention, all pass |
| 3 | Pass Infrastructure: ArkePass protocol + PassPipeline with ≥2 passes | `pytest tests/test_pass_infra.py` — Pass protocol + Pipeline with ≥2 passes | ShapeInference + SSAValidation on matmul/softmax/layernorm |
| 4 | SSA Validator: validates all 46 ops; rejects ≥5 invalid IR examples | `pytest tests/test_ssa_validator.py` — all 46 ops valid; ≥5 invalid rejected | All 46 ops pass; broken IR (dup SSA, shape mismatch, etc.) rejected |
| 5 | Backend Abstraction: ArkeBackend protocol + TritonBackend implements it | `pytest tests/test_backend_protocol.py` — `ArkeBackend` protocol + `TritonBackend` | matmul/relu/softmax codegen via backend protocol |
| 6 | Codegen + GPU: 46 ops via TritonBackend, correctness 100%, perf ≥1.00× P3 | `arke bench --bl 4 --layer l1` — all 46 ops correct, perf ≥1.00× eager | Full BL4 (OT0-4 × ST1-2): every op GPU-verified |
| 7 | Non-regression: ≥422 tests passed, ≤6 skipped, 0 new failures | `pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures | — |

---

## Known Limitations & S7 Optimization Targets

**G6.6 Performance Benchmark Status: 46/46 ops complete (100% coverage)** — closed 2026-06-06

### Closure path

The original 2026-04-26 closure marked S6 ✅ at 40/45 ops. The 2026-06-06
reconciliation that brought G6 to a real 7/7 PASS:

1. **D8-X1 catalog growth (45→46 ops):** `silu_and_mul` / `gelu_and_mul`
   renamed from `silu`/`gelu` aliases; `swiglu_packed` onboarded as the
   46th op (audit-degraded, PyTorch-eager baseline).
2. **BL4×L1 re-run for the 3 new/renamed ops** in
   `benchmarks/results/phase1/stage6/track1/l1/` so G6.6 sees them.
3. **5 OT4 attention ops** (`flash_attention`, `cross_attention`,
   `grouped_query_attention`, `multi_latent_attention`, `paged_attention`)
   re-run at tier-3 shapes. tier-1/2 attention shape sets are empty by
   spec (attention is BL5/L1 territory); tier-3 small shapes fit in
   6 GB VRAM and emit PyTorch-eager rows, which is what G6.6 counts.
4. **G6.7 `qkv_fa-shape3` flake fixed**: the L2 correctness probe was
   being hijacked by FlagGems' global `aten::mm` registration once any
   earlier baseline enabled it in the same pytest session, causing
   Triton codegen failures on unusual fp16/fp64 shapes. The probe now
   computes the QKV reference on CPU in fp64, escaping the dispatcher
   override entirely. This is a probe-only change; perf measurements
   still run on GPU.

### Performance ladder (all 46 ops ≥1.00× P3 eager baseline)

OT4 attention values reflect tier-3 measurements; tier-1/2 attention
shape sets are empty by spec.

**OT0 (Elementwise):** add, cast, copy_, exp, mul, neg, sigmoid, tanh, rsqrt ✅

**OT1 (Move):** permute, transpose, gather, scatter, split, concat ✅

**OT2 (Compute):** matmul, batch_matmul, grouped_matmul ✅

**OT3 (Reduce + fused):** softmax, layernorm, rmsnorm, rmsnorm_residual, reduce_sum, reduce_mean, reduce_max, argmax, topk, **silu_and_mul** ✅, **gelu_and_mul** ✅, **swiglu_packed** ✅ (audit-degraded)

**OT4 (Attention/Special):** rope, embedding, cross_entropy, fused_linear_cross_entropy, cumsum, where_, **flash_attention** ✅, **cross_attention** ✅, **grouped_query_attention** ✅, **multi_latent_attention** ✅, **paged_attention** ✅

**Quantization:** quantize_per_token, dequantize_per_channel ✅

### S7 forward-looking notes

flash_attention / GQA / MLA / paged_attention all clear BL4×L1 at
tier-3 small shapes on 6 GB; tier-2 large attention shapes (llama2-7b,
llama3-7b 4k+) still OOM on this hardware. That's a BL5/L2 concern for
S7+ — out of scope for G6.

---

## Pre-Refactor Reference (G6 v1, commit fd2cbe0)

The following were completed under the **old architecture** before the Lang/IR/Compiler redesign. They serve as reference, but all features and tests need re-implementation and re-validation under the new architecture:

- 46 `.ak` example files, grammar fixes, 4D tensor support
- SemanticIR op catalog 46 ops, Attention/Rope/Quantize/MLA fields
- `ast_to_strategy()` converter, StrategyIR JSON round-trip
- 10 Triton template classes, V1 validator extensions
- 422 tests, BL5×L1+L2: 9/9 criteria, 46/46 E2E correct

> ⚠️ All above are pre-refactor. Post-refactor re-implementation and re-validation required.

---

## Tasks

### Track 1: OpRegistry + SemanticInterpreter (P0)

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| C1.1 | Design `OpSchema` dataclass + `OpRegistry` class | P0 | 0.5d | Validate with `relu` (OT0) + `matmul` (OT2) — simplest ops | ✅ |
| C1.2 | Migrate all 46 ops from `catalog.py` to `OpRegistry` | P0 | 1d | All 46 ops registered; spot-check OT0-OT4 one each | ✅ |
| C1.3 | Remove op-specific if/elif from `shape_inference.py` | P0 | 0.5d | Shape inference correct for `matmul` (2D→2D), `softmax` (reduce), `flash_attention` (4D) | ✅ |
| C1.4 | Implement `SemanticInterpreter` (PyTorch eager executor) | P0 | 1d | Start with OT0 (12 elem ops), then OT1-OT4; correctness 100% | ✅ |
| C1.5 | Migrate `numerical_check.py` to use `SemanticInterpreter` | P0 | 0.5d | `matmul` + `softmax` + `layernorm` numerical check unchanged | ✅ |
| C1.6 | Update `kernel_cache.py` to use parser instead of `_build_ir()` | P1 | 0.5d | `matmul.ak` → parse → IR → cache round-trip | ✅ |
| C1.7 | Update `triton_template_engine.py` to use registry lookup | P0 | 0.5d | `matmul` + `relu` codegen via registry lookup matches old output | ✅ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §3 (OpRegistry), §5 (SemanticInterpreter)

### Track 2: Pass Infrastructure + SSA Validator (P1, depends on Track 1)

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| C2.1 | Define `ArkePass` protocol + `PassContext` + `PassPipeline` | P1 | 0.5d | Pipeline runs on `matmul` IR (smoke test) | ✅ |
| C2.2 | Implement `ShapeInferencePass` (wraps `shape_inference.py`) | P1 | 0.5d | `matmul` (2D), `batch_matmul` (3D), `flash_attention` (4D) shape correct | ✅ |
| C2.3 | Implement `SSAValidator` + `SSAValidationPass` | P1 | 1d | All 46 ops pass; 5+ crafted invalid IR rejected (dup def, undefined use, shape mismatch, type mismatch, cycle) | ✅ |
| C2.4 | Implement `RationalePreservationPass` | P1 | 0.5d | `matmul.ak` with @rationale → IR → codegen comments preserved | ✅ |
| C2.5 | Integrate `PassPipeline` into `ArkePipeline.run()` | P1 | 0.5d | Full pipeline: `softmax.ak` → parse → passes → codegen → GPU correct | ✅ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §4 (Pass Infrastructure), §8 (SSA Validator)

### Track 3: Backend Abstraction (P1, independent)

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| C3.1 | Define `ArkeBackend` protocol + `BackendArtifact` hierarchy | P1 | 0.5d | Protocol compiles with `matmul` + `relu` backend artifacts | ✅ |
| C3.2 | Wrap `TritonBackend` to implement `ArkeBackend` | P1 | 0.5d | `matmul` + `softmax` + `layernorm` codegen via TritonBackend identical to old path | ✅ |
| C3.3 | Update `ArkePipeline` to use backend via protocol | P1 | 0.5d | Full E2E: `matmul.ak` → pipeline → TritonBackend → GPU → correct result | ✅ |
| C3.4 | Implement `MockBackend` for testing | P1 | 0.5d | MockBackend returns deterministic output for `relu`/`add` (OT0 smoke test) | ✅ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §7 (Backend Abstraction)

### Track 4: Agent Tool Infrastructure

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| Agent-G6-M2 | Declarative `ToolMeta` + `ArkeTool` ABC (Migration 2) | P1 | 1d | `compile_and_profile` tool on `matmul` returns structured JSON | ⬜ |
| Agent-G6-CLI | Structured `--json-log` output + consistent exit codes | P1 | 0.5d | `arke codegen matmul.ak --json-log` outputs valid JSON | ⬜ |

**Design ref:** `docs/architecture/arke-harness.md` §6 (Tools — declarative `ToolMeta`)

### Track 5: Full Operator Verification + Non-regression

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| G6-BL4 | BL4×L1 full run: 46 ops × ST1-ST2, correctness 100%, perf ≥1.00× P3 | P0 | 2d | All OT0-OT4 GPU-verified; any failure blocks gate | ✅ (46/46) |
| D8 | Full non-regression run + fix regressions (ARCH.10) | P0 | 1d | ≥422 tests, 0 new failures | ✅ |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: OpRegistry live | Track 1 (C1.1-C1.3, C1.7) | Day 2 | G6[1] partial |
| M2: SemanticInterpreter 46 ops correct | Track 1 (C1.4-C1.6) | Day 4 | G6[2] |
| M3: Pass pipeline + SSA | Track 2 (C2.1-C2.5) | Day 7 | G6[3], G6[4] |
| M4: Backend abstraction | Track 3 (C3.1-C3.4) | Day 5 | G6[5] |
| M5: Agent tools | Track 4 | Day 6 | — |
| M6: BL4×L1 full operator verification | Track 5 (G6-BL4) | Day 9 | G6[6] |
| M7: Non-regression + gate | Track 5 (D8) | Day 10 | G6[7] |

**Critical path:** Track 1 → Track 2 → BL4 verification → Non-regression

**Note:** Track 3 and Track 4 are independent and can be parallelized with Track 1.

---

## Dependencies

- **Depends on:** S0-S5 (all passed)
- **Blocks:** S7 (Lang & IR v0.1.0 needs Pass pipeline, OpRegistry, Backend abstraction)
