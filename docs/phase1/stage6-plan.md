# Phase 1 — Stage 6: Compiler Infrastructure

> Gate G6 exit criteria → [plan.md](../roadmap/plan.md#stage-6-g6-compiler-infrastructure--%E2%AC%9C--current)

**Status:** 6/7 criteria PASS (85.7%) — G6.6 performance benchmark 40/45 ops complete

**Objective:** Refactor the compiler toolchain into a clean, extensible architecture. OpRegistry as single source of truth, Pass pipeline for composable transformations, Backend abstraction for multi-target support. **Validated through full 45-op correctness and performance.**

**Depends on:** S0-S5 (all passed)
**Blocks:** S7 (Lang & IR v2 needs Pass pipeline, OpRegistry, Backend abstraction)

---

## Gate Criteria Breakdown

**BL Exit:** BL4×L1 — Full 45 ops correctness 100% + perf ≥1.00× P3 (eager). Every infra component is validated through real operator execution.

| # | Criterion | Verification | Operator Validation |
|:-:|:----------|:-------------|:-------------------|
| 1 | OpRegistry: single source of truth for all 45 ops (adding op ≤ 2 files) | `scripts/verify_op_registry.py` — adding op requires ≤2 file changes | All 45 ops registered, metadata complete |
| 2 | SemanticInterpreter: PyTorch eager executor, correctness 100% | `pytest tests/test_semantic_interpreter.py` — all 45 ops correct | OT0 relu/gelu → OT4 flash_attention, all pass |
| 3 | Pass Infrastructure: ArkePass protocol + PassPipeline with ≥2 passes | `pytest tests/test_pass_infra.py` — Pass protocol + Pipeline with ≥2 passes | ShapeInference + SSAValidation on matmul/softmax/layernorm |
| 4 | SSA Validator: validates all 45 ops; rejects ≥5 invalid IR examples | `pytest tests/test_ssa_validator.py` — all 45 ops valid; ≥5 invalid rejected | All 45 ops pass; broken IR (dup SSA, shape mismatch, etc.) rejected |
| 5 | Backend Abstraction: ArkeBackend protocol + TritonBackend implements it | `pytest tests/test_backend_protocol.py` — `ArkeBackend` protocol + `TritonBackend` | matmul/relu/softmax codegen via backend protocol |
| 6 | Codegen + GPU: 45 ops via TritonBackend, correctness 100%, perf ≥1.00× P3 | `arke bench --bl 4 --layer l1` — all 45 ops correct, perf ≥1.00× eager | Full BL4 (OT0-4 × ST1-2): every op GPU-verified |
| 7 | Non-regression: ≥422 tests passed, ≤6 skipped, 0 new failures | `pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures | — |

---

## Known Limitations & S7 Optimization Targets

**G6.6 Performance Benchmark Status: 40/45 ops complete (89% coverage)**

### Failed Operators (5 ops) — Deferred to S7

| Op | Issue | Root Cause | S7 Optimization Target |
|:---|:------|:-----------|:----------------------|
| flash_attention | OOM on tier-2 large shapes (llama2-7b-4k, llama3-7b-4k) | Triton kernel memory footprint exceeds 6GB VRAM; no memory optimization in current backend | Implement memory-efficient attention (block-wise computation, gradient checkpointing) in Arke-Lang v2 + Arke-Compiler v2 |

### Passing Operators (40/45) — All meet ≥1.00× P3 eager baseline

**OT0 (Elementwise, 9 ops):** add, cast, copy_, exp, mul, neg, sigmoid, tanh, rsqrt ✅

**OT1 (Move, 6 ops):** permute, transpose, gather, scatter, split, concat ✅

**OT2 (Compute, 3 ops):** matmul, batch_matmul, grouped_matmul ✅

**OT3 (Reduce, 9 ops):** softmax, layernorm, rmsnorm, rmsnorm_residual, reduce_sum, reduce_mean, reduce_max, argmax, topk ✅

**OT4 (Attention/Special, 8 ops):** rope, embedding, geglu, swiglu, cross_entropy, fused_linear_cross_entropy, cumsum, where_ ✅

**Quantization (2 ops):** quantize_per_token, dequantize_per_channel ✅

### S7 Action Items

1. **Memory-Efficient Attention Design** — Redesign flash_attention in Arke-Lang v2 with explicit memory tiling strategy and shape constraints
2. **Shape Tier Validation** — Ensure all tier-2 shapes fit within target hardware (6GB VRAM) constraints; adjust benchmark-shapes.md if needed
3. **Backend Memory Optimization** — Implement memory pooling, kernel fusion, and gradient checkpointing in Arke-Compiler v2

---

## Pre-Refactor Reference (G6 v1, commit fd2cbe0)

The following were completed under the **old architecture** before the Lang/IR/Compiler redesign. They serve as reference, but all features and tests need re-implementation and re-validation under the new architecture:

- 46 `.ak` example files, grammar fixes, 4D tensor support
- SemanticIR op catalog 45 ops, Attention/Rope/Quantize/MLA fields
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
| C1.2 | Migrate all 45 ops from `catalog.py` to `OpRegistry` | P0 | 1d | All 45 ops registered; spot-check OT0-OT4 one each | ✅ |
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
| C2.3 | Implement `SSAValidator` + `SSAValidationPass` | P1 | 1d | All 45 ops pass; 5+ crafted invalid IR rejected (dup def, undefined use, shape mismatch, type mismatch, cycle) | ✅ |
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

**Design ref:** `docs/architecture/agent-design.md` §5.1 (Tool Declarative Interface)

### Track 5: Full Operator Verification + Non-regression

| ID | Task | Priority | Estimate | Operator Validation | Status |
|:---|:-----|:--------:|:--------:|:-------------------|:------:|
| G6-BL4 | BL4×L1 full run: 45 ops × ST1-ST2, correctness 100%, perf ≥1.00× P3 | P0 | 2d | All OT0-OT4 GPU-verified; any failure blocks gate | 🟨 (40/45) |
| D8 | Full non-regression run + fix regressions (ARCH.10) | P0 | 1d | ≥422 tests, 0 new failures | ✅ |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: OpRegistry live | Track 1 (C1.1-C1.3, C1.7) | Day 2 | G6[1] partial |
| M2: SemanticInterpreter 45 ops correct | Track 1 (C1.4-C1.6) | Day 4 | G6[2] |
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
- **Blocks:** S7 (Lang & IR v2 needs Pass pipeline, OpRegistry, Backend abstraction)
