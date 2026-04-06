# Phase 1 — Stage 6: Compiler Infrastructure

> Gate G6 exit criteria → [plan.md](../roadmap/plan.md#stage-6-g6-compiler-infrastructure--%E2%AC%9C--current)

**Objective:** Refactor the compiler toolchain into a clean, extensible architecture. OpRegistry as single source of truth, Pass pipeline for composable transformations, Backend abstraction for multi-target support.

**Depends on:** S0-S5 (all passed)
**Blocks:** S7 (Lang & IR v2 needs Pass pipeline, OpRegistry, Backend abstraction)

---

## Gate Criteria Breakdown

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | OpRegistry: single source of truth for all 45 ops (adding op ≤ 2 files) | `scripts/verify_op_registry.py` — adding op requires ≤2 file changes |
| 2 | SemanticInterpreter: PyTorch eager executor, all 45 ops correct | `pytest tests/test_semantic_interpreter.py` — all 45 ops correct |
| 3 | Pass Infrastructure: ArkePass protocol + PassPipeline with ≥2 passes | `pytest tests/test_pass_infra.py` — Pass protocol + Pipeline with ≥2 passes |
| 4 | SSA Validator: validates all 45 ops; rejects ≥5 invalid IR examples | `pytest tests/test_ssa_validator.py` — all 45 ops valid; ≥5 invalid examples rejected |
| 5 | Backend Abstraction: ArkeBackend protocol + TritonBackend implements it | `pytest tests/test_backend_protocol.py` — `ArkeBackend` protocol + `TritonBackend` implements it |
| 6 | Non-regression: ≥422 tests passed, ≤6 skipped, 0 new failures | `pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures |

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

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| C1.1 | Design `OpSchema` dataclass + `OpRegistry` class | P0 | 0.5d | ⬜ |
| C1.2 | Migrate all 45 ops from `catalog.py` to `OpRegistry` | P0 | 1d | ⬜ |
| C1.3 | Remove op-specific if/elif from `shape_inference.py` | P0 | 0.5d | ⬜ |
| C1.4 | Implement `SemanticInterpreter` (PyTorch eager executor) | P0 | 1d | ⬜ |
| C1.5 | Migrate `numerical_check.py` to use `SemanticInterpreter` | P0 | 0.5d | ⬜ |
| C1.6 | Update `kernel_cache.py` to use parser instead of `_build_ir()` | P1 | 0.5d | ⬜ |
| C1.7 | Update `triton_template_engine.py` to use registry lookup | P0 | 0.5d | ⬜ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §3 (OpRegistry), §5 (SemanticInterpreter)

### Track 2: Pass Infrastructure + SSA Validator (P1, depends on Track 1)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| C2.1 | Define `ArkePass` protocol + `PassContext` + `PassPipeline` | P1 | 0.5d | ⬜ |
| C2.2 | Implement `ShapeInferencePass` (wraps `shape_inference.py`) | P1 | 0.5d | ⬜ |
| C2.3 | Implement `SSAValidator` + `SSAValidationPass` | P1 | 1d | ⬜ |
| C2.4 | Implement `RationalePreservationPass` | P1 | 0.5d | ⬜ |
| C2.5 | Integrate `PassPipeline` into `ArkePipeline.run()` | P1 | 0.5d | ⬜ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §4 (Pass Infrastructure), §8 (SSA Validator)

### Track 3: Backend Abstraction (P1, independent)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| C3.1 | Define `ArkeBackend` protocol + `BackendArtifact` hierarchy | P1 | 0.5d | ⬜ |
| C3.2 | Wrap `TritonBackend` to implement `ArkeBackend` | P1 | 0.5d | ⬜ |
| C3.3 | Update `ArkePipeline` to use backend via protocol | P1 | 0.5d | ⬜ |
| C3.4 | Implement `MockBackend` for testing | P1 | 0.5d | ⬜ |

**Design ref:** `docs/architecture/arke-compiler-infrastructure.md` §7 (Backend Abstraction)

### Track 4: Agent Tool Infrastructure

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| Agent-G6-M2 | Declarative `ToolMeta` + `ArkeTool` ABC (Migration 2) | P1 | 1d | ⬜ |
| Agent-G6-CLI | Structured `--json-log` output + consistent exit codes | P1 | 0.5d | ⬜ |

**Design ref:** `docs/architecture/agent-design.md` §5.1 (Tool Declarative Interface)

### Track 5: Non-regression

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8 | Full non-regression run + fix regressions (ARCH.10) | P0 | 1d | ⬜ |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: OpRegistry live | Track 1 (C1.1-C1.3, C1.7) | Day 2 | G6[1] partial |
| M2: SemanticInterpreter live | Track 1 (C1.4-C1.6) | Day 4 | G6[2] |
| M3: Pass pipeline + SSA | Track 2 (C2.1-C2.5) | Day 7 | G6[3], G6[4] |
| M4: Backend abstraction | Track 3 (C3.1-C3.4) | Day 5 | G6[5] |
| M5: Agent tools | Track 4 | Day 6 | — |
| M6: Non-regression + gate | Track 5 (D8) | Day 8 | G6[6] |

**Critical path:** Track 1 → Track 2 → Non-regression

**Note:** Track 3 and Track 4 are independent and can be parallelized with Track 1.

---

## Dependencies

- **Depends on:** S0-S5 (all passed)
- **Blocks:** S7 (Lang & IR v2 needs Pass pipeline, OpRegistry, Backend abstraction)
