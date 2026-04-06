# Phase 1 — Stage 7: Lang & IR v2

> Gate G7 exit criteria → [plan.md](../roadmap/plan.md#stage-7-g7-lang--ir-v2-)

**Objective:** Implement the multi-layer IR architecture (Layer 4/3/2/1), upgrade Arke Lang with `where` clause and backend-agnostic strategy, complete spec documents, assess dynamic shape feasibility, establish MLIR framework skeleton.

**Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
**Blocks:** S8 (Agent Autonomy needs v2 IR/Lang, MLIR skeleton, full op coverage)

---

## Gate Criteria Breakdown

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | Arke Lang Spec v2.0 document finalized | `docs/spec/arke-lang-spec-v2.md` exists and complete |
| 2 | Arke IR Spec v2.0 document finalized (Layer 4/3/2/1 defined) | `docs/spec/arke-ir-spec-v2.md` exists, defines Layer 4/3/2/1 |
| 3 | `where` clause MVP: parses + SemanticIR `symbolic_dims` populated | `pytest tests/test_symbolic_shape.py` — ≥5 ops with `where` clause |
| 4 | Dynamic Shape feasibility assessment document complete | `docs/phase1/dynamic-shape-feasibility.md` exists |
| 5 | MLIR framework skeleton: MLIREmitter exists, BL1 matmul verified | MLIREmitter skeleton exists; BL1 matmul verified via MLIR skeleton |
| 6 | All 45 ops: .ak → SemanticIR → StrategyIR full round-trip | `python -m arke.compiler.pipeline --ak examples/<op>.ak --dry-run` passes all 45 ops |
| 7 | Token efficiency: .ak lines < Triton lines for all OT0-OT4 | OT0-OT4: `.ak` lines < Triton lines @ equivalent performance |
| 8 | Backend-agnostic strategy: 0 Triton-specific fields in StrategyIR core | `scripts/check_backend_agnostic.py` — 0 Triton-specific fields in `StrategyIR.decisions[]` |
| 9 | Non-regression: ≥422 tests, 0 new failures | `pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures |

---

## Context: Already Completed (from G6 v1)

The following Lang/IR items were completed during the earlier G6 v1 pass and form the foundation for S7 work:

| ID | Description | Status |
|:---|:------------|:------:|
| D6-L1 | `.ak` 4D tensor syntax extension | ✅ Done |
| D6-L2 | gather/scatter semantic nodes | ✅ Done |
| D6-L3 | quantize primitive syntax | ✅ Done |
| D6-L4 | paged memory semantic annotation (stub) | ✅ Done (stub) |
| D6-L5 | grammar fix (array literal, float constant) | ✅ Done |
| D6-L6 | `.ak` example files for all 45 ops | ✅ Done (46 files) |
| D6-IR1 | SemanticIR op catalog → 45 ops | ✅ Done |
| D6-IR2 | AttentionSemanticIR (mask_type, num_kv_heads, head_dim) | ✅ Done |
| D6-IR3 | RopeSemanticIR (theta, base, rotary_dim) | ✅ Done |
| D6-IR4 | QuantizeSemanticIR (scale_dtype, group_size, zero_point) | ✅ Done |
| D6-IR5 | `ast_to_strategy()` converter | ✅ Done |
| D6-IR6 | StrategyIR JSON round-trip (all 45 ops) | ✅ Done |
| D6-IR7 | MLA-specific fields (latent_dim, kv_lora_rank) | ✅ Done |
| D6-E1 | 10 Triton template classes (OT3/OT4 full) | ✅ Done |
| D6-E6 | V1 validator extension (attention + quantization tolerance) | ✅ Done |

---

## Tasks

### Track 1: Spec Documents (P1)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| ARCH.1 | Arke Lang Spec v2.0 document (`docs/spec/arke-lang-spec-v2.md`) | P1 | 1d | ⬜ |
| ARCH.2 | Arke IR Multi-Layer Architecture spec v2.0 (`docs/spec/arke-ir-spec-v2.md`, Layer 4/3/2/1 defined) | P1 | 1.5d | ⬜ |
| ARCH.9 | Layer 3/2/1 spec documents (stub) — `docs/spec/arke-ir-layer{3,2,1}-spec.md` + MLIR mapping updated | P2 | 0.5d | ⬜ |
| ARCH.12 | Dynamic Shape feasibility assessment (`docs/phase1/dynamic-shape-feasibility.md`) — covers where clause design, symbolic_dims IR, shape constraint propagation, Triton/MLIR integration points, risk assessment | P1 | 1d | ⬜ |

**Design ref:** `docs/spec/arke-ir-spec-design.md` (multi-layer architecture), `docs/spec/arke-lang-spec-design.md` (lang spec)

### Track 2: `where` Clause + Symbolic Shapes (P2, depends on ARCH.12 feasibility)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D3 | Implement `where` clause in Lark grammar (ARCH.8 MVP) | P2 | 0.5d | ⬜ |
| D4 | Add `symbolic_dims` field to SemanticIR + converter (ARCH.8 MVP) | P2 | 0.5d | ⬜ |
| D5 | Shape propagation for symbolic dims in `ShapeInferencePass` (ARCH.8 MVP) | P2 | 1d | ⬜ |
| D6 | Write `tests/test_symbolic_shape.py` (G6-LI.7) | P2 | 0.5d | ⬜ |

### Track 3: MLIR Framework Skeleton (P1)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| ARCH.11 | MLIR framework skeleton — MLIREmitter exists, BL1 matmul verified via MLIR skeleton | P1 | 1d | ⬜ |
| D9 | Layer 3/2/1 spec stub implementation + MLIR integration (ARCH.9, ARCH.11) | P2 | 1d | ⬜ |

**Design ref:** `docs/spec/arke-ir-spec-design.md` §10 (MLIR Integration Design)

### Track 4: Lang & IR Completeness Validation (P0)

These items correspond to the G6-LI criteria that verify completeness and correctness:

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| G6-LI.1 | All 45 ops expressible and parseable in `.ak` | P0 | — | ✅ (v1) |
| G6-LI.2 | `.ak → SemanticIR → StrategyIR → Pass pipeline` full round-trip | P0 | 0.5d | ⬜ |
| G6-LI.3 | `@rationale` annotations preserved through full pipeline | P1 | 0.5d | ⬜ |
| G6-LI.4 | Token efficiency: `.ak` ≤ Triton line count (OT0-OT4) | P1 | 0.5d | ⬜ |
| G6-LI.5 | Python interop IR round-trip (`pytest tests/test_ir_roundtrip.py`) | P0 | 0.5d | ⬜ |
| G6-LI.6 | Grammar completeness: 0 parse failures (`arke parse examples/ --strict`) | P0 | — | ✅ (v1) |
| G6-LI.7 | Symbolic shape `.ak` → SemanticIR with `where` clause | P2 | — | → Track 2 |
| G6-LI.8 | Backend-agnostic strategy (no Triton-specific fields in StrategyIR core) | P1 | 0.5d | ⬜ |

### Track 5: Remaining Open G6 v1 Items (Lang/IR/Eng)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D6-IR8 | PaddingStrategy decision type | P2 | 0.5d | ⬜ |
| D6-A1 | Attention prompt template | P1 | 0.5d | ⬜ |
| D6-A2 | Rope prompt + rationale template | P1 | 0.5d | ⬜ |
| D6-A3 | Fusion opportunity detection | P2 | 0.5d | ⬜ |
| D6-A4 | Quantize/dequantize prompt template | P2 | 0.5d | ⬜ |
| D6-A5 | Batch optimize pipeline (45 ops parallel sessions) | P2 | 1d | ⬜ |
| D6-A6 | Non-aligned shape rationale template | P2 | 0.5d | ⬜ |
| D6-E2 | bench_l1 routing extension (45 ops + shape_registry) | P1 | 1d | ⬜ |
| D6-E3 | bench_l2 OT3/OT4 fused benchmark runner | P1 | 0.5d | ⬜ |
| D6-E4 | Baseline adaptation (FlashAttn-2, Liger, FlagGems GQA) | P1 | 0.5d | ⬜ |
| D6-E5 | CSV output `L1/OT{n}/perf_{op}.csv` | P2 | 0.5d | ⬜ |
| D7 | Write `scripts/check_backend_agnostic.py` (G6-LI.8) | P1 | 0.5d | ⬜ |

---

## BL5 Performance Targets (Inherited from original G6)

S7 inherits the BL5 performance/correctness targets. These were fully satisfied in G6 v1 and must not regress:

### L1 @ BL5 (OT0-4, ST1-4)

| Op Group | Correctness | Performance |
|:---------|:------------|:------------|
| **OT0** Elementwise (12 ops) | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.90 P1 (FlagGems elem) |
| **OT1** Reduction (10 ops) | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.85 P1 (FlagGems norm/softmax) |
| **OT2** Compute-Dense (11 ops) | 100%(ST1-3) + ≥95%(ST4) | matmul geomean ≥ 0.90 P0; others ≥ P3 |
| **OT3** Gated Activation (7 ops) | 100%(ST1-3) + ≥95%(ST4) | swiglu/rope geomean ≥ 0.85 P1 (Liger/FlagGems) |
| **OT4** Attention (5 ops) | 100%(ST1-4, excl. OOM) | FA geomean ≥ 0.80 P1 (FlashAttn-2); GQA ≥ 0.80 |

### L2 @ BL5 (Fused Operators)

| Fusion | Requirement | Baseline |
|:-------|:------------|:---------|
| matmul+relu, matmul+gelu | ≥ 1.05× unfused | P3 unfused |
| swiglu, geglu | ≥ 0.90× Liger | P1 |
| linear+cross_entropy | ≥ 1.05× unfused | P3 |
| QKV+flash_attention | ≥ 0.80× FlashAttn-2 | P1 |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: Spec documents drafted | Track 1 | Day 3 | G7[1], G7[2] |
| M2: Dynamic shape feasibility | Track 1 (ARCH.12) | Day 4 | G7[4] |
| M3: Full round-trip validated | Track 4 | Day 6 | G7[6], G7[7], G7[8] |
| M4: where clause MVP | Track 2 | Day 8 | G7[3] |
| M5: MLIR skeleton | Track 3 | Day 10 | G7[5] |
| M6: Remaining items + regression | Track 5 | Day 12 | G7[9] |

**Critical path:** Spec documents → where clause feasibility → MLIR skeleton → non-regression

---

## Dependencies

- **Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
- **Blocks:** S8 (Agent Autonomy needs v2 IR/Lang, full op coverage, MLIR skeleton)
