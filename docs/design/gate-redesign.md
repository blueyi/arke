# Stage 1 Gate System v3 — Function-First + Accuracy-Always + Performance-Progressive

> **Shape definitions:** All Tier 1/2/3 shapes (including non-aligned cases for every op category)
> are defined in [docs/design/BENCHMARK.md § Shape Matrix](../../docs/design/BENCHMARK.md#3-shape-matrix).
> Gate criteria reference those shapes by tier and count.

---

## Design Principles

**Gate priority: Function > Accuracy > Performance**

Each gate has a distinct essential purpose:
- **Function gates:** Verify a capability exists (can it do this at all?)
- **Accuracy gates:** Verify numerical correctness across all shapes (does it do it right?)
- **Performance gates:** Verify acceptable speed at the corresponding development stage (does it do it fast enough?)

Performance thresholds increase progressively — no unrealistic targets at early gates.

---

## Gate ↔ Purpose Mapping

| Gate | Essential Type | Core Question | Accuracy Req | Performance Req |
|:-----|:---------------|:-------------|:-------------|:----------------|
| G0 | **Function** | Can the environment run? | — | — |
| G1 | **Function + Accuracy** | Can IR express + validate correctly? | Tier 3 full 100% | — |
| G2 | **Function + Accuracy + Performance** | Can codegen produce correct, usable kernels? | Tier 3 full 100% | Initial baseline |
| G3 | **Function + Accuracy** | Can the LLM agent complete closed-loop optimization autonomously? | Tier 3 sampled 100% | Initial baseline |
| G4 | **Accuracy + Performance** | Is Arke better than LLM-direct? | Tier 3 full 100% | Comparative advantage |
| G5 | **Accuracy + Performance** | Does it work in real models? | Multi-config 100% | E2E acceptable |

---

## G0: Environment Feasibility

**Type: Function**
**Core question:** Are CUDA + Triton + PyTorch toolchains usable on the target hardware?

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G0.1 | CUDA detection | Function | `torch.cuda.is_available()` | `True` |
| G0.2 | Triton compilation | Function | Triton matmul kernel compiles | exit 0, no compile errors |
| G0.3 | GPU execution | Function | Triton matmul [128,128,128] executes | Returns non-zero tensor |
| G0.4 | Test framework | Function | `pytest tests/ -q` | ≥ 100 passed, 0 failed |

**Exit command:** `arke gate G0`
**Exit artifacts:** CI log (make test passes)

---

## G1: IR Expressiveness & Validation Correctness

**Type: Function + Accuracy**
**Core question:** Can the IR system fully express computational intent and optimization strategies? Is the validator correct across all shapes?

### Function Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G1.1 | OP_CATALOG coverage | Function | `len(OP_CATALOG)` | ≥ 10 ops |
| G1.2 | Strategy decision types | Function | Enumerate decision kinds | ≥ 6 kinds |
| G1.3 | IR serialization completeness | Function | All 10 ops × `from_json(to_json(ir))` | 100% round-trip match |
| G1.4 | `.ak` parse → IR | Function | ≥ 3 `.ak` files parse → AST → IR | Matches KernelBuilder output |
| G1.5 | V0 static validation available | Function | V0 validator on 10 ops | 100% complete, latency < 1ms |
| G1.6 | Unit test coverage | Function | `pytest tests/ -q` | ≥ 200 passed, 0 failed |

### Accuracy Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G1.7 | V1 numerical — matmul | Accuracy | **Tier 3 matmul 50 shapes**, 3 random seeds, f16 | **100% pass** (atol=0.1, rtol=0.05) |
| G1.8 | V1 numerical — softmax | Accuracy | **Tier 3 softmax 25 shapes**, 3 random seeds, f16 | **100% pass** |
| G1.9 | V1 numerical — elementwise | Accuracy | **Tier 3 elementwise 15 shapes**, 3 seeds, relu/gelu/silu | **100% pass** |
| G1.10 | V1 numerical — layernorm | Accuracy | **Tier 3 layernorm/rmsnorm 15 shapes**, 3 seeds, f16 | **100% pass** |

> **No performance requirement.** G1 only validates IR expressiveness and numerical correctness, not kernel speed.

**Exit command:** `arke gate G1`
**Exit artifacts:**
- `gate_results/G1/validation_matrix.csv` — shape × op × seed → pass/fail
- `gate_results/G1/unit_tests.log`

---

## G2: Codegen Correctness & Baseline Performance

**Type: Function + Accuracy + Performance (initial baseline)**
**Core question:** Can code generation produce correct GPU kernels? What performance level are we at?

### Function Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G2.1 | Pipeline connectivity | Function | IR → Strategy → Codegen → Compile → Run | Single kernel end-to-end passes |
| G2.2 | Multi-op templates | Function | matmul + softmax + elementwise each generate Triton code | All templates compile and execute |

### Accuracy Criteria (Tier 3 full, hard gate)

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G2.3 | matmul correctness | Accuracy | L1 Tier 3 **50 shapes**, f16, vs NumPy | **100% allclose** (atol=0.1) |
| G2.4 | softmax correctness | Accuracy | L1 Tier 3 **25 shapes**, f16, vs NumPy | **100% allclose** |
| G2.5 | elementwise correctness | Accuracy | L1 Tier 3 **15 shapes**, relu/gelu/silu | **100% allclose** |
| G2.6 | layernorm correctness | Accuracy | L1 Tier 3 **15 shapes**, f16, vs NumPy | **100% allclose** |

### Performance Criteria (initial baseline — relaxed thresholds)

> This is Arke's first performance data. Thresholds correspond to "template basically usable" level.

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G2.7 | matmul perf pass rate | Performance | Tier 3 50 shapes vs cuBLAS (excl. M×N×K < 2^20) | **≥ 50%** shapes achieve ≥ 50% cuBLAS |
| G2.8 | matmul perf geomean | Performance | Tier 3 50 shapes geomean (excl. M×N×K < 2^20) | **geomean ≥ 60%** cuBLAS |
| G2.9 | softmax perf pass rate | Performance | Tier 3 25 shapes vs cuDNN (excl. M×N < 2^19) | **≥ 40%** shapes achieve ≥ 50% cuDNN |
| G2.10 | elementwise perf | Performance | Tier 3 15 shapes vs PyTorch eager (excl. numel < 2^20) | **≥ 50%** shapes achieve ≥ 50% PyTorch |
| G2.11 | layernorm perf | Performance | Tier 3 15 shapes vs cuDNN (excl. M×N < 2^19) | **≥ 40%** shapes achieve ≥ 50% cuDNN |

> **Performance thresholds are low.** This is the Phase 1.2 exit — manual strategy template codegen, no LLM optimization or autotune yet. Goal is "usable" not "fastest".

**Exit command:** `arke gate G2`
**Exit artifacts:**
- `gate_results/G2/matmul_tier3.csv` — 50 shapes × baselines
- `gate_results/G2/softmax_tier3.csv` — 25 shapes × baselines
- `gate_results/G2/elementwise_tier3.csv` — 15 shapes × baselines
- `gate_results/G2/layernorm_tier3.csv` — 15 shapes × baselines
- `gate_results/G2/summary.json` — pass rates / geomean / worst case

---

## G3: LLM Agent Autonomous Optimization

**Type: Function + Accuracy (+ performance observation)**
**Core question:** Can the LLM agent complete the optimization loop without human intervention? Does it work across diverse shapes?

### Function Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G3.1 | Tool usage breadth | Function | Single session trajectory | ≥ 8 distinct tools |
| G3.2 | Decision capability | Function | Single session trajectory | ≥ 4 strategy decisions |
| G3.3 | Closed-loop completeness | Function | Agent runner | start → finish, 0 human steps |
| G3.4 | Error recovery | Function | Trajectory analysis | ≥ 1 rollback → successful recovery |
| G3.5 | Multi-provider support | Function | Anthropic + OpenAI each complete 1 run | Both providers complete |
| G3.6 | Trajectory recording | Function | JSONL output | header + ≥ 6 step records, valid format |

### Accuracy Criteria (agent-produced kernels must all be correct)

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G3.7 | Agent kernel correctness — generalization | Accuracy | Agent generates kernels on **≥ 10 diverse shapes** (sampled from Tier 3: square×3, rect×2, non-aligned×2, LLM×3) | **100% correct** |

### Performance Observation (record but don't gate)

| # | Criterion | Type | Verification | Notes |
|---|-----------|:----:|-------------|-------|
| G3.P1 | Agent kernel performance | Observe | Agent kernels vs cuBLAS geomean | Record to CSV, no threshold |
| G3.P2 | Agent vs G2 template | Observe | Agent kernel vs template kernel perf ratio | Observe whether LLM optimization improves template baseline |

> **G3 performance is observation, not a gate.** LLM agent performance is unstable when first running. The key is proving "can close the loop" + "output is correct". Performance improvement is G4's goal.

**Exit command:** `arke gate G3`
**Exit artifacts:**
- `gate_results/G3/agent_trajectories/` — one JSONL per shape
- `gate_results/G3/agent_kernels/` — generated kernel per shape
- `gate_results/G3/correctness.csv` — shape → correct/incorrect
- `gate_results/G3/performance.csv` — shape → latency → vs_cublas (observation data)

---

## G4: Comparative Advantage over Direct LLM

**Type: Accuracy + Performance (comparative)**
**Core question:** Where is Arke better than LLM writing Triton directly? By how much?

### Accuracy Criteria (Tier 3 full comparison)

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G4.1 | Arke correct rate ≥ LLM-direct | Accuracy | Tier 3 matmul 50 shapes × 3 trials | `arke_correct_rate ≥ direct_correct_rate` |
| G4.2 | Arke consistency ≥ LLM-direct | Accuracy | Tier 3 × 3 trials variance | `arke_stddev ≤ direct_stddev` |

### Performance Criteria (progressive targets based on development stage)

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G4.3 | vs LLM-direct performance | Performance | Tier 3 matmul geomean (excl. M≤32) | Arke geomean **≥ 90%** LLM-direct geomean |
| G4.4 | vs P1 Expert (FlagGems) | Performance | Tier 3 matmul geomean (excl. M≤32) | Arke geomean **≥ 70%** FlagGems geomean |
| G4.5 | Token efficiency | Performance | End-to-end token consumption stats | Arke total tokens **≤ 60%** LLM-direct tokens |

### L2 Observation (record but don't gate)

| # | Criterion | Type | Verification | Notes |
|---|-----------|:----:|-------------|-------|
| G4.P1 | L2 fused operators | Observe | matmul+gelu Tier 3 shapes | Record Arke fused vs separate vs FlagGems |

> **G4 performance thresholds are reduced from v2.** Stage 1 Arke doesn't have autotune or MLIR backend — template codegen is naturally weaker on small shapes. vs FlagGems 70% is a pragmatic target.

**Exit command:** `arke gate G4`
**Exit artifacts:**
- `gate_results/G4/comparison_tier3.csv` — Tier 3 × method × trial
- `gate_results/G4/token_efficiency.json`
- `gate_results/G4/summary.json`

---

## G5: End-to-End Model Integration

**Type: Accuracy + Performance (E2E)**
**Core question:** Do Arke kernels work in real models? Is performance acceptable?

### Accuracy Criteria (multi-config, hard gate)

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G5.1 | Inference correctness — multi seq_len | Accuracy | GPT-2, seq=128/256/512 | **100%** top-1 match, max_logit_diff < 5.0 |
| G5.2 | Inference correctness — multi batch | Accuracy | GPT-2, batch=1/4/8, seq=128 | **100%** top-1 match |

### Performance Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G5.3 | Latency — seq=128 | Performance | L3 benchmark | Arke ≤ **1.15×** eager |
| G5.4 | Latency — seq=512 | Performance | L3 benchmark | Arke ≤ **1.20×** eager |
| G5.5 | Latency generalization | Performance | L3, 3 seq_lens | ≥ **2/3** seq_lens: Arke ≤ 1.15× eager |
| G5.6 | Memory | Performance | L3, all configs | **100%** peak_mem ≤ 6144 MB |

### Function Criteria

| # | Criterion | Type | Verification | Pass Condition |
|---|-----------|:----:|-------------|----------------|
| G5.7 | Replacement coverage | Function | Patch statistics | ≥ 48 Conv1D/Linear replaced |

> **E2E performance thresholds are relaxed.** seq=128 raised from 1.1× to 1.15×; seq=512 set at 1.20×. Reason: monkey-patching overhead is not eliminable in Stage 1 (requires Stage 2's torch.compile backend integration). Accuracy is non-negotiable.

**Exit command:** `arke gate G5`
**Exit artifacts:**
- `gate_results/G5/e2e_results.csv` — seq × batch × mode → latency/mem/correct
- `gate_results/G5/summary.json`

---

## Progressive Performance Roadmap

Shows how performance standards increase across development stages:

```
           G2 (Template)  G3 (Agent)    G4 (Compare)   G5 (E2E)
           ────────────   ──────────    ────────────   ────────
Function:  ✓ required     ✓ core        —              ✓ coverage
Accuracy:  100%           100%          ≥ LLM-direct   100% multi-config
Perf goal: ≥50% cuBLAS    observe only  ≥90% direct    ≤1.15× eager
                                        ≥70% FlagGems
Perf type: absolute floor  no gate      relative edge  E2E overhead
```

**Why no perf gate at G3?**
- Agent strategy quality is unstable on first runs
- Agent value lies in "can close the loop autonomously", not "fastest on first try"
- Performance improvement is an iterative process, validated at G4

**Why does G4 gate relative advantage rather than absolute level?**
- G4's question is "Is Arke better than LLM-direct?", not "How fast is Arke?"
- Absolute performance baseline was already established at G2

---

## Exclusion Rules

Performance measurements exclude shapes where Triton kernel launch overhead
(~30µs) dominates over actual compute time. Accuracy is ALWAYS tested on ALL shapes.

| Rule | Effect | Rationale |
|:-----|:-------|:----------|
| matmul: M×N×K < 2^20 | perf excluded from stats | Launch overhead > compute |
| softmax: M×N < 2^19 | perf excluded from stats | Same |
| elementwise: numel < 2^20 | perf excluded from stats | Same |
| layernorm: M×N < 2^19 | perf excluded from stats | Same |
| softmax: N > 131072 | accuracy+perf excluded | Single-block template limit |
| OOM shapes | Skip, record "OOM" | 6GB VRAM limit |
| Triton compile timeout (>60s) | Record "TIMEOUT", accuracy marked fail | Template may need fix |

---

## CLI Integration

```bash
# Gate verification (default Tier 3)
arke gate G0                      # Environment check
arke gate G1                      # IR + validator + Tier 3 numerical
arke gate G2                      # L1 Tier 3 full bench (all ops)
arke gate G3                      # Agent 10-shape closed-loop
arke gate G4                      # Tier 3 comparison + token stats
arke gate G5                      # L3 multi-config E2E
arke gate --all                   # Full suite (~30-60 min)

# Quick check (Tier 1, daily development)
arke gate G2 --tier 1             # 15 shapes fast regression

# Example output
arke gate G2

  G2: Codegen Correctness & Baseline Performance
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Function:
    G2.1 Pipeline connectivity       ✅ PASS
    G2.2 Multi-op templates           ✅ PASS  4/4
  Accuracy:
    G2.3 matmul correctness (50)      ✅ PASS  50/50 (100%)
    G2.4 softmax correctness (25)     ✅ PASS  25/25 (100%)
    G2.5 elementwise correctness (15) ✅ PASS  15/15 (100%)
    G2.6 layernorm correctness (15)   ✅ PASS  15/15 (100%)
  Performance:
    G2.7  matmul ≥50% rate            ✅ PASS  36/46 (78% ≥ 50%)
    G2.8  matmul geomean              ✅ PASS  72% cuBLAS (≥ 60%)
    G2.9  softmax ≥50% rate           ✅ PASS  12/22 (55% ≥ 40%)
    G2.10 elementwise ≥50% rate       ✅ PASS  10/13 (77% ≥ 50%)
    G2.11 layernorm ≥50% rate         ✅ PASS  8/14 (57% ≥ 40%)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  G2: PASS (11/11)
  Artifacts: gate_results/G2/
```

---

## Mapping to plan-v3.0

| Gate | Phase | Previous Target | Current Target |
|:-----|:------|:----------------|:---------------|
| G0 | 1.0 | "Triton matmul runs" | Function: 4 environment checks |
| G1 | 1.1 | "Known-good strategy representable" | Function 6 items + Accuracy Tier 3 full 4 ops |
| G2 | 1.2 | "perf ≥ 70% cuBLAS" | Function 2 + Accuracy Tier 3 full 4 ops + Performance 5 (relaxed) |
| G3 | 1.3-1.4 | "matmul perf ≥ 50% cuBLAS" | Function 6 + Accuracy Tier 3 sampled 1 + Performance observation |
| G4 | 1.5 | "Arke ≥ LLM-direct across ≥5" | Accuracy comparison 2 + Performance comparison 3 |
| G5 | 1.7 | "latency ≤ torch.compile" | Accuracy multi-config 2 + Performance 4 (relaxed) + Function 1 |
