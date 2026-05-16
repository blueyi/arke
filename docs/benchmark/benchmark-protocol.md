# Arke Benchmark — Protocol, Scoring & Implementation

Measurement protocol, scoring system, CLI interface, output structure, and implementation status.

→ Parent: [`benchmark-design.md`](./benchmark-design.md)

---

## Design Goal

**Default target: all operators correct on all shapes, performance ≥ same-backend reference (within ε tolerance).**

```
Correctness: 100% pass rate across all OT × ST combinations
Performance: Arke latency ≤ same-backend reference latency × (1 + ε)   with ε = 0.03
```

The reference baseline is chosen to be **architecturally fair to the
backend Arke currently targets** (see *Same-Backend Fairness* below).
For Phase 1 (Triton path), the reference is the **best Triton-only
implementation per op** in the ladder (FlagGems / Liger / Unsloth /
vLLM-Triton / flash-attn). For later phases the reference shifts in
lock-step with the backend (Triton-Ascend, MLIR, C-like vendor DSL,
LLVM IR).

When no same-backend reference exists for a given op-shape, the row is
recorded with `perf_oracle_unavailable_<backend>=true` and treated as
**audit-only** (excluded from Gate scoring; still present in PERF_ALL).

> **Benchmark vs Gate separation:** the BL/OT/ST/L benchmark **measurement**
> layer is frozen — shape sets, op coverage, latency capture method, and
> PERF_ALL schema do not change. What may change per Gate is the
> **acceptance** layer (which baseline is the denominator; what ε
> tolerance applies; which rows are audit-only). See
> `docs/roadmap/plan.md` § Gate Governance for the locked rules.

---

## Same-Backend Fairness

> **Locked principle (2026-05-16, Leon-approved).** The Gate performance
> comparison denominator must use the **same compiler backend** as the
> Arke kernel under test. This isolates Arke's compilation quality from
> cross-backend architectural advantages.

| Arke Backend | Reference Baseline (denominator) | Audit-only when missing |
|:---|:---|:---|
| Phase 1 — Triton (NVIDIA) | best Triton-only kernel in ladder | no Triton reference exists for op-shape |
| Phase 2 — Triton-Ascend | best Triton-Ascend kernel | no Ascend Triton reference |
| Phase 3 — MLIR | MLIR-native reference (linalg/transform) | no MLIR reference |
| Phase 4 — C-like vendor DSL (CUDA-C / CCE-C / Bang-C) | hand-tuned vendor C-like reference (e.g. CUTLASS for CUDA-C) | no vendor reference for op-shape |
| Phase 5 — LLVM IR | LLVM-IR-direct hand-written reference | no LLVM reference |

**ε tolerance:** universally `ε = 0.03` (3% measurement-noise band).
**Pass criterion:** `arke_latency ≤ reference_latency × (1 + ε)`,
equivalently `ratio = reference / arke ≥ 1 / (1 + ε) ≈ 0.971`.

**Per-op reference selection.** Within a backend, pick the
**ladder-fastest** implementation for the (op, dtype, shape) combination,
following the PRIMARY+FALLBACK ordering documented in
[`golden-kernel-ladder.md`](./golden-kernel-ladder.md). Filter the ladder
to entries marked `backend = <current-phase-backend>`; cross-backend
entries are skipped for Gate scoring (they may still be recorded for
audit and reporting).

**Why same-backend.** Cross-backend comparisons conflate compiler
quality with backend-architectural advantages — e.g. comparing
Arke-Triton against PyTorch eager-fused-dispatch measures Triton's
kernel-launch overhead vs PyTorch's C-level dispatch path, *not* whether
Arke's Triton codegen is competitive with peer Triton kernels. The
same-backend rule answers a sharper question: *given this backend, how
close is Arke's compiled output to the best hand-tuned kernel humans
have produced on the same backend?*

---

## Measurement Protocol

### L1: Single Operator

```python
# 1. Warmup: 200 iterations (triggers autotune, JIT)
for _ in range(200):
    kernel(inputs)
torch.cuda.synchronize()

# 2. Measure: CUDA events, 500 iterations
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(500):
    kernel(inputs)
end.record()
torch.cuda.synchronize()
latency_us = start.elapsed_time(end) / 500 * 1000

# Alternative: triton.testing.do_bench (FlagGems standard)
from triton.testing import do_bench
latency_ms = do_bench(lambda: kernel(inputs), warmup=200, rep=500)
```

### L2: Fused Operator

Same protocol as L1. Compares: (a) unfused sequential, (b) torch.compile fusion,
(c) expert fusion (FlagGems/Liger), (d) Arke fusion.

### L3: E2E Model (= BL6)

```python
# 1. Load model, apply Arke kernel patches (KernelCache)
# 2. Warmup: 50 forward passes
# 3. Measure: 200 forward passes with CUDA events
# 4. Report: mean latency (ms), throughput (tok/s)
# 5. Correctness: top-1 logit match + max absolute diff
```

### Correctness Tolerances

Per dtype the absolute / relative tolerances passed to `torch.allclose`:

| Dtype | atol | rtol | Method |
|:------|-----:|-----:|:-------|
| f16 | 0.1 | 0.05 | `torch.allclose` + max/mean diff |
| f32 | 1e-5 | 1e-4 | `torch.allclose` |
| bf16 | 0.2 | 0.1 | `torch.allclose` |

The **reference output** that a candidate kernel is compared against
comes from the **Golden Kernel** for that op (see below) — *not* from a
hardcoded `torch.*` path. Picking a Golden per the locked ladder ensures
correctness is measured against the closest available production kernel,
the same one that anchors the perf ratio.

### Golden Kernel Protocol

For each op, **one designated Golden Kernel** plays both roles:

1. **Correctness oracle** — its output on a given input is the expected
   value all candidate kernels are compared against (within tolerances
   above).
2. **Perf denominator** — its latency on the same `(op, shape)` is the
   baseline against which `ratio_vs_baseline` is computed.

Selection is automatic: `benchmarks.golden_ladder.golden_runner_for(op)`
iterates registered runners in priority order (P0 → P5) and returns the
first one whose `supports(op)` is true and `available` is true.

The complete per-op assignment is locked in
[`golden-kernel-ladder.md`](./golden-kernel-ladder.md). At a glance:

| Tier  | Default Golden          | Notes |
|:------|:------------------------|:------|
| P0    | `cuBLAS/cuDNN`          | PyTorch vendor backends — most OT0/OT1/OT2 |
| P1    | `FlagGems` / `Liger-Kernel` | Liger preferred for OT3 fused ops |
| P2    | `flash-attn`, `FlashMLA`, `vLLM` | Specific OT4 attention ops |
| P3    | `PyTorch-eager`         | Fallback for ops with no production kernel |
| P4    | `torch.compile`         | Inductor — separate runner, never the golden |
| P5    | `Arke`, `LLM-direct`    | Our own — never the golden |

#### Audit semantics

Every PERF_ALL row carries `golden_runner` and `golden_priority` columns
identifying which kernel served as oracle. When the designated Golden
cannot produce an output:

| Status | When emitted |
|:-------|:-------------|
| `golden_unavailable_pending_baseline` | No registered runner declares supports(op) AND available=true. Row falls back to PyTorch-eager reference and the gate flags the gap. |
| `mla_golden_degraded=true` (in `correctness_reason`) | FlashMLA selected as primary for `multi_latent_attention` but returns None (e.g. on sm<9.0). Row uses PyTorch-eager fallback; gate-G7 surfaces the degradation. |
| `golden_runner=<name> returned None; used PyTorch-eager reference fallback` | Picked Golden cannot service this shape; PyTorch-eager covers the gap. |

#### Overrides

`bench_l1` exposes two override mechanisms for ad-hoc experimentation:

```bash
# Pin a single op
python -m benchmarks.bench_l1 --op softmax --golden softmax=FlagGems

# Pin many ops via YAML
python -m benchmarks.bench_l1 --all --golden-file ./golden_overrides.yaml
```

The override-pinned runner must declare `supports(op)` and be `available`;
otherwise `GoldenUnavailable` fires and the row is marked
`golden_unavailable_pending_baseline`.

#### Locked ladder preferences (`LADDER_PREFERENCES`)

A small protocol-level dict in `benchmarks/golden_ladder.py` pins a Golden
for ops where the strict P0-first rule chooses a runner that is *fast but
not a stable oracle*. These pins are part of the design contract;
adding/removing entries requires Leon's sign-off.

| Op   | Pinned Golden       | Rationale (commit / date)                            |
|:-----|:--------------------|:-----------------------------------------------------|
| rope | `PyTorch-eager` (P3) | Liger rope crashes on odd-D head dims and select non-aligned shapes (`tests/test_benchmark_correctness_probe_linea12.py::test_rope_odd_head_dim*`, commits `ad28665`+`c80d182`). A Golden must cover every measured shape; eager satisfies the entire OT3 grid and is the analytical reference. Liger remains a benchmark candidate — only its Golden role is removed. (G7.8c, 2026-05-12) |

Caller-supplied `--golden` / `--golden-file` overrides take precedence
over `LADDER_PREFERENCES` so ad-hoc experiments aren't blocked. To
benchmark Liger as Golden for one run:

```bash
python -m benchmarks.bench_l1 --op rope --golden rope=Liger-Kernel
```

### Metrics Collected Per Run

See [`benchmark-csv-spec.md`](./benchmark-csv-spec.md) for the full CSV
schema. Key metrics:

| Metric | Unit | Description |
|:-------|:-----|:------------|
| `latency_us` | μs | Median kernel latency |
| `tflops` | TFLOPS | Achieved throughput (compute-bound ops) |
| `gbps` | GB/s | Achieved bandwidth (memory-bound ops) |
| `ratio_vs_baseline` | ratio | `golden_latency / candidate_latency` (>1 = candidate faster than golden) |
| `golden_runner` | str | Designated Golden Kernel for this row's op |
| `golden_priority` | int | Ladder priority (0..5) of the chosen golden |
| `correct` | bool | Passes numerical tolerance vs golden output |
| `compile_time_s` | s | Time to generate + compile kernel |

---

## Scoring System

### Correctness Gate (binary)

Every (operator, shape, dtype) must pass correctness against its Golden
Kernel. **No exceptions.** A single correctness failure blocks the entire
benchmark level from passing. Rows tagged
`golden_unavailable_pending_baseline` are not failures themselves — they
flag a *coverage gap* that gate-G7 audits separately.

#### Typed `unsupported` rows (audit-only)

A row with `correctness_status="unsupported"` is exempted from
correctness fail counting **only** when `correctness_reason` matches one
of the recognised typed-decline templates:

| Template (regex, case-insensitive)            | Semantics                                                                 |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `\.get_fn\s+declined\b`                       | Runner refused the (op, shape) at dispatch — typed runner-side decline.   |
| `does not implement\s+run_with_inputs\b`      | Runner has no probe implementation for this op — infra-side decline.      |
| `requires even head_dim`                      | Op-level math guard (e.g. RoPE) — shape mathematically ill-defined.       |
| `mathematically ill-defined`                  | Generic op-level math guard.                                              |
| `no correctness probe for`                    | Harness has no correctness probe for a fused op yet — probe-infra gap.    |

Untyped `unsupported` rows — empty `correctness_reason` or any reason
that does not match a typed template — **remain correctness failures**.
This keeps the silent-decline escape hatch closed: a runner cannot opt
out of correctness checking without writing a machine-readable
justification. Typed-unsupported rows are surfaced in the gate summary
under `typed_unsupported=N` for audit visibility.

The same rule applies to performance scoring: typed-unsupported rows
have no Arke-vs-baseline comparison (empty `perf_pass` / `perf_actual`)
and are excluded from the perf denominator rather than flagged as
malformed. Implementation: `benchmarks/gate_g7.py::_is_typed_unsupported`.

### Performance Score (per shape)

```
ratio = golden_latency / candidate_latency     (>1.0 = candidate faster than golden)
```

The `golden_latency` is the median latency of the runner identified by
`golden_runner` on the same `(op, shape, dtype)`. Per the protocol it is
*always* the same runner that produced the correctness reference — no
divergence between the two roles.

When the golden is `PyTorch-eager` (P3 fallback for ops without a
production kernel), the ratio is informational only and the row is
excluded from gate scoring.

### Aggregation

```
op_score     = geomean(ratio across all shapes for one operator)
tier_score   = geomean(op_scores across all operators in one OT tier)
level_score  = geomean(tier_scores across all OT tiers in one BL level)
arke_score   = 0.3 × L1_level_score + 0.3 × L2_level_score + 0.4 × L3_level_score
```

L3 weighted highest because real-world E2E impact matters most.

### Report Indicators

| Indicator | Meaning |
|:---------:|:--------|
| 🟢 | ratio ≥ 1.0 (Arke ≥ vendor) |
| 🟡 | ratio ≥ 0.8 (within 20%) |
| 🔴 | ratio < 0.8 |

### Exclusion Rules

| Scenario | Handling | Reason |
|:---------|:---------|:-------|
| M ≤ 32 (matmul) | Correctness required; perf excluded from score | Triton ~55μs launch floor |
| N ≤ 32 (softmax) | Correctness required; perf excluded from score | Same |
| M×N ≤ 1024 (elementwise) | Correctness required; perf excluded from score | Kernel-launch dominated |
| OOM shapes | Skip, record "OOM" | Hardware VRAM limit |
| Triton compile timeout (>60s) | Record "TIMEOUT", correctness = fail | Template may need fix |

---

## CLI Interface

### Recommended Invocation Modes

The benchmark stack currently has **two supported entry paths**:

1. `arke bench ...` — the canonical benchmark CLI for BL/OT/ST/layer selection.
2. `python -m benchmarks ...` — module entry point that routes to the same benchmark CLI, plus the `gate` subcommand.

Use them as follows:

| Goal | Recommended command | Notes |
|:-----|:--------------------|:------|
| Run a benchmark suite | `arke bench ...` | Preferred human-facing interface |
| Run from Python/module context | `python -m benchmarks ...` | Equivalent to `arke bench ...` for benchmark runs |
| Run gate verification | `python -m benchmarks gate G6 --tier 2` | Gate runner remains a `python -m benchmarks gate ...` flow |

### Standard Local Workflow

From the repository root:

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate
```

Then use one of the standard benchmark entry patterns below.

### Standard Benchmark Commands

```bash
# Canonical CLI
arke bench
arke bench --bl 5
arke bench --bl 5 --ot 4 --layer L1
arke bench --bl 5 --layer L2

# Module-entry equivalent
python -m benchmarks
python -m benchmarks --bl 5
python -m benchmarks --bl 5 --ot 4 --layer L1
python -m benchmarks --bl 5 --layer L2

# Gate verification (separate entry path)
python -m benchmarks gate G6 --tier 2
python -m benchmarks gate G7 --tier 2
```

### Current Command Mapping Rules

- `python -m benchmarks` is the module wrapper for benchmark runs and is equivalent to `arke bench`.
- `python -m benchmarks gate ...` is **not** an `arke bench` alias; it dispatches to the dedicated gate runner.
- Prefer uppercase layer names in docs and examples: `L1`, `L2`, `L3`.
- Use `arke bench` in user-facing docs unless the context specifically needs the module form or gate runner.

### Design Principle

CLI parameters directly map to the benchmark classification system:

| Parameter | Maps to | Values |
|:----------|:--------|:-------|
| `--bl` | Benchmark Level | `1`–`6` (default: `2`) |
| `--ot` | Operator Tier filter | `0`–`4`, comma-separated |
| `--st` | Shape Tier filter | `1`–`4`, comma-separated |
| `--layer` | Evaluation Layer | `L1`, `L2`, `L3` |
| `--op` | Specific operator(s) | operator name, comma-separated |

**`--bl` is the primary control.** It determines the default OT and ST ranges.
`--ot`, `--st`, `--layer`, `--op` are overrides for fine-grained control.

### Default Behavior

```
arke bench              → BL2 (OT0–OT2 × ST1–ST2, L1 only)
arke bench --bl 5       → BL5 (OT0–OT4 × ST1–ST4, L1+L2)
arke bench --bl 6       → BL6 (Model-Complete, L1+L2+L3)
```

### BL → Default Expansion

| `--bl` | Default OT | Default ST | Default Layer | Description |
|:------:|:-----------|:-----------|:--------------|:------------|
| `1` | OT0–OT2 | ST1 | L1 | Smoke test, <30s |
| `2` | OT0–OT2 | ST1–ST2 | L1 | Daily CI, ~5 min |
| `3` | OT0–OT2 | ST1–ST3 | L1 | Gate validation |
| `4` | OT0–OT4 | ST1–ST2 | L1, L2 | Operator completeness |
| `5` | OT0–OT4 | ST1–ST4 | L1, L2 | Complete suite |
| `6` | Model-Complete | Model-Real | L1, L2, L3 | E2E model validation |

### Examples

```bash
# Quick smoke test (BL1)
arke bench --bl 1

# Daily CI (BL2, default)
arke bench

# Gate validation with full stress shapes
arke bench --bl 3

# All operators, standard shapes
arke bench --bl 4

# Complete benchmark (all ops × all shapes)
arke bench --bl 5

# E2E model validation
arke bench --bl 6
arke bench --bl 6 --model gpt2                     # Specific model
arke bench --bl 6 --model llama2-7b --seq-len 512,2048

# Filter by Operator Tier
arke bench --ot 0                                   # Elementwise only
arke bench --ot 2,4                                 # Dense + Attention only
arke bench --bl 5 --ot 4                            # All shapes, attention only

# Filter by Shape Tier
arke bench --st 4                                   # Production shapes only
arke bench --bl 3 --st 3                            # Stress shapes only

# Filter by Evaluation Layer
arke bench --layer L1                               # Single ops only
arke bench --layer L2                               # Fused ops only
arke bench --layer L3                               # E2E only (implies BL6)

# Filter by specific operator
arke bench --op matmul                              # All shapes for matmul
arke bench --op matmul --st 4                       # matmul production shapes
arke bench --op matmul,softmax --bl 3               # matmul+softmax, stress shapes

# Specific shapes
arke bench --op matmul --shapes square-1k,square-4k

# Baseline control
arke bench --baselines cublas,flaggems,arke         # Only these baselines
arke bench --baselines all                          # All available baselines

# Report & comparison
arke bench report {run_id}                          # Generate report
arke bench diff {run_id_1} {run_id_2}               # Compare two runs
arke bench history --op matmul --shape square-4k    # Performance trend
```

### Validation Rules

- `--layer L3` automatically sets `--bl 6` (L3 ≡ BL6)
- `--layer L2` requires `--bl ≥ 4` (L2 needs OT3+)
- `--ot 4` requires `--st 4` (attention ops only have ST4 shapes)
- `--bl 6 --op matmul` is valid (runs only matmul shapes from the model graph)

### Module Entry Equivalents

`python -m benchmarks` routes to the same benchmark semantics as `arke bench`, while `python -m benchmarks gate ...` dispatches the dedicated Gate runner.

```bash
python -m benchmarks --all                → arke bench --bl 6
python -m benchmarks --layer L1           → arke bench --layer L1 --bl 2
python -m benchmarks --op matmul          → arke bench --op matmul
python -m benchmarks --op matmul --tier 2 → arke bench --op matmul --st 2
python -m benchmarks --report             → arke bench report latest
```

---

## Output Structure & Provenance Tracking

### Directory Layout

```
benchmarks/results/{run_id}/
├── config.json              # Run configuration (bl, ot, st, layer)
├── hardware.json            # GPU, driver, CUDA, PyTorch/Triton versions
├── L1/
│   ├── OT0/                 # Elementwise results
│   │   ├── perf_relu.csv
│   │   ├── perf_gelu.csv
│   │   └── ...
│   ├── OT1/                 # Reduction results
│   ├── OT2/                 # Compute-dense results
│   ├── OT3/                 # Gated activation results
│   └── OT4/                 # Attention results
├── L2/
│   ├── perf_matmul_relu.csv
│   └── ...
├── L3/
│   └── {model}/
│       ├── perf_e2e.csv
│       └── config.json      # Model, seq_len, patches
├── summary.json             # Aggregated scores by BL/OT/ST
├── PERF_ALL.csv             # All rows in unified CSV v2.0 schema
└── report.md                # Human-readable report
```

### Provenance Tracking

Every result carries full source attribution:
- **CSV schema** — unified 41-column format ([`benchmark-csv-spec.md`](./benchmark-csv-spec.md))
- **`config.json`** — run parameters: bl, ot, st, layer, baselines
- **`hardware.json`** — GPU name, CUDA version, driver, framework versions

---

## Benchmark-Driven Development

The benchmark is the **target state definition** for Arke development.

### Capability Mapping

> **Legend:** ✅ = done, 🔶 = IR defined but no codegen template, ⬜ = not started

| Benchmark Target | Primary Baseline | IR | Template | Codegen | Strategy |
|:-----------------|:-----------------|:--:|:--------:|:-------:|:--------:|
| **OT0 Elementwise** | | | | | |
| L1 relu | PyTorch `F.relu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 gelu | PyTorch `F.gelu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 silu | PyTorch `F.silu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 add | PyTorch `torch.add` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 mul | PyTorch `torch.mul` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| **OT1 Reduction** | | | | | |
| L1 softmax | cuDNN/PyTorch (P0/P3) | ✅ | ✅ `softmax.py.j2` | ✅ | rows_per_prog ✅ |
| L1 layernorm | cuDNN/PyTorch (P0/P3) | ✅ | ✅ `layernorm.py.j2` | ✅ | block_size ✅ |
| L1 rmsnorm | FlagGems (P1) | ✅ | ✅ `layernorm.py.j2` | ✅ | block_size ✅ |
| L1 rmsnorm_residual | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 reduce_sum | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| L1 reduce_max | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| **OT2 Compute-Dense** | | | | | |
| L1 matmul ≥ P0 | cuBLAS (P0) | ✅ | ✅ `matmul.py.j2` | ✅ | tile, split-k, swizzle ✅ |
| L1 batch_matmul ≥ P0 | cuBLAS (P0) | ✅ | ✅ `matmul.py.j2` | ✅ | batch dim ✅ |
| L1 grouped_matmul | CUTLASS (P0) | ✅ | 🔶 | ❌ | ⬜ |
| L1 transpose | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| **OT3 Gated Activation** | | | | | |
| L1 swiglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 geglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| **OT4 Attention** | | | | | |
| L1 flash_attention ≥ P1 | FlashAttention (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 grouped_query_attention | FlashAttention (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 multi_latent_attention | DeepSeek ref | ✅ | 🔶 | ❌ | ⬜ |
| **L2 Fused** | | | | | |
| L2 matmul+gelu ≥ P1 | FlagGems fusion (P1) | ✅ | ✅ epilogue | ✅ | fusion decision ✅ |
| L2 matmul+relu ≥ P1 | FlagGems fusion (P1) | ✅ | ✅ epilogue | ✅ | fusion decision ✅ |
| L2 swiglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| **L3/BL6 E2E** | | | | | |
| GPT-2 ≤ eager | E2E eager | ✅ | ✅ | ✅ | KernelCache ✅ |
| LLaMA-2 7B ≤ eager | E2E eager | partial | 🔶 | ❌ | ⬜ |
| DeepSeek-V2 ≤ eager | E2E eager | partial | 🔶 | ❌ | ⬜ |

**Summary:** 11/20 operators have working codegen (Triton template). 9 operators
(reduce_sum/max, transpose, rmsnorm_residual, grouped_matmul, swiglu, geglu,
flash_attention, GQA, MLA) have IR + numerical validation but no Triton
template yet — this is the primary Phase 2 codegen gap.

---

## Implementation Status

### Runner Implementations

| Runner | Tier | Status | Ops Supported |
|:-------|:----:|:------:|:------|
| `CuBLASRunner` | P0 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `FlagGemsRunner` | P1 | ✅ | matmul, softmax, layernorm, rmsnorm, gelu, relu, silu |
| `LigerRunner` | P1 | ✅ | rmsnorm, gelu, silu, rope |
| `FlashAttnRunner` | P1 | ⬜ | flash_attention (planned) |
| `TritonTutorialRunner` | P2 | ✅ | matmul, softmax |
| `PyTorchEagerRunner` | P3 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `InductorRunner` | P4 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `LLMDirectRunner` | P5 | ⬜ | (planned: all ops via LLM codegen) |
| `ArkeRunner` | — | ✅ | matmul, softmax |

### Benchmark Components

| Component | Status | Description |
|:----------|:------:|:------------|
| `baselines/` | ✅ | BaselineRunner ABC + 8 runner classes |
| `shapes.py` | ✅ | Shape registry with ST1–ST4 tagging |
| `perf_csv.py` | ✅ | PerfRow + PerfCSVWriter (CSV v2.0, 41 columns) |
| `measure.py` | ✅ | CUDA event timing |
| `bench_l1.py` | ✅ | L1 single operator benchmarks |
| `bench_l2.py` | ✅ | L2 fused operator benchmarks |
| `bench_l3.py` | ✅ | L3 E2E model benchmarks (GPT-2) |
| `gate.py` | ✅ | Gate verification CLI |
| `cli.py` | ✅ | Unified CLI entry point |
| `op_registry.py` | ✅ | Parses benchmark-ops.md → OT_OPS / OP_TIER / ALL_OPS (single source of truth) |
| `report.py` | ✅ | Markdown report generator |
| Hardware info | ✅ | `hardware.json` per run |
| Provenance | ✅ | CSV source column + per-run manifest |
| BL/OT/ST CLI | ✅ | `arke bench --bl/--ot/--st/--layer` |
| Op catalog auto-sync | ✅ | `tests/conftest.py` + `scripts/sync_ops.py` detect md changes on every pytest run |
| `shape_registry.py` | ✅ | Parses benchmark-shapes.md → SHAPE_TABLES / SHAPES_BY_TIER / ALL_SHAPE_TAGS (single source of truth) |
| Shape catalog auto-sync | ✅ | `tests/conftest.py` + `scripts/sync_shapes.py` detect shape changes via SHA-256 tags hash |
| Cross-run diff | ⬜ | `arke bench diff` |
| CI integration | ⬜ | GitHub Actions regression mode |

---

## Dependencies

### Required

```bash
pip install torch triton
```

### Benchmark Baselines

```bash
pip install flag-gems        # FlagGems — 200+ Triton ops (P1)
pip install liger-kernel     # Liger — LLM training kernels (P1)
```

### Optional

```bash
pip install flash-attn --no-build-isolation   # FlashAttention (P1)
pip install triton-kernels                     # HuggingFace community kernels
pip install nvidia-cutlass                     # CUTLASS C++ GEMM baselines
```

### Graceful Degradation

Missing baseline packages are skipped with a warning; results show `N/A`.

---

## Resume / Incremental Persistence

Long L1/L2 runs persist every measurement to disk as it completes, so a
crash, OOM-kill, or terminal session loss never wastes more than the
in-flight test point.

### Behaviour

* `bench_l1` / `bench_l2` append each `(op, shape_tag, baseline|approach)`
  row to the canonical `<op>_results.csv` immediately and `fsync` it.
* On re-launch, existing rows are scanned. The default `--retry-policy auto`
  policy:
  * **skips** rows with `status=ok` (already passing)
  * **skips** rows with `status ∈ {oom, skipped, unsupported, incompatible}`
    (a known limitation already recorded)
  * **retries** rows with `status ∈ {error, timeout}` (likely transient)
* Other policies: `--retry-policy none` (skip everything that has any row)
  and `--retry-policy all` (rerun every non-`ok` row).
* A `progress.jsonl` event log under the layer directory captures every
  measurement, op start/finish, and resume skip count.
* A `status.json` snapshot is written at run end, plus a `.bench.lock`
  PID file while a process holds the directory.

### Configuration drift guard

The layer directory's `config.json` carries a fingerprint over
`ops + shape_tags + tier + warmup + reps + phase + stage + track + layer`.
A resume aborts with a clear error when the fingerprint changes; pass
`--force-restart` to override (which also breaks live locks).

### Output path normalization

`--output` accepts either the bare results root
(`benchmarks/results`) or any prefix that already contains
`phase{N}/stage{N}/track{N}[/{layer}]`. The runner strips the duplicate
suffix so artifacts always land at
`<root>/phase{N}/stage{N}/track{N}/{layer}/`. This eliminates the
former `track{N}/phase{N}/stage{N}/track{N}/{layer}/` nested directory
bug that hid in-progress data from the gate / dashboard tooling.

### Inspecting progress

```bash
python -m benchmarks status \
    --output benchmarks/results \
    --phase 1 --stage 7 --track 6 --layer l2 --recent 10
```

Reports per-op `rows / ok / permanent / retryable` counts, lock
liveness, fingerprint, and the most recent progress events. Add
`--json` for machine-readable output.

---

*Last updated: 2026-05-10*
