# Arke Benchmark System

Comprehensive GPU operator benchmark framework with multi-tier baselines,
automated provenance tracking, and a three-layer evaluation architecture.

---

## Table of Contents

1. [Overview & Benchmark Level System](#1-overview--benchmark-level-system)
2. [Operator Tier (OT)](#2-operator-tier-ot)
3. [Shape Tier (ST)](#3-shape-tier-st)
4. [Evaluation Layers (L1/L2/L3)](#4-evaluation-layers-l1l2l3)
5. [Baselines](#5-baselines)
6. [Measurement Protocol](./benchmark-protocol.md#measurement-protocol)
7. [Scoring System](./benchmark-protocol.md#scoring-system)
8. [Output & CSV Spec](./benchmark-csv-spec.md)
9. [Operator Source Registry](./operator-source-registry.md)
10. [CLI Interface](./benchmark-protocol.md#cli-interface)
11. [Benchmark-Driven Development](./benchmark-protocol.md#benchmark-driven-development)
12. [Implementation Status](./benchmark-protocol.md#implementation-status)
13. [Dependencies](./benchmark-protocol.md#dependencies)
14. [Op Catalog — Single Source of Truth](#op-catalog--single-source-of-truth)
15. [Shape Catalog — Single Source of Truth](#shape-catalog--single-source-of-truth)

**Sub-documents** (all in [`benchmark/`](./)):
- [`benchmark-ops.md`](./benchmark-ops.md) — Op Tier definitions, operator catalog, per-op baseline selection
- [`benchmark-shapes.md`](./benchmark-shapes.md) — Shape Tier definitions, full shape matrices per operator
- [`benchmark-protocol.md`](./benchmark-protocol.md) — Measurement protocol, scoring, CLI, implementation status
- [`benchmark-csv-spec.md`](./benchmark-csv-spec.md) — Unified CSV output schema (39 columns, Excel-friendly)
- [`operator-source-registry.md`](./operator-source-registry.md) — Complete baseline source catalog (14 sources)

---

## 1. Overview & Benchmark Level System

### Benchmark Level (BL)

A **Benchmark Level** is a specific combination of **Operator Tier** (OT) and **Shape Tier** (ST).
Higher levels cover more complex operators and/or larger/harder shapes.

```
Benchmark Level = Operator Tier (OT) × Shape Tier (ST)
```

| Level | Operator Coverage | Shape Coverage | Description | Typical Use |
|:-----:|:-----------------|:--------------|:------------|:------------|
| **BL1** | OT0–OT2 | ST1 | Basic ops × micro shapes | Quick smoke test, <30s |
| **BL2** | OT0–OT2 | ST1–ST2 | Basic ops × standard shapes | Daily CI, ~5 min |
| **BL3** | OT0–OT2 | ST1–ST3 | Basic ops × full shapes (incl. non-aligned) | Gate validation |
| **BL4** | OT0–OT4 | ST1–ST2 | **All ops** × standard shapes | Operator completeness |
| **BL5** | OT0–OT4 | ST1–ST4 | **All ops × all shapes** | Complete benchmark suite |
| **BL6** | Model-Complete | Model-Real | True model graph: all ops + exact production shapes | E2E model validation |

**BL6 (Model-Complete)** is special: instead of a Cartesian product of OT×ST, it captures the
exact operator+shape combinations that appear in a real model's forward pass:

| Model | Operators (from compute graph) | Shape Source |
|:------|:------------------------------|:-------------|
| GPT-2 Small | matmul, layernorm, gelu, softmax, add, transpose | Model params × seq ∈ {128, 512, 1024} |
| LLaMA-2 7B | matmul, rmsnorm, silu_and_mul, flash_attention, rope, transpose | Model params × seq ∈ {512, 2048, 4096} |
| DeepSeek-V2 | matmul, grouped_matmul, rmsnorm, silu_and_mul, multi_latent_attention, grouped_query_attention, rope | Model params × seq ∈ {512, 2048, 8192} |

> BL6 catches correctness and performance issues that BL5 misses because real models use
> specific dimension combinations (e.g. LLaMA's head_dim=128 with n_kv_heads=8) that may
> not be covered by the general shape matrix.

### OT × ST Coverage Matrix

```
              ST1(micro)  ST2(standard)  ST3(stress)  ST4(production)
OT0 (elem)      BL1          BL2            BL3            ─
OT1 (reduce)    BL1          BL2            BL3            ─
OT2 (dense)     BL1          BL2            BL3           BL5
OT3 (gated)     BL4          BL4            ─             BL5
OT4 (attn)       ─            ─             ─             BL5
Model-Complete  ──────────────── BL6 ────────────────────────
```

### Layer × BL Coverage Matrix

Benchmark Level (BL) defines **coverage scope**; Evaluation Layer (L) defines **what is measured**.
They are orthogonal — a benchmark run specifies both (e.g. "L1 at BL3").

```
         BL1  BL2  BL3  BL4  BL5  BL6
L1        ✓    ✓    ✓    ✓    ✓    ─     Single operator perf
L2        ─    ─    ─    ✓    ✓    ─     Fused operator perf
L3 ≡ BL6  ─    ─    ─    ─    ─    ✓     E2E model perf
```

> **L3 ≡ BL6**: L3 is the execution of BL6's model-complete operator+shape set as an
> end-to-end forward pass. They are the same concept viewed from two angles —
> BL6 defines the coverage, L3 defines the evaluation method.

---

## 2. Operator Tier (OT)

Operators are classified by **computational complexity and kernel design difficulty**.

> Full details: [`benchmark-ops.md`](./benchmark-ops.md)

| Tier | Name | Operators | Core Characteristic |
|:----:|:-----|:----------|:--------------------|
| **OT0** | Elementwise | `relu`, `gelu`, `silu`, `add`, `mul` | No reduction, pure memory-bound |
| **OT1** | Reduction | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max` | Row-wise reduction, warp-level cooperation |
| **OT2** | Compute-Dense | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose` | Matrix multiply, tensor core tiling, shared memory staging |
| **OT3** | Gated Activation | `silu_and_mul`, `gelu_and_mul` | Split + nonlinear + elementwise mul; output dim = input/2 |
| **OT4** | Attention | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` | Multi-stage fused kernel, online softmax, causal mask, KV compression |

**Design rationale:**
- OT0→OT1: Adds reduction (shared memory, warp shuffle)
- OT1→OT2: Shifts from memory-bound to compute-bound (tensor cores, tiling strategy)
- OT2→OT3: Composite semantics — split + nonlinearity + multiply, output shape changes
- OT3→OT4: Multi-stage fusion (QKᵀ → online-softmax → AV), most complex tiling and memory management

---

## 3. Shape Tier (ST)

Shapes are classified by **scale, alignment, and production relevance**.

> Full shape matrices: [`benchmark-shapes.md`](./benchmark-shapes.md)

| Tier | Name | Description | Count (approx) |
|:----:|:-----|:------------|:--------------:|
| **ST1** | Micro | Small, power-of-2 aligned shapes; launch-overhead dominated | ~15 shapes/op |
| **ST2** | Standard | Medium scale + typical LLM production shapes (GPT-2, LLaMA) | ~30 shapes/op |
| **ST3** | Stress | Non-power-of-2, off-by-one, extreme aspect ratios; full alignment coverage | ~50 shapes/op |
| **ST4** | Production | LLM production shapes at scale (DeepSeek-V2/V3, LLaMA-3, Qwen2.5); long-context variants | per-op, see shapes doc |

**ST4 applies only to OT2–OT4** (compute-dense and attention ops), since elementwise/reduction
operators at production scale are subsumed by ST3.

---

## 4. Evaluation Layers (L1/L2/L3)

Benchmark Level (BL) defines the **coverage scope** (how many ops × shapes).
Evaluation Layers define the **measurement dimension** (what question is answered).

| Layer | Question | What is Measured | BL Range | Metric |
|:-----:|:---------|:-----------------|:---------|:-------|
| **L1** | Can Arke match hand-tuned single-op kernels? | Individual operator perf | BL1–BL5 | μs latency, TFLOPS/GB·s, % of baseline |
| **L2** | Can Arke fuse operators as well as expert code? | Fused operator perf | BL4–BL5 | TFLOPS, vs best fused baseline |
| **L3** | Does it make a real model faster end-to-end? | Full model forward pass | **BL6** | Wall-clock latency, tok/s, peak memory |

> **L3 ≡ BL6**: L3 executes BL6's model-complete operator+shape set as an end-to-end
> forward pass. BL6 defines the coverage; L3 defines the evaluation method.

### L1: Single Operators

Benchmarks each of the 20 OP_CATALOG operators across the shape matrix.
Reports μs latency, TFLOPS (compute-bound ops), GB/s (memory-bound ops),
and ratio vs each available baseline tier.

### L2: Fused Operators

Compares three approaches:
1. **Separate ops** — sequential unfused kernels (baseline)
2. **torch.compile** — Inductor auto-fusion
3. **FlagGems / Liger** — expert Triton fusion

Currently supports: `matmul+relu`, `matmul+gelu`, `silu_and_mul`, `gelu_and_mul`

### L3: E2E Model (= BL6)

Full model forward pass benchmarks using BL6 model-complete shape sets:

| Mode | Description |
|:-----|:------------|
| **Eager** | PyTorch default (SDPA attention) |
| **torch.compile** | Inductor-optimized |
| **Arke** | Arke kernels via KernelCache patch |

Reports: latency (mean/min/max/median), correctness (logit diff, top-1 match), peak memory.

---

## 5. Baselines

### Baseline Priority Tiers

Each operator is benchmarked against multiple baseline tiers, ranked by expected performance:

| Tier | Name | Source | License |
|:----:|:-----|:-------|:--------|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS | NVIDIA EULA / BSD-3 |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention | Apache-2.0 / BSD-2 / BSD-3 |
| **P2** | Reference Triton | Triton official tutorials | MIT |
| **P3** | PyTorch eager | `torch.nn.functional` | BSD-3-Clause |
| **P4** | Inductor-generated | `torch.compile` output | BSD-3-Clause |
| **P5** | LLM-direct | LLM writes Triton directly | — |

### Where to find details

| What | Where |
|:-----|:------|
| **Per-operator baseline selection** (primary + expert baselines) | [`benchmark-ops.md`](./benchmark-ops.md) — each op card lists its P0–P5 mapping |
| **Source installation, API, version, full op lists** | [`operator-source-registry.md`](./operator-source-registry.md) — complete catalog of all 14 baseline sources |
| **Fused operator baselines** | [`benchmark-ops.md` §OT3/OT4](./benchmark-ops.md) + table below |

### Fused Operator → Baseline Summary (L2)

| Fused Op | P0 Vendor | P1 Expert | P3 PyTorch | P5 LLM-direct |
|:---------|:----------|:----------|:-----------|:--------------|
| matmul + relu | — | FlagGems (ATen fusion) | `torch.compile` | ✓ |
| matmul + gelu | — | FlagGems | `torch.compile` | ✓ |
| linear + cross_entropy | — | Liger `fused_linear_ce` | separate ops | ✓ |
| QKV + attention | cuDNN SDPA | FlashAttention | `F.scaled_dot_product_attention` | ✓ |
| silu_and_mul | — | Liger `silu_and_mul` | manual impl | ✓ |
| gelu_and_mul | — | Liger `gelu_and_mul` | manual impl | ✓ |

---

## 6. Standard Benchmark Usage

### Recommended entry points

The benchmark system is documented around `arke bench`, but the codebase currently supports two practical entry paths:

- `arke bench ...` — canonical benchmark CLI.
- `python -m benchmarks ...` — module wrapper for the same benchmark CLI.

Gate verification remains separate:

- `python -m benchmarks gate ...`

### Standard local invocation

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate
```

### Common commands

```bash
# Default daily benchmark selection (BL2, L1)
arke bench

# Full BL5 coverage
arke bench --bl 5

# OT / layer scoped runs
arke bench --bl 5 --ot 0 --layer L1
arke bench --bl 5 --ot 4 --layer L1
arke bench --bl 5 --layer L2

# Module-entry form
python -m benchmarks --bl 5 --ot 4 --layer L1

# Gate checks
python -m benchmarks gate G0 --tier 2
python -m benchmarks gate G6 --tier 2
python -m benchmarks gate G7 --tier 2
```

### Documentation convention

- Use `arke bench` in benchmark design/spec examples by default.
- Use `python -m benchmarks` when documenting module execution behavior.
- Use `python -m benchmarks gate ...` for Gate verification examples.

For the full CLI contract and option semantics, see [`benchmark-protocol.md`](./benchmark-protocol.md#cli-interface).

---

*For full shape matrices, see [`benchmark-shapes.md`](./benchmark-shapes.md).*
*For baseline source installation and API details, see [`operator-source-registry.md`](./operator-source-registry.md).*
*For measurement protocol, scoring, and CLI, see [`benchmark-protocol.md`](./benchmark-protocol.md).*

---

## Op Catalog — Single Source of Truth

`benchmark-ops.md` is the **authoritative definition** of all operators and their OT tier.
All code consumes it through `benchmarks/op_registry.py` — no hardcoded op lists anywhere else.

### Data Flow

```
benchmark-ops.md  ←── edit here to add/remove/move operators
       │
       ▼
benchmarks/op_registry.py     ← parses OT Summary table at import time
       │                         exports: OT_OPS, OP_TIER, ALL_OPS, TOTAL_OPS
       │
       ├──▶ benchmarks/cli.py        (OT_OPS — BL expansion & op filtering)
       └──▶ benchmarks/shapes.py     (OP_TIER — op→tier mapping)

tests/conftest.py              ← pytest hook, runs on every test session
       │
       ├─ snapshot unchanged → silent pass (zero overhead)
       └─ snapshot changed   → diff report + scripts/sync_ops.py
                                └─ validates registry + checks shape coverage
                                └─ updates .benchmark_ops_snapshot.json
```

### Auto-Sync Behavior

| Scenario | Result |
|:---------|:-------|
| `benchmark-ops.md` unchanged | conftest silent; tests run normally |
| Operator added / removed | conftest prints `Added`/`Removed` diff, runs sync, updates snapshot |
| Operator moved between tiers | conftest prints `Moved: op(OT0→OT1)`, runs sync |
| Count declared in md ≠ parsed count | `op_registry` raises `ValueError`; pytest exits with error |
| New op has no shape route yet | `sync_ops.py` prints `WARNING`; tests continue (fallback shape used) |
| `arke` not on PATH | Tests use `sys.executable -m arke.cli bench` — always portable |

### Key Files

| File | Role |
|:-----|:-----|
| `docs/benchmark/benchmark-ops.md` | **Edit this** — OT Summary table is the catalog |
| `benchmarks/op_registry.py` | Parser + module-level singletons (`OT_OPS`, `OP_TIER`, `ALL_OPS`) |
| `benchmarks/cli.py` | Imports `OT_OPS` from `op_registry` |
| `benchmarks/shapes.py` | Imports `OP_TIER` from `op_registry`; routes all 45 ops to shape sets |
| `tests/conftest.py` | `pytest_configure` hook — detects catalog changes before any test runs |
| `scripts/sync_ops.py` | Validates registry + shape coverage; called by conftest |
| `.benchmark_ops_snapshot.json` | Persisted OT catalog hash — change detection baseline |

### How to Add a New Operator

1. Edit `benchmark-ops.md` → add the op to the OT Summary table
   - Increment the **Count** column for that tier
   - Add `` `op_name` `` to the Operators cell
   - Add a full operator spec section in the tier's body
2. Run `pytest tests/test_benchmark_protocol.py` (or any pytest)
   - conftest detects the count change, runs `sync_ops.py`
   - If the op has no shape route: a WARNING is printed
3. Add a shape table in `benchmark-shapes.md` (or map to an existing table)
4. Add a shape route in `benchmarks/shapes.py` `get_shapes()` if needed
5. All other files (`cli.py`, tests) pick up the new op automatically

---

## Shape Catalog — Single Source of Truth

`benchmark-shapes.md` is the **authoritative definition** of all benchmark shapes and their ST tier.
All code consumes it through `benchmarks/shape_registry.py` — shapes are parsed from markdown at import time.

### Data Flow

```
benchmark-shapes.md  ←── edit here to add/remove/modify shapes
       │
       ▼
benchmarks/shape_registry.py   ← parses all markdown tables at import time
       │                          exports: SHAPE_TABLES, ALL_SHAPE_TAGS,
       │                                   TOTAL_SHAPES, SHAPES_BY_TIER
       │
       └──▶ benchmarks/shapes.py   get_shapes(op) prefers registry data;
                                   _dict_to_shape() converts rows → dataclass
                                   hardcoded dataclasses remain as fallback

tests/conftest.py               ← pytest hook (after op catalog check)
       │
       ├─ tags_hash unchanged → silent pass (zero overhead)
       └─ tags_hash changed   → diff report + scripts/sync_shapes.py
                                 └─ validates registry + checks op coverage
                                 └─ updates .benchmark_shapes_snapshot.json
```

### Auto-Sync Behavior

| Scenario | Result |
|:---------|:-------|
| `benchmark-shapes.md` unchanged | conftest silent; tests run normally |
| Shape added / removed | conftest prints `Added`/`Removed` tag diff, runs sync, updates snapshot |
| Shape table modified (new rows) | tags_hash changes; sync validates and updates snapshot |
| Markdown format error | `shape_registry` logs warning; fallback to hardcoded shapes |
| New op has no shape table yet | `sync_shapes.py` prints `WARNING`; get_shapes() falls back to hardcoded route |
| `shape_registry.py` import fails | `_REGISTRY_AVAILABLE = False`; all shapes served from hardcoded dataclasses |

### Key Files

| File | Role |
|:-----|:-----|
| `docs/benchmark/benchmark-shapes.md` | **Edit this** — shape tables are the catalog (30 tables, ~358 shapes) |
| `benchmarks/shape_registry.py` | Parser + module-level singletons (`SHAPE_TABLES`, `SHAPES_BY_TIER`, `ALL_SHAPE_TAGS`) |
| `benchmarks/shapes.py` | `get_shapes()` prefers registry → fallback to hardcoded; `_dict_to_shape()` converter |
| `tests/conftest.py` | `pytest_configure` hook — detects shape catalog changes (tags_hash via SHA-256) |
| `scripts/sync_shapes.py` | Validates registry + per-op coverage; called by conftest |
| `.benchmark_shapes_snapshot.json` | Persisted tags hash + per-table row counts — change detection baseline |

### How to Add a New Shape

1. Edit `benchmark-shapes.md` → add rows to an existing table (or create a new `###` section)
   - Include Tag, dimensions, Tier (1–4), Source, and Notes columns
   - Follow ST1 = power-of-2, ST2 = production, ST3 = non-aligned/stress, ST4 = large-scale
2. Run `pytest` (any test file)
   - conftest detects the tags_hash change, runs `sync_shapes.py`
   - sync validates and updates the snapshot
3. `get_shapes()` automatically picks up the new shape on next import
4. No code changes needed unless a completely new table requires a new routing entry

### Combined Architecture

```
                 ┌─────────────────┐     ┌──────────────────────┐
                 │ benchmark-ops.md│     │ benchmark-shapes.md  │
                 │   (45 operators)│     │ (30 tables, 358 shapes)│
                 └────────┬────────┘     └──────────┬───────────┘
                          │                         │
                          ▼                         ▼
                 op_registry.py            shape_registry.py
                 OT_OPS, OP_TIER           SHAPE_TABLES, SHAPES_BY_TIER
                          │                         │
              ┌───────────┼──────────┐              │
              ▼           ▼          ▼              ▼
          cli.py     shapes.py   tests     shapes.py (get_shapes)
         (OT_OPS)   (OP_TIER)  (ALL_OPS)  (registry → fallback)
                          │
                  ┌───────┴───────┐
                  ▼               ▼
          conftest.py        conftest.py
      (ops snapshot)      (shapes snapshot)
              │                   │
              ▼                   ▼
        sync_ops.py         sync_shapes.py
```

---

*Last updated: 2026-04-05*
