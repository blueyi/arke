---
name: swiglu-packed-fusion
description: Onboard and optimize the OT3 fused operator `swiglu_packed` (D8-X1 demo op). Use when the user asks to optimize, benchmark, or generate kernels for SwiGLU packed FFN projection — i.e., the fused chain `split(X, 2, dim=-1) → silu(gate) * up → @ W`. Triggers on "swiglu_packed", "swiglu fused", "SwiGLU FFN down projection", "D8-X1 demo op", "OT3 swiglu", "fused gated projection".
inputs:
  - shape_tier   # ST1 | ST2 | ST3 (default ST1)
  - dtype        # fp16 | bf16 | fp32 (default fp16)
budgets:
  decisions: 60
  compiles: 20
references:
  - benchmarks/op_registry.py
  - benchmarks/bench_l1.py
  - benchmarks/baselines/pytorch_eager.py
  - arke/ir/ops/catalog.py
  - arke/ir/ops/reference_impls.py
  - docs/benchmark/golden-kernel-ladder.md
  - docs/benchmark/audit/swiglu_packed_baseline_audit_2026-05-30.md
---

# SwiGLU-Packed Fusion (OT3, D8-X1)

Optimize the **fused gated projection** operator that combines a SwiGLU
non-linearity with the FFN down-projection matmul in one kernel. This is
the D8-X1 extensibility-demo operator for Stage 8 Tier 1.

## When this skill applies

The kernel under optimization is `swiglu_packed`:

```
gate, up = split(X, 2, dim=-1)     # X: [M, 2K]  →  gate, up : [M, K]
H        = silu(gate) * up         # H: [M, K]
Y        = H @ W                   # W: [K, N], Y: [M, N]
```

In Arke catalog terms (`arke/ir/ops/catalog.py::SWIGLU_PACKED`):

- `category="gated"`, `template_hint.template_name="gated_activation"`,
  `extra_ctx={"op_variant":"swiglu_packed"}`.
- Inputs `{"X": Tensor[M, 2K], "W": Tensor[K, N]}` with the constraint
  `X.shape[-1] % 2 == 0 and W.shape[0] == X.shape[-1] / 2`.
- Output `Tensor[M, N]`.

This is **distinct from** the OT3 `silu_and_mul` op (payload-only,
no matmul). Do not collapse them.

## Baseline ladder (audit-degraded)

Per `docs/benchmark/golden-kernel-ladder.md` OT3 row 8 and the audit
under `docs/benchmark/audit/swiglu_packed_baseline_audit_2026-05-30.md`:

| Slot     | Runner                | Notes                                                |
|:---------|:----------------------|:-----------------------------------------------------|
| Golden   | `PyTorch-eager` (P3)  | Audit-degraded — no fallback                         |
| Fallback | —                     | None: 9 audited libraries (FlagGems · Liger · vLLM · flash-attn · FlashMLA · xformers · TransformerEngine · Megatron-LM · FlashInfer · NVIDIA Apex · DeepSpeed) all decompose into separate `silu_and_mul` + `matmul` kernels |

This means `ratio_vs_baseline` here compares against an **unfused**
PyTorch eager decomposition. Beating it is the *purpose* of this demo
op — a true fused single-kernel implementation should comfortably win
on memory-bandwidth-bound shapes.

## Shapes

Defined in `benchmarks/shapes.py` under the matmul-class grid (see also
`benchmarks/shape_registry.py`). swiglu_packed reuses matmul shape
semantics: `(M, K_packed=2K, N)` triplets. Pick by tier:

```bash
# Smoke: one small shape
python -m benchmarks.bench_l1 --op swiglu_packed --shapes <one_tag> --warmup 3 --reps 20

# Tier 1 (full shape grid for this op)
python -m benchmarks.bench_l1 --op swiglu_packed --tier 1

# Tier 2 (extended, ~3× shapes)
python -m benchmarks.bench_l1 --op swiglu_packed --tier 2
```

## Optimization procedure

1. **Analyze.** Call `analyze_compute()` on the SemanticIR. Confirm
   `op = swiglu_packed`, `category = gated`, K is the contraction dim,
   N is the output dim, M is batch.
2. **Identify the regime.** Compute arithmetic intensity ≈ `2·M·K·N /
   (M·2K + K·N + M·N) · sizeof(dtype)`. For ST1 typical shapes
   (`M ≤ 4096`, `K = N = 4096`) it's compute-bound; for small `K` or
   `N < 1024` it can be bandwidth-bound. Decide block tile sizes
   accordingly.
3. **Decide fusion strategy.** Two viable structures:
   - **Tile-and-fuse** (preferred): compute `H[block] = silu(gate)*up`
     in registers, then accumulate `Y[block] += H[block] @ W[block]`
     in the same kernel. One pass over `X`, no intermediate writeback.
   - **Two-pass fallback**: write `H` to global, then call a matmul
     kernel. Use only if register pressure or shared-memory budget
     blows up — record the `@rationale`.
4. **Pick block tile.** Use `list_legal_actions()`; typical winners on
   Ampere SM 8.6 (6 GB VRAM):
   - `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32` for fp16 with K=4096.
   - Raise `BLOCK_K` only when shared memory has slack (check via the
     hardware profile).
5. **Set num_warps / num_stages.** Start `num_warps=4, num_stages=3`.
   Promote `num_stages=4` only on K ≥ 8192 with software pipelining
   headroom.
6. **Verify (V1).** Call `verify_correctness()` against
   `ref_swiglu_packed` (`arke/ir/ops/reference_impls.py::ref_swiglu_packed`).
   Tolerances per `docs/benchmark/benchmark-design.md`:
   `fp16: atol=0.1, rtol=0.05`; `bf16: atol=0.2, rtol=0.1`;
   `fp32: atol=1e-5, rtol=1e-4`.
7. **Profile (V2).** Call `compile_and_profile()` against the Golden
   (PyTorch-eager). Target: `ratio_vs_baseline ≥ 1.20×` on ST1
   (low bar — the baseline is unfused, audit-degraded).
8. **Generalize.** Re-run across the tier's shape set. If geomean
   speedup < 1.10×, surface a bottleneck-shape recommendation in the
   trajectory (`OnSessionEnd` hook is enough).

## @rationale checklist

Every `apply_decision` for swiglu_packed should explain at least one of:

- **Why fused vs two-pass.** Cite arithmetic-intensity numbers or
  observed register pressure.
- **Why this block tile.** Tie back to `K`, `N`, dtype, and SM
  shared-memory budget.
- **Why num_warps/num_stages.** Tie back to the pipeline depth
  achievable for the chosen `BLOCK_K`.
- **What you would change if `N` doubles** (the most common shape
  perturbation). Demonstrates generalization.

## Anti-patterns (don't do)

- ❌ Lowering to `silu_and_mul` then `matmul` *as the optimization
  target*. That's the baseline. The whole point of the op is to beat
  the unfused chain.
- ❌ Pinning `--golden swiglu_packed=Liger-Kernel`. Liger has no
  single-kernel implementation (audit). The CLI will fire
  `GoldenUnavailable`.
- ❌ Hardcoding shapes inside the kernel. Use the shape grid; if a
  shape isn't supported, register it in `benchmarks/shape_registry.py`
  rather than special-casing the kernel.
- ❌ Skipping V1 verify on a new tile choice. SwiGLU's `silu` interacts
  non-linearly with the matmul accumulator; a wrong layout silently
  produces near-correct but biased output that passes loose tolerances
  by accident.

## Acceptance for D8-X1

This skill is the SKILL.md deliverable for Stage 8 Track 0 D8-X1
("Extensibility Demo A — new operator onboarding"). The companion
deliverables already on disk:

- ✅ Catalog registration (`arke/ir/ops/catalog.py::SWIGLU_PACKED`).
- ✅ Reference impl (`arke/ir/ops/reference_impls.py::ref_swiglu_packed`).
- ✅ Baseline runner support (`benchmarks/baselines/pytorch_eager.py`).
- ✅ Shape mapping (`benchmarks/shapes.py` + `shape_registry.py`).
- ✅ SSOT entry (`docs/benchmark/benchmark-ops.md § swiglu_packed`).
- ✅ Ladder row (`docs/benchmark/golden-kernel-ladder.md` OT3 #8).
- ✅ Audit doc (`docs/benchmark/audit/swiglu_packed_baseline_audit_2026-05-30.md`).
- ✅ Onboarding test (`tests/test_swiglu_packed_onboarding.py`, 3/3 PASS).
- ✅ Catalog count 45 → 46 (`tests/test_ssot_op_registry.py` PASS).

BL1 evidence (C5, in progress) lands the runtime numbers under
`benchmarks/results/phase1/stage8/extensibility/bl1_new_op.csv`.
