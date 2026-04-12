# Stage 7 BL5 Gap Report

## Objective

Stage 7 must ultimately satisfy **BL5 × (L1 + L2)** for all required operators and shape families, with both:

- correctness / accuracy evidence
- performance evidence against the required baselines and thresholds

This report reverse-decomposes the remaining work from that final contract.

## Machine-readable source of truth

The current target matrix is generated at:

- `benchmarks/stage7_bl5_target_matrix.json`

It is intended to be the machine-readable planning substrate for Stage 7 closure.

## Current completion snapshot

From the generated matrix:

- **L1 total ops required:** 45
- **L1 ops with Stage 7 track6 results:** 37
- **L1 op coverage ratio:** 0.8222
- **L1 required shapes total:** 685
- **L1 required shapes observed:** 37
- **L1 shape coverage ratio:** 0.0540

- **L2 total fusions required:** 6
- **L2 fusions with Stage 7 track6 results:** 1
- **L2 fusion coverage ratio:** 0.1667
- **L2 required shapes total:** 120
- **L2 required shapes observed:** 1
- **L2 shape coverage ratio:** 0.0083

- **Performance artifacts present:** yes
- **Correctness / accuracy artifacts present:** yes (currently partial; verified live for L2 `matmul_relu` and a growing L1 subset including dense linear algebra, elementwise, reduction, normalization, activations, batched GEMM, data-movement/indexing, quantization, and attention; `matmul`, `gelu`, `silu`, `softmax`, and `layernorm` now also have live correctness evidence across representative required shapes, alongside `cross_attention`, `flash_attention`, and `grouped_query_attention`)

## What is currently verified

Current Stage 7 track6 artifacts currently provide benchmark evidence for:

- **L1:** `matmul` on dense GEMM shapes such as `square-1k`; `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `add`, `mul`, `neg`, `exp`, `rsqrt`, `where_`, `cast`, `copy_`, and `transpose` on representative square/2D or FFN shapes such as `gpt2-ffn`; `softmax` on attention/logit shapes such as `attn-gpt2-128`; `layernorm`, `rmsnorm`, and `rmsnorm_residual` on normalization shapes such as `gpt2-small`; `reduce_sum`, `reduce_max`, `reduce_mean`, and `argmax` on reduction shapes such as `small`; `cumsum` on row/column cumulative shapes such as `gpt2-row`; `topk` on routing/sampling shapes such as `moe-top2-small`; `batch_matmul` on `gpt2-attn-128`; `concat` / `split` on QKV-merge/split shapes such as `gpt2-qkv-merge` and `gpt2-qkv-split`; `gather` on dispatch/cache-read shapes such as `moe-dispatch-top2`; `scatter` on combine/cache-write shapes such as `moe-combine-top2`; `permute` on attention layout shapes such as `gpt2-bhsd`; `embedding` on vocab/sequence shapes such as `gpt2-small`; `quantize_per_token` and `dequantize_per_channel` on quantization shapes such as `gpt2-int8`; `cross_attention` on encoder/decoder attention shapes such as `whisper-enc-dec`; `flash_attention` on causal self-attention shapes such as `gpt2-sm-128`; `grouped_query_attention` on grouped-KV attention shapes such as `llama3-8b-512`
- **L2:** `matmul_relu` on `square-1k`

This is enough to prove that the harness and artifact path are alive, but it is far from proving BL5 closure.

## What remains fundamentally incomplete

### 1. Coverage is far below BL5 exit requirements

Stage 7 final closure requires:

- all BL5 L1 operators to be benchmark-runnable
- all required BL5 shape families to be benchmark-runnable
- all required L2 fusion cases to be benchmark-runnable

Current coverage is only a tiny fraction of the requirement set.

### 2. Correctness evidence is now persisted, but coverage is still narrow

Current benchmark artifacts now persist machine-readable correctness fields such as:

- `allclose`
- `max_abs_diff`
- `mean_abs_diff`
- `rtol`
- `atol`
- `correctness_status`
- `correctness_reason`

This unblocks credible Stage 7 correctness accounting, but it is still far from BL5 correctness closure because only a small subset of required points currently have live correctness evidence.

### 3. Performance targets are documented, but not yet point-wise enforced in artifacts

The Stage 7 plan already states BL5 performance goals by operator group and fusion family. However, current artifacts do not yet persist explicit per-point fields such as:

- `perf_target`
- `perf_actual`
- `perf_pass`
- `perf_gap`
- baseline identity used for the pass decision

### 4. Benchmark gaps must directly drive Lang / IR / lowering work

Track 6 in Stage 7 is already defined as reverse-decomposition from the BL5 benchmark target. This means missing benchmark evidence should directly generate follow-up work across the Arke component stack.

## Reverse decomposition into Arke 4-piece suite

This report organizes the remaining work into four component tracks.

### A. Lang / Parser

Goal:
- ensure the Stage 7 language surface can express all BL5-required operator and shape families, including conditional strategies and symbolic constraints.

Current status:
- foundational parser / symbolic-shape support exists
- representative gaps have already been fixed (for example `dim("B") <= ...` support)
- full BL5 benchmark-driven coverage is not yet proven

Next objectives:
1. audit every BL5 operator family against the current surface syntax
2. add missing `.ak` examples for any unsupported benchmark-driven patterns
3. maintain a coverage audit linking BL5 targets to parseable examples

### B. SemanticIR / StrategyIR

Goal:
- ensure every BL5 target maps cleanly into SemanticIR and StrategyIR with stable semantics

Current status:
- Stage 7 IR path is alive
- kernel identity alignment work has landed
- full BL5 target matrix is not yet validated benchmark-first

Next objectives:
1. validate every target-matrix op against SemanticIR / StrategyIR generation
2. add metadata hooks needed for benchmark evidence and diagnosability
3. ensure fusion cases are represented canonically, not via aliases or ad hoc surface workarounds

### C. Lowering / Compiler pipeline

Goal:
- ensure every required BL5 operator and fusion can traverse the Stage 7 lowering path far enough for benchmark execution and artifact production

Current status:
- MLIR skeleton and several Stage 7 naming/lowering inconsistencies have been fixed
- current proof is still narrow relative to BL5 full coverage

Next objectives:
1. build a per-op / per-fusion lowering compatibility audit from the target matrix
2. classify failures by stage: parse / semantic / strategy / lowering / emitter / runtime
3. drive remaining lowering fixes directly from benchmark gaps

### D. Benchmark / Gate / Evidence

Goal:
- make benchmark evidence authoritative for BL5 closure, rather than anecdotal

Current status:
- benchmark runners, summary files, Stage 7 result directories, and G7 artifacts exist
- correctness artifacts are now present for a small but real subset of L1/L2 benchmark points
- point-wise performance pass/fail evidence is still missing

Next objectives:
1. persist correctness / accuracy metrics into all benchmark artifacts
2. persist performance target evaluation into all benchmark artifacts
3. generate coverage dashboards and gap reports from the target matrix
4. promote these checks into formal gate criteria beyond today's minimal G7 state

## Phased goals

### Phase S7-A — BL5 target matrix and gap accounting

Deliverables:
- `benchmarks/stage7_bl5_target_matrix.json`
- this gap report

Success condition:
- every required L1 operator and L2 fusion has a machine-readable required-shape inventory
- current observed coverage is measurable, not anecdotal

### Phase S7-B — correctness-first artifact schema

Deliverables:
- benchmark runners write correctness metrics and tolerances
- `PERF_ALL.csv` and per-op result files include correctness fields
- summary generation aggregates correctness pass/fail

Success condition:
- every benchmark point has machine-readable correctness evidence

### Phase S7-C — coverage closure

Deliverables:
- Stage 7 benchmark routing covers all BL5 L1 ops and all Stage 7 L2 fusions
- missing points are surfaced automatically

Success condition:
- required operator coverage = 100%
- required shape coverage = 100% (with explicit OOM-policy evidence where applicable)

### Phase S7-D — performance contract enforcement

Deliverables:
- point-wise performance targets persisted in artifacts
- group/fusion pass logic implemented
- gate-readable summaries expose coverage + correctness + performance status

Success condition:
- Stage 7 can machine-check the BL5 performance contract, not just log raw latency

### Phase S7-E — final BL5 exit

Success condition:
- full required coverage achieved
- correctness pass achieved for all required benchmark points
- performance targets satisfied at the required group/fusion level
- Stage 7 has gate-level evidence strong enough to claim BL5 closure

## Immediate next actions

1. Continue extending `run_with_inputs(...)`/reference coverage from the current verified L1 subset into the remaining unsupported BL5 operators
2. Audit other write/index ops for similar probe-semantics pitfalls (for example repeated-index nondeterminism) before counting mismatches as implementation bugs
3. Extend artifact writing so performance pass/fail is preserved in `PERF_ALL.csv` and summaries alongside the new correctness fields
4. Generate a coverage-oriented report that highlights missing BL5 ops / shape tags directly from the target matrix
