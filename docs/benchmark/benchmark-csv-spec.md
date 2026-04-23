# Benchmark CSV Specification

> Version: 2.0 | Date: 2026-04-05
> All performance CSVs must follow this schema for cross-stage analysis in Excel.

→ Parent: [`benchmark-design.md`](./benchmark-design.md)

## Design Principles

1. **Single schema** — one CSV format for all performance data (L1/L2/L3/Gate)
2. **Excel-friendly** — sortable columns, no nested data, UTF-8 with BOM
3. **Self-contained** — each row has full context (no need to cross-reference)
4. **Append-safe** — new stages/runs append rows; filter by `stage`/`run_id`
5. **Tier-aware** — every row carries `op_tier`/`shape_tier`/`benchmark_level`
   for direct filtering by the OT/ST/BL classification system

## Column Specification

| # | Column | Type | Required | Description | Example |
|---|--------|------|:--------:|-------------|---------|
| 1 | `stage` | str | ✅ | Stage identifier | `phase1` |
| 2 | `gate` | str | | Gate that produced this data (if applicable) | `G2`, `G6` |
| 3 | `run_id` | str | ✅ | Unique run identifier (timestamp) | `2026-04-05_012345` |
| 4 | `operator` | str | ✅ | Operator name | `matmul`, `flash_attention` |
| 5 | `op_tier` | int | ✅ | Operator Tier (0–4, see benchmark-design.md §2) | `2` |
| 6 | `category` | str | ✅ | Operator category (A–G, legacy compat) | `A` |
| 7 | `shape_tag` | str | ✅ | Human-readable shape tag | `llama2-7b-512`, `ds-v2-2k` |
| 8 | `shape_tier` | int | ✅ | Shape Tier (1–4, see benchmark-design.md §3) | `2` |
| 9 | `benchmark_level` | int | ✅ | Benchmark Level (1–6, see benchmark-design.md §1) | `3` |
| 10 | `eval_layer` | str | | Evaluation Layer (L1/L2/L3) | `L1` |
| 11 | `M` | int | | Dimension M (matmul/elementwise) | `1024` |
| 9 | `N` | int | | Dimension N | `4096` |
| 10 | `K` | int | | Dimension K (matmul only) | `1024` |
| 11 | `batch` | int | | Batch size (attention ops) | `1` |
| 12 | `seq_len` | int | | Sequence length (attention ops) | `2048` |
| 13 | `num_heads` | int | | Number of heads (attention ops) | `32` |
| 14 | `head_dim` | int | | Head dimension (attention ops) | `128` |
| 16 | `dtype` | str | ✅ | Data type | `f16`, `bf16`, `f32` |
| 17 | `backend` | str | ✅ | Backend target | `nvidia`, `ascend`, `mlir`, `llvm` |
| 18 | `method` | str | ✅ | Which implementation produced this result | `arke`, `cublas`, `flaggems`, `liger`, `inductor`, `llm_direct`, `eager` |
| 19 | `baseline_tier` | str | | Baseline tier of this method (P0–P5) | `P0` |
| 20 | `latency_us` | float | ✅ | Median latency in microseconds | `44.2` |
| 21 | `latency_min_us` | float | | Minimum latency in microseconds | `42.1` |
| 22 | `latency_max_us` | float | | Maximum latency in microseconds | `48.7` |
| 23 | `latency_std_us` | float | | Standard deviation | `1.3` |
| 24 | `tflops` | float | | Throughput in TFLOPS | `2.45` |
| 25 | `bandwidth_gbps` | float | | Memory bandwidth in GB/s | `450.2` |
| 26 | `correct` | bool | ✅ | Numerical correctness pass | `TRUE` |
| 27 | `max_abs_err` | float | | Maximum absolute error vs reference | `1.2e-4` |
| 28 | `max_rel_err` | float | | Maximum relative error vs reference | `2.3e-3` |
| 29 | `baseline_method` | str | | Primary baseline for ratio calculation | `cublas` |
| 30 | `baseline_latency_us` | float | | Baseline median latency | `44.2` |
| 31 | `ratio_vs_baseline` | float | | `baseline_latency / arke_latency` (>1 = Arke faster) | `1.09` |
| 32 | `warmup_iters` | int | | Number of warmup iterations | `200` |
| 33 | `bench_iters` | int | | Number of benchmark iterations | `500` |
| 34 | `gpu_name` | str | | GPU model | `RTX 3060 Laptop` |
| 35 | `gpu_mem_mb` | int | | GPU memory in MB | `6144` |
| 36 | `cuda_version` | str | | CUDA version | `12.4` |
| 37 | `triton_version` | str | | Triton version | `3.2.0` |
| 38 | `pytorch_version` | str | | PyTorch version | `2.6.0+cu124` |
| 39 | `notes` | str | | Free-form notes | `known-fail: dispatch overhead` |

## Usage Rules

### File Naming
```
<stage>/gates/<gate>/performance/perf_<operator>.csv    — Gate performance data
<stage>/L1/<run_id>/perf_<operator>.csv                 — L1 benchmark data
<stage>/L2/<run_id>/perf_<operator>.csv                 — L2 benchmark data
<stage>/L3/<run_id>/perf_<operator>.csv                 — L3 benchmark data
<stage>/STAGE_PERF_ALL.csv                              — Consolidated: all data for the stage
```

### Consolidated Stage CSV
At Stage completion, merge all per-run CSVs into `STAGE_PERF_ALL.csv`.
This single file enables cross-gate, cross-operator Excel analysis:
- **PivotTable** by `operator` × `method` → compare Arke vs baselines
- **Filter** by `gate` → see progression across gates
- **Sort** by `ratio_vs_baseline` → find best/worst shapes
- **Chart** `shape_tag` vs `ratio_vs_baseline` → visual performance profile

### Excel Tips
- File is UTF-8 with BOM (Excel auto-detects encoding)
- Boolean values: `TRUE`/`FALSE` (Excel native)
- Empty optional fields: blank (not "N/A" or "null")
- Decimal separator: `.` (standard CSV)

### Row Rules
1. **One row per (run_id, operator, shape_tag, method, dtype) tuple**
2. Arke results and baseline results are **separate rows** — Excel can pivot
3. `ratio_vs_baseline` is only set on Arke rows (baselines leave it blank)
4. `correct` must always be filled (even for baselines: always `TRUE`)

## Example

```csv
stage,gate,run_id,operator,op_tier,category,shape_tag,shape_tier,benchmark_level,eval_layer,M,N,K,batch,seq_len,num_heads,head_dim,dtype,backend,method,baseline_tier,latency_us,latency_min_us,latency_max_us,latency_std_us,tflops,bandwidth_gbps,correct,max_abs_err,max_rel_err,baseline_method,baseline_latency_us,ratio_vs_baseline,warmup_iters,bench_iters,gpu_name,gpu_mem_mb,cuda_version,triton_version,pytorch_version,notes
phase1,G2,2026-04-02_134301,matmul,2,A,square-1k,2,2,L1,1024,1024,1024,,,,f16,nvidia,arke,,39.8,38.1,42.3,1.2,54.1,,TRUE,1.2e-4,2.3e-3,cublas,44.2,1.11,200,500,RTX 3060 Laptop,6144,12.4,3.2.0,2.6.0+cu124,
phase1,G2,2026-04-02_134301,matmul,2,A,square-1k,2,2,L1,1024,1024,1024,,,,f16,nvidia,cublas,P0,44.2,43.0,46.1,0.8,48.7,,TRUE,,,,,,200,500,RTX 3060 Laptop,6144,12.4,,2.6.0+cu124,
phase1,G2,2026-04-02_134301,matmul,2,A,square-4k,2,2,L1,4096,4096,4096,,,,f16,nvidia,arke,,2380.0,2350.0,2420.0,18.5,57.7,,TRUE,1.5e-4,1.8e-3,cublas,1486.0,0.62,200,500,RTX 3060 Laptop,6144,12.4,3.2.0,2.6.0+cu124,
```
