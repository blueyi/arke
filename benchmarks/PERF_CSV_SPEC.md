# Benchmark CSV Specification

> Version: 1.0 | Date: 2026-04-05
> All performance CSVs must follow this schema for cross-stage analysis in Excel.

## Design Principles

1. **Single schema** — one CSV format for all performance data (L1/L2/L3/Gate)
2. **Excel-friendly** — sortable columns, no nested data, UTF-8 with BOM
3. **Self-contained** — each row has full context (no need to cross-reference)
4. **Append-safe** — new stages/runs append rows; filter by `stage`/`run_id`

## Column Specification

| # | Column | Type | Required | Description | Example |
|---|--------|------|:--------:|-------------|---------|
| 1 | `stage` | str | ✅ | Stage identifier | `stage1` |
| 2 | `gate` | str | | Gate that produced this data (if applicable) | `G2`, `G6` |
| 3 | `run_id` | str | ✅ | Unique run identifier (timestamp) | `2026-04-05_012345` |
| 4 | `operator` | str | ✅ | Operator name | `matmul`, `flash_attention` |
| 5 | `category` | str | ✅ | Operator category (BENCHMARK.md §2) | `A`, `B`, `C`, `D`, `E` |
| 6 | `shape_tag` | str | ✅ | Human-readable shape tag | `llama2-7b-512`, `ds-v2-2k` |
| 7 | `shape_tier` | int | ✅ | Shape tier (1-4) | `2` |
| 8 | `M` | int | | Dimension M (matmul/elementwise) | `1024` |
| 9 | `N` | int | | Dimension N | `4096` |
| 10 | `K` | int | | Dimension K (matmul only) | `1024` |
| 11 | `batch` | int | | Batch size (attention ops) | `1` |
| 12 | `seq_len` | int | | Sequence length (attention ops) | `2048` |
| 13 | `num_heads` | int | | Number of heads (attention ops) | `32` |
| 14 | `head_dim` | int | | Head dimension (attention ops) | `128` |
| 15 | `dtype` | str | ✅ | Data type | `f16`, `bf16`, `f32` |
| 16 | `backend` | str | ✅ | Backend target | `nvidia`, `ascend`, `mlir`, `llvm` |
| 17 | `method` | str | ✅ | Which implementation produced this result | `arke`, `cublas`, `flaggems`, `liger`, `inductor`, `llm_direct`, `eager` |
| 18 | `latency_us` | float | ✅ | Median latency in microseconds | `44.2` |
| 19 | `latency_min_us` | float | | Minimum latency in microseconds | `42.1` |
| 20 | `latency_max_us` | float | | Maximum latency in microseconds | `48.7` |
| 21 | `latency_std_us` | float | | Standard deviation | `1.3` |
| 22 | `tflops` | float | | Throughput in TFLOPS | `2.45` |
| 23 | `bandwidth_gbps` | float | | Memory bandwidth in GB/s | `450.2` |
| 24 | `correct` | bool | ✅ | Numerical correctness pass | `TRUE` |
| 25 | `max_abs_err` | float | | Maximum absolute error vs reference | `1.2e-4` |
| 26 | `max_rel_err` | float | | Maximum relative error vs reference | `2.3e-3` |
| 27 | `baseline_method` | str | | Primary baseline for ratio calculation | `cublas` |
| 28 | `baseline_latency_us` | float | | Baseline median latency | `44.2` |
| 29 | `ratio_vs_baseline` | float | | `baseline_latency / arke_latency` (>1 = Arke faster) | `1.09` |
| 30 | `warmup_iters` | int | | Number of warmup iterations | `200` |
| 31 | `bench_iters` | int | | Number of benchmark iterations | `500` |
| 32 | `gpu_name` | str | | GPU model | `RTX 3060 Laptop` |
| 33 | `gpu_mem_mb` | int | | GPU memory in MB | `6144` |
| 34 | `cuda_version` | str | | CUDA version | `12.4` |
| 35 | `triton_version` | str | | Triton version | `3.2.0` |
| 36 | `pytorch_version` | str | | PyTorch version | `2.6.0+cu124` |
| 37 | `notes` | str | | Free-form notes | `known-fail: dispatch overhead` |

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
stage,gate,run_id,operator,category,shape_tag,shape_tier,M,N,K,batch,seq_len,num_heads,head_dim,dtype,backend,method,latency_us,latency_min_us,latency_max_us,latency_std_us,tflops,bandwidth_gbps,correct,max_abs_err,max_rel_err,baseline_method,baseline_latency_us,ratio_vs_baseline,warmup_iters,bench_iters,gpu_name,gpu_mem_mb,cuda_version,triton_version,pytorch_version,notes
stage1,G2,2026-04-02_134301,matmul,A,square-1k,2,1024,1024,1024,,,,,,nvidia,arke,39.8,38.1,42.3,1.2,54.1,,TRUE,1.2e-4,2.3e-3,cublas,44.2,1.11,200,500,RTX 3060 Laptop,6144,12.4,3.2.0,2.6.0+cu124,
stage1,G2,2026-04-02_134301,matmul,A,square-1k,2,1024,1024,1024,,,,,,nvidia,cublas,44.2,43.0,46.1,0.8,48.7,,TRUE,,,,,,,200,500,RTX 3060 Laptop,6144,12.4,,2.6.0+cu124,
stage1,G2,2026-04-02_134301,matmul,A,square-4k,2,4096,4096,4096,,,,,,nvidia,arke,2380.0,2350.0,2420.0,18.5,57.7,,TRUE,1.5e-4,1.8e-3,cublas,1486.0,0.62,200,500,RTX 3060 Laptop,6144,12.4,3.2.0,2.6.0+cu124,
```
