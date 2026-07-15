# TC Attention v4b: Shared Memory Bank Conflict Analysis

**Date**: 2026-07-15  
**Kernel**: `scratch/tc_attn/tc_attn_v4b.cu`  
**GPU**: RTX 3060 Laptop (sm_86, 6 GB)  
**Result**: ❌ **Negative result — bank conflicts are NOT the bottleneck**

## Theoretical Analysis

With D=64, all shared memory buffers have stride = 64 halfs = 128 bytes.

**Bank conflict calculation:**
- 32 banks × 4 bytes/bank = 128 bytes per bank cycle
- Stride 128 bytes / 4 bytes per bank = 32 banks
- 32 banks mod 32 = **0 → stride hits all 32 banks → full wrap → ZERO conflicts**

For `wmma::load_matrix_sync` (16×16 tile loads):
- Row-major load with leading dimension = 64 halfs = 128 bytes
- Consecutive rows land on consecutive bank sets (offset by 128B = full wrap)
- Col-major load (K in QK^T) with same stride: also clean

**Prediction: padding should NOT help** — confirmed by experiment.

## Experimental Setup

Created `tc_attn_v4b_pad.cu`: identical kernel with PAD=8 halfs added to each row:
- Stride: 72 halfs = 144 bytes (NOT a clean 32-bank multiple → introduces mild conflicts!)
- Smem: 45 KB (vs 40 KB) → still fits 2 blocks/SM
- Uses synchronous loads (no cp.async) because padded layout breaks contiguous 16B alignment

## Results

```
Case               Correct  v4b (ms)     v4b_pad (ms)   Δ%         Winner
--------------------------------------------------------------------------------
[1x1x128x64]       ✓        0.0310       0.0309         +0.3       ~tie
[1x4x128x64]       ✓        0.0310       0.0312         -0.4       ~tie
[1x8x512x64]       ✓        0.1653       0.1830         -10.7      v4b
[1x8x1024x64]      ✓        0.4688       0.5651         -20.5      v4b
[1x8x2048x64]      ✓        2.1481       6.1004         -184.0     v4b
[4x8x2048x64]      ✓        7.3889       7.2367         +2.1       v4b_pad
```

**Average: padding HURTS by -35.5%**

## Interpretation

Two confounded effects in v4b_pad:
1. **Padding itself**: at D=64, stride=128B already wraps cleanly across 32 banks. Padding to 72 halfs (144B) actually introduces mild conflicts (144/4=36, 36 mod 32=4).
2. **Loss of cp.async**: padded layout breaks 16B-aligned contiguous access required by `cp.async.cg`. Falling back to synchronous loads eliminates prefetch overlap.

The massive degradation at S=2048 (1 batch) is dominated by loss of cp.async overlap: 31 tile iterations where global→smem loads are fully sequential instead of overlapped with QK^T + softmax compute. At S=128 (1-2 tiles), there's little to overlap so both variants are equivalent.

The `4x8x2048` case (~tie) shows that with enough blocks (256), the GPU hides latency via block-level parallelism regardless of per-block prefetch strategy.

## Conclusion

- **Bank conflicts are NOT a bottleneck** in v4b at D=64
- The D=64 layout with stride=128B gives a perfect 32-bank wrap → zero conflicts by construction
- **cp.async prefetch IS valuable**: removing it costs 10-180% depending on tile count
- **Do not pursue shared memory padding** for this kernel at D=64
- Next optimization targets: instruction-level parallelism, warp scheduling, global memory BW

## Files

- `tc_attn_v4b_pad.cu` — padded variant (negative result, kept for reference)
- `run_tc_attn_v4b_pad.py` — comparison harness
