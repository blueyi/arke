// Example 06: RMSNorm (Category C — reduction, LLaMA-style)
//
// Natural Language:
//   "对 LLaMA hidden state (batch=128, hidden=4096) 做 RMSNorm"
//
// Semantics: out[i,j] = x[i,j] / sqrt(mean(x[i,:]^2) + eps) * w[j]
// Category: C (reduction)

kernel rmsnorm(
    X:      Tensor<[128, 4096], f16>,
    weight: Tensor<[4096], f16>
) -> Tensor<[128, 4096], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = rmsnorm(X=X, weight=weight, eps=0.000010);
    return Y;
}

strategy rmsnorm_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("one CTA per row — RMS reduction is row-local");
    vectorize(loop="col", width=8)
        @rationale("vec8 f16 for 128-bit loads — maximize bandwidth");
    place(tensor="X_row", memory="shared")
        @rationale("cache in smem: need two passes (compute rms, then normalize)");
    launch_config(num_warps=8, num_stages=1)
        @rationale("4096/8/32=16 iters per warp — 8 warps for good SM occupancy");
}
