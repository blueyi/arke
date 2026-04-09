// Example 18: RMSNorm + Residual (Category C — normalization, fused)
//
// Natural Language:
//   "RMSNorm with residual add: H = X + residual, Y = rmsnorm(H, W)"
//   "LLaMA/DeepSeek 架构的标准 pre-norm 层"
//
// Semantics: H[i,j] = X[i,j] + residual[i,j]; Y[i,j] = H[i,j] / sqrt(mean(H[i,:]^2) + eps) * W[j]
// Category: C (normalization — fused residual + rmsnorm)

kernel rmsnorm_residual_kernel(
    X: Tensor<[128, 768], f16>,
    residual: Tensor<[128, 768], f16>,
    W: Tensor<[768], f16>
) -> Tensor<[128, 768], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = rmsnorm_residual(X=X, residual=residual, W=W);
    return Y;
}

strategy rmsnorm_residual_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("one row per block: each thread block computes one token's norm");
    launch_config(num_warps=8, num_stages=1)
        @rationale("8 warps for 768-dim: 8*32=256 threads, each handles 3 elements");
}
