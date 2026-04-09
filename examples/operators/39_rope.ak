// Example 39: RoPE — Rotary Position Embedding (Category D — positional encoding)
//
// Natural Language:
//   "旋转位置编码：X:[2, 12, 128, 64] + Cos/Sin:[128, 64] → Y:[2, 12, 128, 64]"
//   "RoPE: 将 Q/K 向量按 pair 旋转，注入位置信息"
//
// Semantics: Y[b,h,s,2k] = X[...,2k]*cos(θ) - X[...,2k+1]*sin(θ)
//            Y[b,h,s,2k+1] = X[...,2k]*sin(θ) + X[...,2k+1]*cos(θ)
// Category: D (positional encoding — RoPE)

kernel rope_kernel(
    X: Tensor<[2, 12, 128, 64], f16>,
    Cos: Tensor<[128, 64], f16>,
    Sin: Tensor<[128, 64], f16>
) -> Tensor<[2, 12, 128, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = rope(X=X, Cos=Cos, Sin=Sin);
    return Y;
}

strategy rope_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 sequence positions per block — pair-wise rotation is elementwise-like");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: elementwise mul+add with sin/cos broadcast, memory-bound");
}
