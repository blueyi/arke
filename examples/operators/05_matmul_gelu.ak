// Example 05: Fused MatMul + GELU (Cat B — fused compute+elementwise)
//
// Natural Language:
//   "计算 GPT-2 FFN 的第一个线性变换并立即应用 GELU 激活:
//    out = gelu(X @ W)，其中 X: [128,768], W: [768,3072]"
//
// Semantics: out[i,j] = gelu(sum_k(X[i,k] * W[k,j]))
// Category: B (fused compute — matmul + activation epilogue)

kernel matmul_gelu(
    X: Tensor<[128, 768], f16>,
    W: Tensor<[768, 3072], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Z = matmul(A=X, B=W);
    let Y = gelu(X=Z);
    return Y;
}

strategy matmul_gelu_strategy for target("nvidia_ampere") {
    tile(loop="M", factors=[32])
        @rationale("M=128 small — 32 tile keeps register pressure low");
    tile(loop="N", factors=[128])
        @rationale("N=3072: 128-tile, multiple blocks cover output columns");
    tile(loop="K", factors=[32])
        @rationale("K-tile=32: A+B smem = 32*32*2*2 = 4096B, fits in L1");
    reorder(order=["M", "N", "K"])
        @rationale("MN outer for parallelism, K inner for reduction accumulation");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("each block owns one (M,N) tile");
    fuse(ops=["matmul", "gelu"], fusion_type="epilogue")
        @rationale("apply gelu in matmul epilogue — saves global memory roundtrip for Z");
    place(tensor="A_tile", memory="shared")
        @rationale("A tile cached in smem to reduce global loads during K-reduction");
    place(tensor="B_tile", memory="shared")
        @rationale("B tile cached in smem for same reason");
    compute(warps=4, num_stages=3)
        @rationale("3 pipeline stages hide global→shared latency for A/B prefetch");
}
