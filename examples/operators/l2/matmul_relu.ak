// Stage 7 L2 surface example: matmul + relu epilogue fusion

kernel matmul_relu_kernel(
    X: Tensor<[M, K], f16>,
    W: Tensor<[K, N], f16>
) -> Tensor<[M, N], f16>
where M: dynamic(max=4096), K: static, N: dynamic(max=4096)
{
    let Z = matmul(A=X, B=W);
    let Y = relu(X=Z);
    return Y;
}

strategy matmul_relu_kernel for target("nvidia_ampere") {
    tile(loop="M", factors=[32])
        @rationale("BL5 L2 epilogue path keeps M tiles modest for CTA occupancy");
    tile(loop="N", factors=[128])
        @rationale("epilogue fusion amortizes relu over 128-column output tiles");
    tile(loop="K", factors=[32])
        @rationale("shared-memory-friendly reduction tile for fused matmul epilogue");
    reorder(order=["M", "N", "K"])
        @rationale("MN outer ordering exposes output parallelism before K reduction");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("map output tiles across CTA grid");
    fuse(ops=["matmul", "relu"], fusion_type="epilogue")
        @rationale("Stage 7 L2 fusion slot: matmul_relu");
    place(tensor="A_tile", memory="shared")
        @rationale("cache lhs tile for repeated K-step reuse");
    place(tensor="B_tile", memory="shared")
        @rationale("cache rhs tile so relu can stay in epilogue registers");
    compute(warps=4, num_stages=3)
        @rationale("triple-buffer global-to-shared pipeline for fused epilogue path");
}
