// Example 02: Fused MatMul + ReLU
// Demonstrates kernel + schedule with @rationale annotations

kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16, row_major>,
    B: Tensor<[512, 2048], f16, col_major>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");

    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing on global memory reads");

    tile(loop="k", factors=[32])
        @rationale("shared memory capacity = 48KB, fits 32x tiled blocks");

    reorder(["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"])
        @rationale("outer loops parallel, inner loops for data reuse");

    parallel(loops=["i_outer", "j_outer"],
             mapping={"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"});

    place(tensor="A_tile", memory=shared)
        @rationale("broadcast along j dimension, reuse across j iterations");

    place(tensor="B_tile", memory=shared)
        @rationale("broadcast along i dimension, reuse across i iterations");

    fuse(ops=["matmul", "relu"], type=epilogue)
        @rationale("relu is elementwise, fuse to avoid extra global memory write");
}
