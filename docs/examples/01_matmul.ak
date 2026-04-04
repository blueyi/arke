// Example 01: Matrix Multiplication (matmul)
//
// Natural Language:
//   "Compute the matrix product C = A × B, where A is 1024×1024
//    and B is 1024×1024, both in half precision (f16)."
//
// Semantics: C[i,j] = sum(A[i,k] * B[k,j], axis=k)
// Category: A (compute)

kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A=A, B=B);
    return C;
}

strategy matmul for target("nvidia_ampere") {
    tile(loop="M", factors=[128])
        @rationale("128 rows per thread block for L1 reuse");
    tile(loop="N", factors=[128])
        @rationale("128 cols for balanced occupancy");
    tile(loop="K", factors=[32])
        @rationale("K=32 for shared memory tiling");
    reorder(order=["M", "N", "K"])
        @rationale("outer M/N for parallelism, inner K for reduction");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("each thread block computes one 128×128 output tile");
    place(tensor="A_tile", memory="shared")
        @rationale("tile A in shared memory for K-loop reuse");
    place(tensor="B_tile", memory="shared")
        @rationale("tile B in shared memory for K-loop reuse");
}
