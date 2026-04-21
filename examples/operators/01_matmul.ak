// Example 01: Matrix Multiplication (matmul)
//
// Natural Language:
//   "计算两个 1024×1024 的半精度矩阵乘法 C = A × B"
//
// Semantics: C[i,j] = sum(A[i,k] * B[k,j], axis=k)
// Category: A (compute)
//
// Note: The `strategy` block is OPTIONAL.
// If omitted, the Arke compiler automatically generates a hardware-aware
// Strategy IR via DefaultStrategyGenerator.
// You can still provide an explicit strategy to override the defaults.

kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16>
{
    let C = matmul(A=A, B=B);
    return C;
}

strategy matmul_strategy for target("nvidia_ampere") {
    tile(loop="M", factors=[128]) @rationale("tile M for tensor-core friendly CTA reuse");
    tile(loop="N", factors=[128]) @rationale("tile N to balance occupancy and shared-memory reuse");
    tile(loop="K", factors=[32]) @rationale("keep K tile aligned with Ampere MMA fragments");
    reorder(order=["M", "N", "K"]) @rationale("outer MN ordering improves cooperative tile traversal");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"}) @rationale("map output tiles to CTA grid");
    place(tensor="A_tile", memory="shared") @rationale("stage lhs tile in shared memory");
    place(tensor="B_tile", memory="shared") @rationale("stage rhs tile in shared memory");
}
