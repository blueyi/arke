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
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A=A, B=B);
    return C;
}

// strategy block omitted intentionally — compiler auto-generates from nvidia_ampere profile:
//
//   tile(loop="M", factors=[64])   — tensor-core aligned, fits L1
//   tile(loop="N", factors=[64])
//   tile(loop="K", factors=[16])   — shared memory budget
//   reorder(order=["M", "N", "K"])
//   parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
//   place(tensor="A_tile", memory="shared")
//   place(tensor="B_tile", memory="shared")
//
// To override, add an explicit strategy block (see docs/examples/01_matmul.md §5).
