// Arke Example: MatMul + GELU fusion
// Computes Y = gelu(A × B) — complex activation epilogue

kernel matmul_gelu(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = gelu(X=C);
    return Y;
}
