// Arke Example: Fused MatMul + ReLU
// Computes Y = relu(A × B) — epilogue fusion eliminates intermediate write

kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(X=C);
    return Y;
}
