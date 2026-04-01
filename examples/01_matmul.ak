// Arke Example: Basic Matrix Multiplication
// Computes C = A × B for 1024×1024 matrices in f16

kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A, B);
    return C;
}
