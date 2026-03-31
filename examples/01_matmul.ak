// Example 01: Basic Matrix Multiplication
// Demonstrates kernel definition syntax

kernel matmul(
    A: Tensor<[1024, 512], f16, row_major>,
    B: Tensor<[512, 2048], f16, col_major>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    return C;
}
