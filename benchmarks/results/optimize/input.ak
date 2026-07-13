kernel matmul_kernel(
    A: Tensor<[64, 64], f16>,
    B: Tensor<[64, 64], f16>
) -> Tensor<[64, 64], f16>
{
    let C = matmul(A=A, B=B);
    return C;
}
