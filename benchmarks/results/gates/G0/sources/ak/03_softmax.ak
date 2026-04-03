// Arke Example: Softmax
// Computes Y = softmax(X) along the last dimension

kernel softmax(
    X: Tensor<[4096, 4096], f16>
) -> Tensor<[4096, 4096], f16> {
    let Y = softmax(X=X);
    return Y;
}
