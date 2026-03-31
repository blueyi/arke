// Example 03: Softmax
// Demonstrates reduction operator and multi-pass computation

kernel softmax(
    X: Tensor<[1024, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let max_val = reduce_max(X, axis=1);
    let shifted = sub(X, max_val);
    let exp_val = exp(shifted);
    let sum_val = reduce_sum(exp_val, axis=1);
    let Y = div(exp_val, sum_val);
    return Y;
}

schedule softmax for target("nvidia_ampere") {
    // Online softmax: fuse max and sum into single pass
    fuse(ops=["reduce_max", "sub", "exp", "reduce_sum", "div"], type=vertical)
        @rationale("online softmax algorithm — single pass over data");

    tile(loop="j", factors=[256])
        @rationale("process 256 elements per thread for register reuse");

    parallel(loops=["i"],
             mapping={"i": "blockIdx.x"})
        @rationale("each row is independent, parallelize across rows");
}
