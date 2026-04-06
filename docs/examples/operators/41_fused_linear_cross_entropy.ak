// Example 41: Fused Linear + Cross Entropy (Category C — fused matmul+loss)
//
// Natural Language:
//   "融合线性层+交叉熵：Hidden:[128, 4096] × W:[4096, 32000] → CE loss"
//   "避免物化 32000-wide logits，直接分块计算 loss"
//
// Semantics: logits = Hidden @ W^T; loss[i] = cross_entropy(logits[i,:], Labels[i])
// Category: C (fused — matmul + cross entropy)

kernel fused_linear_cross_entropy_kernel(
    Hidden: Tensor<[128, 4096], f16>,
    Weight: Tensor<[32000, 4096], f16>,
    Labels: Tensor<[128], i64>
) -> Tensor<[128], f32> {
    let Loss = fused_linear_cross_entropy(Hidden=Hidden, Weight=Weight, Labels=Labels);
    return Loss;
}

strategy fused_linear_cross_entropy_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("1 row per CTA — each row computes partial logits + streaming CE");
    tile(loop="col", factors=[4096])
        @rationale("4096-col tile over vocab: chunked matmul avoids materializing full logits");
    launch_config(num_warps=8, num_stages=2)
        @rationale("2-stage: overlap next weight chunk load with current matmul+softmax");
}
