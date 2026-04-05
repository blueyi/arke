// Example 40: Cross Entropy Loss (Category E — reduction, loss function)
//
// Natural Language:
//   "交叉熵损失：Logits:[128, 32000] + Labels:[128] → Loss:[128]"
//   "loss[i] = -log(softmax(Logits[i, :])[Labels[i]])"
//
// Semantics: loss[i] = -logits[i, label[i]] + log(sum_j(exp(logits[i,j])))
// Category: E (reduction — cross entropy loss)

kernel cross_entropy_kernel(
    Logits: Tensor<[128, 32000], f32>,
    Labels: Tensor<[128], i64>
) -> Tensor<[128], f32> {
    let Loss = cross_entropy(Logits=Logits, Labels=Labels);
    return Loss;
}

strategy cross_entropy_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("1 row per CTA — 32000-wide softmax+log needs full CTA cooperation");
    launch_config(num_warps=8, num_stages=1)
        @rationale("8 warps: large vocab dim reduction needs parallelism within CTA");
}
