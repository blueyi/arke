// Example 38: Copy (Category A — memory/layout, in-place copy)
//
// Natural Language:
//   "原位拷贝：Src:[128, 4096] → Dst:[128, 4096]，in-place copy"
//
// Semantics: Dst[:] = Src[:]
// Category: A (memory — in-place copy)

kernel copy_kernel(
    Src: Tensor<[128, 4096], f16>,
    Dst: Tensor<[128, 4096], f16>
) -> Tensor<[128, 4096], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = copy_(Src=Src, Dst=Dst);
    return Y;
}

strategy copy_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — pure memcpy, maximize occupancy and bandwidth");
    compute(warps=4, num_stages=2)
        @rationale("2-stage pipeline: overlap load and store for bandwidth saturation");
}
