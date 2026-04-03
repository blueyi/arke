"""Auto-generated Triton softmax kernel by Arke."""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_1024_1024_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    x_ptrs = X_ptr + row * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptrs, mask=n_mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp

    y_ptrs = Y_ptr + row * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, y, mask=n_mask)


def softmax_1024_1024(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2, "softmax expects 2D input"
    M, N = X.shape
    Y = torch.empty_like(X)
    BLOCK_N = triton.next_power_of_2(N)
    # Heuristic num_warps selection based on row width
    if BLOCK_N <= 256:
        num_warps = 2
    elif BLOCK_N <= 2048:
        num_warps = 4
    elif BLOCK_N <= 8192:
        num_warps = 8
    else:
        num_warps = 16
    grid = (M,)
    softmax_1024_1024_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return Y
