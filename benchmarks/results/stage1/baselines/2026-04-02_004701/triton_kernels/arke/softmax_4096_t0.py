"""Auto-generated Triton softmax kernel by Arke."""
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_4096_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptrs = X_ptr + pid_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # Numerically stable softmax: subtract max, exp, normalize
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp

    y_ptrs = Y_ptr + pid_m * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, y, mask=mask)


def softmax_4096(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2, "softmax expects 2D input"
    M, N = X.shape
    Y = torch.empty_like(X)
    # BLOCK_N must be >= N and a power of 2
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    softmax_4096_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return Y
