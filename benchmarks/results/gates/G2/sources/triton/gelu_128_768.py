"""Auto-generated Triton elementwise kernel by Arke."""
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_128_768_kernel(
    X_ptr, Y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask)
    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    xf = x.to(tl.float32)
    yf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    y = yf.to(x.dtype)
    tl.store(Y_ptr + offsets, y, mask=mask)


def gelu_128_768(X: torch.Tensor) -> torch.Tensor:
    Y = torch.empty_like(X)
    n_elements = X.numel()
    # Heuristic: larger blocks reduce grid overhead for large tensors
    if n_elements <= 65536:
        BLOCK_SIZE = 1024
        num_warps = 4
    elif n_elements <= 524288:
        BLOCK_SIZE = 4096
        num_warps = 4
    else:
        BLOCK_SIZE = 8192
        num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_128_768_kernel[grid](
        X, Y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return Y
