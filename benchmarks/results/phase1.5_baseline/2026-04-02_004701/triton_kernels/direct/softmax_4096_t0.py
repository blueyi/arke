import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    X_ptr,
    Out_ptr,
    N_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    X_row_ptr = X_ptr + row_idx * N_cols
    Out_row_ptr = Out_ptr + row_idx * N_cols
    
    # First pass: compute max
    m = float('-inf')
    for col_off in tl.static_range(0, N_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        x = tl.load(X_row_ptr + cols, mask=mask, other=float('-inf'))
        m = tl.maximum(m, tl.max(x, axis=0))
    
    # Second pass: compute sum of exp(x - max)
    s = 0.0
    for col_off in tl.static_range(0, N_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        x = tl.load(X_row_ptr + cols, mask=mask, other=float('-inf'))
        exp_x = tl.exp(x - m)
        s += tl.sum(exp_x, axis=0)
    
    # Third pass: write normalized values
    for col_off in tl.static_range(0, N_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        x = tl.load(X_row_ptr + cols, mask=mask, other=float('-inf'))
        exp_x = tl.exp(x - m)
        out = exp_x / s
        tl.store(Out_row_ptr + cols, out.to(tl.float16), mask=mask)


def softmax_4096(X: torch.Tensor) -> torch.Tensor:
    assert X.shape == (4096, 4096), f"Expected shape (4096, 4096), got {X.shape}"
    assert X.dtype == torch.float16, f"Expected dtype float16, got {X.dtype}"
    
    out = torch.empty_like(X)
    
    N_rows, N_cols = X.shape
    
    # Use BLOCK_SIZE of 1024 to process 4096 columns in 4 iterations
    # This fits well within register budget on Ampere
    BLOCK_SIZE = 1024
    
    grid = (N_rows,)
    
    _softmax_kernel[grid](
        X,
        out,
        N_cols=N_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    
    return out