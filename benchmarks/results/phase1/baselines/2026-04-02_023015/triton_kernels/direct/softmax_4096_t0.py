import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    X_ptr,
    OUT_ptr,
    N_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    X_row_ptr = X_ptr + row_idx * N_cols
    OUT_row_ptr = OUT_ptr + row_idx * N_cols
    
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
        x = x - m
        e = tl.exp(x)
        e = tl.where(mask, e, 0.0)
        s += tl.sum(e, axis=0)
    
    # Third pass: compute softmax and store
    for col_off in tl.static_range(0, N_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        x = tl.load(X_row_ptr + cols, mask=mask, other=float('-inf'))
        x = x - m
        e = tl.exp(x)
        out = e / s
        out = out.to(tl.float16)
        tl.store(OUT_row_ptr + cols, out, mask=mask)


def softmax_4096(X: torch.Tensor) -> torch.Tensor:
    assert X.shape == (4096, 4096), f"Expected shape (4096, 4096), got {X.shape}"
    assert X.dtype == torch.float16
    
    out = torch.empty_like(X)
    
    N_rows, N_cols = X.shape
    
    # Use BLOCK_SIZE of 4096 to process entire row if possible, but that requires
    # 4096 * 2 bytes = 8KB per block which is fine for registers
    # However, 4096 elements per thread block might be too many for registers
    # Use a smaller block size and loop
    BLOCK_SIZE = 4096  # Process entire row in one shot if possible
    
    # For 4096 columns, we can try to fit in one block
    # But 4096 floats = 16KB which might be tight for registers
    # Let's use 1024 and loop 4 times
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


if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    
    out = softmax_4096(X)
    ref = torch.softmax(X.float(), dim=-1).half()
    
    print(f"Max absolute error: {(out.float() - ref.float()).abs().max().item():.6f}")
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), "Mismatch!"
    print("PASSED")