import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_gelu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask_a = offs_k[None, :] < K - k * BLOCK_K
        k_mask_b = offs_k[:, None] < K - k * BLOCK_K
        
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        accumulator = tl.dot(a, b, accumulator)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = accumulator
    cst = 0.7978845608028654  # sqrt(2/pi)
    inner = cst * (x + 0.044715 * x * x * x)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    
    c = gelu.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_matmul_gelu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul"
    M, K = A.shape
    K, N = B.shape
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    fused_matmul_gelu_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=4,
    )
    
    return C