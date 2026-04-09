"""BL1 (Instruction Level) matmul implementation in MLIR."""

from typing import Tuple


class BL1MatMul:
    """BL1 matmul: direct MLIR linalg.matmul without fusion."""
    
    @staticmethod
    def emit_mlir(
        lhs_name: str,
        rhs_name: str,
        result_name: str,
        lhs_shape: Tuple[int, ...],
        rhs_shape: Tuple[int, ...],
        result_shape: Tuple[int, ...],
        dtype: str = 'f32'
    ) -> str:
        """Emit BL1 matmul MLIR code."""
        
        # Convert shapes to MLIR format
        lhs_type = f'tensor<{lhs_shape[0]}x{lhs_shape[1]}x{dtype}>'
        rhs_type = f'tensor<{rhs_shape[0]}x{rhs_shape[1]}x{dtype}>'
        result_type = f'tensor<{result_shape[0]}x{result_shape[1]}x{dtype}>'
        
        mlir_code = f'''
  %{result_name} = linalg.matmul
    ins(%{lhs_name}, %{rhs_name} : {lhs_type}, {rhs_type})
    outs(%{result_name} : {result_type})
    -> {result_type}
'''
        return mlir_code.strip()
    
    @staticmethod
    def emit_llvm(
        lhs_name: str,
        rhs_name: str,
        result_name: str,
        m: int,
        n: int,
        k: int,
        dtype: str = 'f32'
    ) -> str:
        """Emit BL1 matmul LLVM IR code."""
        
        # LLVM IR for matmul (simplified)
        llvm_code = f'''
  %{result_name} = call {{i64, i64, i64}} @matmul_kernel(
    %{lhs_name}, %{rhs_name},
    i64 {m}, i64 {n}, i64 {k}
  )
'''
        return llvm_code.strip()
    
    @staticmethod
    def emit_cuda(
        lhs_name: str,
        rhs_name: str,
        result_name: str,
        m: int,
        n: int,
        k: int,
        dtype: str = 'f32'
    ) -> str:
        """Emit BL1 matmul CUDA code."""
        
        cuda_code = f'''
  // BL1 MatMul: {m}x{k} @ {k}x{n} -> {m}x{n}
  cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    {n}, {m}, {k},
    &alpha,
    {rhs_name}, CUDA_R_{dtype.upper()}, {n},
    {lhs_name}, CUDA_R_{dtype.upper()}, {k},
    &beta,
    {result_name}, CUDA_R_{dtype.upper()}, {n},
    CUDA_R_{dtype.upper()},
    CUBLAS_GEMM_DEFAULT
  );
'''
        return cuda_code.strip()


def benchmark_bl1_matmul(m: int, n: int, k: int, dtype: str = 'f32') -> dict:
    """Benchmark BL1 matmul performance."""
    import torch
    import time
    
    # Create test tensors
    lhs = torch.randn(m, k, dtype=torch.float32)
    rhs = torch.randn(k, n, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(lhs, rhs)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        result = torch.matmul(lhs, rhs)
    elapsed = time.time() - start
    
    # Calculate FLOPS
    flops = 2 * m * n * k * 100 / elapsed
    
    return {
        'shape': (m, n, k),
        'dtype': dtype,
        'time_ms': elapsed * 10,
        'flops': flops,
        'tflops': flops / 1e12,
    }
