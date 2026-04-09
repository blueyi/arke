"""Tests for MLIR backend."""

import pytest
from arke.backends.mlir import MLIREmitter, register_mlir_ops
from arke.backends.mlir.bl1_matmul import BL1MatMul, benchmark_bl1_matmul


def test_mlir_emitter_basic():
    """Test basic MLIR emitter."""
    emitter = MLIREmitter()
    register_mlir_ops(emitter)
    
    # Test that emitter is initialized
    assert emitter is not None
    assert len(emitter.ops_registry) > 0


def test_bl1_matmul_mlir():
    """Test BL1 matmul MLIR emission."""
    mlir_code = BL1MatMul.emit_mlir(
        lhs_name='lhs',
        rhs_name='rhs',
        result_name='result',
        lhs_shape=(128, 64),
        rhs_shape=(64, 128),
        result_shape=(128, 128),
        dtype='f32'
    )
    
    assert 'linalg.matmul' in mlir_code
    assert 'tensor<128x64xf32>' in mlir_code
    assert 'tensor<64x128xf32>' in mlir_code


def test_bl1_matmul_cuda():
    """Test BL1 matmul CUDA emission."""
    cuda_code = BL1MatMul.emit_cuda(
        lhs_name='d_lhs',
        rhs_name='d_rhs',
        result_name='d_result',
        m=128,
        n=128,
        k=64,
        dtype='f32'
    )
    
    assert 'cublasGemmEx' in cuda_code
    assert '128' in cuda_code


def test_bl1_matmul_benchmark():
    """Test BL1 matmul benchmark."""
    result = benchmark_bl1_matmul(128, 128, 64)
    
    assert result['shape'] == (128, 128, 64)
    assert result['dtype'] == 'f32'
    assert result['tflops'] > 0
    assert result['time_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
