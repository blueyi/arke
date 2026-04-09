"""MLIR operation implementations."""

from typing import List
from arke.ir.semantic import Operation


class MLIROp:
    """Base class for MLIR operation emitters."""
    
    def emit(self, op: Operation, indent: str) -> str:
        raise NotImplementedError


class MatMulOp(MLIROp):
    """MLIR matmul operation."""
    
    def emit(self, op: Operation, indent: str) -> str:
        # Get inputs
        lhs = op.inputs[0].name if op.inputs else '%lhs'
        rhs = op.inputs[1].name if len(op.inputs) > 1 else '%rhs'
        
        # Emit linalg.matmul
        return f'{indent}%result = linalg.matmul ins(%{lhs}, %{rhs} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%result : tensor<?x?xf32>) -> tensor<?x?xf32>'


class ReluOp(MLIROp):
    """MLIR relu operation."""
    
    def emit(self, op: Operation, indent: str) -> str:
        inp = op.inputs[0].name if op.inputs else '%inp'
        return f'{indent}%result = arith.maxf %{inp}, %cst_zero : f32'


class SoftmaxOp(MLIROp):
    """MLIR softmax operation."""
    
    def emit(self, op: Operation, indent: str) -> str:
        inp = op.inputs[0].name if op.inputs else '%inp'
        return f'{indent}%result = arke.softmax %{inp} : tensor<?x?xf32> -> tensor<?x?xf32>'


# Registry
MLIR_OPS = {
    'matmul': MatMulOp(),
    'batch_matmul': MatMulOp(),
    'relu': ReluOp(),
    'softmax': SoftmaxOp(),
}


def register_mlir_ops(emitter):
    """Register all MLIR ops with emitter."""
    for op_name, op_impl in MLIR_OPS.items():
        emitter.register_op(op_name, op_impl)
