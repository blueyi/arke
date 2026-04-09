"""MLIR code emitter for Arke IR."""

from typing import List, Dict, Any
from arke.ir.semantic import SemanticIR, Operation, Tensor
from arke.ir.strategy import StrategyIR


class MLIREmitter:
    """Emits MLIR code from StrategyIR."""
    
    def __init__(self):
        self.ops_registry = {}
        self.indent_level = 0
    
    def emit(self, strategy_ir: StrategyIR) -> str:
        """Emit MLIR code from StrategyIR."""
        lines = []
        
        # Module header
        lines.append('module {')
        self.indent_level += 1
        
        # Emit function
        lines.append(self._emit_func(strategy_ir))
        
        self.indent_level -= 1
        lines.append('}')
        
        return '\n'.join(lines)
    
    def _emit_func(self, strategy_ir: StrategyIR) -> str:
        """Emit MLIR function."""
        lines = []
        indent = '  ' * self.indent_level
        
        # Function signature
        func_name = strategy_ir.kernel_name
        params = ', '.join([
            f'%{p.name}: {self._mlir_type(p)}'
            for p in strategy_ir.params
        ])
        return_type = self._mlir_type(strategy_ir.return_type)
        
        lines.append(f'{indent}func.func @{func_name}({params}) -> {return_type} {{')
        self.indent_level += 1
        
        # Emit operations
        for op in strategy_ir.ops:
            lines.append(self._emit_op(op))
        
        # Return
        lines.append(f'{"  " * self.indent_level}return %result : {return_type}')
        
        self.indent_level -= 1
        lines.append(f'{indent}}}')
        
        return '\n'.join(lines)
    
    def _emit_op(self, op: Operation) -> str:
        """Emit MLIR operation."""
        indent = '  ' * self.indent_level
        op_name = op.name.lower()
        
        # Get MLIR op mapping
        if op_name not in self.ops_registry:
            return f'{indent}// TODO: {op_name}'
        
        mlir_op = self.ops_registry[op_name]
        return mlir_op.emit(op, indent)
    
    def _mlir_type(self, tensor: Tensor) -> str:
        """Convert Arke tensor type to MLIR type."""
        shape = ', '.join([
            f'?'  # Dynamic dimensions
            if hasattr(d, 'is_dynamic') and d.is_dynamic
            else str(d)
            for d in tensor.shape
        ])
        dtype = self._mlir_dtype(tensor.dtype)
        return f'tensor<{shape}x{dtype}>'
    
    def _mlir_dtype(self, dtype: str) -> str:
        """Convert Arke dtype to MLIR dtype."""
        mapping = {
            'f16': 'f16',
            'f32': 'f32',
            'f64': 'f64',
            'i32': 'i32',
            'i64': 'i64',
            'bool': 'i1',
        }
        return mapping.get(dtype, dtype)
    
    def register_op(self, op_name: str, emitter):
        """Register MLIR op emitter."""
        self.ops_registry[op_name] = emitter
