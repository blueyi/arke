# Arke Pass Infrastructure Specification

> **Version:** 1.0.0  
> **Status:** Specification  
> **Date:** 2026-04-09  
> **Purpose:** Define universal pass interface, composition, and execution model for IR transformation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pass Abstraction](#2-pass-abstraction)
3. [Pass Lifecycle](#3-pass-lifecycle)
4. [Pass Composition](#4-pass-composition)
5. [Data Flow & Dependencies](#5-data-flow--dependencies)
6. [Error Handling](#6-error-handling)
7. [Built-in Passes](#7-built-in-passes)
8. [Custom Pass Development](#8-custom-pass-development)
9. [Pass Registry](#9-pass-registry)
10. [Examples](#10-examples)

---

## 1. Overview

### 1.1 Purpose

A **Pass** is a composable IR transformation that:
- Takes IR as input (SemanticIR, StrategyIR, ScheduleIR, or InstructionIR)
- Applies a specific transformation (analysis, optimization, lowering)
- Produces transformed IR as output
- Preserves or updates metadata (rationale, provenance)

### 1.2 Design Principles

1. **Single Responsibility** — Each pass does one thing well
2. **Composable** — Passes can be chained in arbitrary order (with dependency tracking)
3. **Verifiable** — Each pass has pre/post-conditions that can be checked
4. **Debuggable** — Passes emit structured logs and can be inspected
5. **Hardware-Agnostic** — Passes work on canonical IR, not hardware-specific code

### 1.3 Pass Hierarchy

```
Pass (abstract base)
├── AnalysisPass (read-only, produces analysis result)
│   ├── ShapeInferencePass
│   ├── ConstraintAnalysisPass
│   └── DataFlowAnalysisPass
├── TransformPass (modifies IR)
│   ├── OptimizationPass
│   │   ├── FusionPass
│   │   ├── TilingPass
│   │   └── MemoryOptimizationPass
│   └── LoweringPass
│       ├── ScheduleIRLoweringPass
│       └── InstructionIRLoweringPass
└── VerificationPass (checks invariants)
    ├── SSAValidatorPass
    ├── ConstraintValidatorPass
    └── CorrectnessValidatorPass
```

---

## 2. Pass Abstraction

### 2.1 Base Pass Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class PassContext:
    """Execution context for a pass."""
    hardware_target: str                    # "nvidia_ampere", "ascend_a3", etc.
    optimization_level: int = 2             # 0=none, 1=basic, 2=aggressive
    debug: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PassResult:
    """Result of pass execution."""
    status: str                             # "success", "failed", "skipped"
    ir: Any                                 # transformed IR (or original if failed)
    analysis: Dict[str, Any] = field(default_factory=dict)  # analysis results
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)  # perf metrics

class Pass(ABC):
    """Abstract base class for all passes."""
    
    def __init__(self, name: str, pass_type: str):
        self.name = name
        self.pass_type = pass_type  # "analysis", "transform", "verification"
        self.preconditions: List[str] = []
        self.postconditions: List[str] = []
        self.dependencies: List[str] = []  # other passes that must run first
    
    @abstractmethod
    def run(self, ir: Any, context: PassContext) -> PassResult:
        """Execute the pass.
        
        Args:
            ir: Input IR (SemanticIR, StrategyIR, ScheduleIR, or InstructionIR)
            context: Execution context
        
        Returns:
            PassResult with transformed IR and metadata
        """
        pass
    
    def check_preconditions(self, ir: Any) -> bool:
        """Verify preconditions before running pass."""
        # Default: no preconditions
        return True
    
    def check_postconditions(self, ir: Any) -> bool:
        """Verify postconditions after running pass."""
        # Default: no postconditions
        return True
```

### 2.2 Analysis Pass

```python
class AnalysisPass(Pass):
    """Pass that analyzes IR without modifying it."""
    
    def __init__(self, name: str):
        super().__init__(name, "analysis")
    
    @abstractmethod
    def analyze(self, ir: Any, context: PassContext) -> Dict[str, Any]:
        """Perform analysis and return results."""
        pass
    
    def run(self, ir: Any, context: PassContext) -> PassResult:
        """Execute analysis pass."""
        try:
            if not self.check_preconditions(ir):
                return PassResult(status="failed", ir=ir, 
                    errors=["Preconditions not met"])
            
            analysis = self.analyze(ir, context)
            
            return PassResult(
                status="success",
                ir=ir,  # unchanged
                analysis=analysis
            )
        except Exception as e:
            return PassResult(status="failed", ir=ir, errors=[str(e)])
```

### 2.3 Transform Pass

```python
class TransformPass(Pass):
    """Pass that transforms IR."""
    
    def __init__(self, name: str):
        super().__init__(name, "transform")
    
    @abstractmethod
    def transform(self, ir: Any, context: PassContext) -> Any:
        """Transform IR and return modified version."""
        pass
    
    def run(self, ir: Any, context: PassContext) -> PassResult:
        """Execute transform pass."""
        try:
            if not self.check_preconditions(ir):
                return PassResult(status="failed", ir=ir,
                    errors=["Preconditions not met"])
            
            transformed_ir = self.transform(ir, context)
            
            if not self.check_postconditions(transformed_ir):
                return PassResult(status="failed", ir=ir,
                    errors=["Postconditions not met"])
            
            return PassResult(status="success", ir=transformed_ir)
        except Exception as e:
            return PassResult(status="failed", ir=ir, errors=[str(e)])
```

### 2.4 Verification Pass

```python
class VerificationPass(Pass):
    """Pass that verifies IR invariants."""
    
    def __init__(self, name: str):
        super().__init__(name, "verification")
    
    @abstractmethod
    def verify(self, ir: Any, context: PassContext) -> bool:
        """Verify IR invariants. Return True if valid."""
        pass
    
    def run(self, ir: Any, context: PassContext) -> PassResult:
        """Execute verification pass."""
        try:
            valid = self.verify(ir, context)
            
            if valid:
                return PassResult(status="success", ir=ir)
            else:
                return PassResult(status="failed", ir=ir,
                    errors=["Verification failed"])
        except Exception as e:
            return PassResult(status="failed", ir=ir, errors=[str(e)])
```

---

## 3. Pass Lifecycle

### 3.1 Execution Phases

```
┌─────────────────────────────────────────────────────────┐
│ 1. Initialization                                       │
│    ├─ Create pass instance                             │
│    ├─ Set preconditions/postconditions                 │
│    └─ Register dependencies                            │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 2. Dependency Resolution                               │
│    ├─ Topologically sort passes                        │
│    ├─ Verify all dependencies are available            │
│    └─ Detect circular dependencies                     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 3. Precondition Check                                  │
│    ├─ Verify IR meets preconditions                    │
│    └─ Abort if preconditions fail                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 4. Execution                                           │
│    ├─ Run pass.run(ir, context)                        │
│    ├─ Emit structured logs                            │
│    └─ Collect metrics                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 5. Postcondition Check                                 │
│    ├─ Verify transformed IR meets postconditions       │
│    └─ Abort if postconditions fail                     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 6. Result Reporting                                    │
│    ├─ Return PassResult                                │
│    ├─ Update IR for next pass                          │
│    └─ Log errors/warnings                              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Preconditions & Postconditions

```python
# Example: ShapeInferencePass
class ShapeInferencePass(AnalysisPass):
    def __init__(self):
        super().__init__("shape_inference")
        
        # Preconditions: IR must have valid kernel definition
        self.preconditions = [
            "kernel_id is defined",
            "inputs have concrete or symbolic shapes",
            "outputs are declared"
        ]
        
        # Postconditions: all shapes must be inferred
        self.postconditions = [
            "all output shapes are concrete or symbolic",
            "no unresolved shape variables",
            "shape constraints are satisfied"
        ]
    
    def check_preconditions(self, ir: Any) -> bool:
        return (
            hasattr(ir, 'kernel_id') and
            hasattr(ir, 'inputs') and
            hasattr(ir, 'outputs')
        )
    
    def check_postconditions(self, ir: Any) -> bool:
        # All outputs must have inferred shapes
        return all(
            hasattr(output, 'shape') and output.shape is not None
            for output in ir.outputs
        )
```

---

## 4. Pass Composition

### 4.1 Pass Pipeline

```python
class PassPipeline:
    """Ordered sequence of passes."""
    
    def __init__(self, name: str):
        self.name = name
        self.passes: List[Pass] = []
    
    def add_pass(self, pass_: Pass) -> None:
        """Add pass to pipeline."""
        self.passes.append(pass_)
    
    def run(self, ir: Any, context: PassContext) -> PassResult:
        """Execute all passes in order."""
        current_ir = ir
        results = []
        
        for pass_ in self.passes:
            result = pass_.run(current_ir, context)
            results.append(result)
            
            if result.status == "failed":
                return PassResult(
                    status="failed",
                    ir=current_ir,
                    errors=[f"Pass {pass_.name} failed: {result.errors}"]
                )
            
            current_ir = result.ir
        
        return PassResult(status="success", ir=current_ir)
```

### 4.2 Pass Manager

```python
class PassManager:
    """Manages pass registration, dependency resolution, and execution."""
    
    def __init__(self):
        self.passes: Dict[str, Pass] = {}
        self.pipelines: Dict[str, PassPipeline] = {}
    
    def register_pass(self, pass_: Pass) -> None:
        """Register a pass."""
        self.passes[pass_.name] = pass_
    
    def create_pipeline(self, name: str, pass_names: List[str]) -> PassPipeline:
        """Create pipeline from pass names."""
        pipeline = PassPipeline(name)
        
        # Topologically sort passes by dependencies
        sorted_names = self._topological_sort(pass_names)
        
        for pass_name in sorted_names:
            if pass_name not in self.passes:
                raise ValueError(f"Pass {pass_name} not registered")
            pipeline.add_pass(self.passes[pass_name])
        
        return pipeline
    
    def _topological_sort(self, pass_names: List[str]) -> List[str]:
        """Sort passes by dependencies."""
        # Implementation: Kahn's algorithm or DFS
        pass
```

---

## 5. Data Flow & Dependencies

### 5.1 Pass Dependencies

```python
# Example: FusionPass depends on ShapeInferencePass
class FusionPass(TransformPass):
    def __init__(self):
        super().__init__("fusion")
        
        # Must run after shape inference
        self.dependencies = ["shape_inference"]
        
        # Preconditions: all shapes must be known
        self.preconditions = [
            "all shapes are concrete or symbolic",
            "no unresolved shape variables"
        ]
```

### 5.2 Analysis Result Propagation

```python
# Analysis passes produce results that transform passes consume
class PassContext:
    def __init__(self):
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
    
    def get_analysis(self, pass_name: str) -> Dict[str, Any]:
        """Retrieve analysis results from previous pass."""
        return self.analysis_results.get(pass_name, {})
    
    def set_analysis(self, pass_name: str, results: Dict[str, Any]) -> None:
        """Store analysis results for downstream passes."""
        self.analysis_results[pass_name] = results

# Usage in transform pass
class TilingPass(TransformPass):
    def transform(self, ir: Any, context: PassContext) -> Any:
        # Get shape analysis from ShapeInferencePass
        shape_analysis = context.get_analysis("shape_inference")
        
        # Use shape information to guide tiling decisions
        for kernel in ir.kernels:
            shapes = shape_analysis.get(kernel.id, {})
            # ... apply tiling based on shapes
        
        return ir
```

---

## 6. Error Handling

### 6.1 Error Categories

```python
class PassError(Exception):
    """Base class for pass errors."""
    pass

class PreconditionError(PassError):
    """Precondition not met."""
    pass

class PostconditionError(PassError):
    """Postcondition not met."""
    pass

class DependencyError(PassError):
    """Dependency not available."""
    pass

class TransformError(PassError):
    """Transformation failed."""
    pass
```

### 6.2 Error Recovery

```python
class PassPipeline:
    def run(self, ir: Any, context: PassContext, 
            on_error: str = "abort") -> PassResult:
        """Execute pipeline with error handling.
        
        Args:
            on_error: "abort" (stop), "skip" (continue), "rollback" (restore)
        """
        current_ir = ir
        checkpoint_ir = ir
        
        for pass_ in self.passes:
            try:
                result = pass_.run(current_ir, context)
                
                if result.status == "failed":
                    if on_error == "abort":
                        return result
                    elif on_error == "skip":
                        continue
                    elif on_error == "rollback":
                        current_ir = checkpoint_ir
                        continue
                
                checkpoint_ir = current_ir
                current_ir = result.ir
            
            except Exception as e:
                if on_error == "abort":
                    return PassResult(status="failed", ir=current_ir, 
                        errors=[str(e)])
                elif on_error == "rollback":
                    current_ir = checkpoint_ir
        
        return PassResult(status="success", ir=current_ir)
```

---

## 7. Built-in Passes

### 7.1 Analysis Passes

| Pass | Input | Output | Purpose |
|:-----|:------|:-------|:--------|
| `shape_inference` | SemanticIR | Shape analysis | Infer output shapes from inputs |
| `constraint_analysis` | SemanticIR | Constraints | Extract and validate constraints |
| `dataflow_analysis` | StrategyIR | Data flow graph | Build data dependency graph |
| `memory_analysis` | ScheduleIR | Memory usage | Estimate memory footprint |

### 7.2 Transform Passes

> **Note (K-H5.1, 2026-07-29):** `schedule_lowering` / `instruction_lowering`
> below are structurally implemented but produce **skeleton** ScheduleIR /
> InstructionIR — see the honest-downgrade status note in
> `arke-ir-spec.md §3.4`. Backends currently do their own scheduling; these
> passes do not yet drive codegen. The rows describe the target pass contract.

| Pass | Input | Output | Purpose |
|:-----|:------|:-------|:--------|
| `fusion` | StrategyIR | StrategyIR | Fuse adjacent operations |
| `tiling` | StrategyIR | StrategyIR | Apply tiling decisions |
| `memory_optimization` | StrategyIR | StrategyIR | Optimize memory layout |
| `schedule_lowering` | StrategyIR | ScheduleIR | Lower to schedule IR |
| `instruction_lowering` | ScheduleIR | InstructionIR | Lower to instruction IR |

### 7.3 Verification Passes

| Pass | Input | Purpose |
|:-----|:------|:--------|
| `ssa_validator` | Any IR | Verify SSA form |
| `constraint_validator` | SemanticIR | Verify constraints |
| `correctness_validator` | InstructionIR | Verify correctness |

---

## 8. Custom Pass Development

### 8.1 Template

```python
from arke.compiler.pass_infrastructure import TransformPass, PassContext, PassResult

class MyCustomPass(TransformPass):
    """Custom pass template."""
    
    def __init__(self):
        super().__init__("my_custom_pass")
        
        # Define preconditions
        self.preconditions = [
            "IR has valid kernel definition",
            "All shapes are inferred"
        ]
        
        # Define postconditions
        self.postconditions = [
            "IR is still valid",
            "No new shape variables introduced"
        ]
        
        # Define dependencies
        self.dependencies = ["shape_inference"]
    
    def check_preconditions(self, ir: Any) -> bool:
        # Implement precondition check
        return True
    
    def check_postconditions(self, ir: Any) -> bool:
        # Implement postcondition check
        return True
    
    def transform(self, ir: Any, context: PassContext) -> Any:
        """Implement transformation logic."""
        # Get analysis results from previous passes
        shape_analysis = context.get_analysis("shape_inference")
        
        # Apply transformation
        transformed_ir = self._apply_transformation(ir, shape_analysis)
        
        # Return transformed IR
        return transformed_ir
    
    def _apply_transformation(self, ir: Any, analysis: Dict) -> Any:
        # Implementation details
        pass
```

### 8.2 Registration

```python
# In arke/compiler/passes/__init__.py
from arke.compiler.pass_infrastructure import PassManager
from .my_custom_pass import MyCustomPass

pass_manager = PassManager()
pass_manager.register_pass(MyCustomPass())
```

---

## 9. Pass Registry

### 9.1 Built-in Pass Registry

```python
# arke/compiler/passes/registry.py
from arke.compiler.pass_infrastructure import PassManager

def create_default_pass_manager() -> PassManager:
    """Create pass manager with all built-in passes."""
    pm = PassManager()
    
    # Analysis passes
    pm.register_pass(ShapeInferencePass())
    pm.register_pass(ConstraintAnalysisPass())
    pm.register_pass(DataFlowAnalysisPass())
    pm.register_pass(MemoryAnalysisPass())
    
    # Transform passes
    pm.register_pass(FusionPass())
    pm.register_pass(TilingPass())
    pm.register_pass(MemoryOptimizationPass())
    pm.register_pass(ScheduleLoweringPass())
    pm.register_pass(InstructionLoweringPass())
    
    # Verification passes
    pm.register_pass(SSAValidatorPass())
    pm.register_pass(ConstraintValidatorPass())
    pm.register_pass(CorrectnessValidatorPass())
    
    return pm
```

### 9.2 Standard Pipelines

```python
# arke/compiler/passes/pipelines.py

# Pipeline 1: Semantic IR → Strategy IR
SEMANTIC_TO_STRATEGY_PIPELINE = [
    "shape_inference",
    "constraint_analysis",
    "dataflow_analysis"
]

# Pipeline 2: Strategy IR → Schedule IR
STRATEGY_TO_SCHEDULE_PIPELINE = [
    "fusion",
    "tiling",
    "memory_optimization",
    "schedule_lowering"
]

# Pipeline 3: Schedule IR → Instruction IR
SCHEDULE_TO_INSTRUCTION_PIPELINE = [
    "instruction_lowering"
]

# Pipeline 4: Full compilation
FULL_COMPILATION_PIPELINE = (
    SEMANTIC_TO_STRATEGY_PIPELINE +
    STRATEGY_TO_SCHEDULE_PIPELINE +
    SCHEDULE_TO_INSTRUCTION_PIPELINE
)

# Pipeline 5: Verification
VERIFICATION_PIPELINE = [
    "ssa_validator",
    "constraint_validator",
    "correctness_validator"
]
```

---

## 10. Examples

### 10.1 Shape Inference Pass

```python
class ShapeInferencePass(AnalysisPass):
    """Infer output shapes from input shapes and semantic rules."""
    
    def __init__(self):
        super().__init__("shape_inference")
        self.preconditions = ["kernel has inputs with shapes"]
        self.postconditions = ["all outputs have inferred shapes"]
    
    def analyze(self, ir: Any, context: PassContext) -> Dict[str, Any]:
        """Infer shapes for all outputs."""
        results = {}
        
        for kernel in ir.kernels:
            kernel_results = {}
            
            # Get semantic rules for this kernel
            rules = kernel.semantic_rules
            
            # Apply shape inference rules
            for rule in rules:
                if rule.type == "shape_inference":
                    # Execute shape inference rule
                    output_shapes = self._execute_rule(
                        rule, kernel.inputs
                    )
                    kernel_results.update(output_shapes)
            
            results[kernel.id] = kernel_results
        
        return results
    
    def _execute_rule(self, rule: Any, inputs: List) -> Dict:
        # Implementation: execute shape inference rule
        pass
```

### 10.2 Fusion Pass

```python
class FusionPass(TransformPass):
    """Fuse adjacent operations in strategy IR."""
    
    def __init__(self):
        super().__init__("fusion")
        self.dependencies = ["shape_inference"]
        self.preconditions = ["all shapes are inferred"]
        self.postconditions = ["fused operations are valid"]
    
    def transform(self, ir: Any, context: PassContext) -> Any:
        """Fuse adjacent operations."""
        # Get shape analysis
        shape_analysis = context.get_analysis("shape_inference")
        
        # Identify fusion opportunities
        fusion_groups = self._identify_fusion_groups(ir, shape_analysis)
        
        # Apply fusions
        for group in fusion_groups:
            ir = self._apply_fusion(ir, group)
        
        return ir
    
    def _identify_fusion_groups(self, ir: Any, analysis: Dict) -> List:
        # Implementation: identify fusible operation groups
        pass
    
    def _apply_fusion(self, ir: Any, group: List) -> Any:
        # Implementation: fuse operations in group
        pass
```

---

## References

- `docs/spec/arke-ir-spec.md` — IR layer definitions
- `docs/architecture/arke-compiler-infrastructure.md` — Compiler architecture
- `docs/architecture/e2e-flow.md` — End-to-end flow

---

**End of Pass Infrastructure Specification**
