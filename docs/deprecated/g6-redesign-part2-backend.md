# G6 重构方案二：多级后端扩展性设计

> **文档目的：** 为 Arke 编译器提供清晰的多级后端抽象架构，使同一套 SemanticIR + StrategyIR 能路由到 Triton（Phase 1-2）、MLIR（Phase 2-3）、LLVM IR（Phase 3+）等不同后端，同时为 StrategyIR 建立后端无关的分层设计。
>
> **版本：** 1.0  
> **日期：** 2026-04-06  
> **关联：** G6 重构（g6-redesign-overview.md）、IR 规范（arke-ir-spec-v1.md）、IR-MLIR 映射（ir-mlir-mapping.md）、Phase 2/3 审视（phase2-3-review.md）

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [Backend 抽象层设计](#2-backend-抽象层设计)
3. [StrategyIR 的后端无关性分析](#3-strategyir-的后端无关性分析)
4. [StrategyIR Level 分层设计](#4-strategyir-level-分层设计)
5. [Template Engine → Lowering Engine 演进路径](#5-template-engine--lowering-engine-演进路径)
6. [Arke Lang/IR Spec 对后端扩展的支持](#6-arke-langir-spec-对后端扩展的支持)
7. [具体实现拆解](#7-具体实现拆解)
8. [向后兼容保障](#8-向后兼容保障)
9. [风险与权衡](#9-风险与权衡)

---

## 1. 背景与动机

### 1.1 当前架构的限制

Phase 1 已验证 Arke 的核心假设：结构化协议让 LLM kernel 更正确（100% vs 83%），@rationale 让决策可追溯。但当前架构存在几个关键限制：

**硬耦合问题：** `ArkePipeline.run()` 中，codegen 路径直接硬编码了 `from arke.backend.triton_backend import TritonBackend`，无法在不修改 pipeline 核心逻辑的情况下切换后端。

**StrategyIR 的 Triton 污染：** `launch_config` 的 `num_warps`、`num_stages` 是 Triton/NVIDIA 特有概念，`autotune` 的 config 格式也是 Triton-specific 的。当 Phase 2 对接 MLIR 时，这些字段对其他后端毫无意义。

**Template Engine 的角色模糊：** `TritonTemplateEngine` 既做了模板选择（策略层面的逻辑），又做了 Jinja2 渲染（实现层面的逻辑），两者没有清晰的边界，导致未来替换 lowering 策略时会牵一发而动全身。

**target_hw 未被利用：** `StrategyIR.target_hw` 存在但未被用于后端路由，只是一个装饰性字段。

### 1.2 目标软件栈

AI 硬件算子的完整软件栈是：

```
Python/Triton/TVM (高阶编程)
        ↓
   MLIR/TIR dialects
        ↓
   CUDA/AscendC/HIP
        ↓
   LLVM IR
        ↓
   PTX/ISA
```

Arke 通过 3 个 Stage 逐步下探这个栈：

| Stage | 后端 | 切入层 | 价值 |
|:-----:|:-----|:-------|:-----|
| **Phase 1** | Triton → NVIDIA | Python 层 | 快速验证，LLM 友好 |
| **Phase 2** | Triton → Ascend 或 MLIR | Triton/MLIR 层 | 摆脱 dispatch overhead，多硬件 |
| **Phase 3** | MLIR → LLVM IR | LLVM 层 | 最大灵活性，LLM 控制 Level 2 决策 |

G6 的重构必须在架构上为这个演进做好准备，而不是等到 Phase 3 再重构一次。**架构成本最低的时机是现在，因为代码库还较小，测试覆盖完整（397+ tests），有安全网。**

### 1.3 设计原则

1. **SemanticIR 不变**：SemanticIR 已经是后端无关的（描述"计算什么"），不需要改动
2. **StrategyIR 分层而非重写**：Level 1 决策保持兼容，Level 2/3 作为可选 extension sections
3. **Backend Protocol 先行**：先定义接口，再把现有 TritonBackend 适配进去
4. **渐进迁移**：Phase 1 的所有 .ak 文件和测试不能破坏
5. **显式路由**：`target_hw` 字段驱动后端选择，不靠运行时猜测

---

## 2. Backend 抽象层设计

### 2.1 BackendArtifact 类型层次

在定义 Backend Protocol 之前，需要先定义"翻译结果"的类型层次。不同后端的中间产物格式差异极大：

```python
# arke/backend/artifact.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendArtifact:
    """后端翻译的中间产物基类。
    
    所有后端的 translate() 都返回此类的子类。
    上层代码可以按后端特定类型处理，也可以只持有基类引用。
    """
    backend_id: str           # 产物归属的后端标识，e.g. "triton", "mlir", "llvm"
    kernel_id: str            # 对应的 kernel_id
    target_hw: str            # 目标硬件，e.g. "nvidia_ampere"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TritonArtifact(BackendArtifact):
    """Triton 后端的翻译产物：Python 源码字符串。
    
    由 TritonTemplateEngine 生成，传入 TritonCompiler 编译。
    """
    source_code: str = ""            # 完整的 Triton Python 源码
    template_name: str = ""          # 使用的 Jinja2 模板名称（用于调试）
    primary_op: str = ""             # 主算子名称


@dataclass
class MLIRArtifact(BackendArtifact):
    """MLIR 后端的翻译产物：MLIR 文本表示（或 Python MLIR API 描述）。
    
    Phase 2 预留。包含 linalg dialect 计算 + transform dialect 优化序列。
    """
    mlir_module: str = ""            # MLIR textual form (mlir-print 格式)
    transform_sequence: str = ""     # transform dialect ops 序列
    dialects_required: list[str] = field(default_factory=list)  # e.g. ["linalg", "transform", "gpu"]


@dataclass
class LLVMArtifact(BackendArtifact):
    """LLVM IR 后端的翻译产物：LLVM IR 文本表示。
    
    Phase 3 预留。由 Loop Nest IR 经 lowering passes 生成。
    """
    llvm_ir: str = ""                # LLVM IR textual form (.ll 文件格式)
    target_triple: str = ""          # e.g. "nvptx64-nvidia-cuda"
    data_layout: str = ""            # LLVM data layout string


@dataclass
class CompiledKernel:
    """编译后的可执行 kernel（后端无关的包装器）。
    
    持有运行时可调用的对象及其元数据。
    """
    backend_id: str
    kernel_id: str
    callable: Any                    # 实际可调用对象（后端特定）
    source_artifact: BackendArtifact | None = None
    compile_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    # 性能估算（可选，由后端填充）
    estimated_tflops: float | None = None
    register_count: int | None = None


@dataclass  
class ProfileResult:
    """内核性能分析结果（后端无关）。"""
    latency_us: float
    tflops: float | None = None
    roofline_efficiency: float | None = None
    memory_bandwidth_gb_s: float | None = None
    backend_id: str = ""
    raw_stats: dict[str, Any] = field(default_factory=dict)
```

### 2.2 ArkeBackend Protocol 定义

```python
# arke/backend/base.py  （重写现有 base.py）

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import torch

from arke.backend.artifact import (
    BackendArtifact, CompiledKernel, ProfileResult
)
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

logger = logging.getLogger(__name__)


@runtime_checkable
class ArkeBackend(Protocol):
    """Arke 后端协议。
    
    所有后端必须实现此 Protocol 的三个核心方法：
    - translate():  SemanticIR + StrategyIR → BackendArtifact（IR 翻译）
    - compile():    BackendArtifact → CompiledKernel（代码编译/JIT）
    - execute():    CompiledKernel + inputs → torch.Tensor（内核执行）
    
    可选实现 profile() 以支持性能分析。
    
    设计说明：
    - translate() 是纯 IR 变换，不依赖运行时，可在无 GPU 环境执行
    - compile() 可能需要运行时（JIT 编译），但不执行 kernel
    - execute() 需要真实硬件
    - 三阶段分离使得测试、缓存、分布式编译各自独立
    """
    
    # 后端标识符，必须唯一，e.g. "triton", "mlir", "llvm"
    name: str
    
    def translate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
    ) -> BackendArtifact:
        """将 SemanticIR + StrategyIR 翻译为后端特定的中间产物。
        
        此方法是纯 IR 变换：
        - 不执行任何编译或运行
        - 不依赖 GPU 或运行时
        - 相同输入必须产生相同输出（deterministic）
        
        Args:
            semantic: 计算语义，描述"计算什么"
            strategy: 优化策略，描述"如何优化"
            
        Returns:
            后端特定的 BackendArtifact 子类实例
            
        Raises:
            TranslationError: 当 IR 无法被此后端表达时
        """
        ...

    def compile(
        self,
        artifact: BackendArtifact,
    ) -> CompiledKernel:
        """将后端产物编译为可执行 kernel。
        
        对于 Triton 后端：exec_compile Triton Python → PTX（JIT）
        对于 MLIR 后端：MLIR module → LLVM IR → PTX（AOT 或 JIT）
        对于 LLVM 后端：LLVM IR → PTX → cubin（AOT）
        
        Args:
            artifact: translate() 的返回值
            
        Returns:
            CompiledKernel，其中 callable 是可执行的 Python 对象
            
        Raises:
            CompileError: 编译失败时
        """
        ...

    def execute(
        self,
        kernel: CompiledKernel,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """执行编译后的 kernel。
        
        Args:
            kernel: compile() 的返回值
            inputs: 输入张量字典，key 对应 SemanticIR.params[].name
            
        Returns:
            输出张量字典，通常包含 "output" key
            
        Raises:
            ExecutionError: 运行时错误（如内存不足）
        """
        ...

    def profile(
        self,
        kernel: CompiledKernel,
        inputs: dict[str, torch.Tensor],
        warmup: int = 5,
        runs: int = 20,
    ) -> ProfileResult:
        """（可选）分析 kernel 性能。
        
        默认实现：执行 warmup + runs 次，取平均延迟。
        后端可覆盖以使用硬件计数器等更精确的方法。
        """
        ...

    def supports(self, target_hw: str) -> bool:
        """查询此后端是否支持给定的目标硬件。
        
        Args:
            target_hw: StrategyIR.target_hw 的值，e.g. "nvidia_ampere"
            
        Returns:
            True 如果此后端能为该硬件生成代码
        """
        ...
```

**为何选择 Protocol 而非 ABC：**

`runtime_checkable` Protocol 允许我们用 `isinstance(backend, ArkeBackend)` 进行运行时检查，同时不强制继承（duck typing 友好）。这对于未来通过插件机制注册第三方后端很重要。如果需要共享基础设施（如 `profile()` 的默认实现），可以提供一个 `BaseBackend` ABC 作为可选基类，但 Protocol 定义接口契约。

### 2.3 BackendRegistry（后端发现与注册）

```python
# arke/backend/registry.py

from __future__ import annotations

import importlib
import logging
from typing import Callable, Type

from arke.backend.base import ArkeBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """全局后端注册表。
    
    职责：
    1. 维护 backend_id → backend_class 的映射
    2. 根据 target_hw 路由到合适的后端
    3. 支持插件式后端注册（第三方可注册自己的后端）
    
    线程安全说明：注册操作在模块 import 时发生（单线程），
    查询操作是只读的，无需加锁。
    """
    
    _backends: dict[str, type] = {}
    
    # target_hw 前缀 → 默认后端 ID 的映射
    # 当同一 target_hw 有多个后端时，用此表决定默认
    _hw_to_backend: dict[str, str] = {
        "nvidia_": "triton",      # nvidia_ampere, nvidia_hopper, ...
        "amd_": "triton",         # amd_mi300, ... (ROCm Triton)
        "ascend_": "mlir",        # ascend_910b, ... (AscendC via MLIR)
        "cpu_": "llvm",           # cpu_x86, cpu_arm, ...
    }
    
    @classmethod
    def register(cls, backend_id: str, backend_class: type) -> None:
        """注册一个后端实现。
        
        Args:
            backend_id: 唯一标识符，e.g. "triton", "mlir"
            backend_class: 实现 ArkeBackend Protocol 的类
            
        Example:
            @register_backend("my_backend")
            class MyBackend:
                name = "my_backend"
                ...
        """
        if backend_id in cls._backends:
            logger.warning(
                "Backend '%s' already registered; overwriting with %s",
                backend_id, backend_class.__name__
            )
        cls._backends[backend_id] = backend_class
        logger.debug("Registered backend: %s → %s", backend_id, backend_class.__name__)
    
    @classmethod
    def get(cls, backend_id: str) -> type:
        """按 ID 获取后端类。
        
        Raises:
            KeyError: backend_id 未注册
        """
        if backend_id not in cls._backends:
            available = list(cls._backends.keys())
            raise KeyError(
                f"Backend '{backend_id}' not registered. "
                f"Available backends: {available}"
            )
        return cls._backends[backend_id]
    
    @classmethod
    def create(cls, backend_id: str, **kwargs) -> ArkeBackend:
        """实例化一个后端。
        
        Args:
            backend_id: 注册的后端 ID
            **kwargs: 传给后端构造函数的参数
        """
        backend_class = cls.get(backend_id)
        return backend_class(**kwargs)
    
    @classmethod
    def for_target(cls, target_hw: str, preferred: str | None = None) -> ArkeBackend:
        """根据 target_hw 自动路由到合适的后端。
        
        路由优先级：
        1. 显式指定 preferred 后端（如果已注册且支持该硬件）
        2. _hw_to_backend 前缀匹配
        3. 第一个声明支持该 target_hw 的后端
        4. 默认使用 "triton"（向后兼容）
        
        Args:
            target_hw: e.g. "nvidia_ampere", "ascend_910b"
            preferred: 显式指定的后端 ID（可选）
        """
        # 1. 显式指定
        if preferred and preferred in cls._backends:
            backend = cls.create(preferred)
            if backend.supports(target_hw):
                return backend
            logger.warning(
                "Preferred backend '%s' does not support '%s'; falling back",
                preferred, target_hw
            )
        
        # 2. 前缀匹配
        for prefix, backend_id in cls._hw_to_backend.items():
            if target_hw.startswith(prefix) and backend_id in cls._backends:
                return cls.create(backend_id)
        
        # 3. 遍历已注册后端，找支持该硬件的
        for backend_id, backend_class in cls._backends.items():
            backend = backend_class()
            if backend.supports(target_hw):
                logger.debug(
                    "Routing target_hw='%s' to backend='%s' via supports() check",
                    target_hw, backend_id
                )
                return backend
        
        # 4. Fallback：triton（Phase 1 向后兼容）
        if "triton" in cls._backends:
            logger.warning(
                "No backend found for target_hw='%s'; falling back to triton",
                target_hw
            )
            return cls.create("triton")
        
        raise RuntimeError(
            f"No suitable backend for target_hw='{target_hw}'. "
            f"Registered backends: {list(cls._backends.keys())}"
        )
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """返回所有已注册的后端 ID 列表。"""
        return list(cls._backends.keys())
    
    @classmethod
    def _auto_discover(cls) -> None:
        """自动发现并加载内置后端。
        
        扫描 arke/backend/ 下所有 *_backend.py 文件，
        import 它们以触发 @register_backend 装饰器。
        """
        import pkgutil
        import arke.backend as backend_pkg
        
        for finder, name, ispkg in pkgutil.iter_modules(backend_pkg.__path__):
            if name.endswith("_backend") and name != "base":
                try:
                    importlib.import_module(f"arke.backend.{name}")
                    logger.debug("Auto-discovered backend module: arke.backend.%s", name)
                except ImportError as e:
                    logger.debug("Backend module arke.backend.%s not available: %s", name, e)


def register_backend(backend_id: str) -> Callable[[type], type]:
    """装饰器：注册后端类到全局注册表。
    
    Usage:
        @register_backend("triton")
        class TritonBackend:
            name = "triton"
            ...
    """
    def decorator(cls: type) -> type:
        BackendRegistry.register(backend_id, cls)
        return cls
    return decorator


# 模块级注册函数（兼容当前 triton_backend.py 中的调用方式）
def register_backend_fn(backend_id: str, backend_class: type) -> None:
    """函数式注册接口（兼容旧代码）。"""
    BackendRegistry.register(backend_id, backend_class)
```

### 2.4 当前三个后端的定位

#### TritonBackend（Phase 1-2，当前实现）

```python
# arke/backend/triton_backend.py  （适配现有实现）

from arke.backend.base import ArkeBackend
from arke.backend.artifact import BackendArtifact, TritonArtifact, CompiledKernel, ProfileResult
from arke.backend.registry import register_backend
from arke.backend.triton_template_engine import TritonTemplateEngine
from arke.backend.compiler import TritonCompiler
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

# 支持的目标硬件前缀
_SUPPORTED_HW_PREFIXES = ("nvidia_", "amd_")


@register_backend("triton")
class TritonBackend:
    """Triton 代码生成后端，支持 NVIDIA GPU（Phase 1）和 AMD GPU（Phase 2）。
    
    实现说明：
    - translate()：委托给 TritonTemplateEngine，生成 TritonArtifact
    - compile()：委托给 TritonCompiler，JIT 编译 Triton Python 代码
    - execute()：调用 TritonCompiler.run()
    - TritonTemplateEngine 成为此后端的内部实现细节，不暴露给上层
    """
    
    name = "triton"
    
    def __init__(self) -> None:
        self._engine = TritonTemplateEngine()
        self._compiler = TritonCompiler()
    
    def translate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
    ) -> TritonArtifact:
        """SemanticIR + StrategyIR → Triton Python 源码。"""
        template_name, primary_op = self._engine._select_template(semantic)
        source_code = self._engine.translate(semantic, strategy)
        return TritonArtifact(
            backend_id="triton",
            kernel_id=semantic.kernel_id,
            target_hw=strategy.target_hw,
            source_code=source_code,
            template_name=template_name,
            primary_op=primary_op,
        )
    
    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Triton Python 源码 → 编译结果。"""
        assert isinstance(artifact, TritonArtifact), \
            f"TritonBackend.compile() expects TritonArtifact, got {type(artifact)}"
        
        import time
        t0 = time.perf_counter()
        compile_result = self._compiler.compile(artifact.source_code)
        compile_time_ms = (time.perf_counter() - t0) * 1000
        
        return CompiledKernel(
            backend_id="triton",
            kernel_id=artifact.kernel_id,
            callable=compile_result,
            source_artifact=artifact,
            compile_time_ms=compile_time_ms,
            success=compile_result.success,
            error_message=getattr(compile_result, "error", "") or "",
        )
    
    def execute(
        self,
        kernel: CompiledKernel,
        inputs: dict,
    ) -> dict:
        """执行编译后的 Triton kernel。"""
        assert kernel.backend_id == "triton"
        output = self._compiler.run(kernel.callable, inputs)
        return {"output": output}
    
    def profile(
        self,
        kernel: CompiledKernel,
        inputs: dict,
        warmup: int = 5,
        runs: int = 20,
    ) -> ProfileResult:
        """使用 TritonCompiler.profile() 分析性能。"""
        assert isinstance(kernel.source_artifact, TritonArtifact)
        result = self._compiler.profile(
            kernel.source_artifact.source_code, inputs,
            warmup=warmup, runs=runs
        )
        return ProfileResult(
            latency_us=result.latency_us,
            tflops=result.tflops,
            roofline_efficiency=result.roofline_efficiency,
            backend_id="triton",
        )
    
    def supports(self, target_hw: str) -> bool:
        """支持所有 nvidia_ 和 amd_ 前缀的硬件。"""
        return any(target_hw.startswith(p) for p in _SUPPORTED_HW_PREFIXES)
```

#### MLIRBackend（Phase 2，接口预留）

```python
# arke/backend/mlir_backend.py  （新增，Phase 2 骨架）

from arke.backend.base import ArkeBackend
from arke.backend.artifact import BackendArtifact, MLIRArtifact, CompiledKernel, ProfileResult
from arke.backend.registry import register_backend
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

_SUPPORTED_HW_PREFIXES = ("nvidia_", "amd_", "ascend_")


@register_backend("mlir")
class MLIRBackend:
    """MLIR 代码生成后端（Phase 2 预留）。
    
    实现路径：
    - translate()：SemanticIR → linalg dialect，StrategyIR → transform dialect
    - compile()：mlir-opt + mlir-translate → LLVM IR → PTX/cubin
    - execute()：加载 cubin，通过 CUDA driver API 执行
    
    Phase 2 实现参考：ir-mlir-mapping.md §2-4
    """
    
    name = "mlir"
    
    def translate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
    ) -> MLIRArtifact:
        """SemanticIR + StrategyIR → MLIR module（linalg + transform dialect）。
        
        Phase 2 实现步骤：
        1. 遍历 semantic.nodes，按 ir-mlir-mapping.md §2.3 生成 linalg ops
        2. 遍历 strategy.decisions（Level 1 + Level 2），按 §3.1 生成 transform ops
        3. 处理 FusionGroup → transform.structured.fuse_into_containing_op
        4. @rationale → transform.annotate "arke.rationale"
        """
        raise NotImplementedError(
            "MLIRBackend is reserved for Phase 2. "
            "See docs/spec/ir-mlir-mapping.md for the implementation plan."
        )
    
    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """MLIR module → 可执行 kernel。
        
        Phase 2 实现步骤：
        1. mlir-opt：linalg → scf → gpu → nvvm/rocdl/spirv
        2. mlir-translate：nvvm → LLVM IR
        3. llc / ptxas：LLVM IR → PTX → cubin
        4. CUDA driver API：加载 cubin，返回 CUfunction
        """
        raise NotImplementedError("MLIRBackend.compile() reserved for Phase 2")
    
    def execute(self, kernel: CompiledKernel, inputs: dict) -> dict:
        """通过 CUDA driver API 执行编译后的 kernel。"""
        raise NotImplementedError("MLIRBackend.execute() reserved for Phase 2")
    
    def supports(self, target_hw: str) -> bool:
        return any(target_hw.startswith(p) for p in _SUPPORTED_HW_PREFIXES)
```

#### LLVMBackend（Phase 3，接口预留）

```python
# arke/backend/llvm_backend.py  （新增，Phase 3 骨架）

from arke.backend.base import ArkeBackend
from arke.backend.artifact import BackendArtifact, LLVMArtifact, CompiledKernel, ProfileResult
from arke.backend.registry import register_backend
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

_SUPPORTED_HW_PREFIXES = ("nvidia_", "amd_", "ascend_", "cpu_")


@register_backend("llvm")
class LLVMBackend:
    """LLVM IR 直接路径后端（Phase 3 预留）。"""
    
    name = "llvm"
    
    def translate(self, semantic, strategy):
        raise NotImplementedError("LLVMBackend reserved for Phase 3")
    
    def compile(self, artifact):
        raise NotImplementedError("LLVMBackend.compile() reserved for Phase 3")
    
    def execute(self, kernel, inputs):
        raise NotImplementedError("LLVMBackend.execute() reserved for Phase 3")
    
    def supports(self, target_hw):
        return any(target_hw.startswith(p) for p in _SUPPORTED_HW_PREFIXES)
```

---

## 3. StrategyIR 的后端无关性分析

### 3.1 Decision Kind 分类

| Kind | 后端无关？ | Triton | MLIR | LLVM |
|------|----------|--------|------|------|
| `tile` | ✅ | BLOCK_SIZE constexpr | `transform.structured.tile_using_for` | loop tiling pass |
| `reorder` | ✅ | loop nest order | `transform.structured.interchange` | loop interchange |
| `fuse` | ✅ | epilogue fusion | `transform.structured.fuse_into_containing_op` | loop fusion |
| `parallel` | ⚠️ 部分 | program_id mapping | `transform.structured.tile_using_forall` | OpenMP/GPU mapping |
| `vectorize` | ⚠️ 部分 | implicit (Triton handles) | `transform.structured.vectorize` | LLVM vectorize pass |
| `place` | ⚠️ 部分 | shared/register annotation | `gpu.workgroup` / `gpu.private` | alloca / shmem |
| `launch_config` | ❌ 后端特定 | num_warps, num_stages | gpu.launch params | N/A |
| `unroll` | ✅ | `tl.static_range` | `transform.loop.unroll` | LLVM unroll |
| `algorithm` | ✅ | 影响 template 选择 | 影响 lowering 路径 | 影响 lowering 路径 |
| `autotune` | ⚠️ 部分 | `@triton.autotune` | multi-version + dispatch | multi-version + dispatch |

### 3.2 设计：Core Decisions vs Backend-Specific Extensions

```python
@dataclass
class Decision:
    kind: str           # 后端无关的 core kind
    params: dict        # 后端无关的参数
    rationale: Rationale | None = None
    step: int = 0
    level: int = 1
    
    # 新增：后端特定扩展
    backend_hints: dict[str, dict] = field(default_factory=dict)
    # e.g., {"triton": {"num_warps": 4}, "mlir": {"vector_size": 4}}
```

**规则：**
- `kind` 和 `params` 是后端无关的（tile, fuse, algorithm 等）
- `backend_hints` 是后端特定的调优参数
- Backend 只消费它认识的 hints，忽略其他

---

## 4. StrategyIR Level 分层设计

### 4.1 三层 Decision 模型

```
Level 1 — Algorithm（算法层）[Phase 1 已有]
  → tile, fuse, algorithm, reorder
  → 后端无关，LLM Agent 主要操作的层

Level 2 — Loop Nest（循环层）[Phase 2 引入]  
  → vectorize width, memory access pattern, prefetch distance
  → MLIR transform dialect 操作的层

Level 3 — Hardware Mapping（硬件层）[Phase 3 引入]
  → register allocation hints, instruction selection, bank conflict avoidance
  → LLVM pass 操作的层
```

### 4.2 数据模型

```python
@dataclass
class Decision:
    kind: str
    params: dict
    rationale: Rationale | None = None
    step: int = 0
    level: int = 1  # 已有字段，1=algo, 2=loop, 3=hw
    backend_hints: dict[str, dict] = field(default_factory=dict)
```

### 4.3 Level 消费规则

| Backend | 消费 Level 1 | 消费 Level 2 | 消费 Level 3 |
|---------|:---:|:---:|:---:|
| Triton | ✅ | 部分（autotune configs） | ❌ |
| MLIR | ✅ | ✅ | 部分（GPU mapping） |
| LLVM | ✅ | ✅ | ✅ |

**向后兼容：** Phase 1 只生成 Level 1 decisions。Level 2/3 decisions 在 Phase 2/3 引入时，不影响 Triton backend。

---

## 5. Template Engine → Lowering Engine 演进路径

```
Phase 1 (当前):
  SemanticIR + StrategyIR
    → TritonTemplateEngine._select_template()  [if/elif 路由]
    → Jinja2 render (template_hint from OpRegistry)
    → Triton Python source
    → triton.compile() + run()

Phase 2 (MLIR):
  SemanticIR + StrategyIR
    → MLIREmitter.emit()  [SemanticIR → linalg/tensor dialect]
    → StrategyApplicator.apply()  [StrategyIR L1+L2 → transform dialect]
    → MLIR pass pipeline  [canonicalize → bufferize → gpu-mapping]
    → LLVM IR → PTX (via MLIR's gpu-to-nvvm)

Phase 3 (LLVM):
  SemanticIR + StrategyIR
    → CustomLowering.lower()  [SemanticIR → Loop Nest IR]
    → ScheduleApplicator.apply()  [StrategyIR L1+L2+L3 → loop transforms]
    → LLVM IR Builder  [loop nest → LLVM IR]
    → LLVM backend → PTX/AMDGPU/etc.
```

**关键设计：** 每个 Stage 的 lowering 是 `ArkeBackend.translate()` 的实现。上层 pipeline 不关心后端细节——它只调用 `backend.translate(semantic, strategy)`。

---

## 6. Arke Lang/IR Spec 对后端扩展的支持

### 6.1 `target()` 语法扩展

```ak
// v1.0 — 只支持 nvidia_ampere
strategy my_strategy for target("nvidia_ampere") { ... }

// v1.1 — 扩展目标硬件
strategy my_strategy for target("nvidia_hopper") { ... }
strategy my_strategy for target("ascend_910b") { ... }
strategy my_strategy for target("amd_mi300") { ... }
```

target 字符串的命名规范：`<vendor>_<arch>`

### 6.2 backend_hints 在 .ak 中的表达

```ak
strategy my_strategy for target("nvidia_ampere") {
    tile(loop="M", factors=[64])
        @rationale("64-tile for tensor core alignment")
        @hint(triton={"num_warps": 4})         // 后端特定
        @hint(mlir={"vector_size": 4});        // 后端特定
}
```

**向后兼容：** `@hint` 是可选注解。没有 `@hint` 时，后端使用默认值。

### 6.3 StrategyIR JSON 扩展

```json
{
    "kind": "tile",
    "params": {"loop": "M", "factors": [64]},
    "rationale": {"text": "..."},
    "level": 1,
    "backend_hints": {
        "triton": {"num_warps": 4},
        "mlir": {"vector_size": 4}
    }
}
```

`from_dict()` 忽略未知的 `backend_hints` key（前向兼容）。

---

## 7. 具体实现拆解

| 文件 | 操作 | 工作内容 | 估时 | 优先级 |
|------|------|---------|------|-------|
| `arke/backend/artifact.py` | **新增** | BackendArtifact 类型层次 | 0.5d | P0 |
| `arke/backend/base.py` | **重写** | ArkeBackend Protocol + 辅助类 | 1d | P0 |
| `arke/backend/registry.py` | **新增** | BackendRegistry（发现+注册+路由）| 1d | P0 |
| `arke/backend/triton_backend.py` | **改造** | 适配 ArkeBackend Protocol | 1d | P0 |
| `arke/backend/mlir_backend.py` | **新增** | Phase 2 骨架（NotImplementedError）| 0.5d | P2 |
| `arke/backend/llvm_backend.py` | **新增** | Phase 3 骨架（NotImplementedError）| 0.5d | P3 |
| `arke/ir/strategy.py` | **扩展** | Decision.backend_hints 字段 | 0.5d | P1 |
| `arke/pipeline.py` | **改造** | 通过 BackendRegistry 路由后端 | 1d | P0 |
| `arke/parser/parser.py` | **扩展** | @hint 注解解析 | 0.5d | P1 |
| `tests/test_backend_*.py` | **新增** | 后端抽象层测试 | 1d | P0 |

**总计：~8 人天**（P0 部分 ~5 人天）

---

## 8. 向后兼容保障

| 变更 | 兼容策略 |
|------|---------|
| ArkeBackend Protocol | 现有 TritonBackend 适配 Protocol，公共 API 不变 |
| BackendRegistry | 默认注册 TritonBackend，`get_backend("nvidia_ampere")` 返回 Triton |
| Decision.backend_hints | 新增可选字段，默认空 dict |
| pipeline.py 路由 | 如果没有 BackendRegistry，fallback 到直接调用 TritonBackend |
| .ak target 扩展 | 新 target 字符串不影响旧文件 |

所有 422 tests + 6 skipped 在重构后必须继续 PASS。

---

## 9. 风险与权衡

| 风险 | 影响 | 缓解 |
|------|------|------|
| MLIR/LLVM backend 空壳增加维护成本 | 代码膨胀 | 骨架文件 <100 行，只定义接口 |
| backend_hints 滥用导致策略不可移植 | 策略耦合后端 | 规范：Level 1 decisions 不应依赖 backend_hints |
| BackendRegistry 引入 import-time 开销 | 启动变慢 | lazy import MLIR/LLVM backend |
| 多后端测试矩阵指数增长 | CI 时间 | Phase 1 只测 Triton，Phase 2 加 MLIR |

---

*Created: 2026-04-06 | Author: Arke Architecture Team*
