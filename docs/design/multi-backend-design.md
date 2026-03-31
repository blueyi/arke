# Arke — 多硬件后端抽象设计

> 优先支持：NVIDIA GPU (Triton) + 华为 Ascend A3 (AscendC)
> Date: 2026-03-31
> 补充 v2.1 plan 中缺失的多后端架构设计

---

## 一、问题陈述

v2.1 的设计虽然在 HW Profile 里提到了 Ascend A3，但整个工具链（codegen、验证、profiling）都只围绕 NVIDIA/Triton 设计。如果 Ascend 是一等公民，需要：

1. **统一的硬件抽象层（HAL）**——LLM 的决策接口不因硬件而变
2. **可插拔的后端（Backend）**——每种硬件一套 codegen + compiler + profiler
3. **硬件感知的合法动作枚举**——不同硬件合法的 tile/parallel/place 不同
4. **统一的验证接口**——不同硬件上的数值验证和性能验证走同一协议

---

## 二、两种硬件的编程模型对比

### NVIDIA GPU (Triton)

```
编程模型：SIMT（Single Instruction Multiple Threads）
并行层级：Grid → Block → Warp → Thread
内存层级：Global (HBM) → L2 → Shared Memory → Register
矩阵计算：Tensor Core (mma 指令)
编程语言：Triton (Python DSL) → PTX → CUDA Binary
关键特征：
  - Block-level 编程（Triton 隐藏了 thread 级别细节）
  - Shared memory 需显式管理
  - Warp-level 原语（tl.dot 自动映射到 tensor core）
  - 自动向量化
```

### Huawei Ascend A3 (AscendC)

```
编程模型：SPMD（Single Program Multiple Data）
并行层级：AI Core Group → AI Core → Cube/Vector/Scalar Unit
内存层级：Global (HBM) → L2 → L1 Buffer (1MB) → L0 Buffer → Register
矩阵计算：Cube Unit
编程语言：AscendC (C/C++ 扩展) → CANN Compiler → Ascend Binary
关键特征：
  - AI Core 级别编程（每个 core 独立执行相同程序）
  - 显式 DMA 数据搬运（global ↔ L1 ↔ L0）
  - Pipeline 编程范式（CopyIn → Compute → CopyOut 流水线）
  - Cube Unit 做矩阵乘（类似 Tensor Core，但编程接口不同）
  - L1 Buffer 容量大（1MB vs NVIDIA 48KB shared）
```

### 关键差异映射

| 概念 | NVIDIA (Triton) | Ascend (AscendC) | Arke 抽象 |
|------|-----------------|-------------------|-----------|
| **并行外层** | Grid of Blocks | 多 AI Core | `parallel_outer` |
| **并行内层** | Block of Threads | Core 内向量化 | `parallel_inner` |
| **快速存储** | Shared Memory (48KB) | L1 Buffer (1MB) | `fast_memory` |
| **最快存储** | Register File | L0 Buffer | `local_memory` |
| **矩阵单元** | Tensor Core | Cube Unit | `matrix_unit` |
| **向量单元** | CUDA Core | Vector Unit | `vector_unit` |
| **数据搬运** | 隐式（tl.load/store） | 显式 DMA | `data_transfer` |
| **同步** | __syncthreads | Pipeline Queue | `sync_primitive` |
| **编程粒度** | Block 级 | Core 级 | `compute_unit` |

---

## 三、硬件抽象层（HAL）设计

### 3.1 HW Profile 统一 Schema

```json
{
  "$id": "arke/hw-profile/v1",
  "type": "object",
  "required": ["name", "vendor", "arch", "compute_units", "memory_hierarchy", "constraints"],
  "properties": {
    "name": {"type": "string"},
    "vendor": {"type": "string", "enum": ["nvidia", "huawei", "amd", "intel"]},
    "arch": {"type": "string"},

    "compute_units": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "name": {"type": "string"},
        "matrix_unit": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "tile_shapes": {"type": "array"},
            "dtypes": {"type": "array"},
            "peak_tflops_f16": {"type": "number"}
          }
        },
        "vector_unit": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "width_bits": {"type": "integer"}
          }
        }
      }
    },

    "memory_hierarchy": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "arke_role"],
        "properties": {
          "name": {"type": "string"},
          "arke_role": {
            "type": "string",
            "enum": ["global", "l2_cache", "fast_memory", "local_memory", "register"]
          },
          "size_per_cu": {"type": "integer"},
          "size_total": {"type": "integer"},
          "bandwidth_gbps": {"type": "number"},
          "latency_cycles": {"type": "integer"},
          "programmable": {"type": "boolean"},
          "transfer_mode": {
            "type": "string",
            "enum": ["implicit", "explicit_dma", "explicit_load_store"]
          }
        }
      }
    },

    "constraints": {
      "type": "object",
      "properties": {
        "max_fast_memory_per_block": {"type": "integer"},
        "max_local_memory_per_thread": {"type": "integer"},
        "max_parallel_inner": {"type": "integer"},
        "warp_or_vector_width": {"type": "integer"}
      }
    },

    "programming_model": {
      "type": "string",
      "enum": ["simt_block", "spmd_core", "simd_lane"]
    },

    "data_transfer": {
      "type": "string",
      "enum": ["implicit", "explicit_dma", "explicit_load_store"]
    }
  }
}
```

### 3.2 具体 HW Profile

**nvidia_ampere_rtx3060.json：**
```json
{
  "name": "nvidia_ampere_rtx3060_laptop",
  "vendor": "nvidia",
  "arch": "ampere",
  "compute_units": {
    "count": 30,
    "name": "SM",
    "matrix_unit": {
      "name": "tensor_core",
      "tile_shapes": [[16, 8, 16], [16, 8, 8]],
      "dtypes": ["f16", "bf16", "tf32"],
      "peak_tflops_f16": 21.7
    },
    "vector_unit": {
      "name": "cuda_core",
      "width_bits": 32
    }
  },
  "memory_hierarchy": [
    {"name": "register",      "arke_role": "register",     "size_per_cu": 65536,  "latency_cycles": 1,   "programmable": false, "transfer_mode": "implicit"},
    {"name": "shared_memory", "arke_role": "fast_memory",  "size_per_cu": 49152,  "latency_cycles": 20,  "bandwidth_gbps": 19000, "programmable": true, "transfer_mode": "implicit"},
    {"name": "l2_cache",      "arke_role": "l2_cache",     "size_total": 3145728, "latency_cycles": 200, "bandwidth_gbps": 2000, "programmable": false, "transfer_mode": "implicit"},
    {"name": "hbm",           "arke_role": "global",       "latency_cycles": 500, "bandwidth_gbps": 336, "programmable": false, "transfer_mode": "implicit"}
  ],
  "constraints": {
    "max_fast_memory_per_block": 49152,
    "max_local_memory_per_thread": 255,
    "max_parallel_inner": 1024,
    "warp_or_vector_width": 32
  },
  "programming_model": "simt_block",
  "data_transfer": "implicit"
}
```

**huawei_ascend_a3.json：**
```json
{
  "name": "huawei_ascend_a3",
  "vendor": "huawei",
  "arch": "ascend_a3",
  "compute_units": {
    "count": 32,
    "name": "ai_core",
    "matrix_unit": {
      "name": "cube_unit",
      "tile_shapes": [[16, 16, 16]],
      "dtypes": ["f16", "f32", "i8"],
      "peak_tflops_f16": 640
    },
    "vector_unit": {
      "name": "vector_unit",
      "width_bits": 2048
    }
  },
  "memory_hierarchy": [
    {"name": "register",   "arke_role": "register",     "size_per_cu": 0,       "latency_cycles": 1,   "programmable": false, "transfer_mode": "implicit"},
    {"name": "l0_buffer",  "arke_role": "local_memory", "size_per_cu": 65536,   "latency_cycles": 1,   "programmable": true,  "transfer_mode": "explicit_dma"},
    {"name": "l1_buffer",  "arke_role": "fast_memory",  "size_per_cu": 1048576, "latency_cycles": 10,  "bandwidth_gbps": 48000, "programmable": true, "transfer_mode": "explicit_dma"},
    {"name": "l2_cache",   "arke_role": "l2_cache",     "size_total": 67108864, "latency_cycles": 100, "bandwidth_gbps": 12000, "programmable": false, "transfer_mode": "implicit"},
    {"name": "hbm",        "arke_role": "global",        "latency_cycles": 300, "bandwidth_gbps": 3200, "programmable": false, "transfer_mode": "explicit_dma"}
  ],
  "constraints": {
    "max_fast_memory_per_block": 1048576,
    "max_local_memory_per_thread": 65536,
    "max_parallel_inner": 0,
    "warp_or_vector_width": 256
  },
  "programming_model": "spmd_core",
  "data_transfer": "explicit_dma"
}
```

---

## 四、Strategy IR 的多硬件设计

### 4.1 设计原则

Strategy IR 的决策分为两类：

| 类别 | 含义 | 跨硬件？ | 示例 |
|------|------|:---:|------|
| **算法决策** | 选择什么算法/融合模式 | ✅ 是 | online softmax, epilogue fusion |
| **映射决策** | 如何映射到硬件资源 | ❌ 否 | tile size, parallel mapping, memory placement |

**LLM 在做决策时用的是 Arke 抽象概念（fast_memory, parallel_outer 等），不是硬件原生概念（shared_memory, blockIdx 等）。**

### 4.2 统一的决策 vocabulary

```yaml
# 算法决策（硬件无关）
algorithm:
  softmax: "online" | "standard" | "flash"
  # 更多算法选择...

fuse:
  type: "epilogue" | "prologue" | "vertical" | "horizontal"
  ops: [op_ids]

# 映射决策（用 Arke 抽象，后端翻译为硬件原生）
tile:
  loop: str
  factors: int[]
  # LLM 不需要知道 tile 大小对应的是 threadIdx 还是 DMA block

parallel:
  loops: str[]
  mapping:
    loop_name: "parallel_outer.x" | "parallel_outer.y" | "parallel_outer.z" | "parallel_inner"
  # 后端翻译：
  #   NVIDIA: parallel_outer.x → blockIdx.x, parallel_inner → threadIdx 内循环
  #   Ascend: parallel_outer.x → ai_core_id, parallel_inner → 隐式向量化

place:
  tensor: str
  memory: "fast_memory" | "local_memory" | "global"
  transfer: "async" | "sync" | "double_buffer" | "prefetch"
  # 后端翻译：
  #   NVIDIA: fast_memory → shared, async → cp.async
  #   Ascend: fast_memory → l1_buffer, async → DMA async

compute:
  op: str
  unit: "matrix_unit" | "vector_unit" | "scalar"
  # 后端翻译：
  #   NVIDIA: matrix_unit → tensor core (tl.dot)
  #   Ascend: matrix_unit → cube unit
```

### 4.3 后端翻译映射表

```python
# arke/backend/hal.py

BACKEND_TRANSLATION = {
    "nvidia": {
        "parallel_outer": {"x": "blockIdx.x", "y": "blockIdx.y", "z": "blockIdx.z"},
        "parallel_inner": "threadIdx",
        "fast_memory": "shared_memory",
        "local_memory": "register",
        "matrix_unit": "tensor_core",
        "vector_unit": "cuda_core",
        "async_transfer": "cp.async",
        "sync": "__syncthreads()",
        "double_buffer": "triton double_buffer"
    },
    "huawei": {
        "parallel_outer": {"x": "ai_core_id", "y": "block_id", "z": "core_group_id"},
        "parallel_inner": "implicit_vectorization",
        "fast_memory": "l1_buffer",
        "local_memory": "l0_buffer",
        "matrix_unit": "cube_unit",
        "vector_unit": "vector_unit",
        "async_transfer": "DMA_async",
        "sync": "pipe.Drain()",
        "double_buffer": "DataPipe double_buffer"
    }
}
```

---

## 五、可插拔后端架构

### 5.1 Backend 接口定义

```python
# arke/backend/base.py

from abc import ABC, abstractmethod

class ArkeBackend(ABC):
    """所有硬件后端的基类"""

    @property
    @abstractmethod
    def vendor(self) -> str:
        """e.g., 'nvidia', 'huawei'"""

    @property
    @abstractmethod
    def name(self) -> str:
        """e.g., 'triton', 'ascendc'"""

    @abstractmethod
    def translate_strategy(self, strategy_ir: dict, hw_profile: dict) -> dict:
        """将 Arke 抽象 strategy 翻译为硬件原生参数
        
        例如：
          place(tensor="A_tile", memory="fast_memory")
          → NVIDIA: tl.load to shared_memory
          → Ascend: DMA to l1_buffer
        """

    @abstractmethod
    def generate_code(self, semantic_ir: dict, translated_strategy: dict) -> str:
        """生成目标平台代码
        
        NVIDIA → Triton Python 代码
        Ascend → AscendC C++ 代码
        """

    @abstractmethod
    def compile(self, code: str) -> 'CompiledKernel':
        """编译代码为可执行 kernel"""

    @abstractmethod
    def run(self, kernel: 'CompiledKernel', inputs: list) -> list:
        """执行 kernel"""

    @abstractmethod
    def profile(self, kernel: 'CompiledKernel', inputs: list,
                warmup: int = 5, runs: int = 20) -> 'ProfileResult':
        """性能 profiling"""

    @abstractmethod
    def get_baseline(self, semantic_ir: dict, inputs: list) -> 'ProfileResult':
        """获取 vendor baseline 性能（cuBLAS / Ascend CANN kernel library）"""

    @abstractmethod
    def generate_numpy_reference(self, semantic_ir: dict) -> callable:
        """生成 NumPy 参考实现（硬件无关，但放在 backend 方便复用）"""
```

### 5.2 NVIDIA Triton 后端

```python
# arke/backend/triton_backend.py

class TritonBackend(ArkeBackend):
    vendor = "nvidia"
    name = "triton"

    def translate_strategy(self, strategy_ir, hw_profile):
        """将 Arke 抽象翻译为 Triton 参数"""
        translated = {"codegen_params": {}}
        for decision in strategy_ir["decisions"]:
            if decision["kind"] == "tile":
                # tile factors 直接映射为 Triton BLOCK 常量
                loop = decision["params"]["loop"]
                factors = decision["params"]["factors"]
                translated["codegen_params"][f"BLOCK_{loop.upper()}"] = factors[0]

            elif decision["kind"] == "parallel":
                # parallel_outer → grid dims
                mapping = decision["params"]["mapping"]
                translated["grid_mapping"] = {
                    loop: self._translate_parallel(target)
                    for loop, target in mapping.items()
                }

            elif decision["kind"] == "place":
                # fast_memory → shared_memory (Triton 隐式管理)
                memory = decision["params"]["memory"]
                translated["codegen_params"][f"use_shared_{decision['params']['tensor']}"] = (
                    memory == "fast_memory"
                )

            elif decision["kind"] == "fuse":
                translated["codegen_params"]["fused_epilogue"] = decision["params"].get("ops", [])[-1]

        return translated

    def generate_code(self, semantic_ir, translated_strategy):
        """从 Triton Jinja2 模板生成代码"""
        pattern = self._detect_pattern(semantic_ir)  # "matmul", "softmax", etc.
        template = self._load_template(pattern)
        return template.render(**translated_strategy["codegen_params"])

    def compile(self, code):
        """动态编译 Triton 代码"""
        # exec(code) → 提取 kernel 函数 → 返回 CompiledKernel

    def profile(self, kernel, inputs, warmup=5, runs=20):
        """使用 torch.cuda.Event 做 profiling"""

    def get_baseline(self, semantic_ir, inputs):
        """调用 cuBLAS / PyTorch 作为 baseline"""
```

### 5.3 Ascend AscendC 后端

```python
# arke/backend/ascendc_backend.py

class AscendCBackend(ArkeBackend):
    vendor = "huawei"
    name = "ascendc"

    def translate_strategy(self, strategy_ir, hw_profile):
        """将 Arke 抽象翻译为 AscendC 参数"""
        translated = {"codegen_params": {}}
        for decision in strategy_ir["decisions"]:
            if decision["kind"] == "tile":
                loop = decision["params"]["loop"]
                factors = decision["params"]["factors"]
                # Ascend tiling: 对应 DMA 传输块大小
                translated["codegen_params"][f"TILE_{loop.upper()}"] = factors[0]

            elif decision["kind"] == "parallel":
                mapping = decision["params"]["mapping"]
                translated["core_mapping"] = {
                    loop: self._translate_to_core_mapping(target)
                    for loop, target in mapping.items()
                }

            elif decision["kind"] == "place":
                memory = decision["params"]["memory"]
                transfer = decision["params"].get("transfer", "sync")
                # fast_memory → L1 buffer + DMA 配置
                translated["codegen_params"][f"buffer_{decision['params']['tensor']}"] = {
                    "level": "l1_buffer" if memory == "fast_memory" else "l0_buffer",
                    "dma_mode": "async" if transfer in ["async", "prefetch"] else "sync",
                    "double_buffer": transfer == "double_buffer"
                }

            elif decision["kind"] == "compute":
                unit = decision["params"].get("unit", "matrix_unit")
                translated["codegen_params"]["use_cube"] = (unit == "matrix_unit")

        return translated

    def generate_code(self, semantic_ir, translated_strategy):
        """生成 AscendC C++ 代码"""
        pattern = self._detect_pattern(semantic_ir)
        template = self._load_template(pattern)  # AscendC Jinja2 模板
        return template.render(**translated_strategy["codegen_params"])

    def compile(self, code):
        """调用 CANN 编译器编译 AscendC 代码"""
        # 写入 .cpp → ascendc compile → .o → 加载

    def profile(self, kernel, inputs, warmup=5, runs=20):
        """Ascend profiling"""

    def get_baseline(self, semantic_ir, inputs):
        """调用 CANN kernel library 作为 baseline"""
```

### 5.4 后端注册表

```python
# arke/backend/registry.py

BACKEND_REGISTRY: dict[str, type[ArkeBackend]] = {}

def register_backend(backend_class: type[ArkeBackend]):
    key = f"{backend_class.vendor}_{backend_class.name}"
    BACKEND_REGISTRY[key] = backend_class

def get_backend(hw_profile: dict) -> ArkeBackend:
    """根据 HW Profile 自动选择后端"""
    vendor = hw_profile["vendor"]
    if vendor == "nvidia":
        return BACKEND_REGISTRY["nvidia_triton"]()
    elif vendor == "huawei":
        return BACKEND_REGISTRY["huawei_ascendc"]()
    else:
        raise ValueError(f"No backend for vendor: {vendor}")

# 注册
register_backend(TritonBackend)
register_backend(AscendCBackend)
```

---

## 六、ArkeEnv 的多硬件适配

### 6.1 ArkeEnv 不直接依赖任何后端

```python
class ArkeEnv:
    def __init__(self, semantic_ir, hw_profile):
        self.semantic = semantic_ir
        self.hw = hw_profile
        self.backend = get_backend(hw_profile)   # 自动选择后端
        self.strategy = StrategyIR(target_hw=hw_profile["name"])
        # ...

    def apply_decision(self, kind, params, rationale):
        # 1. 用 Arke 抽象验证合法性（硬件无关 + 硬件感知）
        validation = self.validator.validate(self.semantic, self.strategy, self.hw)

        # 2. 添加到 strategy（用 Arke 抽象词汇）
        self.strategy.add_decision(kind, params, rationale)

        return {"success": validation.pass_, ...}

    def compile_and_profile(self):
        # 1. 后端翻译 strategy → 硬件原生参数
        translated = self.backend.translate_strategy(self.strategy.to_dict(), self.hw)

        # 2. 后端生成代码
        code = self.backend.generate_code(self.semantic, translated)

        # 3. 后端编译
        kernel = self.backend.compile(code)

        # 4. 后端执行 + profiling
        inputs = self._generate_inputs()
        result = self.backend.profile(kernel, inputs)

        # 5. 后端获取 baseline
        baseline = self.backend.get_baseline(self.semantic, inputs)

        return {"performance": result, "vs_baseline": result / baseline}
```

### 6.2 合法动作枚举的硬件感知

```python
class LegalActionsEngine:
    def enumerate_tiling(self, semantic, strategy, hw):
        """不同硬件的合法 tiling 范围不同"""
        max_fast_mem = hw["constraints"]["max_fast_memory_per_block"]

        # NVIDIA shared = 48KB → tile 较小
        # Ascend L1 = 1MB → tile 可以更大
        candidates = []
        for loop in self._get_untiled_loops(semantic, strategy):
            loop_bound = self._get_loop_bound(semantic, loop)
            for factor in self._power_of_2_factors(loop_bound):
                mem_estimate = self._estimate_memory_for_tile(semantic, loop, factor)
                if mem_estimate <= max_fast_mem:
                    candidates.append(TileAction(loop=loop, factors=[factor]))

        return candidates

    def enumerate_placement(self, semantic, strategy, hw):
        """不同硬件的内存层级选项不同"""
        available_levels = [
            level["arke_role"]
            for level in hw["memory_hierarchy"]
            if level.get("programmable", False)
        ]
        # NVIDIA: ["fast_memory"] (shared)
        # Ascend: ["fast_memory", "local_memory"] (L1, L0)

        # 数据传输方式也不同
        transfer_mode = hw.get("data_transfer", "implicit")
        if transfer_mode == "explicit_dma":
            transfer_options = ["sync", "async", "double_buffer", "prefetch"]
        else:
            transfer_options = ["sync", "async"]  # Triton 简化

        # ...
```

---

## 七、LLM 视角：硬件差异如何体现

### 7.1 LLM 看到的是 Arke 抽象

LLM 做决策时**不需要知道**它在优化 NVIDIA 还是 Ascend。它看到的是：

```json
{
  "hw_summary": {
    "fast_memory_per_unit": 49152,      // NVIDIA: 48KB; Ascend 会显示 1048576 (1MB)
    "matrix_unit_available": true,
    "matrix_unit_tile": [16, 8, 16],
    "data_transfer": "implicit",        // NVIDIA; Ascend 会显示 "explicit_dma"
    "peak_tflops_f16": 21.7
  }
}
```

### 7.2 硬件差异通过 hint 体现

`list_legal_actions()` 返回的 hint 会因硬件不同而不同：

**NVIDIA：**
```json
{
  "hint": "fast_memory is 48KB. For compute-bound matmul, typical good tiles: i=64-128, j=64-256, k=32-64. Aim for occupancy > 0.5."
}
```

**Ascend：**
```json
{
  "hint": "fast_memory is 1MB (20x larger than GPU shared). For compute-bound matmul, use larger tiles: i=128-512, j=128-512, k=64-128. Explicit DMA: use double_buffer or prefetch for data transfer overlap."
}
```

### 7.3 同一个 kernel，LLM 的决策自然不同

**matmul 在 NVIDIA 上的典型 LLM 决策：**
```
tile(i=[64], j=[128], k=[32])    — 受 48KB shared memory 约束
place(A_tile, fast_memory, async)  — Triton 隐式搬运
parallel(i_outer→outer.y, j_outer→outer.x)
```

**matmul 在 Ascend 上的典型 LLM 决策：**
```
tile(i=[256], j=[256], k=[64])   — 1MB L1 允许大 tile
place(A_tile, fast_memory, double_buffer)  — 显式 DMA + 双缓冲
place(acc, local_memory)          — L0 buffer 做累加
parallel(batch→outer.x, head→outer.y)
```

**LLM 自然会做出不同决策，因为 `list_legal_actions()` 返回的候选项和 `get_hw_profile()` 返回的参数不同。不需要给 LLM 硬编码"NVIDIA 要怎样、Ascend 要怎样"。**

---

## 八、Codegen 模板的多后端设计

### 8.1 模板目录结构

```
arke/backend/
├── base.py                    # Backend 抽象基类
├── registry.py                # 后端注册表
├── hal.py                     # 硬件抽象翻译表
├── triton/                    # NVIDIA Triton 后端
│   ├── backend.py
│   ├── compiler.py
│   ├── profiler.py
│   └── templates/
│       ├── matmul.py.j2
│       ├── matmul_fused.py.j2
│       ├── softmax.py.j2
│       └── _common.py.j2
├── ascendc/                   # Huawei AscendC 后端
│   ├── backend.py
│   ├── compiler.py
│   ├── profiler.py
│   └── templates/
│       ├── matmul.cpp.j2
│       ├── matmul_fused.cpp.j2
│       ├── softmax.cpp.j2
│       ├── _common.cpp.j2
│       └── host_wrapper.cpp.j2    # AscendC Host 侧调用代码
└── llm_gen/                   # LLM 生成后端（实验性，两种硬件共用）
    ├── generator.py
    └── prompts/
        ├── triton_codegen.md
        └── ascendc_codegen.md
```

### 8.2 AscendC matmul 模板骨架

```cpp
// templates/matmul.cpp.j2 — AscendC matmul kernel 模板
#include "ascendc.h"
using namespace ascendc;

template <typename T>
class MatmulKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c,
        uint32_t M, uint32_t N, uint32_t K)
    {
        this->a_gm = a; this->b_gm = b; this->c_gm = c;
        this->M = M; this->N = N; this->K = K;
    }

    __aicore__ inline void Process() {
        uint32_t coreId = GetBlockId();
        uint32_t coreNum = GetBlockNum();

        // Tiling: 每个 core 处理 M 维度的一个分块
        uint32_t tileM = {{ TILE_I }};
        uint32_t tileN = {{ TILE_J }};
        uint32_t tileK = {{ TILE_K }};

        uint32_t mStart = coreId * tileM;
        if (mStart >= M) return;

        // 申请 L1 buffer
        LocalTensor<T> aLocal = AllocTensor<T>(tileM * tileK);
        LocalTensor<T> bLocal = AllocTensor<T>(tileK * tileN);
        LocalTensor<T> cLocal = AllocTensor<T>(tileM * tileN);

        // 初始化累加器
        // ...

        for (uint32_t k = 0; k < K; k += tileK) {
            {% if buffer_A_tile.dma_mode == "async" %}
            // 异步 DMA: global → L1
            DataCopy(aLocal, a_gm + mStart * K + k, tileM * tileK);
            DataCopy(bLocal, b_gm + k * N, tileK * tileN);
            {% endif %}

            // Cube 矩阵乘
            {% if use_cube %}
            Mmad(cLocal, aLocal, bLocal, tileM, tileN, tileK);
            {% endif %}
        }

        {% if fused_epilogue == "relu" %}
        // 融合 ReLU
        Relu(cLocal, cLocal, tileM * tileN);
        {% endif %}

        // 结果写回 global
        DataCopy(c_gm + mStart * N, cLocal, tileM * tileN);

        FreeTensor(aLocal);
        FreeTensor(bLocal);
        FreeTensor(cLocal);
    }

private:
    GM_ADDR a_gm, b_gm, c_gm;
    uint32_t M, N, K;
};

extern "C" __global__ __aicore__ void matmul_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR params)
{
    auto* p = reinterpret_cast<uint32_t*>(params);
    MatmulKernel<half> op;
    op.Init(a, b, c, p[0], p[1], p[2]);
    op.Process();
}
```

---

## 九、Plan v2.1 修正：多后端任务补充

### Phase 1（Week 1-8）：NVIDIA 优先 + Ascend 骨架

| 新增任务 | Week | 说明 |
|----------|:----:|------|
| 统一 HW Profile Schema | W1 | 两种硬件的 JSON profile |
| Backend 抽象基类 | W1 | `ArkeBackend` ABC |
| 后端注册表 | W1 | `registry.py` |
| Strategy IR 统一词汇表 | W1 | Arke 抽象 → 硬件原生映射 |
| Ascend HW Profile | W1 | `huawei_ascend_a3.json` |
| NVIDIA 后端完整实现 | W2-4 | Triton codegen + compile + profile |
| 合法动作引擎——硬件感知 | W2 | 根据 HW Profile 过滤候选 |
| AscendC matmul 模板 | W5 | 骨架代码（可能无法实际编译） |
| AscendC backend 骨架 | W5 | compile/profile 为 stub |

### Phase 2（Week 9-16，MVP 之后）：Ascend 全面落地

| 任务 | 说明 |
|------|------|
| 获取 Ascend A3 硬件/远程环境 | 实际硬件或华为云 |
| AscendC 完整 codegen | matmul + softmax |
| AscendC compile + profile | 对接 CANN 编译器 |
| 跨硬件评估 | 同一个 LLM session，NVIDIA vs Ascend 的优化效果 |
| AscendC LLM 生成实验 | 路径 B 在 AscendC 上的可行性 |

### 为什么 Ascend 在 Phase 1 只做骨架

1. **当前没有 Ascend 硬件**——WSL2 上有 RTX 3060，没有 Ascend NPU
2. **CANN SDK 需要 Ascend 环境**——无法在 x86+NVIDIA 上编译 AscendC
3. **但架构必须现在设计对**——否则 Phase 2 接入时要大改

**Phase 1 的 Ascend 目标**：
- ✅ HW Profile 定义完整
- ✅ Strategy IR 用 Arke 抽象（不绑定 NVIDIA）
- ✅ 合法动作引擎能为 Ascend 枚举合理的 tile/place 选项
- ✅ AscendC 模板可以生成代码（但无法编译执行）
- ✅ LLM 可以为 Ascend 目标做 tool-use 优化（只是最后 compile_and_profile 会返回 "backend not available"）

---

## 十、更新后的项目结构

```
arke/
├── arke/
│   ├── ir/
│   │   ├── targets/                    # HW Profile
│   │   │   ├── hw_profile.schema.json  # 统一 schema
│   │   │   ├── nvidia_ampere_rtx3060.json
│   │   │   └── huawei_ascend_a3.json
│   │   └── ...
│   ├── engine/
│   │   ├── env.py                      # ArkeEnv（硬件无关）
│   │   ├── legal_actions.py            # 硬件感知的合法动作枚举
│   │   └── ...
│   ├── backend/                        # 可插拔后端
│   │   ├── base.py                     # ArkeBackend ABC
│   │   ├── registry.py                 # 后端注册表
│   │   ├── hal.py                      # 抽象→原生翻译表
│   │   ├── triton/                     # NVIDIA 后端
│   │   │   ├── backend.py
│   │   │   ├── compiler.py
│   │   │   ├── profiler.py
│   │   │   └── templates/
│   │   │       ├── matmul.py.j2
│   │   │       └── softmax.py.j2
│   │   ├── ascendc/                    # Ascend 后端
│   │   │   ├── backend.py
│   │   │   ├── compiler.py             # Phase 1: stub
│   │   │   ├── profiler.py             # Phase 1: stub
│   │   │   └── templates/
│   │   │       ├── matmul.cpp.j2
│   │   │       └── softmax.cpp.j2
│   │   └── llm_gen/                    # LLM 生成后端
│   │       └── generator.py
│   └── ...
```

---

## 十一、关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| Strategy IR 用抽象还是原生词汇？ | **Arke 抽象** | LLM 不应感知硬件差异；后端做翻译 |
| 后端如何选择？ | **根据 HW Profile 自动** | `get_backend(hw_profile)` |
| Ascend Phase 1 做到什么程度？ | **架构 + 模板 + stub** | 无硬件不能编译，但架构不能欠债 |
| LLM 需要为不同硬件用不同 prompt？ | **不需要** | 同一个 prompt + tool schema；差异通过 hw_profile 和 legal_actions 体现 |
| 合法动作是否硬件感知？ | **是** | tile 范围、placement 选项、transfer 模式因硬件而异 |
| 验证系统是否多后端？ | **V0/V1 硬件无关，V2 后端专属** | 静态/数值验证用 NumPy；性能验证必须用目标硬件 |

---

*文档版本：v1.0 | 创建日期：2026-03-31*
*补充 v2.1 plan 的多硬件后端设计缺失*